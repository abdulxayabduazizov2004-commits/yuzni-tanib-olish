"""
Yuz va tana tanib olish tizimi - veb server.

Brauzer kamerani o'zi ochadi (ruxsatni brauzer so'raydi), kadrlarni serverga yuboradi,
server esa tanish natijalarini JSON ko'rinishida qaytaradi.

Ishga tushirish:  .venv/bin/python server.py   ->  http://127.0.0.1:8765
"""
from __future__ import annotations

import argparse
import base64
import io
import os
import tempfile
import threading
import time
import webbrowser
from dataclasses import asdict, fields
from pathlib import Path

import cv2
import numpy as np
from flask import Flask, Response, jsonify, request, send_from_directory, session

from auth import PAGES, ROLES, Accounts, secret_key

from recognizer import DATA_DIR, EVENT_TYPES, PROFILE_FIELDS, SETTINGS_FILE, Engine, Settings

BASE = Path(__file__).resolve().parent
app = Flask(__name__, static_folder=str(BASE / "web"), static_url_path="/static")
app.config["MAX_CONTENT_LENGTH"] = 200 * 1024 * 1024

app.config.update(
    SECRET_KEY=os.environ.get("SECRET_KEY") or secret_key(DATA_DIR / "secret.key"),
    SESSION_COOKIE_HTTPONLY=True, SESSION_COOKIE_SAMESITE="Strict", PERMANENT_SESSION_LIFETIME=12 * 3600,
    # bulutda (HTTPS ortida) cookie faqat HTTPS orqali yuboriladi
    SESSION_COOKIE_SECURE=os.environ.get("SECURE_COOKIES") == "1",
)
if os.environ.get("BEHIND_PROXY") == "1":
    # Netlify proksi -> backend: haqiqiy mijoz IP'si X-Forwarded-For'da keladi (parol bloklash uchun kerak)
    from werkzeug.middleware.proxy_fix import ProxyFix
    app.wsgi_app = ProxyFix(app.wsgi_app, x_for=int(os.environ.get("PROXY_HOPS", "2")), x_proto=1, x_host=1)


@app.get("/healthz")
def healthz():
    return jsonify(ok=True, users=len(db.people) if "db" in globals() else 0)

engine = Engine(Settings.load())
db = engine.db
accounts = Accounts(db)
require = accounts.require
ADMIN, DIRECTOR, USER = "admin", "director", "user"
frame_lock = threading.Lock()
fps_state = {"t": time.time(), "n": 0, "fps": 0.0}


def _jpeg_response(img: np.ndarray | None) -> Response:
    if img is None:
        return Response(status=404)
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 88])
    return Response(buf.tobytes(), mimetype="image/jpeg", headers={"Cache-Control": "no-cache"})


def _data_url(img: np.ndarray, quality: int = 85) -> str:
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, quality])
    return "data:image/jpeg;base64," + base64.b64encode(buf.tobytes()).decode()


def _drain_events() -> list[str]:
    out = []
    while not engine.events.empty():
        out.append(engine.events.get_nowait())
    return out


def _clean_profile(data: dict) -> tuple[dict, str | None]:
    prof = {k: str(data.get(k, "") or "").strip() for k, _, _ in PROFILE_FIELDS}
    missing = [label for k, label, req in PROFILE_FIELDS if req and not prof[k]]
    if missing:
        return prof, "To'ldirilishi shart: " + ", ".join(missing)
    if prof["birth_date"]:
        try:
            time.strptime(prof["birth_date"], "%Y-%m-%d")
        except ValueError:
            return prof, "Tug'ilgan sana YYYY-MM-DD ko'rinishida bo'lsin"
    if prof["email"] and ("@" not in prof["email"] or "." not in prof["email"].split("@")[-1]):
        return prof, "Email manzili noto'g'ri"
    return prof, None


# ------------------------------------------------------------------ sahifa
@app.get("/")
def index():
    return send_from_directory(BASE / "web", "index.html")


@app.get("/api/meta")
@require()
def meta():
    return jsonify(fields=[{"key": k, "label": l, "required": r} for k, l, r in PROFILE_FIELDS],
                   event_types=EVENT_TYPES, attributes=engine.attr_labels if engine.attr else {})


# ------------------------------------------------------------------ kirish / ro'yxatdan o'tish
def _me_json(row) -> dict:
    me = accounts.public(row)
    me["pages"] = PAGES[me["role"]]
    return me


@app.get("/api/auth/me")
def auth_me():
    row = accounts.current()
    return jsonify(user=_me_json(row) if row else None, has_accounts=accounts.count() > 0)


@app.post("/api/auth/login")
def auth_login():
    data = request.get_json(force=True) or {}
    row, err = accounts.login(data.get("username"), data.get("password"), request.remote_addr or "")
    if err:
        return jsonify(error=err), 401
    session.clear()
    session.permanent = bool(data.get("remember", True))
    session["uid"] = row["id"]
    return jsonify(user=_me_json(accounts.get(row["id"])))


@app.post("/api/auth/register")
def auth_register():
    data = request.get_json(force=True) or {}
    row, err = accounts.register(data.get("username"), data.get("full_name"), data.get("password"))
    if err:
        return jsonify(error=err), 400
    session.clear()
    session.permanent = True
    session["uid"] = row["id"]
    return jsonify(user=_me_json(accounts.get(row["id"])), first_admin=row["role"] == "admin")


@app.post("/api/auth/logout")
def auth_logout():
    session.clear()
    return jsonify(ok=True)


@app.post("/api/auth/password")
@require()
def auth_password():
    data = request.get_json(force=True) or {}
    err = accounts.change_own_password(session["uid"], data.get("old"), data.get("new"))
    return (jsonify(error=err), 400) if err else jsonify(ok=True)


# ------------------------------------------------------------------ hisoblarni boshqarish (admin)
@app.get("/api/accounts")
@require(ADMIN)
def accounts_list():
    return jsonify(accounts=accounts.list(), roles=ROLES)


@app.put("/api/accounts/<int:uid>")
@require(ADMIN)
def accounts_update(uid):
    data = request.get_json(force=True) or {}
    err = accounts.update(uid, session["uid"], role=data.get("role"), active=data.get("active"),
                          password=data.get("password"))
    return (jsonify(error=err), 400) if err else jsonify(ok=True)


@app.delete("/api/accounts/<int:uid>")
@require(ADMIN)
def accounts_delete(uid):
    err = accounts.delete(uid, session["uid"])
    return (jsonify(error=err), 400) if err else jsonify(ok=True)


# ------------------------------------------------------------------ kamera kadri
@app.post("/api/frame")
@require(ADMIN, USER)
def frame():
    data = np.frombuffer(request.get_data(), np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if img is None:
        return jsonify(error="kadr o'qilmadi"), 400
    with frame_lock:
        dets = engine.process_frame(img)
        result = engine.detections_json(dets)
        reg = engine.registration_json()
        done = engine.last_registration
        if done is not None and done.done:
            engine.last_registration = None
            done = {"name": done.name, "collected": done.collected, "pid": done.result_pid}
        else:
            done = None
    fps_state["n"] += 1
    if time.time() - fps_state["t"] >= 1:
        fps_state["fps"] = fps_state["n"] / (time.time() - fps_state["t"])
        fps_state.update(t=time.time(), n=0)
    return jsonify(detections=result, registration=reg, registration_done=done,
                   events=_drain_events(), fps=round(fps_state["fps"], 1),
                   size=[img.shape[1], img.shape[0]])


@app.post("/api/analyze")
@require(ADMIN)
def analyze():
    f = request.files.get("image")
    if f is None:
        return jsonify(error="rasm yuborilmadi"), 400
    img = cv2.imdecode(np.frombuffer(f.read(), np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        return jsonify(error="rasm o'qilmadi"), 400
    with frame_lock:
        rgb, results = engine.analyze_image(img)
    return jsonify(image=_data_url(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)),
                   results=[{"name": n, "score": round(s, 3), "via": v,
                             "attrs": [engine.attr_labels.get(k, k) for k in at] if engine.attr else []}
                            for n, s, v, at in results])


# ------------------------------------------------------------------ ro'yxatdan o'tkazish
@app.post("/api/registration/start")
@require(ADMIN)
def reg_start():
    data = request.get_json(force=True) or {}
    pid = data.get("pid")
    if pid:
        if pid not in db.people:
            return jsonify(error="foydalanuvchi topilmadi"), 404
        engine.start_registration(db.name(pid), pid)
    else:
        prof, err = _clean_profile(data.get("profile") or {})
        if err:
            return jsonify(error=err), 400
        engine.start_registration(f"{prof['first_name']} {prof['last_name']}", None, prof)
    return jsonify(ok=True)


@app.post("/api/registration/cancel")
@require(ADMIN)
def reg_cancel():
    engine.cancel_registration()
    return jsonify(ok=True)


# ------------------------------------------------------------------ foydalanuvchilar
@app.get("/api/users")
@require(ADMIN)
def users():
    return jsonify(db.user_rows())


@app.post("/api/users")
@require(ADMIN)
def create_user():
    prof, err = _clean_profile(request.get_json(force=True) or {})
    if err:
        return jsonify(error=err), 400
    p = db.create(prof)
    return jsonify(id=p.pid, name=p.name)


@app.put("/api/users/<pid>")
@require(ADMIN)
def update_user(pid):
    if pid not in db.people:
        return jsonify(error="topilmadi"), 404
    prof, err = _clean_profile(request.get_json(force=True) or {})
    if err:
        return jsonify(error=err), 400
    db.update_profile(pid, prof)
    return jsonify(ok=True)


@app.delete("/api/users/<pid>")
@require(ADMIN)
def delete_user(pid):
    db.delete(pid)
    return jsonify(ok=True)


@app.get("/api/users/<pid>/photo")
@require(ADMIN, DIRECTOR, USER)
def user_photo(pid):
    return _jpeg_response(db.photo(pid))


@app.get("/api/users/<pid>/samples")
@require(ADMIN)
def user_samples(pid):
    return jsonify([{"image": _data_url(img, 80), "source": src, "created_at": c}
                    for img, src, c in db.samples(pid, 80) if img is not None])


@app.post("/api/users/<pid>/images")
@require(ADMIN)
def user_images(pid):
    if pid not in db.people:
        return jsonify(error="topilmadi"), 404
    files = request.files.getlist("images")
    with tempfile.TemporaryDirectory() as tmp:
        paths = []
        for i, f in enumerate(files):
            ext = Path(f.filename or "").suffix or ".jpg"
            path = Path(tmp) / f"{i:04d}{ext}"
            f.save(path)
            paths.append(str(path))
        added, skipped = engine.import_images(pid, paths)
    if request.args.get("delete_if_empty") and pid in db.people and not len(db.people[pid].face):
        db.delete(pid)
        return jsonify(added=0, skipped=skipped, deleted=True)
    return jsonify(added=added, skipped=skipped)


# ------------------------------------------------------------------ dashboard
@app.get("/api/dashboard")
@require(ADMIN, DIRECTOR)
def dashboard():
    etype = request.args.get("type") or None
    events = [dict(r) for r in db.recent_events(200, etype)]
    rows = db.user_rows()
    return jsonify(summary=db.summary(), daily=db.daily_counts(14), hourly=db.hourly_today(),
                   events=events, top=sorted(rows, key=lambda r: -r["seen"])[:10])


@app.get("/api/events/<int:eid>/snapshot")
@require(ADMIN, DIRECTOR)
def event_snapshot(eid):
    return _jpeg_response(db.event_snapshot(eid))


@app.get("/api/events.csv")
@require(ADMIN, DIRECTOR)
def events_csv():
    with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as t:
        path = t.name
    db.export_csv(path)
    data = Path(path).read_bytes()
    Path(path).unlink(missing_ok=True)
    return Response(data, mimetype="text/csv", headers={
        "Content-Disposition": f"attachment; filename=hodisalar_{time.strftime('%Y%m%d')}.csv"})


# ------------------------------------------------------------------ sozlamalar
@app.get("/api/settings")
@require(ADMIN)
def get_settings():
    return jsonify(asdict(engine.s))


@app.put("/api/settings")
@require(ADMIN)
def put_settings():
    data = request.get_json(force=True) or {}
    for f in fields(Settings):
        if f.name in data:
            cur = getattr(engine.s, f.name)
            setattr(engine.s, f.name, type(cur)(data[f.name]))
    SETTINGS_FILE.write_text(__import__("json").dumps(asdict(engine.s), indent=2), encoding="utf-8")
    return jsonify(asdict(engine.s))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--no-browser", action="store_true")
    a = ap.parse_args()
    url = f"http://127.0.0.1:{a.port}"
    print(f"\n  Tizim ishga tushdi:  {url}\n  To'xtatish: Ctrl+C\n")
    if not a.no_browser:
        threading.Timer(1.2, lambda: webbrowser.open(url)).start()
    try:
        app.run(host=a.host, port=a.port, threaded=True, debug=False)
    finally:
        db.close()


if __name__ == "__main__":
    main()
