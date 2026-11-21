"""
Tizim hisoblari (sign in / sign up) va rollar.

Rollar:
  admin    - barcha imkoniyatlar + hisoblarni boshqarish
  director - faqat Dashboard
  user     - faqat jonli kamera

Sign up orqali yaratilgan hisob "user" rolini oladi; rolni faqat admin o'zgartiradi.
Eng birinchi yaratilgan hisob avtomatik "admin" bo'ladi.
Parollar werkzeug (scrypt/pbkdf2) bilan xeshlanadi, ochiq holda saqlanmaydi.
"""
from __future__ import annotations

import re
import secrets
import threading
import time
from functools import wraps
from pathlib import Path

from flask import current_app, g, jsonify, request, session
from itsdangerous import BadSignature, URLSafeTimedSerializer
from werkzeug.security import check_password_hash, generate_password_hash

ROLES = {"admin": "Administrator", "director": "Direktor", "user": "Foydalanuvchi"}
# har bir rol ko'ra oladigan sahifalar (frontend shu ro'yxat bo'yicha menyuni quradi)
PAGES = {
    "admin": ["camera", "register", "users", "dashboard", "settings", "accounts"],
    "director": ["dashboard"],
    "user": ["camera"],
}
USERNAME_RE = re.compile(r"^[A-Za-z0-9_.-]{3,32}$")
MAX_FAILS, LOCK_SECONDS = 5, 300
TOKEN_MAX_AGE = 12 * 3600  # token amal qilish muddati (s)

SCHEMA = """
CREATE TABLE IF NOT EXISTS accounts (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    username TEXT NOT NULL UNIQUE COLLATE NOCASE,
    full_name TEXT NOT NULL,
    password_hash TEXT NOT NULL,
    role TEXT NOT NULL DEFAULT 'user' CHECK (role IN ('admin','director','user')),
    active INTEGER NOT NULL DEFAULT 1,
    created_at TEXT NOT NULL,
    last_login TEXT
);
"""


def secret_key(path: Path) -> bytes:
    """Sessiya kalitini data/secret.key faylida saqlaydi (server qayta ishga tushganda ham sessiyalar saqlanadi)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.is_file():
        path.write_bytes(secrets.token_bytes(32))
        try:
            path.chmod(0o600)
        except OSError:
            pass
    return path.read_bytes()


class Accounts:
    def __init__(self, db):
        self.db = db
        self._fails: dict[str, tuple[int, float]] = {}
        self._flock = threading.Lock()
        with db.lock:
            db.conn.executescript(SCHEMA)
            db.conn.commit()

    # --- yordamchi ---
    def _q(self, sql, args=()):
        with self.db.lock:
            return self.db.conn.execute(sql, args).fetchall()

    def _exec(self, sql, args=()):
        with self.db.lock:
            cur = self.db.conn.execute(sql, args)
            self.db.conn.commit()
            return cur

    @staticmethod
    def public(row) -> dict:
        r = dict(row)
        r.pop("password_hash", None)
        r["active"] = bool(r["active"])
        r["role_label"] = ROLES.get(r["role"], r["role"])
        return r

    def count(self) -> int:
        return self._q("SELECT COUNT(*) FROM accounts")[0][0]

    def get(self, uid: int):
        rows = self._q("SELECT * FROM accounts WHERE id=?", (uid,))
        return rows[0] if rows else None

    def admins(self) -> int:
        return self._q("SELECT COUNT(*) FROM accounts WHERE role='admin' AND active=1")[0][0]

    # --- sign up / sign in ---
    def register(self, username: str, full_name: str, password: str) -> tuple[dict | None, str | None]:
        username, full_name = (username or "").strip(), (full_name or "").strip()
        if not USERNAME_RE.match(username):
            return None, "Login 3-32 belgi bo'lsin: lotin harflari, raqamlar, _ . -"
        if len(full_name) < 2:
            return None, "Ism-familiyani kiriting"
        if len(password or "") < 6:
            return None, "Parol kamida 6 belgidan iborat bo'lsin"
        with self.db.lock:
            if self._q("SELECT 1 FROM accounts WHERE username=?", (username,)):
                return None, "Bu login band"
            role = "admin" if self.count() == 0 else "user"
            cur = self._exec(
                "INSERT INTO accounts (username, full_name, password_hash, role, created_at) VALUES (?, ?, ?, ?, ?)",
                (username, full_name, generate_password_hash(password), role, time.strftime("%Y-%m-%d %H:%M:%S")))
        return self.public(self.get(cur.lastrowid)), None

    def login(self, username: str, password: str, client: str) -> tuple[dict | None, str | None]:
        key = f"{client}|{(username or '').lower()}"
        with self._flock:
            n, until = self._fails.get(key, (0, 0.0))
            if until > time.time():
                return None, f"Ko'p noto'g'ri urinish. {int(until - time.time()) // 60 + 1} daqiqadan so'ng qayta urining"
        rows = self._q("SELECT * FROM accounts WHERE username=?", ((username or "").strip(),))
        row = rows[0] if rows else None
        if row is None or not check_password_hash(row["password_hash"], password or ""):
            with self._flock:
                n += 1
                self._fails[key] = (0, time.time() + LOCK_SECONDS) if n >= MAX_FAILS else (n, 0.0)
            return None, "Login yoki parol noto'g'ri"
        if not row["active"]:
            return None, "Hisobingiz bloklangan. Administratorga murojaat qiling"
        with self._flock:
            self._fails.pop(key, None)
        self._exec("UPDATE accounts SET last_login=? WHERE id=?", (time.strftime("%Y-%m-%d %H:%M:%S"), row["id"]))
        return self.public(self.get(row["id"])), None

    # --- admin boshqaruvi ---
    def list(self) -> list[dict]:
        return [self.public(r) for r in self._q("SELECT * FROM accounts ORDER BY id")]

    def update(self, uid: int, me: int, role: str | None = None, active: bool | None = None,
               password: str | None = None) -> str | None:
        row = self.get(uid)
        if row is None:
            return "Hisob topilmadi"
        demoting = (role is not None and role != "admin") or active is False
        if row["role"] == "admin" and row["active"] and demoting and self.admins() <= 1:
            return "Tizimda kamida bitta faol administrator qolishi kerak"
        if uid == me and demoting:
            return "O'zingizning admin huquqingizni olib tashlay olmaysiz"
        if role is not None:
            if role not in ROLES:
                return "Noto'g'ri rol"
            self._exec("UPDATE accounts SET role=? WHERE id=?", (role, uid))
        if active is not None:
            self._exec("UPDATE accounts SET active=? WHERE id=?", (int(bool(active)), uid))
        if password is not None:
            if len(password) < 6:
                return "Parol kamida 6 belgidan iborat bo'lsin"
            self._exec("UPDATE accounts SET password_hash=? WHERE id=?", (generate_password_hash(password), uid))
        return None

    def delete(self, uid: int, me: int) -> str | None:
        row = self.get(uid)
        if row is None:
            return "Hisob topilmadi"
        if uid == me:
            return "O'z hisobingizni o'chira olmaysiz"
        if row["role"] == "admin" and self.admins() <= 1:
            return "Oxirgi administratorni o'chirib bo'lmaydi"
        self._exec("DELETE FROM accounts WHERE id=?", (uid,))
        return None

    def change_own_password(self, uid: int, old: str, new: str) -> str | None:
        row = self.get(uid)
        if row is None or not check_password_hash(row["password_hash"], old or ""):
            return "Joriy parol noto'g'ri"
        return self.update(uid, uid, password=new)

    # --- himoya dekoratori ---
    # --- tokenlar (proksi/tunnel orqali cookie yo'qolsa ham ishlaydi) ---
    @staticmethod
    def _signer() -> URLSafeTimedSerializer:
        return URLSafeTimedSerializer(current_app.config["SECRET_KEY"], salt="faceid-auth")

    def issue_token(self, row) -> str:
        # "pv" - parol xeshining bir qismi: parol o'zgarsa, eski tokenlar bekor bo'ladi
        return self._signer().dumps({"uid": row["id"], "pv": row["password_hash"][-12:]})

    def _from_token(self, token: str):
        try:
            data = self._signer().loads(token, max_age=TOKEN_MAX_AGE)
        except BadSignature:
            return None
        row = self.get(data.get("uid"))
        if row is None or row["password_hash"][-12:] != data.get("pv"):
            return None
        return row

    def current(self):
        """Joriy foydalanuvchi: Authorization: Bearer <token>, ?t=<token> (faqat GET rasmlar uchun) yoki cookie."""
        if "auth_row" in g:
            return g.auth_row
        row = None
        auth = request.headers.get("Authorization", "")
        if auth.startswith("Bearer "):
            row = self._from_token(auth[7:].strip())
        elif request.method == "GET" and request.args.get("t"):
            row = self._from_token(request.args["t"])
        elif session.get("uid"):
            row = self.get(session["uid"])
        if row is not None and not row["active"]:
            row = None
        g.auth_row = row
        return row

    def current_id(self) -> int:
        return self.current()["id"]

    def require(self, *roles: str):
        """API'ni faqat ko'rsatilgan rollarga ochadi (rol bazadan har safar qayta o'qiladi)."""
        def deco(fn):
            @wraps(fn)
            def wrapper(*a, **kw):
                row = self.current()
                if row is None:
                    return jsonify(error="Tizimga kiring", auth=False), 401
                if roles and row["role"] not in roles:
                    return jsonify(error="Bu amal uchun ruxsatingiz yo'q"), 403
                return fn(*a, **kw)
            return wrapper
        return deco
