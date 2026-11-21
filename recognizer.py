"""
Yuz va tana tanib olish dvigateli.

Modellar (OpenCV Zoo, ONNX):
  * YuNet   - yuzni aniqlash (detektor) + 5 ta nuqta (ko'zlar, burun, og'iz)
  * SFace   - yuzdan 128 o'lchamli "barmoq izi" (embedding) olish
  * Youtu ReID - tanadan (kiyim, gavda) 768 o'lchamli embedding olish
  * HOG     - yuz ko'rinmaganda butun tanani topish

Uzluksiz o'rganish g'oyasi:
  Har bir foydalanuvchi uchun yuz va tana namunalari "galereyasi" saqlanadi.
  Kamera odamni ishonchli tanigan sari (bir necha kadr ketma-ket, yuqori
  o'xshashlik, boshqa odamlar bilan adashish xavfi yo'q) va ushbu ko'rinish
  galereyadagi namunalardan yetarlicha farq qilsa (boshqa burchak, yorug'lik,
  soch turmagi, ko'zoynak...), u yangi namuna sifatida qo'shiladi. Galereya
  to'lsa, yuz uchun eng takroriy namuna, tana uchun eng eski namuna
  (kiyim o'zgaradi) o'chiriladi.
"""
from __future__ import annotations

import json
import os
import queue
import sqlite3
import threading
import time
import uuid
from collections import Counter, deque
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps

BASE_DIR = Path(__file__).resolve().parent
# Bulutda (Docker) yo'llarni muhit o'zgaruvchilari orqali berish mumkin: DATA_DIR=/data (doimiy disk)
MODEL_DIR = Path(os.environ.get("MODEL_DIR", BASE_DIR / "models"))
DATA_DIR = Path(os.environ.get("DATA_DIR", BASE_DIR / "data"))
SETTINGS_FILE = Path(os.environ.get("SETTINGS_FILE", BASE_DIR / "settings.json"))
DB_FILE = DATA_DIR / "database.db"

FACE_DET_MODEL = MODEL_DIR / "face_detection_yunet_2023mar.onnx"
FACE_REC_MODEL = MODEL_DIR / "face_recognition_sface_2021dec.onnx"
BODY_REID_MODEL = MODEL_DIR / "person_reid_youtu_2021nov.onnx"
MODEL_FILES = [FACE_DET_MODEL, FACE_REC_MODEL, BODY_REID_MODEL]

MAX_SAMPLE_IMAGES = 200  # har bir odam uchun bazada saqlanadigan yuz rasmlari


# --------------------------------------------------------------------------- #
# Sozlamalar
# --------------------------------------------------------------------------- #
@dataclass
class Settings:
    face_det_score: float = 0.80   # yuz detektori ishonchi
    face_match: float = 0.42       # SFace kosinus o'xshashligi: >= bo'lsa o'sha odam
    face_learn: float = 0.55       # avtomatik o'rganish uchun minimal o'xshashlik
    face_novelty: float = 0.80     # galereyadagi eng yaqin namuna bundan past => yangi ko'rinish
    face_gallery_max: int = 150
    body_match: float = 0.72       # ReID kosinus o'xshashligi
    body_novelty: float = 0.90
    body_gallery_max: int = 60
    min_face_px: int = 60          # o'rganish uchun yuzning minimal o'lchami (piksel)
    min_sharpness: float = 25.0    # Laplasian dispersiyasi (xiralik filtri)
    learn_interval: float = 1.5    # bir odamni ikki marta o'rganish orasidagi minimal vaqt (s)
    stable_frames: int = 5         # o'rganishdan oldin necha kadr ketma-ket bir xil natija
    body_enabled: bool = True
    auto_learn: bool = True
    attributes_enabled: bool = True  # ko'zoynak, kepka, soqol... aniqlash

    @classmethod
    def load(cls, path: Path = SETTINGS_FILE) -> "Settings":
        s = cls()
        if path.exists():
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
                for f in fields(cls):
                    if f.name in data:
                        setattr(s, f.name, type(getattr(s, f.name))(data[f.name]))
            except Exception as e:  # noqa: BLE001
                print(f"[ogohlantirish] settings.json o'qilmadi: {e}")
        else:
            try:
                path.write_text(json.dumps(asdict(s), indent=2), encoding="utf-8")
            except OSError:
                pass
        return s


# --------------------------------------------------------------------------- #
# Yordamchi funksiyalar
# --------------------------------------------------------------------------- #
def _normalize(v) -> np.ndarray:
    v = np.asarray(v, np.float32).reshape(-1)
    n = float(np.linalg.norm(v))
    return v / n if n > 0 else v


def _atomic_write_bytes(path: Path, data: bytes) -> None:
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "wb") as f:
        f.write(data)
    os.replace(tmp, path)



def read_image(path: str | Path) -> np.ndarray | None:
    """Unicode yo'llar va EXIF burilishini hisobga olgan holda rasmni BGR ko'rinishida o'qiydi."""
    try:
        pil = Image.open(path)
        pil = ImageOps.exif_transpose(pil).convert("RGB")
        return cv2.cvtColor(np.asarray(pil), cv2.COLOR_RGB2BGR)
    except Exception:  # noqa: BLE001
        try:
            data = np.fromfile(str(path), dtype=np.uint8)
            return cv2.imdecode(data, cv2.IMREAD_COLOR)
        except Exception:  # noqa: BLE001
            return None


def _iou(a, b) -> float:
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    x1, y1 = max(ax, bx), max(ay, by)
    x2, y2 = min(ax + aw, bx + bw), min(ay + ah, by + bh)
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    union = aw * ah + bw * bh - inter
    return inter / union if union > 0 else 0.0


_FONT_CANDIDATES = [
    "/System/Library/Fonts/Supplemental/Arial Unicode.ttf",
    "/Library/Fonts/Arial Unicode.ttf",
    "/System/Library/Fonts/Supplemental/Arial.ttf",
    "/System/Library/Fonts/Helvetica.ttc",
    "C:/Windows/Fonts/arial.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
]
_font_cache: dict[int, ImageFont.ImageFont] = {}


def _font(size: int):
    if size not in _font_cache:
        font = None
        for p in _FONT_CANDIDATES:
            if os.path.exists(p):
                try:
                    font = ImageFont.truetype(p, size)
                    break
                except OSError:
                    continue
        _font_cache[size] = font or ImageFont.load_default(size=size)
    return _font_cache[size]


# --------------------------------------------------------------------------- #
# Ma'lumotlar bazasi (SQLite: data/database.db)
# --------------------------------------------------------------------------- #
PROFILE_FIELDS = [  # (maydon, sarlavha, majburiy)
    ("last_name", "Familiya", True),
    ("first_name", "Ism", True),
    ("middle_name", "Otasining ismi", False),
    ("birth_date", "Tug'ilgan sana (YYYY-MM-DD)", False),
    ("gender", "Jinsi", False),
    ("phone", "Telefon", False),
    ("email", "Email", False),
    ("department", "Bo'lim / guruh", False),
    ("position", "Lavozim / maqom", False),
    ("employee_id", "ID raqam (xodim/talaba)", False),
    ("notes", "Izoh", False),
]

SCHEMA = """
PRAGMA journal_mode=WAL;
PRAGMA foreign_keys=ON;
CREATE TABLE IF NOT EXISTS users (
    id TEXT PRIMARY KEY,
    first_name TEXT NOT NULL, last_name TEXT NOT NULL, middle_name TEXT DEFAULT '',
    birth_date TEXT DEFAULT '', gender TEXT DEFAULT '', phone TEXT DEFAULT '', email TEXT DEFAULT '',
    department TEXT DEFAULT '', position TEXT DEFAULT '', employee_id TEXT DEFAULT '', notes TEXT DEFAULT '',
    photo BLOB,
    created_at TEXT NOT NULL, updated_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS embeddings (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id TEXT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    kind TEXT NOT NULL CHECK (kind IN ('face','body')),
    vector BLOB NOT NULL, source TEXT NOT NULL, created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS ix_emb_user ON embeddings(user_id, kind);
CREATE TABLE IF NOT EXISTS samples (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id TEXT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    image BLOB NOT NULL, source TEXT NOT NULL, created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS ix_samples_user ON samples(user_id);
CREATE TABLE IF NOT EXISTS events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id TEXT REFERENCES users(id) ON DELETE SET NULL,
    user_name TEXT, type TEXT NOT NULL, method TEXT DEFAULT '', score REAL,
    note TEXT DEFAULT '', snapshot BLOB, created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS ix_events_time ON events(created_at);
CREATE INDEX IF NOT EXISTS ix_events_user ON events(user_id);
"""

EVENT_TYPES = {
    "registered": "Ro'yxatdan o'tdi",
    "recognized": "Aniqlandi",
    "unknown": "Noma'lum shaxs",
    "learned": "O'rgandi",
    "updated": "Tahrirlandi",
    "deleted": "O'chirildi",
    "imported": "Rasmdan o'rgatildi",
}


def _now_str() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def _jpeg(img: np.ndarray | None, quality: int = 90) -> bytes | None:
    if img is None:
        return None
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, quality])
    return buf.tobytes() if ok else None


def decode_jpeg(data: bytes | None) -> np.ndarray | None:
    if not data:
        return None
    return cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)


class Person:
    """Xotiradagi nusxa: profil + yuz/tana galereyalari (tez qidirish uchun)."""

    def __init__(self, pid: str, profile: dict, face_dim: int, body_dim: int):
        self.pid = pid
        self.profile = {k: (profile.get(k) or "") for k, _, _ in PROFILE_FIELDS}
        self.created = profile.get("created_at") or _now_str()
        self.face = np.zeros((0, face_dim), np.float32)
        self.body = np.zeros((0, body_dim), np.float32)
        self.face_ids: list[int] = []
        self.body_ids: list[int] = []
        self.last_face_learn = 0.0
        self.last_body_learn = 0.0

    @property
    def name(self) -> str:
        p = self.profile
        return f"{p['first_name']} {p['last_name']}".strip() or "Nomsiz"


class Database:
    """Barcha ma'lumotlar bitta SQLite faylida: foydalanuvchilar, embeddinglar,
    namuna rasmlar va hodisalar (aniqlash tarixi)."""

    def __init__(self, path: Path, face_dim: int, body_dim: int, settings: Settings):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self.face_dim, self.body_dim = face_dim, body_dim
        self.s = settings
        self.lock = threading.RLock()
        self.conn = sqlite3.connect(str(path), check_same_thread=False, timeout=10)
        self.conn.row_factory = sqlite3.Row
        self.conn.executescript(SCHEMA)
        self.people: dict[str, Person] = {}
        self.version = 0
        self._cache: dict[str, tuple] = {}
        self._load()

    def _load(self) -> None:
        with self.lock:
            for row in self.conn.execute("SELECT * FROM users"):
                p = Person(row["id"], dict(row), self.face_dim, self.body_dim)
                self.people[p.pid] = p
            vecs: dict[tuple, list] = {}
            for row in self.conn.execute("SELECT id, user_id, kind, vector FROM embeddings ORDER BY id"):
                vecs.setdefault((row["user_id"], row["kind"]), []).append(
                    (row["id"], np.frombuffer(row["vector"], np.float32)))
            for (uid, kind), items in vecs.items():
                p = self.people.get(uid)
                dim = self.face_dim if kind == "face" else self.body_dim
                items = [(i, v) for i, v in items if v.shape[0] == dim]
                if p is None or not items:
                    continue
                setattr(p, kind, np.vstack([v for _, v in items]).astype(np.float32))
                setattr(p, f"{kind}_ids", [i for i, _ in items])
            self.version += 1

    def save(self, force_all: bool = False) -> None:  # noqa: ARG002 - moslik uchun
        with self.lock:
            self.conn.commit()

    def close(self) -> None:
        with self.lock:
            self.conn.commit()
            self.conn.close()

    # --- foydalanuvchilar ---
    def create(self, profile: dict, photo: np.ndarray | None = None) -> Person:
        with self.lock:
            pid = time.strftime("%Y%m%d") + "_" + uuid.uuid4().hex[:8]
            now = _now_str()
            p = Person(pid, profile, self.face_dim, self.body_dim)
            cols = [k for k, _, _ in PROFILE_FIELDS]
            self.conn.execute(
                f"INSERT INTO users (id, {', '.join(cols)}, photo, created_at, updated_at) "
                f"VALUES (?, {', '.join('?' * len(cols))}, ?, ?, ?)",
                [pid, *[p.profile[c] for c in cols], _jpeg(photo), now, now])
            self.people[pid] = p
            self.log_event("registered", pid)
            self.conn.commit()
            self.version += 1
            return p

    def update_profile(self, pid: str, profile: dict) -> None:
        with self.lock:
            p = self.people.get(pid)
            if p is None:
                return
            p.profile.update({k: (profile.get(k) or "") for k, _, _ in PROFILE_FIELDS if k in profile})
            cols = [k for k, _, _ in PROFILE_FIELDS]
            self.conn.execute(
                f"UPDATE users SET {', '.join(c + '=?' for c in cols)}, updated_at=? WHERE id=?",
                [*[p.profile[c] for c in cols], _now_str(), pid])
            self.log_event("updated", pid)
            self.conn.commit()
            self.version += 1

    def set_photo(self, pid: str, img: np.ndarray) -> None:
        with self.lock:
            self.conn.execute("UPDATE users SET photo=? WHERE id=?", (_jpeg(img), pid))
            self.conn.commit()
            self.version += 1

    def photo(self, pid: str) -> np.ndarray | None:
        with self.lock:
            row = self.conn.execute("SELECT photo FROM users WHERE id=?", (pid,)).fetchone()
        return decode_jpeg(row["photo"]) if row else None

    def delete(self, pid: str) -> None:
        with self.lock:
            p = self.people.pop(pid, None)
            if p:
                self.log_event("deleted", None, note=p.name)
            self.conn.execute("DELETE FROM users WHERE id=?", (pid,))
            self.conn.commit()
            self.version += 1

    def name(self, pid: str | None) -> str:
        p = self.people.get(pid) if pid else None
        return p.name if p else "Noma'lum"

    # --- hodisalar ---
    def log_event(self, etype: str, pid: str | None, method: str = "", score: float | None = None,
                  note: str = "", snapshot: np.ndarray | None = None) -> None:
        with self.lock:
            self.conn.execute(
                "INSERT INTO events (user_id, user_name, type, method, score, note, snapshot, created_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (pid, self.name(pid) if pid else (note or None), etype, method, score, note,
                 _jpeg(snapshot, 80), _now_str()))

    def mark_seen(self, pid: str, method: str = "", score: float | None = None,
                  snapshot: np.ndarray | None = None, note: str = "") -> None:
        with self.lock:
            self.log_event("recognized", pid, method, score, note=note, snapshot=snapshot)
            self.conn.commit()
            self.version += 1

    # --- o'rganish ---
    def add(self, pid: str, kind: str, emb: np.ndarray | None, image: np.ndarray | None = None,
            novelty: float | None = None, source: str = "auto") -> bool:
        """Yangi namuna qo'shadi. Galereyada juda o'xshashi bo'lsa qo'shmaydi (False)."""
        with self.lock:
            p = self.people.get(pid)
            if p is None or emb is None:
                return False
            gal, ids = getattr(p, kind), getattr(p, f"{kind}_ids")
            if novelty is None:
                novelty = self.s.face_novelty if kind == "face" else self.s.body_novelty
            if len(gal) and float((gal @ emb).max()) >= novelty:
                return False
            cap = self.s.face_gallery_max if kind == "face" else self.s.body_gallery_max
            if len(gal) >= cap:
                if kind == "face":   # eng takroriy (boshqalarga eng o'xshash) namunani o'chiramiz
                    sim = gal @ gal.T
                    np.fill_diagonal(sim, -1.0)
                    drop = int(sim.max(axis=1).argmax())
                else:                # tana: eng eski namuna (kiyim o'zgarib turadi)
                    drop = 0
                self.conn.execute("DELETE FROM embeddings WHERE id=?", (ids[drop],))
                gal = np.delete(gal, drop, axis=0)
                ids = ids[:drop] + ids[drop + 1:]
            cur = self.conn.execute(
                "INSERT INTO embeddings (user_id, kind, vector, source, created_at) VALUES (?, ?, ?, ?, ?)",
                (pid, kind, emb.astype(np.float32).tobytes(), source, _now_str()))
            setattr(p, kind, np.vstack([gal, emb[None, :]]).astype(np.float32))
            setattr(p, f"{kind}_ids", ids + [cur.lastrowid])
            if image is not None:
                self.conn.execute("INSERT INTO samples (user_id, image, source, created_at) VALUES (?, ?, ?, ?)",
                                  (pid, _jpeg(image), source, _now_str()))
                self.conn.execute(
                    "DELETE FROM samples WHERE user_id=? AND id NOT IN "
                    "(SELECT id FROM samples WHERE user_id=? ORDER BY id DESC LIMIT ?)",
                    (pid, pid, MAX_SAMPLE_IMAGES))
            if source == "auto":
                self.log_event("learned", pid, method=kind)
            self.conn.commit()
            self.version += 1
            return True

    # --- qidiruv ---
    def rank(self, emb: np.ndarray | None, kind: str) -> list[tuple[str, float]]:
        """Har bir odam uchun galereyadagi eng yuqori o'xshashlik, kamayish tartibida."""
        if emb is None:
            return []
        with self.lock:
            c = self._cache.get(kind)
            if c is None or c[0] != self.version:
                pids, mats, starts, n = [], [], [], 0
                for pid, p in self.people.items():
                    g = getattr(p, kind)
                    if len(g) == 0:
                        continue
                    pids.append(pid)
                    starts.append(n)
                    mats.append(g)
                    n += len(g)
                mat = np.vstack(mats) if mats else None
                c = (self.version, pids, mat, np.asarray(starts, dtype=np.intp))
                self._cache[kind] = c
            _, pids, mat, starts = c
        if mat is None:
            return []
        per_person = np.maximum.reduceat(mat @ emb, starts)
        order = np.argsort(-per_person)
        return [(pids[i], float(per_person[i])) for i in order]

    # --- dashboard uchun so'rovlar ---
    def _q(self, sql: str, args=()) -> list[sqlite3.Row]:
        with self.lock:
            return self.conn.execute(sql, args).fetchall()

    def summary(self) -> dict:
        today = time.strftime("%Y-%m-%d")
        one = lambda sql, a=(): self._q(sql, a)[0][0] or 0  # noqa: E731
        return {
            "users": len(self.people),
            "new_today": one("SELECT COUNT(*) FROM users WHERE created_at LIKE ?", (today + "%",)),
            "recognized_today": one("SELECT COUNT(*) FROM events WHERE type='recognized' AND created_at LIKE ?",
                                    (today + "%",)),
            "unique_today": one("SELECT COUNT(DISTINCT user_id) FROM events WHERE type='recognized' "
                                "AND created_at LIKE ?", (today + "%",)),
            "unknown_today": one("SELECT COUNT(*) FROM events WHERE type='unknown' AND created_at LIKE ?",
                                 (today + "%",)),
            "learned_today": one("SELECT COUNT(*) FROM events WHERE type='learned' AND created_at LIKE ?",
                                 (today + "%",)),
            "face_samples": sum(len(p.face) for p in self.people.values()),
            "body_samples": sum(len(p.body) for p in self.people.values()),
            "db_size_mb": self.path.stat().st_size / 1e6 if self.path.exists() else 0.0,
        }

    def daily_counts(self, days: int = 7) -> list[tuple[str, int, int]]:
        """Oxirgi N kun: (sana, aniqlashlar, noma'lumlar)."""
        rows = self._q(
            "SELECT substr(created_at,1,10) d, SUM(type='recognized') r, SUM(type='unknown') u FROM events "
            "WHERE created_at >= date('now','localtime', ?) GROUP BY d", (f"-{days - 1} days",))
        got = {r["d"]: (r["r"] or 0, r["u"] or 0) for r in rows}
        out = []
        for i in range(days - 1, -1, -1):
            d = time.strftime("%Y-%m-%d", time.localtime(time.time() - i * 86400))
            out.append((d, *got.get(d, (0, 0))))
        return out

    def hourly_today(self) -> list[int]:
        rows = self._q("SELECT CAST(substr(created_at,12,2) AS INTEGER) h, COUNT(*) c FROM events "
                       "WHERE type='recognized' AND created_at LIKE ? GROUP BY h", (time.strftime("%Y-%m-%d") + "%",))
        hours = [0] * 24
        for r in rows:
            hours[r["h"]] = r["c"]
        return hours

    def recent_events(self, limit: int = 100, etype: str | None = None) -> list[sqlite3.Row]:
        if etype:
            return self._q("SELECT id, user_id, user_name, type, method, score, note, created_at FROM events "
                           "WHERE type=? ORDER BY id DESC LIMIT ?", (etype, limit))
        return self._q("SELECT id, user_id, user_name, type, method, score, note, created_at FROM events "
                       "ORDER BY id DESC LIMIT ?", (limit,))

    def event_snapshot(self, event_id: int) -> np.ndarray | None:
        rows = self._q("SELECT snapshot FROM events WHERE id=?", (event_id,))
        return decode_jpeg(rows[0]["snapshot"]) if rows else None

    def user_rows(self) -> list[dict]:
        stats = {r["user_id"]: (r["c"], r["last"]) for r in self._q(
            "SELECT user_id, COUNT(*) c, MAX(created_at) last FROM events WHERE type='recognized' GROUP BY user_id")}
        rows = []
        with self.lock:
            for pid, p in self.people.items():
                c, last = stats.get(pid, (0, ""))
                rows.append({"id": pid, "name": p.name, **p.profile, "created_at": p.created,
                             "faces": len(p.face), "bodies": len(p.body), "seen": c, "last_seen": last or ""})
        return sorted(rows, key=lambda r: r["name"].lower())

    def samples(self, pid: str, limit: int = 60) -> list[tuple[np.ndarray, str, str]]:
        return [(decode_jpeg(r["image"]), r["source"], r["created_at"]) for r in self._q(
            "SELECT image, source, created_at FROM samples WHERE user_id=? ORDER BY id DESC LIMIT ?", (pid, limit))]

    def export_csv(self, path: str) -> None:
        import csv
        with open(path, "w", newline="", encoding="utf-8-sig") as f:
            w = csv.writer(f)
            w.writerow(["Sana", "Hodisa", "Foydalanuvchi", "Usul", "Ishonch", "Izoh"])
            for r in self._q("SELECT * FROM events ORDER BY id"):
                w.writerow([r["created_at"], EVENT_TYPES.get(r["type"], r["type"]), r["user_name"] or "",
                            r["method"], f"{r['score']:.3f}" if r["score"] is not None else "", r["note"]])


# --------------------------------------------------------------------------- #
# Aniqlash va kuzatish (tracking)
# --------------------------------------------------------------------------- #
@dataclass
class Detection:
    box: tuple                       # x, y, w, h  (yuz, yoki yuz yo'q bo'lsa tana)
    face_row: np.ndarray | None      # YuNet natijasi (15 ta son)
    body_box: tuple | None
    face_emb: np.ndarray | None = None
    aligned: np.ndarray | None = None
    quality_ok: bool = False
    yaw: float = 0.0
    body_emb: np.ndarray | None = None
    pid: str | None = None
    score: float = 0.0
    via: str = ""
    state: str = ""                  # "learned" | "reg" | ""
    attrs: list = field(default_factory=list)          # aniqlangan atributlar kalitlari
    attr_probs: dict = field(default_factory=dict)


class Track:
    def __init__(self, tid: int, kind: str, box: tuple):
        self.tid, self.kind, self.box = tid, kind, box
        self.votes: deque = deque(maxlen=10)
        self.missed = 0
        self.age = 0
        self.body_emb = None
        self.body_rank: list = []
        self.announced: str | None = None
        self.unknown_logged = False
        self.attr_logits: np.ndarray | None = None

    def identity(self) -> str | None:
        if not self.votes:
            return None
        pid, cnt = Counter(self.votes).most_common(1)[0]
        return pid if cnt * 2 > len(self.votes) else None

    def stable(self, pid: str, n: int) -> bool:
        return len(self.votes) >= n and all(v == pid for v in list(self.votes)[-n:])


REG_STEPS = [
    ("front", "Kameraga to'g'ri qarang", 6),
    ("left", "Boshingizni sekin chapga buring", 4),
    ("right", "Boshingizni sekin o'ngga buring", 4),
    ("up", "Biroz yuqoriga qarang", 3),
    ("down", "Biroz pastga qarang", 3),
    ("free", "Yuz ifodasini o'zgartiring (jilmaying, gapiring)", 4),
]
REG_STEP_TIMEOUT = 10.0  # burilish aniqlanmasa ham shu vaqtdan keyin qadam qabul qilinadi


class Registration:
    def __init__(self, name: str, pid: str | None = None, profile: dict | None = None):
        self.name = name
        self.pid = pid  # mavjud foydalanuvchini qo'shimcha o'rgatish bo'lsa
        self.profile = profile or {}
        self.result_pid: str | None = None
        self.step = 0
        self.got = 0
        self.last = 0.0
        self.step_started = time.time()
        self.faces: list[np.ndarray] = []
        self.images: list[np.ndarray] = []
        self.bodies: list[np.ndarray] = []
        self.message = ""
        self.done = False

    @property
    def total_needed(self) -> int:
        return sum(n for _, _, n in REG_STEPS)

    @property
    def collected(self) -> int:
        return len(self.faces)

    @property
    def instruction(self) -> str:
        return REG_STEPS[min(self.step, len(REG_STEPS) - 1)][1]


class Engine:
    def __init__(self, settings: Settings | None = None):
        missing = [p.name for p in MODEL_FILES if not p.is_file()]
        if missing:
            raise FileNotFoundError(
                "Modellar topilmadi: " + ", ".join(missing) + "\n`python download_models.py` ni ishga tushiring."
            )
        self.s = settings or Settings.load()
        self.mlock = threading.Lock()  # OpenCV DNN obyektlari ko'p oqimli emas
        self.detector = cv2.FaceDetectorYN.create(str(FACE_DET_MODEL), "", (320, 320),
                                                  self.s.face_det_score, 0.3, 5000)
        self.recognizer = cv2.FaceRecognizerSF.create(str(FACE_REC_MODEL), "")
        self.reid = cv2.dnn.readNet(str(BODY_REID_MODEL))
        self.hog = cv2.HOGDescriptor()
        self.hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
        self.attr = None
        try:
            import attributes as _attributes
            if _attributes.available():
                self.attr = _attributes.AttributeDetector()
                self.attr_labels = _attributes.LABELS
        except Exception as e:  # noqa: BLE001
            print(f"[ogohlantirish] atribut modeli yuklanmadi: {e}")
        body_dim = self._body_embed_crop(np.zeros((256, 128, 3), np.uint8)).shape[0]
        self.db = Database(DB_FILE, 128, body_dim, self.s)

        self.events: queue.Queue[str] = queue.Queue()
        self.tracks: list[Track] = []
        self._next_tid = 1
        self.frame_idx = 0
        self.hog_boxes: list[tuple] = []
        self.registration: Registration | None = None
        self.last_registration: Registration | None = None

    # ------------------------------------------------------------------ log
    def log(self, msg: str) -> None:
        self.events.put(time.strftime("%H:%M:%S ") + msg)

    # -------------------------------------------------------------- modellar
    def _detect_faces(self, img: np.ndarray) -> np.ndarray:
        h, w = img.shape[:2]
        with self.mlock:
            self.detector.setInputSize((w, h))
            _, faces = self.detector.detect(img)
            # katta portretlarda yuz detektor diapazonidan katta bo'ladi - kichraytirib qayta urinamiz
            for side in (640, 320):
                if faces is not None or max(h, w) <= side:
                    break
                k = side / max(h, w)
                small = cv2.resize(img, (max(1, int(w * k)), max(1, int(h * k))), interpolation=cv2.INTER_AREA)
                self.detector.setInputSize((small.shape[1], small.shape[0]))
                _, faces = self.detector.detect(small)
                if faces is not None:
                    faces = faces.copy()
                    faces[:, :14] /= k
        return faces if faces is not None else np.zeros((0, 15), np.float32)

    def _face_embed(self, img: np.ndarray, row: np.ndarray):
        with self.mlock:
            aligned = self.recognizer.alignCrop(img, row)
            feat = self.recognizer.feature(aligned)
        return _normalize(feat), aligned

    def _body_embed_crop(self, crop: np.ndarray) -> np.ndarray:
        x = cv2.resize(crop, (128, 256)).astype(np.float32)[:, :, ::-1] / 255.0
        x = (x - (0.485, 0.456, 0.406)) / (0.229, 0.224, 0.225)
        blob = np.ascontiguousarray(x.transpose(2, 0, 1)[None], dtype=np.float32)
        with self.mlock:
            self.reid.setInput(blob)
            out = self.reid.forward()
        return _normalize(out)

    def _body_embed(self, img: np.ndarray, box: tuple | None) -> np.ndarray | None:
        if box is None:
            return None
        x, y, w, h = box
        crop = img[y:y + h, x:x + w]
        if crop.size == 0 or w < 16 or h < 32:
            return None
        return self._body_embed_crop(crop)

    def _detect_bodies(self, img: np.ndarray) -> list[tuple]:
        H, W = img.shape[:2]
        scale = 640.0 / W if W > 640 else 1.0
        small = cv2.resize(img, None, fx=scale, fy=scale) if scale != 1.0 else img
        with self.mlock:
            rects, weights = self.hog.detectMultiScale(small, winStride=(8, 8), padding=(8, 8), scale=1.05)
        if len(rects) == 0:
            return []
        weights = np.asarray(weights).reshape(-1)
        keep = cv2.dnn.NMSBoxes([list(map(int, r)) for r in rects], weights.tolist(), 0.6, 0.4)
        out = []
        for i in np.asarray(keep).reshape(-1):
            x, y, w, h = (np.asarray(rects[i]) / scale).astype(int)
            x, y = max(0, x), max(0, y)
            out.append((x, y, min(w, W - x), min(h, H - y)))
        return out

    @staticmethod
    def _body_from_face(fb: tuple, shape) -> tuple | None:
        """Yuz atrofidan tana sohasini taxmin qiladi (kamera oldida turgan odam uchun)."""
        x, y, w, h = fb
        H, W = shape[:2]
        cx = x + w / 2
        x1, x2 = int(max(0, cx - 1.6 * w)), int(min(W, cx + 1.6 * w))
        y1, y2 = int(max(0, y - 0.3 * h)), int(min(H, y + 7.5 * h))
        if (y2 - (y + h)) < 1.5 * h:  # tana kadrda yetarli ko'rinmayapti
            return None
        return (x1, y1, x2 - x1, y2 - y1)

    @staticmethod
    def _yaw(row: np.ndarray) -> float:
        rex, _, lex, _, nx = row[4], row[5], row[6], row[7], row[8]
        eye_dist = abs(lex - rex) + 1e-6
        return float((nx - (rex + lex) / 2) / eye_dist)

    # ---------------------------------------------------------- aniqlash
    def _detect(self, img: np.ndarray, hog_boxes: list[tuple]) -> list[Detection]:
        H, W = img.shape[:2]
        dets: list[Detection] = []
        face_centers = []
        used = set()
        for row in self._detect_faces(img):
            x, y, w, h = (int(round(v)) for v in row[:4])
            x, y = max(0, x), max(0, y)
            w, h = min(w, W - x), min(h, H - y)
            if w < 20 or h < 20:
                continue
            cx, cy = x + w / 2, y + h / 2
            face_centers.append((cx, cy))
            body = None
            for i, (bx, by, bw, bh) in enumerate(hog_boxes):
                if i not in used and bx <= cx <= bx + bw and by <= cy <= by + 0.4 * bh and w < 0.8 * bw:
                    body = (bx, by, bw, bh)
                    used.add(i)
                    break
            if body is None:
                body = self._body_from_face((x, y, w, h), img.shape)
            d = Detection(box=(x, y, w, h), face_row=row, body_box=body)
            d.face_emb, d.aligned = self._face_embed(img, row)
            sharp = cv2.Laplacian(cv2.cvtColor(d.aligned, cv2.COLOR_BGR2GRAY), cv2.CV_64F).var()
            d.quality_ok = min(w, h) >= self.s.min_face_px and row[14] >= 0.88 and sharp >= self.s.min_sharpness
            d.yaw = self._yaw(row)
            dets.append(d)
        # yuzi ko'rinmayotgan odamlar (orqasi bilan, uzoqda)
        for i, b in enumerate(hog_boxes):
            if i in used:
                continue
            bx, by, bw, bh = b
            if any(bx <= cx <= bx + bw and by <= cy <= by + bh for cx, cy in face_centers):
                continue
            dets.append(Detection(box=b, face_row=None, body_box=b))
        return dets

    # ---------------------------------------------------------- tracking
    def _assign_tracks(self, dets: list[Detection]) -> list[Track | None]:
        assigned: list[Track | None] = [None] * len(dets)
        free = set(range(len(self.tracks)))
        pairs = []
        for di, d in enumerate(dets):
            kind = "face" if d.face_row is not None else "body"
            for ti in free:
                t = self.tracks[ti]
                if t.kind == kind:
                    iou = _iou(d.box, t.box)
                    if iou > 0.25:
                        pairs.append((iou, di, ti))
        taken_d, taken_t = set(), set()
        for _, di, ti in sorted(pairs, reverse=True):
            if di in taken_d or ti in taken_t:
                continue
            taken_d.add(di)
            taken_t.add(ti)
            t = self.tracks[ti]
            t.box, t.missed = dets[di].box, 0
            t.age += 1
            assigned[di] = t
        for ti, t in enumerate(self.tracks):
            if ti not in taken_t:
                t.missed += 1
        for di, d in enumerate(dets):
            if assigned[di] is None:
                t = Track(self._next_tid, "face" if d.face_row is not None else "body", d.box)
                self._next_tid += 1
                self.tracks.append(t)
                assigned[di] = t
        self.tracks = [t for t in self.tracks if t.missed <= 5]
        return assigned

    # ---------------------------------------------------------- tanib olish
    def _attributes(self, img, d: Detection, track: Track | None) -> None:
        """Ko'zoynak, kepka, soqol va h.k. Kuzatuvda har 8-kadrda hisoblanadi va silliqlanadi."""
        if self.attr is None or not self.s.attributes_enabled or d.face_row is None:
            return
        lg = track.attr_logits if track is not None else None
        if lg is None or track.age % 8 == 0:
            crop = self.attr.head_crop(img, d.box)
            if crop is not None:
                new = self.attr.logits([crop])[0]
                lg = new if lg is None else 0.6 * lg + 0.4 * new
                if track is not None:
                    track.attr_logits = lg
        if lg is None:
            return
        p = 1.0 / (1.0 + np.exp(-lg))
        d.attrs = self.attr.decide(p)
        d.attr_probs = {k: round(float(v), 2) for k, v in zip(self.attr.keys, p)}

    def attr_text(self, keys: list[str]) -> str:
        return ", ".join(self.attr_labels.get(k, k) for k in keys) if self.attr else ""

    def _recognize(self, img, d: Detection, now: float, track: Track | None, live: bool) -> None:
        s = self.s
        self._attributes(img, d, track)
        if d.face_row is not None:
            rank = self.db.rank(d.face_emb, "face")
            best_pid, best = rank[0] if rank else (None, -1.0)
            second = rank[1][1] if len(rank) > 1 else -1.0
            pid, via = None, ""
            if best >= s.face_match:
                pid, via = best_pid, "yuz"
            elif best >= s.face_match - 0.08 and s.body_enabled and d.body_box is not None:
                # chegaraviy holat: tana ham tasdiqlasa qabul qilamiz
                d.body_emb = self._body_embed(img, d.body_box)
                body_scores = dict(self.db.rank(d.body_emb, "body"))
                if body_scores.get(best_pid, -1.0) >= s.body_match:
                    pid, via = best_pid, "yuz+tana"
            if track is not None:
                track.votes.append(pid)
                shown = track.identity()
            else:
                shown = pid
            d.pid, d.via = shown, via if shown == pid else "yuz"
            d.score = dict(rank).get(shown, best) if shown else best

            if (live and s.auto_learn and track is not None and pid is not None
                    and best >= s.face_learn and second < s.face_match
                    and track.stable(pid, s.stable_frames) and d.quality_ok):
                self._learn(img, d, pid, now)
        else:
            if not s.body_enabled:
                return
            if track is not None and track.body_emb is not None and track.age % 3 != 0:
                rank = track.body_rank
            else:
                d.body_emb = self._body_embed(img, d.body_box)
                rank = self.db.rank(d.body_emb, "body")
                if track is not None:
                    track.body_emb, track.body_rank = d.body_emb, rank
            best_pid, best = rank[0] if rank else (None, -1.0)
            pid = best_pid if best >= s.body_match else None
            if track is not None:
                track.votes.append(pid)
                pid = track.identity()
            d.pid, d.score, d.via = pid, best, "tana"

        if live and track is not None and d.pid and track.announced != d.pid and len(track.votes) >= 3 \
                and track.stable(d.pid, 3):
            track.announced = d.pid
            self.db.mark_seen(d.pid, d.via, d.score, self._snapshot(img, d), note=self.attr_text(d.attrs))
            extra = f" · {self.attr_text(d.attrs)}" if d.attrs else ""
            self.log(f"Aniqlandi: {self.db.name(d.pid)} ({d.via}, {d.score:.2f}){extra}")
        elif (live and track is not None and d.face_row is not None and d.pid is None and d.quality_ok
              and not track.unknown_logged and len(track.votes) >= 8
              and all(v is None for v in track.votes) and self.db.people):
            track.unknown_logged = True
            self.db.log_event("unknown", None, "yuz", d.score, note=self.attr_text(d.attrs),
                              snapshot=self._snapshot(img, d))
            self.db.save()
            self.db.version += 1
            self.log("Noma'lum shaxs kadrda (bazaga qayd etildi)")

    @staticmethod
    def _snapshot(img: np.ndarray, d: Detection) -> np.ndarray | None:
        x, y, w, h = d.box
        H, W = img.shape[:2]
        m = int(0.4 * max(w, h))
        crop = img[max(0, y - m):min(H, y + h + m), max(0, x - m):min(W, x + w + m)]
        if crop.size == 0:
            return None
        k = 160 / max(crop.shape[:2])
        return cv2.resize(crop, None, fx=k, fy=k) if k < 1 else crop

    def _learn(self, img, d: Detection, pid: str, now: float) -> None:
        p = self.db.people.get(pid)
        if p is None:
            return
        if now - p.last_face_learn >= self.s.learn_interval:
            p.last_face_learn = now
            if self.db.add(pid, "face", d.face_emb, d.aligned, source="auto"):
                d.state = "learned"
                self.log(f"O'rgandi: {p.name} - yangi yuz ko'rinishi (jami {len(p.face)})")
        if self.s.body_enabled and d.body_box is not None and now - p.last_body_learn >= 2 * self.s.learn_interval:
            p.last_body_learn = now
            emb = d.body_emb if d.body_emb is not None else self._body_embed(img, d.body_box)
            if self.db.add(pid, "body", emb):
                d.state = "learned"
                self.log(f"O'rgandi: {p.name} - yangi tana/kiyim ko'rinishi (jami {len(p.body)})")

    # ---------------------------------------------------------- ro'yxatdan o'tkazish
    def start_registration(self, name: str, pid: str | None = None, profile: dict | None = None) -> None:
        """pid=None: yangi foydalanuvchi (profile anketadan), aks holda mavjudini qo'shimcha o'rgatish."""
        self.registration = Registration(name, pid, profile)
        what = "qo'shimcha o'rgatish" if pid else "ro'yxatdan o'tkazish"
        self.log(f"{name}: {what} boshlandi. Ko'rsatmalarga amal qiling.")

    def cancel_registration(self) -> None:
        if self.registration:
            self.log(f"{self.registration.name}: bekor qilindi.")
        self.registration = None

    def _registration_step(self, img, dets: list[Detection], now: float) -> None:
        r = self.registration
        if r is None:
            return
        faces = [d for d in dets if d.face_row is not None]
        for d in faces:
            d.state = "reg"
        if not faces:
            r.message = "Yuz ko'rinmayapti"
            return
        if len(faces) > 1:
            r.message = "Kadrda faqat bitta odam bo'lishi kerak"
            return
        d = faces[0]
        key, text, need = REG_STEPS[r.step]
        if not d.quality_ok:
            r.message = "Yaqinroq keling yoki yorug'likni yaxshilang"
            return
        timeout = now - r.step_started > REG_STEP_TIMEOUT
        pose_ok = {"front": abs(d.yaw) < 0.15, "left": d.yaw < -0.10, "right": d.yaw > 0.10}.get(key, True)
        if not (pose_ok or timeout):
            r.message = text
            return
        if now - r.last < 0.35:
            return
        if r.faces and float(max(f @ d.face_emb for f in r.faces)) > 0.97:
            r.message = text + " (biroz harakatlaning)"
            return
        if r.pid is not None:
            p = self.db.people.get(r.pid)
            if p is not None and len(p.face):
                sim = float((p.face @ d.face_emb).max())
                if sim < self.s.face_match - 0.05:
                    r.message = f"Bu yuz '{p.name}' ga o'xshamayapti"
                    return
        elif not r.faces:
            rank = self.db.rank(d.face_emb, "face")
            if rank and rank[0][1] >= self.s.face_learn:
                self.log(f"Diqqat: bu yuz allaqachon '{self.db.name(rank[0][0])}' "
                         f"sifatida ro'yxatda bo'lishi mumkin ({rank[0][1]:.2f}).")
        r.faces.append(d.face_emb)
        r.images.append(d.aligned)
        if self.s.body_enabled and d.body_box is not None:
            emb = self._body_embed(img, d.body_box)
            if emb is not None:
                r.bodies.append(emb)
        r.last = now
        r.got += 1
        r.message = text
        if r.got >= need:
            r.step += 1
            r.got = 0
            r.step_started = now
            if r.step >= len(REG_STEPS):
                self._finish_registration()

    def _finish_registration(self) -> None:
        r = self.registration
        self.registration = None
        if r is None:
            return
        with self.db.lock:
            if r.pid and r.pid in self.db.people:
                p = self.db.people[r.pid]
            else:
                p = self.db.create(r.profile or {"first_name": r.name, "last_name": ""},
                                   photo=r.images[0] if r.images else None)
            nf = sum(self.db.add(p.pid, "face", e, im, novelty=0.99, source="reg")
                     for e, im in zip(r.faces, r.images))
            nb = sum(self.db.add(p.pid, "body", e, novelty=0.97) for e in r.bodies)
        self.db.save()
        r.done = True
        r.result_pid = p.pid
        self.last_registration = r
        self.log(f"Tayyor: {p.name} - {nf} ta yuz va {nb} ta tana namunasi saqlandi.")

    # ---------------------------------------------------------- asosiy sikl
    def process_frame(self, frame: np.ndarray) -> list[Detection]:
        """Bitta kamera kadrini qayta ishlaydi (tanish, o'rganish, ro'yxatga olish)."""
        now = time.time()
        self.frame_idx += 1
        if self.s.body_enabled and self.frame_idx % 6 == 0:
            self.hog_boxes = self._detect_bodies(frame)
        elif not self.s.body_enabled:
            self.hog_boxes = []
        dets = self._detect(frame, self.hog_boxes)
        if self.registration is not None:
            self._registration_step(frame, dets, now)
            return [d for d in dets if d.face_row is not None]
        tracks = self._assign_tracks(dets)
        for d, t in zip(dets, tracks):
            self._recognize(frame, d, now, t, live=True)
        return dets

    def process(self, frame: np.ndarray) -> np.ndarray:
        """Natija: chizilgan RGB rasm."""
        return self._draw(frame, self.process_frame(frame))

    def detections_json(self, dets: list[Detection]) -> list[dict]:
        out = []
        for d in dets:
            if d.face_row is None and d.pid is None:
                continue  # aniqlanmagan "faqat tana" qutilarini ko'rsatmaymiz
            out.append({
                "kind": "face" if d.face_row is not None else "body",
                "box": [int(v) for v in d.box],
                "body": [int(v) for v in d.body_box] if d.body_box and self.s.body_enabled else None,
                "pid": d.pid, "name": self.db.name(d.pid) if d.pid else None,
                "score": round(float(d.score), 3), "via": d.via, "state": d.state,
                "attrs": [{"key": k, "label": self.attr_labels.get(k, k), "p": d.attr_probs.get(k, 0)}
                          for k in d.attrs] if self.attr else [],
            })
        return out

    def registration_json(self) -> dict | None:
        r = self.registration
        if r is None:
            return None
        return {"name": r.name, "message": r.message or r.instruction, "instruction": r.instruction,
                "step": min(r.step + 1, len(REG_STEPS)), "steps": len(REG_STEPS),
                "collected": r.collected, "total": r.total_needed, "existing": r.pid is not None}

    def analyze_image(self, img: np.ndarray):
        """Statik rasmni tekshiradi (o'rganmaydi). Natija: (RGB rasm, [(ism, ball, usul)])."""
        if img is None:
            raise ValueError("Rasm o'qilmadi")
        img = self._limit_size(img, 1600)
        dets = self._detect(img, self._detect_bodies(img) if self.s.body_enabled else [])
        now = time.time()
        for d in dets:
            self._recognize(img, d, now, None, live=False)
        results = [(self.db.name(d.pid), d.score, d.via, d.attrs) for d in dets]
        return self._draw(img, dets), results

    @staticmethod
    def _limit_size(img: np.ndarray, max_side: int) -> np.ndarray:
        h, w = img.shape[:2]
        k = max_side / max(h, w)
        return cv2.resize(img, None, fx=k, fy=k, interpolation=cv2.INTER_AREA) if k < 1 else img

    def import_images(self, pid: str, paths: list[str]) -> tuple[int, int]:
        """Har xil turdagi rasmlardan (telefon, hujjat, guruh surati...) o'rgatish."""
        p = self.db.people.get(pid)
        if p is None:
            return 0, 0
        added = skipped = 0
        for i, path in enumerate(paths, 1):
            img = read_image(path)
            if img is None:
                skipped += 1
                self.log(f"O'qib bo'lmadi: {Path(path).name}")
                continue
            img = self._limit_size(img, 1600)
            best = None
            for rot in (None, cv2.ROTATE_90_CLOCKWISE, cv2.ROTATE_90_COUNTERCLOCKWISE, cv2.ROTATE_180):
                im = img if rot is None else cv2.rotate(img, rot)
                faces = self._detect_faces(im)
                if len(faces):
                    cands = []
                    for row in faces:
                        emb, aligned = self._face_embed(im, row)
                        sim = float((p.face @ emb).max()) if len(p.face) else 0.0
                        cands.append((sim, float(row[2] * row[3]), row, emb, aligned))
                    # rasmda bir nechta odam bo'lsa: shu odamga eng o'xshashi, aks holda eng kattasi
                    matching = [c for c in cands if c[0] >= self.s.face_match - 0.05]
                    pick = max(matching, key=lambda c: c[0]) if matching else max(cands, key=lambda c: c[1])
                    if len(p.face) and not matching and len(faces) > 1:
                        pick = None
                    if pick is not None:
                        best = (im, pick)
                    break
            if best is None:
                skipped += 1
                self.log(f"Yuz topilmadi: {Path(path).name}")
                continue
            im, (sim, _, row, emb, aligned) = best
            if len(p.face) and sim < self.s.face_match - 0.1:
                self.log(f"Ogohlantirish: {Path(path).name} dagi yuz {p.name} ga kam o'xshaydi ({sim:.2f}), "
                         f"baribir qo'shildi.")
            ok = self.db.add(pid, "face", emb, aligned, novelty=0.985, source="photo")
            x, y, w, h = (int(v) for v in row[:4])
            bbox = self._body_from_face((max(0, x), max(0, y), w, h), im.shape)
            if self.s.body_enabled and bbox is not None:
                self.db.add(pid, "body", self._body_embed(im, bbox), novelty=0.97)
            added += int(ok)
            skipped += int(not ok)
            self.log(f"[{i}/{len(paths)}] {Path(path).name}: {"qo'shildi" if ok else "oldingisiga juda o'xshash"}")
        if added:
            self.db.log_event("imported", pid, note=f"{added} ta rasm")
            if self.db.photo(pid) is None and len(p.face):
                last = self.db.samples(pid, 1)
                if last:
                    self.db.set_photo(pid, last[0][0])
        self.db.save()
        return added, skipped

    # ---------------------------------------------------------- chizish
    def _draw(self, frame: np.ndarray, dets: list[Detection]) -> np.ndarray:
        img = frame.copy()
        H, W = img.shape[:2]
        th = max(2, W // 500)
        labels = []
        for d in dets:
            x, y, w, h = d.box
            if d.face_row is None:   # faqat tana
                if d.pid is None:
                    cv2.rectangle(img, (x, y), (x + w, y + h), (140, 140, 140), 1)
                    continue
                color = (255, 170, 0)
                cv2.rectangle(img, (x, y), (x + w, y + h), color, th)
                labels.append((x, y, f"{self.db.name(d.pid)} (tana) {d.score:.2f}", color))
                continue
            if d.state == "reg":
                color, text = (0, 215, 255), "Ro'yxatga olinmoqda..."
            elif d.pid:
                color = (60, 200, 60)
                text = f"{self.db.name(d.pid)}  {d.score:.2f}" + ("  +tana" if d.via == "yuz+tana" else "")
            else:
                color, text = (60, 60, 230), f"Noma'lum  {max(d.score, 0):.2f}"
            if d.attrs:
                text += "  · " + self.attr_text(d.attrs)
            if d.state == "learned":
                text += "  [o'rgandi]"
            if d.body_box is not None and self.s.body_enabled and d.state != "reg":
                bx, by, bw, bh = d.body_box
                cv2.rectangle(img, (bx, by), (bx + bw, by + bh), color, 1)
            cv2.rectangle(img, (x, y), (x + w, y + h), color, th)
            labels.append((x, y, text, color))

        r = self.registration
        if r is not None:
            bar_h = max(70, H // 9)
            overlay = img.copy()
            cv2.rectangle(overlay, (0, 0), (W, bar_h), (30, 30, 30), -1)
            img = cv2.addWeighted(overlay, 0.65, img, 0.35, 0)
            frac = r.collected / r.total_needed
            cv2.rectangle(img, (0, bar_h - 8), (int(W * frac), bar_h), (0, 215, 255), -1)

        pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(pil)
        font = _font(max(16, H // 32))
        for x, y, text, color in labels:
            rgb = (color[2], color[1], color[0])
            l, t, rr, b = draw.textbbox((0, 0), text, font=font)
            tw, tht = rr - l, b - t
            ty = y - tht - 10 if y - tht - 10 > 0 else y + 4
            draw.rectangle((x, ty, x + tw + 10, ty + tht + 8), fill=rgb)
            draw.text((x + 5, ty + 4 - t), text, font=font, fill=(255, 255, 255))
        if r is not None:
            big = _font(max(20, H // 22))
            step = f"Qadam {min(r.step + 1, len(REG_STEPS))}/{len(REG_STEPS)}  ·  {r.collected}/{r.total_needed}"
            draw.text((16, 8), f"{r.name}: {r.message or r.instruction}", font=big, fill=(255, 255, 255))
            draw.text((16, 12 + big.size), step + "   (Esc - bekor qilish)", font=font, fill=(0, 215, 255))
        return np.asarray(pil)
