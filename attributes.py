"""
Tashqi ko'rinish atributlarini aniqlash: ko'zoynak, quyosh ko'zoynagi, kepka, shlyapa, soqol, mo'ylov,
niqob, ro'mol, quloqchin, sirg'a, tabassum.

CLIP (ViT-B/32) modeli "zero-shot" usulida ishlaydi: bosh sohasi rasmi har bir atribut uchun
matnli tavsif juftliklari bilan solishtiriladi ("ko'zoynak taqqan odam" / "odam").
Alohida o'qitish talab qilinmaydi - yangi atribut qo'shish uchun ATTRIBUTES ro'yxatiga
tavsif yozish kifoya.
"""
from __future__ import annotations

import hashlib
import json
import os
import threading
from pathlib import Path

import cv2
import numpy as np

MODEL_DIR = Path(os.environ.get("MODEL_DIR", Path(__file__).resolve().parent / "models"))
VISION = MODEL_DIR / "clip_vision_model_quantized.onnx"
TEXT = MODEL_DIR / "clip_text_model_quantized.onnx"
TOKENIZER = MODEL_DIR / "clip_tokenizer.json"
CACHE = MODEL_DIR / "clip_text_cache.npz"

P = "a photo of a person."
M = "a photo of a man."
# (kalit, o'zbekcha nomi, [(ijobiy tavsif, salbiy tavsif), ...], chegara)
# Har bir juftlik uchun logit farqi hisoblanadi va o'rtachasi olinadi (ansambl).
ATTRIBUTES = [
    ("glasses", "Ko'zoynak", [("glasses", "no glasses"),
                              ("a photo of a person wearing glasses.", P)], 0.55),
    ("sunglasses", "Quyosh ko'zoynagi", [("a photo of a person wearing sunglasses.", P),
                                         ("dark sunglasses", "clear eyeglasses")], 0.60),
    ("cap", "Kepka", [("a photo of a person wearing a baseball cap.", P),
                      ("a photo of a person wearing a cap.", "a photo of a person with bare head.")], 0.62),
    ("hat", "Shlyapa / bosh kiyim", [("a photo of a person wearing a hat.", P),
                                     ("a person wearing a hat", "a person with no hat")], 0.60),
    ("headscarf", "Ro'mol", [("a photo of a woman wearing a headscarf.", "a photo of a woman."),
                             ("a photo of a woman wearing a hijab.", "a photo of a woman wearing a hat.")], 0.65),
    ("beard", "Soqol", [("a photo of a man with a beard.", M),
                        ("a photo of a person with a beard.", "a photo of a person with a shaved face.")], 0.55),
    ("mustache", "Mo'ylov", [("a photo of a man with a mustache.", M),
                             ("a man with a mustache", "a clean-shaven man")], 0.60),
    ("mask", "Tibbiy niqob", [("a photo of a person wearing a medical face mask.", P)], 0.65),
    ("headphones", "Quloqchin", [("a photo of a person wearing headphones.", P),
                                 ("headphones over the ears", "visible ears"),
                                 ("a person wearing headphones", "a person wearing glasses")], 0.75),
    ("earrings", "Sirg'a", [("a photo of a woman wearing earrings.", "a photo of a woman.")], 0.70),
    ("smile", "Tabassum", [("a photo of a smiling person.", "a photo of a person with a neutral expression.")], 0.65),
]
LABELS = {k: n for k, n, _, _ in ATTRIBUTES}
# bir vaqtda faqat bittasi bo'lishi mumkin bo'lganlar (eng ishonchlisi qoladi)
EXCLUSIVE = [("glasses", "sunglasses"), ("cap", "hat", "headscarf")]

MEAN = np.array([0.48145466, 0.4578275, 0.40821073], np.float32)
STD = np.array([0.26862954, 0.26130258, 0.27577711], np.float32)


def available() -> bool:
    return VISION.is_file() and TEXT.is_file() and TOKENIZER.is_file()


class AttributeDetector:
    def __init__(self):
        import onnxruntime as ort

        opts = ort.SessionOptions()
        opts.intra_op_num_threads = 2
        opts.log_severity_level = 3
        self.vision = ort.InferenceSession(str(VISION), opts, providers=["CPUExecutionProvider"])
        self.lock = threading.Lock()
        self.keys = [a[0] for a in ATTRIBUTES]
        self.thr = np.array([a[3] for a in ATTRIBUTES], np.float32)
        self.dirs = self._text_directions(ort, opts)  # har bir atribut: (juftliklar, 512)

    def _text_directions(self, ort, opts) -> list[np.ndarray]:
        sig = hashlib.md5(json.dumps([a[2] for a in ATTRIBUTES]).encode()).hexdigest()
        if CACHE.is_file():
            c = np.load(CACHE, allow_pickle=False)
            if "sig" in c and str(c["sig"]) == sig:
                return [c[f"a{i}"] for i in range(len(ATTRIBUTES))]
        from tokenizers import Tokenizer

        tok = Tokenizer.from_file(str(TOKENIZER))
        text = ort.InferenceSession(str(TEXT), opts, providers=["CPUExecutionProvider"])

        def emb(p: str) -> np.ndarray:
            v = text.run(None, {"input_ids": np.array([tok.encode(p).ids], np.int64)})[0][0]
            return v / np.linalg.norm(v)

        dirs = [np.stack([emb(p) - emb(n) for p, n in a[2]]).astype(np.float32) for a in ATTRIBUTES]
        np.savez(CACHE, sig=sig, **{f"a{i}": d for i, d in enumerate(dirs)})
        return dirs

    @staticmethod
    def head_crop(img: np.ndarray, box) -> np.ndarray | None:
        """Yuz qutisidan bosh sohasini (kepka va soqol ham kirishi uchun) kesib oladi."""
        x, y, w, h = box
        H, W = img.shape[:2]
        x1, x2 = int(max(0, x - 0.45 * w)), int(min(W, x + 1.45 * w))
        y1, y2 = int(max(0, y - 0.75 * h)), int(min(H, y + 1.35 * h))
        crop = img[y1:y2, x1:x2]
        return crop if crop.size and crop.shape[0] > 20 and crop.shape[1] > 20 else None

    @staticmethod
    def _prep(crop: np.ndarray) -> np.ndarray:
        h, w = crop.shape[:2]
        s = 224 / min(h, w)
        r = cv2.resize(crop, (max(224, round(w * s)), max(224, round(h * s))), interpolation=cv2.INTER_CUBIC)
        rh, rw = r.shape[:2]
        y0, x0 = (rh - 224) // 2, (rw - 224) // 2
        r = r[y0:y0 + 224, x0:x0 + 224, ::-1].astype(np.float32) / 255.0
        return ((r - MEAN) / STD).transpose(2, 0, 1)

    def logits(self, crops: list[np.ndarray]) -> np.ndarray:
        """Har bir rasm va atribut uchun logit (N, K). Ehtimol = sigmoid(logit)."""
        if not crops:
            return np.zeros((0, len(self.keys)), np.float32)
        batch = np.stack([self._prep(c) for c in crops]).astype(np.float32)
        with self.lock:
            e = self.vision.run(None, {"pixel_values": batch})[0]
        e = e / np.linalg.norm(e, axis=1, keepdims=True)
        return np.stack([(100.0 * e @ d.T).mean(axis=1) for d in self.dirs], axis=1).astype(np.float32)

    def probs(self, crops: list[np.ndarray]) -> np.ndarray:
        return 1.0 / (1.0 + np.exp(-self.logits(crops)))

    def decide(self, p: np.ndarray) -> list[str]:
        on = {k for k, v, t in zip(self.keys, p, self.thr) if v >= t}
        idx = {k: i for i, k in enumerate(self.keys)}
        for group in EXCLUSIVE:
            hit = [k for k in group if k in on]
            if len(hit) > 1:
                best = max(hit, key=lambda k: p[idx[k]])
                on -= set(hit) - {best}
        if "mask" in on:  # niqob ostida soqol/mo'ylov/tabassum ko'rinmaydi
            on -= {"beard", "mustache", "smile"}
        return [k for k in self.keys if k in on]
