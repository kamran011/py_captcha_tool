"""
Glyph segmentation and a kNN glyph classifier (OpenCV only, no neural net).

Shared by the fallback text solver, the math solver and the image classifier.
Templates are rendered from our own synthetic generators, cached under
data/models/, and rebuilt automatically if the cache is missing.
"""
from __future__ import annotations

import os
import random
import threading
from pathlib import Path

import cv2
import numpy as np

GLYPH = 20
CACHE_DIR = Path("data/models")


def binarize(gray: np.ndarray) -> np.ndarray:
    """Ink = 255. Otsu on a lightly blurred image."""
    blur = cv2.GaussianBlur(gray, (3, 3), 0)
    _, b = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    return b


def clean(b: np.ndarray, k: int = 2) -> np.ndarray:
    """Remove hairline noise (1-px lines, specks) with a small opening."""
    return cv2.morphologyEx(b, cv2.MORPH_OPEN, np.ones((k, k), np.uint8))


def components(b: np.ndarray, min_area: int = 20) -> list[tuple[int, int, int, int, int]]:
    """Connected components as (x, y, w, h, area), left to right."""
    n, _, stats, _ = cv2.connectedComponentsWithStats(b, connectivity=8)
    boxes = [tuple(int(v) for v in stats[i]) for i in range(1, n) if stats[i][4] >= min_area]
    return sorted(boxes, key=lambda bx: bx[0])


def glyph_vector(b: np.ndarray, box: tuple[int, int, int, int, int]) -> np.ndarray:
    x, y, w, h, _ = box
    crop = b[y : y + h, x : x + w]
    s = max(w, h)
    sq = np.zeros((s, s), np.uint8)
    sq[(s - h) // 2 : (s - h) // 2 + h, (s - w) // 2 : (s - w) // 2 + w] = crop
    sq = cv2.resize(sq, (GLYPH, GLYPH), interpolation=cv2.INTER_AREA)
    return (sq.astype(np.float32) / 255.0).ravel()


class GlyphKNN:
    def __init__(self, vectors: np.ndarray, labels: list[str], k: int = 5) -> None:
        # Plain NumPy kNN: OpenCV 5 no longer ships cv2.ml.
        self.k = k
        self.classes = sorted(set(labels))
        idx = {c: i for i, c in enumerate(self.classes)}
        self._x = vectors.astype(np.float32)
        self._sq = (self._x**2).sum(axis=1)
        self._y = np.array([idx[c] for c in labels], np.int64)

    def predict(self, vec: np.ndarray) -> tuple[str, float, float]:
        """Return (label, vote_share, mean squared distance of k neighbours)."""
        v = vec.astype(np.float32)
        d2 = np.maximum(self._sq - 2.0 * (self._x @ v) + float(v @ v), 0.0)
        nn = np.argpartition(d2, self.k)[: self.k]
        votes = self._y[nn]
        label = int(np.bincount(votes, minlength=len(self.classes)).argmax())
        return self.classes[label], float((votes == label).mean()), float(d2[nn].mean())


def _largest(b: np.ndarray, min_area: int):
    comps = components(b, min_area)
    return max(comps, key=lambda c: c[4]) if comps else None


def _load_or_build(name: str, builder) -> tuple[np.ndarray, list[str]]:
    path = CACHE_DIR / name
    if path.exists():
        try:
            with np.load(path, allow_pickle=False) as data:
                return data["vecs"], [str(s) for s in data["labels"]]
        except Exception:  # noqa: BLE001 - truncated/corrupt cache: rebuild
            pass
    vecs, labels = builder()
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(f".{os.getpid()}.tmp.npz")   # write then atomic rename
    np.savez_compressed(tmp, vecs=vecs, labels=np.array(labels))
    os.replace(tmp, path)
    return vecs, labels


def _build_text() -> tuple[np.ndarray, list[str]]:
    from captcha.image import ImageCaptcha

    from ml.common import ALPHABET, IMG_H, IMG_W
    from ml.generate_dataset import make_image

    rng = random.Random(7)
    gen = ImageCaptcha(width=IMG_W * 2, height=IMG_H * 2, font_sizes=(56, 64, 72))
    vecs, labels = [], []
    for ch in ALPHABET:
        for _ in range(70):
            b = clean(binarize(make_image(ch, gen, rng)))
            box = _largest(b, 25)
            if box:
                vecs.append(glyph_vector(b, box))
                labels.append(ch)
    return np.array(vecs), labels


def _build_math() -> tuple[np.ndarray, list[str]]:
    from app.solvers.math_render import MATH_CHARS, render_expression

    rng = random.Random(11)
    vecs, labels = [], []
    for ch in MATH_CHARS:
        for _ in range(120):
            b = binarize(render_expression(ch, rng))
            box = _largest(b, 8)
            if box:
                vecs.append(glyph_vector(b, box))
                labels.append(ch)
    return np.array(vecs), labels


_cache: dict[str, GlyphKNN] = {}
_lock = threading.Lock()     # solvers run on a thread pool: build each table once


def text_knn() -> GlyphKNN:
    with _lock:
        if "text" not in _cache:
            _cache["text"] = GlyphKNN(*_load_or_build("glyphs_text.npz", _build_text))
        return _cache["text"]


def math_knn() -> GlyphKNN:
    with _lock:
        if "math" not in _cache:
            _cache["math"] = GlyphKNN(*_load_or_build("glyphs_math.npz", _build_math))
        return _cache["math"]
