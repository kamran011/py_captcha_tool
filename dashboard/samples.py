"""Demo inputs for the dashboard: held-out samples, math CAPTCHAs, unsupported images, degradations."""
from __future__ import annotations

import csv
import random
import threading
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
TEST_DIR = ROOT / "data" / "synthetic" / "test"
FIXTURES = ROOT / "tests" / "fixtures" / "detector"
DEGRADATIONS = ["none", "blur", "noise", "low contrast", "inverted", "mirrored"]


def _png(img: np.ndarray) -> bytes:
    return cv2.imencode(".png", img)[1].tobytes()


def test_samples(limit: int = 80) -> list[tuple[str, str]]:
    path = TEST_DIR / "labels.csv"
    if not path.exists():
        return []
    with open(path, encoding="utf-8") as f:
        return [(r["filename"], r["text"]) for _, r in zip(range(limit), csv.DictReader(f))]


def load_sample(filename: str) -> bytes:
    return (TEST_DIR / filename).read_bytes()


def math_sample(seed: int) -> tuple[bytes, str, int]:
    from app.solvers.math_render import random_expression, render_expression

    rng = random.Random(seed)
    expr, answer = random_expression(rng)
    return _png(render_expression(expr, rng)), expr, answer


HARD_LEVELS = {
    "Medium": dict(amp=2.0, lines=0, thick=1, angle=6, dots=40, fonts=(56, 64, 72)),
    "Hard": dict(amp=2.5, lines=1, thick=2, angle=8, dots=60, fonts=(56, 64, 72)),
    "Very hard": dict(amp=4.0, lines=3, thick=3, angle=10, dots=120, fonts=(64, 72, 80), skew=0.9, tilt=40),
    "Skewed": dict(amp=1.5, lines=0, thick=1, angle=3, dots=20, fonts=(56, 64, 72), skew=0.6, tilt=25),
    "Hatched": dict(amp=1.5, lines=0, thick=1, angle=4, dots=0, fonts=(56, 64, 72), hatch=12),
    "Crowded": dict(amp=1.5, lines=0, thick=1, angle=4, dots=20, fonts=(64, 72, 80), squeeze=0.62),
    "Ink blob": dict(amp=2.0, lines=0, thick=1, angle=6, dots=0, fonts=(56, 64, 72), blobs=2),
}
HARD_COUNT = 20
_CAPTCHA_LOCK = threading.Lock()


class _SeededSecrets:
    """Stand-in for the `secrets` calls the captcha library makes, so a sample index always draws the same image."""

    def __init__(self, rng: random.Random) -> None:
        self._r = rng

    def randbelow(self, n: int) -> int:
        return self._r.randrange(n)

    def randbits(self, k: int) -> int:
        return self._r.getrandbits(k)

    def choice(self, seq):
        return self._r.choice(seq)


def _skew_letters(g: np.ndarray, n: int, skew: float, tilt: float, rng: random.Random) -> np.ndarray:
    """Give each letter-sized strip its own shear and tilt; overlapping strips keep the darker pixel."""
    h, w = g.shape
    bg = int(np.median(g))
    cols = np.where((g < bg - 40).any(axis=0))[0]
    x0, x1 = (int(cols[0]), int(cols[-1]) + 1) if cols.size else (0, w)
    out = np.full_like(g, bg)
    edges = np.linspace(x0, x1, n + 1).astype(int)
    for a, b in zip(edges[:-1], edges[1:]):
        pad = (b - a) // 3
        lo, hi = max(a - pad, 0), min(b + pad, w)
        strip = np.full_like(g, bg)
        strip[:, a:b] = g[:, a:b]
        cx = (a + b) / 2
        m = cv2.getRotationMatrix2D((cx, h / 2), rng.uniform(-tilt, tilt), rng.uniform(0.85, 1.15))
        sh = rng.uniform(-skew, skew)
        m[0, 1] += sh
        m[0, 2] -= sh * h / 2
        m[1, 2] += rng.uniform(-h / 8, h / 8)
        moved = cv2.warpAffine(strip, m, (w, h), borderValue=bg)
        out[:, lo:hi] = np.minimum(out[:, lo:hi], moved[:, lo:hi])
    return out


def hard_sample(level: str, index: int) -> tuple[bytes, str]:
    """A text CAPTCHA in the training style, distorted beyond it: wave warp, tilt, crossing curves, speckle."""
    import captcha.image as ci
    from captcha.image import ImageCaptcha

    from ml.common import ALPHABET, IMG_H, IMG_W, MAX_LEN, MIN_LEN

    p = HARD_LEVELS[level]
    seed = 1000 + index
    rng = random.Random(seed)
    text = "".join(rng.choice(ALPHABET) for _ in range(rng.randint(MIN_LEN, MAX_LEN)))
    rng = random.Random(seed)
    with _CAPTCHA_LOCK:
        saved = getattr(ci, "secrets", None)
        ci.secrets = _SeededSecrets(rng)
        try:
            pil = ImageCaptcha(width=IMG_W * 2, height=IMG_H * 2, font_sizes=p["fonts"]).generate_image(text)
        finally:
            ci.secrets = saved
    g = cv2.resize(cv2.cvtColor(np.array(pil.convert("RGB")), cv2.COLOR_RGB2GRAY), (IMG_W, IMG_H),
                   interpolation=cv2.INTER_AREA)
    h, w = g.shape
    ink = int(np.percentile(g, 5))
    if p.get("squeeze"):
        narrow = cv2.resize(g, (int(w * p["squeeze"]), h), interpolation=cv2.INTER_AREA)
        left = (w - narrow.shape[1]) // 2
        g = np.full_like(g, int(np.median(g)))
        g[:, left:left + narrow.shape[1]] = narrow
    if p.get("skew"):
        g = _skew_letters(g, len(text), p["skew"], p.get("tilt", 0), rng)
    xs, ys = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
    phase, period = rng.uniform(0, 6.28), rng.uniform(22, 34)
    g = cv2.remap(g, xs, ys + p["amp"] * np.sin(xs / period * 6.28 + phase), cv2.INTER_LINEAR,
                  borderMode=cv2.BORDER_REPLICATE)
    m = cv2.getRotationMatrix2D((w / 2, h / 2), rng.uniform(-p["angle"], p["angle"]), 1.0)
    g = cv2.warpAffine(g, m, (w, h), borderMode=cv2.BORDER_REPLICATE)
    for _ in range(p["lines"]):
        y0, amp, per, ph = h / 2 + rng.uniform(-h / 4, h / 4), rng.uniform(4, 10), rng.uniform(30, 70), rng.uniform(0, 6.28)
        pts = np.array([[x, int(y0 + amp * np.sin(x / per * 6.28 + ph))] for x in range(0, w, 2)], np.int32)
        cv2.polylines(g, [pts], False, ink, p["thick"], cv2.LINE_AA)
    for _ in range(p["dots"]):
        cv2.circle(g, (rng.randrange(w), rng.randrange(h)), rng.choice([0, 1]), ink, -1)
    if p.get("hatch"):
        shade = (ink + int(np.median(g))) // 2
        for x in range(-h, w + h, p["hatch"]):
            cv2.line(g, (x, 0), (x + h, h), shade, 1, cv2.LINE_AA)
            cv2.line(g, (x + h, 0), (x, h), shade, 1, cv2.LINE_AA)
    for _ in range(p.get("blobs", 0)):
        mask = np.zeros_like(g)
        centre = (rng.randrange(w // 5, 4 * w // 5), rng.randrange(h // 3, 2 * h // 3))
        cv2.ellipse(mask, centre, (rng.randrange(14, 26), rng.randrange(10, 18)), rng.uniform(0, 180), 0, 360, 255, -1)
        g = np.where(mask > 0, 255 - g, g).astype(np.uint8)
    return _png(g), text


def grid_sample() -> bytes:
    """A 3x3 tile grid, the shape of an image-selection challenge."""
    rng = np.random.default_rng(1)
    g = (rng.random((150, 150)) * 60 + 150).astype(np.uint8)
    for i in range(0, 151, 50):
        cv2.line(g, (i, 0), (i, 149), 20, 2)
        cv2.line(g, (0, i), (149, i), 20, 2)
    return _png(g)


def degrade(image: bytes, mode: str) -> bytes:
    if mode == "none":
        return image
    g = cv2.imdecode(np.frombuffer(image, np.uint8), cv2.IMREAD_GRAYSCALE)
    if mode == "blur":
        g = cv2.GaussianBlur(g, (0, 0), 3)
    elif mode == "noise":
        g = (np.random.default_rng(0).random(g.shape) * 255).astype(np.uint8)
    elif mode == "low contrast":
        g = (g.astype(np.float32) * 0.5 + 255 * 0.5 * 0.8).clip(0, 255).astype(np.uint8)
    elif mode == "inverted":
        g = 255 - g
    elif mode == "mirrored":
        g = cv2.flip(g, 1)
    return _png(g)


def detector_fixtures() -> list[str]:
    return sorted(p.stem for p in FIXTURES.glob("*.html")) if FIXTURES.exists() else []


def load_fixture(name: str) -> str:
    return (FIXTURES / f"{name}.html").read_text(encoding="utf-8")
