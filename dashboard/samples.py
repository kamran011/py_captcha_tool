"""Demo inputs for the dashboard: held-out samples, math CAPTCHAs, unsupported images, degradations."""
from __future__ import annotations

import csv
import random
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
