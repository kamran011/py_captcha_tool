"""
Render synthetic arithmetic CAPTCHAs ("12+7", "9x3", "15-4") with OpenCV.

Candidate-controlled generator (assignment §7). Used both to build the glyph
templates the math solver learns from and to create evaluation images.
"""
from __future__ import annotations

import random

import cv2
import numpy as np

MATH_CHARS = "0123456789+-x"
OPERATORS = "+-x"
WIDTH, HEIGHT = 192, 56
_FONTS = (cv2.FONT_HERSHEY_SIMPLEX, cv2.FONT_HERSHEY_DUPLEX, cv2.FONT_HERSHEY_TRIPLEX)


def random_expression(rng: random.Random) -> tuple[str, int]:
    op = rng.choice(OPERATORS)
    if op == "+":
        a, b = rng.randint(1, 20), rng.randint(1, 20)
        return f"{a}+{b}", a + b
    if op == "-":
        a = rng.randint(5, 20)
        b = rng.randint(1, a - 1)
        return f"{a}-{b}", a - b
    a, b = rng.randint(2, 9), rng.randint(2, 9)
    return f"{a}x{b}", a * b


def render_expression(expr: str, rng: random.Random) -> np.ndarray:
    """Return a uint8 grayscale image of *expr* with light noise and clutter."""
    img = np.full((HEIGHT, WIDTH), rng.randint(215, 250), np.uint8)
    font = rng.choice(_FONTS)
    scale = rng.uniform(0.95, 1.25)
    thick = rng.choice((2, 2, 3))
    x = rng.randint(8, 22)
    for ch in expr:
        (cw, chh), _ = cv2.getTextSize(ch, font, scale, thick)
        y = HEIGHT // 2 + chh // 2 + rng.randint(-3, 3)
        cv2.putText(img, ch, (x, y), font, scale, rng.randint(0, 70), thick, cv2.LINE_AA)
        x += cw + rng.randint(7, 12)
    # clutter lighter than the ink so Otsu thresholding drops it
    for _ in range(rng.randint(1, 3)):
        p1 = (rng.randint(0, WIDTH), rng.randint(0, HEIGHT))
        p2 = (rng.randint(0, WIDTH), rng.randint(0, HEIGHT))
        cv2.line(img, p1, p2, rng.randint(150, 190), 1, cv2.LINE_8)
    for _ in range(rng.randint(20, 60)):
        img[rng.randrange(HEIGHT), rng.randrange(WIDTH)] = rng.randint(90, 160)
    noise = np.random.default_rng(rng.randrange(1 << 30)).normal(0, 5, img.shape)
    img = np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    angle = rng.uniform(-3, 3)
    m = cv2.getRotationMatrix2D((WIDTH / 2, HEIGHT / 2), angle, 1.0)
    return cv2.warpAffine(img, m, (WIDTH, HEIGHT), borderValue=int(img.mean()))
