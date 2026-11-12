"""
Generate a synthetic text-CAPTCHA dataset (candidate-controlled, per assignment §7).

Uses the `captcha` library's ImageCaptcha with random fonts sizes, plus extra
augmentation (noise, rotation jitter, blur) applied at generation time so
train / val / test splits differ only by random seed.

    python ml/generate_dataset.py --train 20000 --val 1000 --test 1000
    python ml/generate_dataset.py --math 500      # math-expression images (M2)

Output layout (labels are in a CSV, filenames are numeric):
    data/synthetic/{split}/000001.png
    data/synthetic/{split}/labels.csv   (filename,text)
"""
from __future__ import annotations

import argparse
import csv
import random
import sys
from pathlib import Path

import cv2
import numpy as np
from captcha.image import ImageCaptcha

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from ml.common import ALPHABET, IMG_H, IMG_W, MAX_LEN, MIN_LEN  # noqa: E402

OUT = Path("data/synthetic")


def augment(img: np.ndarray, rng: random.Random) -> np.ndarray:
    """Light augmentation on a grayscale uint8 image."""
    h, w = img.shape
    # small random rotation + shift
    angle = rng.uniform(-3, 3)
    tx, ty = rng.uniform(-3, 3), rng.uniform(-2, 2)
    m = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    m[:, 2] += (tx, ty)
    img = cv2.warpAffine(img, m, (w, h), borderValue=int(img.mean()))
    # gaussian noise
    noise = np.random.default_rng(rng.randrange(1 << 30)).normal(0, rng.uniform(2, 10), img.shape)
    img = np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    if rng.random() < 0.3:
        img = cv2.GaussianBlur(img, (3, 3), 0)
    return img


def make_image(text: str, gen: ImageCaptcha, rng: random.Random) -> np.ndarray:
    pil = gen.generate_image(text)
    arr = cv2.cvtColor(np.array(pil.convert("RGB")), cv2.COLOR_RGB2GRAY)
    arr = cv2.resize(arr, (IMG_W, IMG_H), interpolation=cv2.INTER_AREA)
    return augment(arr, rng)


class _SeededSecrets:
    """Drop-in for the three `secrets` functions the captcha library uses, backed by a seeded RNG."""

    def __init__(self, rng: random.Random) -> None:
        self._r = rng

    def randbelow(self, n: int) -> int:
        return self._r.randrange(n)

    def randbits(self, k: int) -> int:
        return self._r.getrandbits(k)

    def choice(self, seq):
        return self._r.choice(seq)


def generate_split(name: str, n: int, seed: int) -> None:
    rng = random.Random(seed)
    # Make the images (not just the labels) reproducible. `captcha` 0.7 draws fonts, warps and
    # noise from the `secrets` module (the OS CSPRNG), which cannot be seeded; older versions
    # used the global `random`. Dataset generation swaps in a seeded stand-in, so the same seed
    # gives byte-identical images on a given environment and the reported accuracy reproduces.
    random.seed(seed)
    import captcha.image as _ci

    if hasattr(_ci, "secrets"):
        _ci.secrets = _SeededSecrets(random.Random(seed))
    # Wider canvas then downscaled so strokes keep detail
    gen = ImageCaptcha(width=IMG_W * 2, height=IMG_H * 2,
                       font_sizes=(56, 64, 72))
    folder = OUT / name
    folder.mkdir(parents=True, exist_ok=True)
    for old in folder.glob("*.png"):
        old.unlink()
    with open(folder / "labels.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["filename", "text"])
        for i in range(n):
            length = rng.randint(MIN_LEN, MAX_LEN)
            text = "".join(rng.choice(ALPHABET) for _ in range(length))
            fname = f"{i:06d}.png"
            cv2.imwrite(str(folder / fname), make_image(text, gen, rng))
            w.writerow([fname, text])
            if (i + 1) % 2000 == 0:
                print(f"  {name}: {i + 1}/{n}", flush=True)
    print(f"{name}: wrote {n} images to {folder}")


def generate_math(name: str, n: int, seed: int) -> None:
    """Arithmetic CAPTCHAs; the label is the ANSWER (what a solver must return)."""
    from app.solvers.math_render import random_expression, render_expression

    rng = random.Random(seed)
    folder = OUT / name
    folder.mkdir(parents=True, exist_ok=True)
    for old in folder.glob("*.png"):
        old.unlink()
    with open(folder / "labels.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["filename", "text"])
        for i in range(n):
            expr, answer = random_expression(rng)
            fname = f"{i:06d}.png"
            cv2.imwrite(str(folder / fname), render_expression(expr, rng))
            w.writerow([fname, answer])
    print(f"{name}: wrote {n} images to {folder}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", type=int, default=20000)
    ap.add_argument("--val", type=int, default=1000)
    ap.add_argument("--test", type=int, default=1000)
    ap.add_argument("--math", type=int, default=0, help="also write N arithmetic CAPTCHAs to data/synthetic/math_test")
    args = ap.parse_args()
    if args.math:
        generate_math("math_test", args.math, seed=4)
        return
    # Distinct seeds guarantee the splits never share an image
    generate_split("train", args.train, seed=1)
    generate_split("val", args.val, seed=2)
    generate_split("test", args.test, seed=3)


if __name__ == "__main__":
    main()
