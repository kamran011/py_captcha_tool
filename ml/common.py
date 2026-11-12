"""Shared constants and image preprocessing for training and inference."""
from __future__ import annotations

import numpy as np

# Characters used by the synthetic generator (ambiguous 0/O/1/I/l removed)
ALPHABET = "23456789ABCDEFGHJKLMNPQRSTUVWXYZ"
BLANK = 0                      # CTC blank index
CHAR_TO_IDX = {c: i + 1 for i, c in enumerate(ALPHABET)}
IDX_TO_CHAR = {i + 1: c for i, c in enumerate(ALPHABET)}
NUM_CLASSES = len(ALPHABET) + 1

IMG_H, IMG_W = 48, 160         # model input size (grayscale)
MIN_LEN, MAX_LEN = 4, 6


def preprocess(gray: np.ndarray) -> np.ndarray:
    """uint8 grayscale HxW -> float32 1xHxW in [0,1], resized to IMG_H x IMG_W."""
    import cv2

    if gray.ndim == 3:
        gray = cv2.cvtColor(gray, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, (IMG_W, IMG_H), interpolation=cv2.INTER_AREA)
    return (gray.astype(np.float32) / 255.0)[None, :, :]


def ctc_greedy_decode(logits: np.ndarray) -> tuple[str, float]:
    """
    logits: (T, C) raw scores for one image.
    Returns (text, confidence) where confidence is the mean of the max
    softmax probability over the non-blank emitted frames.
    """
    text, mean, _ = ctc_decode_detailed(logits)
    return text, mean


def ctc_decode_detailed(logits: np.ndarray) -> tuple[str, float, float]:
    """Like ctc_greedy_decode but also returns the weakest character's probability."""
    e = np.exp(logits - logits.max(axis=1, keepdims=True))
    probs = e / e.sum(axis=1, keepdims=True)
    best = probs.argmax(axis=1)
    chars, confs, prev = [], [], BLANK
    for t, idx in enumerate(best):
        if idx != BLANK and idx != prev:
            chars.append(IDX_TO_CHAR[int(idx)])
            confs.append(float(probs[t, idx]))
        prev = idx
    text = "".join(chars)
    mean = float(np.mean(confs)) if confs else 0.0
    weakest = float(np.min(confs)) if confs else 0.0
    return text, mean, weakest
