"""Solver contract shared by every solving method."""
from __future__ import annotations

import io
from dataclasses import dataclass, field
from typing import Protocol

import cv2
import numpy as np
from PIL import Image

MAX_BYTES = 10 * 1024 * 1024        # reject oversized uploads before decoding
MAX_PIXELS = 16_000_000             # decompression-bomb guard (read from header)


class SolverUnavailable(RuntimeError):
    """Raised when a solver cannot run (model file missing, API key unset...)."""


@dataclass
class SolveResult:
    text: str
    confidence: float
    method: str
    cost_usd: float = 0.0
    ms: float = 0.0
    detail: dict = field(default_factory=dict)


class Solver(Protocol):
    name: str

    def solve(self, gray: np.ndarray) -> SolveResult:  # pragma: no cover - protocol
        ...


def load_gray(source: bytes | np.ndarray) -> np.ndarray:
    """
    Return uint8 grayscale from encoded bytes or an ndarray.

    Raises ValueError with a fixed, non-revealing message on any bad input
    (library error text can contain build paths, so it is never passed on).
    """
    if isinstance(source, (bytes, bytearray)):
        if not source:
            raise ValueError("empty image data")
        if len(source) > MAX_BYTES:
            raise ValueError("image file too large")
        try:
            with Image.open(io.BytesIO(source)) as im:      # header only, no decode
                width, height = im.size
        except Exception:  # noqa: BLE001 - PIL raises many types for non-images
            raise ValueError("could not decode image bytes") from None
        if width * height > MAX_PIXELS:
            raise ValueError("image dimensions too large")
        try:
            arr = cv2.imdecode(np.frombuffer(source, np.uint8), cv2.IMREAD_GRAYSCALE)
        except cv2.error:
            arr = None
        if arr is None:
            raise ValueError("could not decode image bytes")
        return arr

    if not isinstance(source, np.ndarray) or source.dtype != np.uint8:
        raise ValueError("image array must be uint8")
    if source.ndim == 2:
        return source
    if source.ndim == 3 and source.shape[2] in (3, 4):
        code = cv2.COLOR_BGR2GRAY if source.shape[2] == 3 else cv2.COLOR_BGRA2GRAY
        return cv2.cvtColor(source, code)
    raise ValueError("unsupported image array shape")
