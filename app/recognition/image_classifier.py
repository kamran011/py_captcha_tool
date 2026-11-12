"""
Classify a challenge image as `text`, `math` or `unsupported`.

Rule-based and inspectable on purpose. `unsupported` covers anything the
solvers are not built for (image-selection grids, slider/puzzle strips,
photographs, blank images); those are never sent to a solver.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

from app.solvers.glyphs import binarize, components, glyph_vector, math_knn
from app.solvers.math_render import OPERATORS

# Tuned in tests/test_classifier.py against generated samples.
MAX_AREA = 60_000          # px; larger images look like grids / photos
MAX_ASPECT = 6.0           # wider than this looks like a slider strip
GRID_LINE_FRACTION = 0.6   # a separator must span at least this share of the image
EDGE_MARGIN = 4            # ignore lines this close to the border (a frame is not a grid)
MATH_MIN_SHARE = 0.8
MATH_MAX_DIST = 25.0       # squared glyph distance; text glyphs sit well above it


@dataclass
class ImageClass:
    type: str                 # text | math | unsupported
    confidence: float
    reasons: list[str] = field(default_factory=list)


def _interior_lines(mask: np.ndarray, length: int, vertical: bool) -> int:
    """Count separate long straight dark lines, ignoring any on the image border."""
    kernel = (1, length) if vertical else (length, 1)
    opened = cv2.morphologyEx(mask, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, kernel))
    profile = opened.sum(axis=0 if vertical else 1) > 0
    size = profile.size
    hits = [i for i in np.flatnonzero(profile) if EDGE_MARGIN <= i < size - EDGE_MARGIN]
    return int(1 + (np.diff(hits) > 3).sum()) if hits else 0


def _looks_like_grid(gray: np.ndarray) -> bool:
    """
    Tile grids (image-selection challenges) have long straight separators in
    both directions. Morphological opening with a long 1-px kernel keeps only
    runs that long, so textured photo backgrounds (which defeat edge-density
    tests) and short text strokes are both ignored. A plain frame around a text
    challenge has no interior lines and is not a grid.
    """
    h, w = gray.shape
    mask = binarize(gray)
    return (_interior_lines(mask, max(int(w * GRID_LINE_FRACTION), 10), vertical=False) >= 2
            and _interior_lines(mask, max(int(h * GRID_LINE_FRACTION), 10), vertical=True) >= 2)


def _looks_like_math(b: np.ndarray) -> tuple[bool, float]:
    comps = components(b, 8)
    if not 3 <= len(comps) <= 5:
        return False, 0.0
    knn = math_knn()
    labels, shares, dists = [], [], []
    for box in comps:
        label, share, dist = knn.predict(glyph_vector(b, box))
        labels.append(label)
        shares.append(share)
        dists.append(dist)
    ops = [i for i, c in enumerate(labels) if c in OPERATORS]
    ok = (
        len(ops) == 1
        and ops[0] in (1, 2)
        and len(labels) - ops[0] - 1 in (1, 2)
        and min(shares) >= MATH_MIN_SHARE
        and max(dists) <= MATH_MAX_DIST
    )
    return ok, float(np.mean(shares))


def classify_image(gray: np.ndarray) -> ImageClass:
    h, w = gray.shape[:2]
    if h * w > MAX_AREA:
        return ImageClass("unsupported", 0.9, [f"large image {w}x{h}: likely image grid or photo"])
    if max(w / h, h / w) > MAX_ASPECT:
        return ImageClass("unsupported", 0.8, [f"extreme aspect ratio {w}x{h}: likely slider strip"])
    if _looks_like_grid(gray):
        return ImageClass("unsupported", 0.85, ["regular grid lines detected"])

    b = binarize(gray)
    is_math, conf = _looks_like_math(b)
    if is_math:
        return ImageClass("math", conf, ["one operator glyph between digit groups"])

    n = len(components(b, 20))
    if n == 0:
        return ImageClass("unsupported", 0.7, ["no glyphs found"])
    if n > 12:
        return ImageClass("unsupported", 0.6, [f"{n} components: too cluttered for a text challenge"])
    return ImageClass("text", 0.8, [f"{n} glyph-sized components, no operator pattern"])
