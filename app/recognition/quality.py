"""
Pre-solve image quality gate.

The CRNN is overconfident on garbage (pure noise decoded as "MMMMM" at 0.95;
heavy blur as "MMMM" at 0.98), so confidence alone cannot reject unreadable
input. These checks run first. Thresholds come from measurements on generated
data (see docs/design.md, "Edge cases"):

    good text   : intensity range >= 60, sharpness ratio >= 0.108
    good math   : intensity range >= 169, sharpness ratio >= 0.343
    blur sigma>=1.2 : ratio <= ~0.08  (CRNN exact-match 89% at 1.2, 24% at 2.0)
    random noise: ratio ~ 19.7
    sharpness ratio = Laplacian variance / pixel variance (contrast invariant)
"""
from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

MIN_RANGE_REJECT = 20.0       # p99 - p1 below this: blank or flat
STRETCH_BELOW = 62.0          # below the lowest range seen in normal images
MAX_NOISE_RATIO = 5.0
MIN_SHARPNESS_RATIO = 0.09


@dataclass
class QualityReport:
    ok: bool
    reason: str | None
    metrics: dict = field(default_factory=dict)
    gray: np.ndarray | None = None     # possibly contrast-stretched copy


def _stretch(gray: np.ndarray) -> np.ndarray:
    lo, hi = np.percentile(gray, [1, 99])
    out = (gray.astype(np.float32) - lo) * (255.0 / max(hi - lo, 1.0))
    return np.clip(out, 0, 255).astype(np.uint8)


def assess(gray: np.ndarray) -> QualityReport:
    lo, hi = np.percentile(gray, [1, 99])
    rng = float(hi - lo)
    var = float(gray.var())
    ratio = float(cv2.Laplacian(gray, cv2.CV_64F).var() / (var + 1e-9)) if var > 0 else 0.0
    metrics = {"range": round(rng, 1), "sharpness_ratio": round(ratio, 4), "contrast_stretched": False}

    if rng < MIN_RANGE_REJECT:
        return QualityReport(False, "blank_or_flat", metrics)
    if ratio > MAX_NOISE_RATIO:
        return QualityReport(False, "excessive_noise", metrics)
    if ratio < MIN_SHARPNESS_RATIO:
        return QualityReport(False, "too_blurry", metrics)
    if rng < STRETCH_BELOW:
        metrics["contrast_stretched"] = True
        return QualityReport(True, None, metrics, _stretch(gray))
    return QualityReport(True, None, metrics, gray)
