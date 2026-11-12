"""
Fallback text solver: OpenCV segmentation + kNN glyph classification.

Independent of the neural model, so it still works if the ONNX file is
missing. It is weaker on touching or overlapping characters; its accuracy is
measured in ml/evaluate_pipeline.py rather than assumed.
"""
from __future__ import annotations

import time

import cv2
import numpy as np

from app.solvers.base import SolveResult
from app.solvers.glyphs import binarize, clean, components, glyph_vector, text_knn
from ml.common import IMG_H, IMG_W, MAX_LEN


class SegmentSolver:
    name = "segment_knn"

    def solve(self, gray: np.ndarray) -> SolveResult:
        t0 = time.perf_counter()
        g = cv2.resize(gray, (IMG_W, IMG_H), interpolation=cv2.INTER_AREA)
        b = clean(binarize(g))
        comps = [c for c in components(b, 25) if c[3] >= IMG_H * 0.3]
        boxes = self._split_wide(comps)[: MAX_LEN + 2]

        knn = text_knn()
        chars, shares = [], []
        for box in boxes:
            label, share, _ = knn.predict(glyph_vector(b, box))
            chars.append(label)
            shares.append(share)
        return SolveResult(
            text="".join(chars),
            confidence=float(np.mean(shares)) if shares else 0.0,
            method=self.name,
            ms=(time.perf_counter() - t0) * 1000,
            detail={"glyphs": len(boxes)},
        )

    @staticmethod
    def _split_wide(comps: list) -> list:
        """Split components much wider than the median (touching characters)."""
        if not comps:
            return comps
        widths = sorted(c[2] for c in comps)
        median = widths[len(widths) // 2]
        out = []
        for x, y, w, h, area in comps:
            parts = int(round(w / median)) if median and w > 1.6 * median else 1
            parts = max(1, min(parts, 3))
            step = w // parts
            for i in range(parts):
                out.append((x + i * step, y, step if i < parts - 1 else w - step * i, h, area // parts))
        return out
