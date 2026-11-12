"""
Arithmetic CAPTCHA solver: read the expression glyph by glyph, then evaluate
it with the AST-restricted evaluator (never eval()).
"""
from __future__ import annotations

import re
import time

import numpy as np

from app.solvers.base import SolveResult
from app.solvers.glyphs import binarize, components, glyph_vector, math_knn
from app.solvers.safe_math import safe_eval

_EXPR = re.compile(r"^\d{1,2}[+\-x]\d{1,2}$")


def read_expression(gray: np.ndarray) -> tuple[str, list[float], list[float]]:
    """Return (expression, per-glyph vote shares, per-glyph distances)."""
    b = binarize(gray)
    knn = math_knn()
    chars, shares, dists = [], [], []
    for box in components(b, 8):
        label, share, dist = knn.predict(glyph_vector(b, box))
        chars.append(label)
        shares.append(share)
        dists.append(dist)
    return "".join(chars), shares, dists


class MathSolver:
    name = "math_knn"

    def solve(self, gray: np.ndarray) -> SolveResult:
        t0 = time.perf_counter()
        expr, shares, _ = read_expression(gray)
        if not _EXPR.match(expr):
            raise ValueError(f"unreadable arithmetic expression: {expr!r}")
        answer = safe_eval(expr)
        return SolveResult(
            text=str(answer),
            confidence=float(min(shares)),
            method=self.name,
            ms=(time.perf_counter() - t0) * 1000,
            detail={"expression": expr},
        )
