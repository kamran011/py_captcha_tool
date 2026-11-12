"""
Opt-in test of the REAL vision provider. It SPENDS A FEW CENTS and sends three synthetic validation images
to the configured provider, so it only runs when you ask for it AND a key is configured:

    RUN_LIVE=1 VISION_API_KEY=... pytest tests/test_live_vision.py -v

Skipped by default (and in CI): the offline suite proves the request format, parsing, cost arithmetic,
retries and guards against a mock; this one proves the provider still accepts that request.
"""
from __future__ import annotations

import csv
import os
from pathlib import Path

import cv2
import pytest

from app.config import settings
from app.solvers.vision_api_solver import BudgetGuard, VisionApiSolver, availability

VAL = Path("data/synthetic/val")
pytestmark = pytest.mark.skipif(
    os.getenv("RUN_LIVE") != "1" or not availability(settings)[0] or not (VAL / "labels.csv").exists(),
    reason="needs RUN_LIVE=1, a configured VISION_API_KEY and the generated dataset",
)


def test_real_provider_reads_synthetic_captchas_and_reports_a_sane_cost():
    rows = list(csv.DictReader(open(VAL / "labels.csv", encoding="utf-8")))[:3]
    guard = BudgetGuard(0.05, 10)                                   # hard stop at 5 cents, whatever happens
    solver = VisionApiSolver(settings, budget=guard)
    results = [solver.solve(cv2.imread(str(VAL / r["filename"]), cv2.IMREAD_GRAYSCALE)) for r in rows]
    for r, res in zip(rows, results):
        print(f"{r['text']} -> {res.text} (confidence {res.confidence}, ${res.cost_usd:.6f})")
        assert 4 <= len(res.text) <= 6 and 0.0 <= res.confidence <= 1.0
        assert 0 < res.cost_usd < 0.01                                # a few thousandths of a cent to a cent
        assert res.detail["input_tokens"] > 0 and res.detail["output_tokens"] > 0
    assert guard.status()["spent_today_usd"] == pytest.approx(sum(r.cost_usd for r in results))
