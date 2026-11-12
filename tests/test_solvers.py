"""M2 tests: safe math, math solver, classifier, fallback pipeline."""
from __future__ import annotations

import csv
import random
import time
from pathlib import Path

import cv2
import numpy as np
import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.models import Base
from app.recognition.image_classifier import classify_image
from app.solvers.base import SolveResult, SolverUnavailable
from app.solvers.math_render import random_expression, render_expression
from app.solvers.math_solver import MathSolver
from app.solvers.pipeline import SolvePipeline, record_attempt
from app.solvers.safe_math import safe_eval

TEST_DIR = Path("data/synthetic/test")
needs_data = pytest.mark.skipif(
    not (TEST_DIR / "labels.csv").exists() or not Path("data/models/crnn.onnx").exists(),
    reason="run ml/generate_dataset.py and ml/train_crnn.py first",
)


def _text_samples(n):
    rows = list(csv.DictReader(open(TEST_DIR / "labels.csv", encoding="utf-8")))[:n]
    return [(cv2.imread(str(TEST_DIR / r["filename"]), 0), r["text"]) for r in rows]


# ── safe_eval ────────────────────────────────────────────────────────────────
class TestSafeEval:
    @pytest.mark.parametrize("expr,want", [("2+3", 5), ("9x3", 27), ("15-4", 11), ("2+3x4", 14)])
    def test_valid(self, expr, want):
        assert safe_eval(expr) == want

    @pytest.mark.parametrize("expr", [
        "__import__('os').system('echo hi')", "open('x')", "2**999999", "a+1",
        "(1).__class__", "[1,2]", "1 if 1 else 2", "", "9" * 50,
    ])
    def test_rejects_unsafe(self, expr):
        with pytest.raises((ValueError, SyntaxError)):
            safe_eval(expr)


# ── math solver ──────────────────────────────────────────────────────────────
class TestMathSolver:
    def test_reads_and_evaluates(self):
        rng = random.Random(123)
        solver, ok, n = MathSolver(), 0, 100
        for _ in range(n):
            expr, answer = random_expression(rng)
            ok += solver.solve(render_expression(expr, rng)).text == str(answer)
        assert ok / n >= 0.95

    def test_blank_image_raises(self):
        with pytest.raises(ValueError):
            MathSolver().solve(np.full((56, 192), 240, np.uint8))


# ── classifier ───────────────────────────────────────────────────────────────
class TestClassifier:
    def test_math_images(self):
        rng = random.Random(1)
        got = [classify_image(render_expression(random_expression(rng)[0], rng)).type for _ in range(100)]
        assert got.count("math") >= 95

    @needs_data
    def test_text_images(self):
        got = [classify_image(img).type for img, _ in _text_samples(100)]
        assert got.count("text") >= 95

    @pytest.mark.parametrize("n,size,border", [(3, 150, False), (3, 200, True), (4, 220, True)])
    def test_noisy_tile_grids_are_unsupported_even_when_small(self, n, size, border):
        rs = np.random.default_rng(5)
        im = (rs.random((size, size)) * 90 + 110).astype(np.uint8)       # textured, photo-like
        step = size // n
        for i in range(0 if border else step, size, step):
            cv2.line(im, (i, 0), (i, size - 1), 10, 2)
            cv2.line(im, (0, i), (size - 1, i), 10, 2)
        assert classify_image(im).type == "unsupported"

    def test_framed_text_is_not_mistaken_for_a_grid(self):
        # Known limit (documented in design.md): a 2x2 grid has one interior separator
        # per direction and is NOT caught; a plain border must never be flagged.
        img = np.full((48, 160), 240, np.uint8)
        cv2.putText(img, "AB7K", (10, 36), cv2.FONT_HERSHEY_SIMPLEX, 1.2, 20, 3)
        cv2.rectangle(img, (0, 0), (159, 47), 0, 2)
        assert classify_image(img).type == "text"

    def test_unsupported_shapes(self):
        r = np.random.default_rng(0)
        photo = (r.random((300, 400)) * 255).astype(np.uint8)
        grid = np.full((150, 150), 230, np.uint8)
        for i in range(0, 151, 50):
            cv2.line(grid, (i, 0), (i, 149), 20, 2)
            cv2.line(grid, (0, i), (149, i), 20, 2)
        slider = np.full((40, 300), 200, np.uint8)
        blank = np.full((48, 160), 240, np.uint8)
        for img in (photo, grid, slider, blank):
            assert classify_image(img).type == "unsupported"


# ── pipeline: fallback, timeout, unsupported ─────────────────────────────────
class _Stub:
    def __init__(self, name, text="ABCD", conf=0.99, exc=None, delay=0.0):
        self.name, self.text, self.conf, self.exc, self.delay = name, text, conf, exc, delay
        self.calls = 0

    def solve(self, gray):
        self.calls += 1
        time.sleep(self.delay)
        if self.exc:
            raise self.exc
        return SolveResult(self.text, self.conf, self.name)


def _pipe(*solvers, **kw):
    kw.setdefault("calibration", {})
    return SolvePipeline(solvers_for=lambda t: list(solvers), **kw)


TEXT_IMG = np.full((48, 160), 240, np.uint8)
cv2.putText(TEXT_IMG, "AB7K", (10, 36), cv2.FONT_HERSHEY_SIMPLEX, 1.2, 20, 3)


class TestPipeline:
    def test_low_confidence_falls_through(self):
        a, b = _Stub("weak", "ZZZZ", 0.2), _Stub("strong", "ABCD", 0.95)
        res = _pipe(a, b).solve(TEXT_IMG, challenge_type="text")
        assert res.accepted and res.method == "strong" and res.text == "ABCD"
        assert [t["method"] for t in res.attempts] == ["weak", "strong"]
        assert res.attempts[0]["error"] == "low_confidence"

    def test_primary_accepted_skips_rest(self):
        a, b = _Stub("first", conf=0.9), _Stub("second")
        res = _pipe(a, b).solve(TEXT_IMG, challenge_type="text")
        assert res.method == "first" and b.calls == 0

    def test_error_and_unavailable_fall_through(self):
        a = _Stub("boom", exc=RuntimeError("x"))
        b = _Stub("nomodel", exc=SolverUnavailable("missing"))
        c = _Stub("ok", conf=0.9)
        res = _pipe(a, b, c).solve(TEXT_IMG, challenge_type="text")
        assert res.method == "ok"
        assert res.attempts[0]["error"].startswith("RuntimeError")
        assert res.attempts[1]["error"].startswith("unavailable")

    def test_implausible_length_rejected(self):
        a, b = _Stub("short", "AB", 0.99), _Stub("good", "ABCDE", 0.9)
        res = _pipe(a, b).solve(TEXT_IMG, challenge_type="text")
        assert res.method == "good" and res.attempts[0]["error"] == "implausible_length"

    def test_timeout_falls_through(self):
        a, b = _Stub("slow", delay=0.5), _Stub("fast", conf=0.9)
        res = _pipe(a, b, timeout_s=0.1).solve(TEXT_IMG, challenge_type="text")
        assert res.method == "fast" and res.attempts[0]["error"] == "timeout"

    def test_nothing_accepted_returns_best_unaccepted(self):
        res = _pipe(_Stub("a", conf=0.2), _Stub("b", conf=0.4)).solve(TEXT_IMG, challenge_type="text")
        assert not res.accepted and res.method == "b" and res.error == "low_confidence"

    def test_advisory_solver_never_self_accepts(self):
        adv = _Stub("adv", "ABCD", 1.0)
        res = _pipe(adv, calibration={"adv": {"threshold": None}}).solve(TEXT_IMG, challenge_type="text")
        assert res.text == "ABCD" and not res.accepted and res.error == "advisory_only"

    def test_per_solver_threshold_applies(self):
        a = _Stub("a", conf=0.8)
        res = _pipe(a, calibration={"a": {"threshold": 0.9}}).solve(TEXT_IMG, challenge_type="text")
        assert not res.accepted and res.error == "low_confidence"

    def test_primary_beats_advisory_when_both_unaccepted(self):
        res = _pipe(_Stub("p", "ABCD", 0.3), _Stub("adv", "WXYZ", 1.0),
                    calibration={"adv": {"threshold": None}}).solve(TEXT_IMG, challenge_type="text")
        assert res.method == "adv" and not res.accepted  # highest-confidence candidate, flagged

    def test_all_fail(self):
        res = _pipe(_Stub("a", exc=RuntimeError("x"))).solve(TEXT_IMG, challenge_type="text")
        assert res.text is None and res.error == "all_solvers_failed"

    def test_unsupported_never_reaches_solver(self):
        s = _Stub("never")
        photo = (np.random.default_rng(0).random((300, 400)) * 255).astype(np.uint8)
        res = _pipe(s).solve(photo)
        assert res.challenge_type == "unsupported" and res.error == "unsupported_challenge"
        assert s.calls == 0 and res.attempts == []

    def test_bad_bytes(self):
        res = _pipe(_Stub("x")).solve(b"not an image")
        assert res.error.startswith("invalid_image")


# ── real models end to end ───────────────────────────────────────────────────
@needs_data
class TestRealPipeline:
    def test_text_accuracy_and_classification(self):
        pipe, ok, samples = SolvePipeline(), 0, _text_samples(150)
        for img, truth in samples:
            res = pipe.solve(img)
            assert res.challenge_type == "text"
            ok += res.text == truth
        assert ok / len(samples) >= 0.9

    def test_math_end_to_end(self):
        pipe, rng, ok = SolvePipeline(), random.Random(77), 0
        for _ in range(50):
            expr, answer = random_expression(rng)
            res = pipe.solve(render_expression(expr, rng))
            ok += res.challenge_type == "math" and res.text == str(answer)
        assert ok >= 47


# ── persistence ──────────────────────────────────────────────────────────────
def test_record_attempt():
    engine = create_engine("sqlite://")
    Base.metadata.create_all(engine)
    with sessionmaker(bind=engine)() as db:
        res = _pipe(_Stub("m", "ABCD", 0.9)).solve(TEXT_IMG, challenge_type="text")
        row = record_attempt(db, res, expected="ABCD", filename="x.png")
        assert row.success is True and row.method_used == "m"
        wrong = record_attempt(db, res, expected="WXYZ")
        assert wrong.success is False


def test_hard_samples_are_reproducible():
    from dashboard import samples

    for level in samples.HARD_LEVELS:
        assert samples.hard_sample(level, 4) == samples.hard_sample(level, 4)


@needs_data
def test_hard_samples_push_the_chain_into_fallback():
    """Medium stays readable for the CRNN; harder levels are rejected by it and fall through to segmentation."""
    from dashboard import samples

    pipe = SolvePipeline()
    for level in ("Medium", "Hard", "Very hard"):
        read = fell_back = 0
        for i in range(samples.HARD_COUNT):
            image, text = samples.hard_sample(level, i)
            res = pipe.solve(image, "text")
            crnn = res.attempts[0]
            read += crnn["accepted"] and crnn["text"] == text
            fell_back += not crnn["accepted"] and [a["method"] for a in res.attempts][:2] == ["crnn_onnx", "segment_knn"]
        if level == "Medium":
            assert read >= 17, (level, read)
        else:
            assert fell_back >= 18, (level, fell_back)


@needs_data
def test_hard_styles_are_solvable_inputs_not_crashes():
    """The look-alike styles reach the solvers (not refused wholesale by the quality gate) and never error."""
    from dashboard import samples

    pipe = SolvePipeline()
    for level in ("Skewed", "Hatched", "Crowded", "Ink blob"):
        reached = 0
        for i in range(samples.HARD_COUNT):
            res = pipe.solve(samples.hard_sample(level, i)[0], "text")
            assert res.error != "internal_error", level
            reached += bool(res.attempts)
        assert reached >= 18, (level, reached)
