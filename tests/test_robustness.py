"""Failure-mode regression tests: bad input, log/DB secret hygiene, concurrency."""
from __future__ import annotations

import csv
import io
import logging
import struct
import threading
from pathlib import Path

import cv2
import numpy as np
import pytest
from sqlalchemy import create_engine, event
from sqlalchemy.orm import sessionmaker

from app.logging_config import RedactingFilter, redact, register_secrets
from app.models import Attempt, Base
from app.solvers.base import MAX_PIXELS, load_gray
from app.solvers.pipeline import SolvePipeline, record_attempt

TEST_DIR = Path("data/synthetic/test")
needs_data = pytest.mark.skipif(
    not (TEST_DIR / "labels.csv").exists() or not Path("data/models/crnn.onnx").exists(),
    reason="run ml/generate_dataset.py and ml/train_crnn.py first",
)
png = lambda im: cv2.imencode(".png", im)[1].tobytes()  # noqa: E731


@pytest.fixture(scope="module")
def pipe():
    return SolvePipeline()


@pytest.fixture(scope="module")
def good():
    row = next(csv.DictReader(open(TEST_DIR / "labels.csv", encoding="utf-8")))
    return cv2.imread(str(TEST_DIR / row["filename"]), 0), row["text"]


# ── bad images ───────────────────────────────────────────────────────────────
@needs_data
class TestBadInput:
    def test_control_still_solves(self, pipe, good):
        img, truth = good
        res = pipe.solve(png(img))
        assert res.accepted and res.text == truth

    @pytest.mark.parametrize("payload", [
        b"", b"\x00" * 64, np.random.default_rng(0).bytes(2000),
        b"RIFF" + struct.pack("<I", 1000) + b"WAVEfmt " + bytes(100),    # audio
    ])
    def test_non_image_bytes(self, pipe, payload):
        res = pipe.solve(payload)
        assert res.error.startswith("invalid_image") and res.text is None

    def test_truncated_png(self, pipe, good):
        res = pipe.solve(png(good[0])[:100])
        assert res.error.startswith("invalid_image")

    def test_error_text_does_not_leak_library_paths(self, pipe):
        for payload in (b"", b"junk", np.random.default_rng(1).bytes(500)):
            err = pipe.solve(payload).error
            assert "opencv" not in err.lower() and ":\\" not in err and "cpp" not in err

    def test_oversized_dimensions_rejected_before_decode(self, pipe):
        side = int(MAX_PIXELS**0.5) + 100
        res = pipe.solve(png(np.full((side, side), 200, np.uint8)))
        assert res.error == "invalid_image: image dimensions too large"

    def test_oversized_file_rejected(self, pipe):
        res = pipe.solve(b"\x89PNG" + bytes(11 * 1024 * 1024))
        assert res.error == "invalid_image: image file too large"

    @pytest.mark.parametrize("img", [
        np.zeros((48, 160), np.uint8), np.full((48, 160), 255, np.uint8),
        np.full((48, 160), 128, np.uint8), np.zeros((1, 1), np.uint8),
    ])
    def test_blank_and_flat_rejected(self, pipe, img):
        res = pipe.solve(png(img))
        assert not res.accepted and res.text is None

    def test_pure_noise_not_accepted(self, pipe):
        for seed in range(10):
            noise = (np.random.default_rng(seed).random((48, 160)) * 255).astype(np.uint8)
            res = pipe.solve(png(noise))
            assert not res.accepted, res.text
            # rejected either by the classifier (too cluttered) or the quality gate
            assert res.error == "unsupported_challenge" or res.error == "poor_quality: excessive_noise"

    @pytest.mark.parametrize("sigma", [2, 3, 4, 8])
    def test_heavy_blur_not_accepted(self, pipe, good, sigma):
        res = pipe.solve(png(cv2.GaussianBlur(good[0], (0, 0), sigma)))
        assert not res.accepted
        assert res.error == "poor_quality: too_blurry"

    def test_low_contrast_is_rescued_not_rejected(self, pipe):
        rows = list(csv.DictReader(open(TEST_DIR / "labels.csv", encoding="utf-8")))[:150]
        ok = 0
        for r in rows:
            g = cv2.imread(str(TEST_DIR / r["filename"]), 0)
            dim = (g.astype(np.float32) * 0.5 + 255 * 0.5 * 0.8).clip(0, 255).astype(np.uint8)
            res = pipe.solve(png(dim))
            ok += res.accepted and res.text == r["text"]
        # Measured on the reproducible test split (n=300): 61.7% through the pipeline, 25.7% for the
        # raw model. The bar sits well below the first and well above the second.
        assert ok / len(rows) >= 0.5

    def test_inverted_images_never_accepted(self, pipe):
        # Measured: 0 of 60 accepted. NOT covered (known limitation, see design.md):
        # 90-degree-rotated (23/60) and mirrored (59/60) images are accepted with
        # wrong text, because they look statistically normal to every gate.
        rows = list(csv.DictReader(open(TEST_DIR / "labels.csv", encoding="utf-8")))[:60]
        accepted = sum(
            pipe.solve(png(255 - cv2.imread(str(TEST_DIR / r["filename"]), 0))).accepted
            for r in rows
        )
        assert accepted == 0

    def test_audio_challenge_and_grid_are_unsupported(self, pipe):
        grid = np.full((150, 150), 230, np.uint8)
        for i in range(0, 151, 50):
            cv2.line(grid, (i, 0), (i, 149), 20, 2)
            cv2.line(grid, (0, i), (149, i), 20, 2)
        assert pipe.solve(png(grid)).error == "unsupported_challenge"

    def test_never_raises_on_weird_arrays(self, pipe):
        for arr in (np.zeros((48, 160), np.float32), np.zeros((4, 4, 2), np.uint8),
                    np.zeros(10, np.uint8), "not an image", None):
            res = pipe.solve(arr)             # type: ignore[arg-type]
            assert res.text is None and res.error

    def test_degenerate_repeated_letter_output_rejected(self):
        from app.solvers.base import SolveResult

        class Sink:
            name = "sink"
            def solve(self, gray):
                return SolveResult("MMMMM", 0.99, "sink")

        p = SolvePipeline(solvers_for=lambda t: [Sink()], calibration={})
        img = cv2.putText(np.full((48, 160), 240, np.uint8), "AB7K", (10, 36),
                          cv2.FONT_HERSHEY_SIMPLEX, 1.2, 20, 3)
        res = p.solve(img, challenge_type="text")
        assert not res.accepted and res.attempts[0]["error"] == "degenerate_output"


def test_load_gray_accepts_bgra_and_color():
    assert load_gray(np.zeros((8, 8, 4), np.uint8)).shape == (8, 8)
    assert load_gray(np.zeros((8, 8, 3), np.uint8)).shape == (8, 8)


# ── logging: secrets and tokens must not survive ─────────────────────────────
class TestLogRedaction:
    @pytest.mark.parametrize("line", [
        "verify failed token=AbC123xyz987TOKEN reason=bad",
        'payload {"token": "AbC123xyz987TOKEN", "provider": "x"}',
        "payload {'secret': 'AbC123xyz987TOKEN'}",
        "g-recaptcha-response=AbC123xyz987TOKEN&foo=1",
        "Authorization: Bearer AbC123xyz987TOKEN",
        "x-api-key: AbC123xyz987TOKEN",
        "RECAPTCHA_SECRET=AbC123xyz987TOKEN",
        "cf-turnstile-response: AbC123xyz987TOKEN",
    ])
    def test_unregistered_token_values_removed(self, line):
        assert "AbC123xyz987TOKEN" not in redact(line)

    def test_fingerprint_and_counts_survive(self):
        out = redact("token_fingerprint=ab12cd34ef56 max_tokens=500 status=200")
        assert "ab12cd34ef56" in out and "max_tokens=500" in out and "status=200" in out

    def test_registered_secret_removed_anywhere(self):
        register_secrets(["registered-secret-value-1"])
        assert "registered-secret-value-1" not in redact("oops registered-secret-value-1 leaked")

    def test_traceback_text_is_redacted(self):
        stream = io.StringIO()
        handler = logging.StreamHandler(stream)
        handler.addFilter(RedactingFilter())
        handler.setFormatter(logging.Formatter("%(message)s"))
        log = logging.getLogger("redaction-traceback-test")
        log.propagate, log.handlers = False, [handler]
        try:
            raise RuntimeError("upstream said token=AbC123xyz987TOKEN rejected")
        except RuntimeError:
            log.exception("verify blew up")
        assert "verify blew up" in stream.getvalue()
        assert "AbC123xyz987TOKEN" not in stream.getvalue()

    def test_clean_records_keep_args_for_uvicorn_style_formatters(self):
        rec = logging.LogRecord("uvicorn.access", logging.INFO, "", 0,
                                '%s - "%s %s" %d', ("127.0.0.1:1", "GET", "/health", 200), None)
        RedactingFilter().filter(rec)
        assert rec.args == ("127.0.0.1:1", "GET", "/health", 200)

    def test_pipeline_error_stored_in_db_is_redacted(self):
        from app.solvers.base import SolveResult

        class Leaky:
            name = "leaky"
            def solve(self, gray):
                raise RuntimeError("provider echoed token=AbC123xyz987TOKEN")

        p = SolvePipeline(solvers_for=lambda t: [Leaky()], calibration={})
        img = cv2.putText(np.full((48, 160), 240, np.uint8), "AB7K", (10, 36),
                          cv2.FONT_HERSHEY_SIMPLEX, 1.2, 20, 3)
        res = p.solve(img, challenge_type="text")
        assert "AbC123xyz987TOKEN" not in str(res.attempts)
        engine = create_engine("sqlite://")
        Base.metadata.create_all(engine)
        with sessionmaker(bind=engine)() as db:
            row = record_attempt(db, res)
            assert "AbC123xyz987TOKEN" not in (row.error or "")


# ── SQLite under concurrent writers ──────────────────────────────────────────
def test_sqlite_concurrent_writers_do_not_lock(tmp_path):
    from app import db as app_db

    url = f"sqlite:///{tmp_path / 'c.db'}"
    from app.config import settings

    original = settings.database_url
    settings.database_url = url
    try:
        engine = app_db._make_engine()
    finally:
        settings.database_url = original
    Base.metadata.create_all(engine)
    with engine.connect() as c:
        assert c.exec_driver_sql("PRAGMA journal_mode").scalar() == "wal"
    Session = sessionmaker(bind=engine)
    errors: list[Exception] = []

    def worker(n):
        try:
            for i in range(40):
                with Session() as db:
                    db.add(Attempt(challenge_type="text", predicted_text=f"{n}-{i}"))
                    db.commit()
        except Exception as exc:  # noqa: BLE001
            errors.append(exc)

    threads = [threading.Thread(target=worker, args=(n,)) for n in range(8)]
    [t.start() for t in threads]
    [t.join() for t in threads]
    assert not errors, errors[:1]
    with Session() as db:
        assert db.query(Attempt).count() == 320
