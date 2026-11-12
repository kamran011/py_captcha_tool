"""M5 tests: challenge API (classify / solve / detect / listings / stats) and abuse guards."""
from __future__ import annotations

import csv
import logging
import random
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient
from sqlalchemy.orm import sessionmaker

from app import db as app_db
from app.api import limits
from app.api.solve import get_pipeline
from app.config import settings
from app.main import app
from app.models import Attempt, Base, Verification
from app.solvers.base import SolveResult
from app.solvers.math_render import random_expression, render_expression
from app.solvers.pipeline import SolvePipeline

TEST_DIR = Path("data/synthetic/test")
needs_data = pytest.mark.skipif(
    not (TEST_DIR / "labels.csv").exists() or not Path("data/models/crnn.onnx").exists(),
    reason="run ml/generate_dataset.py and ml/train_crnn.py first",
)
KEY = {"X-API-Key": settings.api_key.get_secret_value()}
png = lambda im: cv2.imencode(".png", im)[1].tobytes()  # noqa: E731


def sample(i: int = 0):
    row = list(csv.DictReader(open(TEST_DIR / "labels.csv", encoding="utf-8")))[i]
    return png(cv2.imread(str(TEST_DIR / row["filename"]), 0)), row["text"]


def text_png():
    """A legible image that passes the quality gate (stub-solver tests need one)."""
    return png(cv2.putText(np.full((48, 160), 240, np.uint8), "AB7K", (10, 36),
                           cv2.FONT_HERSHEY_SIMPLEX, 1.2, 20, 3))


def grid_png():
    g = np.full((150, 150), 230, np.uint8)
    for i in range(0, 151, 50):
        cv2.line(g, (i, 0), (i, 149), 20, 2)
        cv2.line(g, (0, i), (149, i), 20, 2)
    return png(g)


class Stub:
    """Counts concurrent solves so the cap can be asserted."""
    name = "stub"
    lock = threading.Lock()
    now = peak = 0

    def __init__(self, text="ABCD", conf=0.95, delay=0.0):
        self.text, self.conf, self.delay = text, conf, delay

    def solve(self, gray):
        with Stub.lock:
            Stub.now += 1
            Stub.peak = max(Stub.peak, Stub.now)
        try:
            time.sleep(self.delay)
            return SolveResult(self.text, self.conf, self.name)
        finally:
            with Stub.lock:
                Stub.now -= 1


@pytest.fixture
def Session(tmp_path):
    original = settings.database_url
    settings.database_url = f"sqlite:///{tmp_path / 'api.db'}"
    try:
        eng = app_db._make_engine()
    finally:
        settings.database_url = original
    Base.metadata.create_all(eng)
    return sessionmaker(bind=eng)


@pytest.fixture
def client(Session):
    def db_override():
        with Session() as db:
            yield db

    limits.limiter.reset()
    saved = (settings.rate_limit_per_min, settings.max_concurrent_solves,
             settings.solve_queue_timeout_s, settings.max_upload_bytes)
    app.dependency_overrides[app_db.get_db] = db_override
    c = TestClient(app)
    c.Session = Session
    yield c
    app.dependency_overrides.clear()
    (settings.rate_limit_per_min, settings.max_concurrent_solves,
     settings.solve_queue_timeout_s, settings.max_upload_bytes) = saved
    limits.limiter.reset()


def use_stub(client, **kw):
    pipe = SolvePipeline(solvers_for=lambda t: [Stub(**kw)], calibration={})
    app.dependency_overrides[get_pipeline] = lambda: pipe


def post_solve(client, data, name="c.png", **form):
    return client.post("/api/v1/solve", headers=KEY, files={"file": (name, data, "image/png")}, data=form)


# ── auth ─────────────────────────────────────────────────────────────────────
@pytest.mark.parametrize("method,path", [
    ("post", "/api/v1/solve"), ("post", "/api/v1/classify"), ("post", "/api/v1/detect"),
    ("get", "/api/v1/attempts"), ("get", "/api/v1/verifications"), ("get", "/api/v1/stats"),
    ("post", "/api/v1/verify"),
])
def test_every_endpoint_requires_the_key(client, method, path):
    assert getattr(client, method)(path).status_code == 401
    assert getattr(client, method)(path, headers={"X-API-Key": "wrong"}).status_code == 401


# ── classify ─────────────────────────────────────────────────────────────────
@needs_data
def test_classify_text_math_unsupported(client):
    img, _ = sample()
    rng = random.Random(4)
    math = png(render_expression(random_expression(rng)[0], rng))
    for data, want in ((img, "text"), (math, "math"), (grid_png(), "unsupported")):
        r = client.post("/api/v1/classify", headers=KEY, files={"file": ("c.png", data, "image/png")})
        assert r.status_code == 200 and r.json()["type"] == want
        assert r.json()["quality"]["ok"] in (True, False) and r.json()["reasons"]


def test_classify_reports_quality_problem(client):
    blank = png(np.full((48, 160), 200, np.uint8))
    r = client.post("/api/v1/classify", headers=KEY, files={"file": ("c.png", blank, "image/png")})
    assert r.status_code == 200 and r.json()["quality"]["ok"] is False


# ── solve ────────────────────────────────────────────────────────────────────
@needs_data
def test_solve_text_records_attempt_and_accuracy(client):
    img, truth = sample(3)
    r = post_solve(client, img, expected=truth.lower())          # lower-case is normalised
    j = r.json()
    assert r.status_code == 200 and j["accepted"] and j["text"] == truth
    assert j["success"] is True and j["attempt_id"] and j["method"] == "crnn_onnx"
    assert j["cost_usd"] == 0.0 and j["attempts"][0]["method"] == "crnn_onnx"
    wrong = post_solve(client, img, expected="ZZZZ").json()
    assert wrong["success"] is False and wrong["accepted"] is True


@needs_data
def test_solve_math(client):
    rng = random.Random(9)
    expr, answer = random_expression(rng)
    j = post_solve(client, png(render_expression(expr, rng)), expected=str(answer)).json()
    assert j["challenge_type"] == "math" and j["text"] == str(answer) and j["success"] is True
    assert j["detail"]["expression"] == expr


@needs_data
def test_solve_unsupported_and_unreadable_are_422_and_recorded(client):
    r = post_solve(client, grid_png())
    assert r.status_code == 422 and r.json()["error"] == "unsupported_challenge" and r.json()["attempts"] == []
    r2 = post_solve(client, png(np.full((48, 160), 200, np.uint8)))
    # flat image: rejected by the classifier ("no glyphs") or the quality gate; never answered
    assert r2.status_code == 422 and not r2.json()["accepted"]
    assert r2.json()["error"] == "unsupported_challenge" or r2.json()["error"].startswith("poor_quality")
    with client.Session() as db:
        assert db.query(Attempt).count() == 2


def test_solve_not_an_image_is_400_and_not_recorded(client):
    r = post_solve(client, b"definitely not an image")
    assert r.status_code == 400 and r.json()["error"].startswith("invalid_image")
    assert r.json()["attempt_id"] is None
    with client.Session() as db:
        assert db.query(Attempt).count() == 0


def test_upload_limits_and_missing_file(client):
    settings.max_upload_bytes = 100
    r = post_solve(client, b"x" * 101)
    assert r.status_code == 413 and r.json() == {"ok": False, "reason": "file_too_large", "detail": "limit is 100 bytes"}
    settings.max_upload_bytes = 5_000_000
    assert post_solve(client, b"").json()["reason"] == "empty_file"
    r = client.post("/api/v1/solve", headers=KEY)
    assert r.status_code == 422 and r.json()["reason"] == "invalid_request"


def test_challenge_type_hint_and_stub_fallback_visible(client):
    pipe = SolvePipeline(solvers_for=lambda t: [Stub("ZZZZ", 0.1), Stub("WXYZ", 0.9)], calibration={})
    app.dependency_overrides[get_pipeline] = lambda: pipe
    j = post_solve(client, text_png(), challenge_type="text").json()
    assert j["accepted"] and [a["error"] for a in j["attempts"]] == ["low_confidence", None]


def test_filename_is_sanitised(client):
    use_stub(client)
    img = text_png()
    post_solve(client, img, name="../../etc/pass\x00wd<script>.png", challenge_type="text")
    with client.Session() as db:
        name = db.query(Attempt).one().image_filename
    assert "/" not in name and "\x00" not in name and "<" not in name and name.endswith(".png")


# ── guards ───────────────────────────────────────────────────────────────────
def test_rate_limit_429_with_retry_after_and_remaining_header(client):
    settings.rate_limit_per_min = 3
    seen = [client.get("/api/v1/attempts", headers=KEY) for _ in range(4)]
    assert [r.status_code for r in seen] == [200, 200, 200, 429]
    assert [r.headers["X-RateLimit-Remaining"] for r in seen[:3]] == ["2", "1", "0"]
    assert int(seen[3].headers["Retry-After"]) >= 1
    assert seen[3].json()["reason"] == "rate_limited"


def test_rate_limit_applies_to_verify_too(client):
    settings.rate_limit_per_min = 1
    client.get("/api/v1/attempts", headers=KEY)
    assert client.post("/api/v1/verify", headers=KEY, json={"provider": "recaptcha"}).status_code == 429


def test_busy_when_all_solver_slots_are_taken(client):
    use_stub(client)
    settings.max_concurrent_solves, settings.solve_queue_timeout_s = 1, 0.05
    img = text_png()
    held, release = threading.Event(), threading.Event()

    def hog():
        with limits.solve_slot():
            held.set()
            release.wait(3)

    t = threading.Thread(target=hog)
    t.start()
    held.wait(2)
    try:
        r = post_solve(client, img, challenge_type="text")
        assert r.status_code == 503 and r.json()["reason"] == "busy" and r.headers["Retry-After"] == "1"
    finally:
        release.set()
        t.join()
    assert post_solve(client, img, challenge_type="text").status_code == 200       # slot freed


def test_concurrent_solves_respect_the_cap_and_all_are_recorded(client):
    pipe = SolvePipeline(solvers_for=lambda t: [Stub(delay=0.05)], calibration={})
    app.dependency_overrides[get_pipeline] = lambda: pipe
    settings.max_concurrent_solves, settings.solve_queue_timeout_s = 2, 10
    Stub.now = Stub.peak = 0
    img = text_png()
    with ThreadPoolExecutor(8) as ex:
        codes = list(ex.map(lambda _: post_solve(client, img, challenge_type="text").status_code, range(8)))
    assert codes == [200] * 8 and Stub.peak <= 2
    with client.Session() as db:
        assert db.query(Attempt).count() == 8


# ── detect ───────────────────────────────────────────────────────────────────
def test_detect(client):
    html = Path("tests/fixtures/detector/hcaptcha_invisible.html").read_text(encoding="utf-8")
    j = client.post("/api/v1/detect", headers=KEY, json={"html": html}).json()
    assert (j["provider"], j["variant"], j["csv_label"]) == ("hcaptcha", "invisible", "hcaptcha")


def test_detect_validation_and_size(client):
    assert client.post("/api/v1/detect", headers=KEY, json={}).json()["reason"] == "invalid_request"
    r = client.post("/api/v1/detect", headers=KEY, json={"html": "x" * 5_000_001})
    assert r.status_code == 413 and r.json()["reason"] == "html_too_large"


# ── listings and stats ───────────────────────────────────────────────────────
def seed(client):
    with client.Session() as db:
        for i in range(5):
            db.add(Attempt(challenge_type="text", method_used="crnn_onnx", success=i != 0, expected_text="A",
                           confidence=0.9, processing_ms=10 + i, cost_usd=0.0))
        db.add(Attempt(challenge_type="math", method_used="math_knn", success=True, confidence=1.0, processing_ms=3, cost_usd=0.001))
        db.add(Attempt(challenge_type="unsupported", method_used=None, success=False, processing_ms=1, cost_usd=0.0))
        db.add(Verification(provider="recaptcha", ok=True, reason="ok", http_status=200, latency_ms=300, token_fingerprint="abc123def456"))
        db.add(Verification(provider="recaptcha", ok=False, reason="reused_token", http_status=409, latency_ms=2, token_fingerprint="abc123def456"))
        db.commit()


def test_attempts_listing_paging_and_filter(client):
    seed(client)
    j = client.get("/api/v1/attempts?limit=3", headers=KEY).json()
    assert j["total"] == 7 and len(j["items"]) == 3 and j["items"][0]["id"] > j["items"][1]["id"]
    assert client.get("/api/v1/attempts?limit=3&offset=6", headers=KEY).json()["items"].__len__() == 1
    assert client.get("/api/v1/attempts?challenge_type=math", headers=KEY).json()["total"] == 1
    for bad in ("limit=0", "limit=201", "offset=-1", "challenge_type=banana"):
        assert client.get(f"/api/v1/attempts?{bad}", headers=KEY).status_code == 422


def test_verifications_listing_has_fingerprints_only(client):
    seed(client)
    j = client.get("/api/v1/verifications", headers=KEY).json()
    assert j["total"] == 2 and j["items"][0]["token_fingerprint"] == "abc123def456"
    assert set(j["items"][0]) == {"id", "created_at", "provider", "ok", "reason", "http_status", "latency_ms", "token_fingerprint"}


def test_stats_breakdowns(client):
    s0 = client.get("/api/v1/stats", headers=KEY).json()
    assert s0["attempts"]["total"] == 0 and s0["attempts"]["latency_ms"] == {"p50": None, "p95": None}
    seed(client)
    s = client.get("/api/v1/stats", headers=KEY).json()
    a, v = s["attempts"], s["verifications"]
    assert (a["total"], a["successful"], a["failed"]) == (7, 5, 2)
    assert a["labelled"] == 5 and a["labelled_accuracy"] == 0.8
    assert {r["key"]: r["total"] for r in a["by_type"]} == {"text": 5, "math": 1, "unsupported": 1}
    methods = {r["key"]: r for r in a["by_method"]}
    assert methods["crnn_onnx"]["successful"] == 4 and methods["none"]["total"] == 1
    assert a["latency_ms"]["p50"] is not None and a["latency_ms"]["p95"] >= a["latency_ms"]["p50"]
    assert a["total_cost_usd"] == 0.001 and a["avg_cost_usd"] == round(0.001 / 7, 6)
    assert v["total"] == 2 and v["successful"] == 1
    assert {r["key"]: r["total"] for r in v["by_reason"]} == {"ok": 1, "reused_token": 1}


# ── startup ──────────────────────────────────────────────────────────────────
def test_default_api_key_triggers_a_startup_warning(caplog):
    caplog.set_level(logging.WARNING)
    with TestClient(app):
        pass
    default = settings.api_key.get_secret_value() == "changeme-local"
    assert any("default" in r.getMessage() and "API_KEY" in r.getMessage() for r in caplog.records) == default
