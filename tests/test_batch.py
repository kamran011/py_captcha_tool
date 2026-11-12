"""M6 tests: metrics, dataset loading safety, batch runs and the batch API."""
from __future__ import annotations

import csv
import io
from pathlib import Path

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient
from sqlalchemy.orm import sessionmaker

from app import batch as batch_mod
from app import db as app_db
from app.api import limits
from app.api.solve import get_pipeline
from app.batch import BatchItem, DatasetError, list_datasets, load_dataset, run_batch
from app.config import settings
from app.main import app
from app.models import BatchRun, Base
from app.solvers.base import SolveResult
from app.solvers.metrics import char_accuracy, latency_summary, levenshtein
from app.solvers.pipeline import SolvePipeline

KEY = {"X-API-Key": settings.api_key.get_secret_value()}
png = lambda im: cv2.imencode(".png", im)[1].tobytes()  # noqa: E731
GOOD = png(cv2.putText(np.full((48, 160), 240, np.uint8), "AB7K", (10, 36), cv2.FONT_HERSHEY_SIMPLEX, 1.2, 20, 3))


# ── metrics ──────────────────────────────────────────────────────────────────
@pytest.mark.parametrize("a,b,d", [("", "", 0), ("abc", "", 3), ("", "abc", 3), ("kitten", "sitting", 3),
                                   ("ABCD", "ABCD", 0), ("ABCD", "ABD", 1), ("ABCD", "ABXD", 1), ("héllo", "hello", 1)])
def test_levenshtein(a, b, d):
    assert levenshtein(a, b) == d == levenshtein(b, a)


def test_char_accuracy():
    assert char_accuracy("ABCD", "ABCD") == 1.0
    assert char_accuracy("ABD", "ABCD") == 0.75            # one dropped character
    assert char_accuracy("XXXX", "ABCD") == 0.0
    assert char_accuracy("ABCDEFGH", "AB") == 0.0           # floored, never negative
    assert char_accuracy(None, "ABCD") == 0.0 and char_accuracy("", "") == 1.0 and char_accuracy("A", "") == 0.0


def test_latency_summary():
    assert latency_summary([]) == {"p50": None, "p95": None, "mean": None}
    s = latency_summary([1.0, 2.0, 3.0, 4.0, 100.0, None])
    assert s["p50"] == 3.0 and s["p95"] > s["p50"] and s["mean"] == 22.0


# ── datasets ─────────────────────────────────────────────────────────────────
@pytest.fixture
def root(tmp_path, monkeypatch):
    for name, labels in (("good", [("a.png", "ab7k"), ("b.png", "AB7K")]), ("tiny", [("a.png", "XY")])):
        d = tmp_path / name
        d.mkdir()
        for fname, _ in labels:
            (d / fname).write_bytes(GOOD)
        with open(d / "labels.csv", "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["filename", "text"])
            w.writerows(labels)
    (tmp_path / "nolabels").mkdir()
    (tmp_path / "bad name!").mkdir()
    monkeypatch.setattr(batch_mod, "DATA_ROOT", tmp_path)
    monkeypatch.setattr("app.batch.DATA_ROOT", tmp_path)
    return tmp_path


def test_list_datasets_only_valid_named_folders_with_labels(root):
    assert list_datasets(root) == [{"name": "good", "count": 2}, {"name": "tiny", "count": 1}]


def test_load_dataset_normalises_labels_and_clamps_limit(root):
    items = load_dataset("good", 10, root)
    assert [i.expected for i in items] == ["AB7K", "AB7K"] and len(load_dataset("good", 1, root)) == 1
    assert len(load_dataset("good", 0, root)) == 1            # clamped to at least one


@pytest.mark.parametrize("name", ["../etc", "..", ".", "/etc/passwd", "a/b", "a\\b", "", "x" * 41, "good\x00", "go od", None, 5])
def test_dataset_names_cannot_escape_the_data_root(root, name):
    with pytest.raises(DatasetError):
        load_dataset(name, 5, root)


def test_unknown_dataset_and_labels_traversal(root):
    with pytest.raises(DatasetError, match="unknown"):
        load_dataset("nolabels", 5, root)
    (root / "outside.png").write_bytes(GOOD)
    with open(root / "tiny" / "labels.csv", "a", newline="") as f:
        csv.writer(f).writerow(["../outside.png", "EVIL"])
    items = load_dataset("tiny", 10, root)
    assert [i.name for i in items] == ["a.png"]                # the traversal entry is skipped


# ── run_batch ────────────────────────────────────────────────────────────────
class Scripted:
    name = "scripted"

    def __init__(self, answers):
        self.answers, self.i = answers, 0

    def solve(self, gray):
        text, conf = self.answers[self.i % len(self.answers)]
        self.i += 1
        return SolveResult(text, conf, self.name)


def pipe(*answers):
    return SolvePipeline(solvers_for=lambda t: [Scripted(list(answers))], calibration={})


def test_run_batch_metrics():
    items = [BatchItem(f"{i}.png", GOOD, "ABCD") for i in range(4)]
    # exact, one wrong char (confident), exact, and a low-confidence wrong answer
    r = run_batch(pipe(("ABCD", 0.99), ("ABXD", 0.99), ("ABCD", 0.95), ("WXYZ", 0.2)), items, "t")
    assert (r["n"], r["labelled"]) == (4, 4)
    assert r["exact_match"] == 0.5 and r["char_accuracy"] == round((1 + 0.75 + 1 + 0) / 4, 4)
    assert r["accepted"] == 3 and r["coverage"] == 0.75
    assert r["precision_on_accepted"] == round(2 / 3, 4) and r["confident_wrong"] == 1
    assert r["worst"][0]["predicted"] == "ABXD"                # wrong answers ranked by confidence
    assert r["by_type"] == {"text": {"n": 4, "exact_match": 0.5}} and r["latency_ms"]["p50"] is not None
    assert len(r["results"]) == 4 and not r["truncated"]


def test_unlabelled_items_report_no_accuracy_but_still_solve():
    r = run_batch(pipe(("ABCD", 0.9)), [BatchItem("a.png", GOOD, None)], "upload")
    assert r["labelled"] == 0 and r["exact_match"] is None and r["char_accuracy"] is None
    assert r["results"][0]["predicted"] == "ABCD" and r["results"][0]["exact"] is None


def test_bad_images_are_counted_as_errors_not_crashes():
    items = [BatchItem("junk.png", b"not an image", "ABCD"), BatchItem("ok.png", GOOD, "ABCD")]
    r = run_batch(pipe(("ABCD", 0.9)), items, "t")
    assert r["by_error"] == {"invalid_image": 1} and r["exact_match"] == 0.5


def test_time_budget_truncates():
    r = run_batch(pipe(("ABCD", 0.9)), [BatchItem(f"{i}.png", GOOD, "ABCD") for i in range(50)], "t", time_budget_s=-1)
    assert r["truncated"] is True and r["n"] == 0 and r["exact_match"] is None


# ── API ──────────────────────────────────────────────────────────────────────
@pytest.fixture
def client(tmp_path, root):
    original = settings.database_url
    settings.database_url = f"sqlite:///{tmp_path / 'b.db'}"
    try:
        eng = app_db._make_engine()
    finally:
        settings.database_url = original
    Base.metadata.create_all(eng)
    Session = sessionmaker(bind=eng)

    def db_override():
        with Session() as db:
            yield db

    limits.limiter.reset()
    app.dependency_overrides[app_db.get_db] = db_override
    app.dependency_overrides[get_pipeline] = lambda: pipe(("AB7K", 0.99))
    c = TestClient(app)
    c.Session = Session
    yield c
    app.dependency_overrides.clear()
    limits.limiter.reset()


def test_batch_endpoints_need_the_key(client):
    for method, path in (("get", "/datasets"), ("post", ""), ("post", "/upload"), ("get", "/runs")):
        assert getattr(client, method)(f"/api/v1/batch{path}").status_code == 401


def test_datasets_endpoint_flags_training_data(client, root):
    d = root / "train"
    d.mkdir()
    (d / "labels.csv").write_text("filename,text\n")
    names = {x["name"]: x["note"] for x in client.get("/api/v1/batch/datasets", headers=KEY).json()["datasets"]}
    assert names["good"] is None and "optimistic" in names["train"]


def test_run_dataset_stores_history_and_hides_per_item_rows(client):
    r = client.post("/api/v1/batch", headers=KEY, json={"dataset": "good", "limit": 5})
    j = r.json()
    assert r.status_code == 200 and j["n"] == 2 and j["exact_match"] == 1.0 and "results" not in j and j["run_id"]
    runs = client.get("/api/v1/batch/runs", headers=KEY).json()["items"]
    assert runs[0]["dataset"] == "good" and runs[0]["exact_match"] == 1.0
    with client.Session() as db:
        assert db.query(BatchRun).count() == 1


@pytest.mark.parametrize("body,status", [
    ({"dataset": "nope"}, 404), ({"dataset": "../etc"}, 422), ({"dataset": "good", "limit": 0}, 422),
    ({"dataset": "good", "limit": 5000}, 422), ({}, 422),
])
def test_run_dataset_errors(client, body, status):
    r = client.post("/api/v1/batch", headers=KEY, json=body)
    assert r.status_code == status


def test_upload_with_and_without_labels(client):
    files = [("files", ("one.png", GOOD, "image/png")), ("files", ("two.png", GOOD, "image/png"))]
    labels = ("labels", ("labels.csv", b"filename,text\none.png,ab7k\ntwo.png,ZZZZ\n", "text/csv"))
    j = client.post("/api/v1/batch/upload", headers=KEY, files=files + [labels]).json()
    assert j["labelled"] == 2 and j["exact_match"] == 0.5 and len(j["results"]) == 2
    j2 = client.post("/api/v1/batch/upload", headers=KEY, files=files).json()
    assert j2["labelled"] == 0 and j2["results"][0]["predicted"] == "AB7K"


def test_upload_limits(client):
    many = [("files", (f"{i}.png", GOOD, "image/png")) for i in range(101)]
    assert client.post("/api/v1/batch/upload", headers=KEY, files=many).status_code == 413
    assert client.post("/api/v1/batch/upload", headers=KEY).status_code == 422
    settings.max_upload_bytes = 50
    try:
        r = client.post("/api/v1/batch/upload", headers=KEY, files=[("files", ("a.png", GOOD, "image/png"))])
        assert r.status_code == 413
    finally:
        settings.max_upload_bytes = 5_000_000


def test_upload_filenames_are_sanitised(client):
    j = client.post("/api/v1/batch/upload", headers=KEY,
                    files=[("files", ("../../evil\x00<b>.png", GOOD, "image/png"))]).json()
    name = j["results"][0]["name"]
    assert "/" not in name and "<" not in name and "\x00" not in name
