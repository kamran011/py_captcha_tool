"""Dashboard smoke tests: every page renders without exceptions, against a faked API client."""
from __future__ import annotations

from pathlib import Path

import pytest
from streamlit.proto.TextInput_pb2 import TextInput
from streamlit.testing.v1 import AppTest

from dashboard import samples
from dashboard.client import ApiClient, ApiResult

APP = str(Path(__file__).resolve().parent.parent / "dashboard" / "streamlit_app.py")

STATS = {
    "attempts": {"total": 10, "successful": 9, "failed": 1, "success_rate": 0.9, "labelled": 8,
                 "labelled_accuracy": 0.875, "avg_confidence": 0.97, "avg_processing_ms": 5.0,
                 "latency_ms": {"p50": 4.0, "p95": 9.0}, "total_cost_usd": 0.0, "avg_cost_usd": 0.0,
                 "by_type": [{"key": "text", "total": 8, "successful": 7}, {"key": "math", "total": 2, "successful": 2}],
                 "by_method": [{"key": "crnn_onnx", "total": 8, "successful": 7}]},
    "verifications": {"total": 3, "successful": 2, "failed": 1, "latency_ms": {"p50": 50.0, "p95": 90.0},
                      "by_reason": [{"key": "ok", "total": 2}, {"key": "reused_token", "total": 1}],
                      "by_provider": []},
    "sites": {"total": 0, "scanned": 0},
}
SOLVED = {"challenge_type": "text", "classifier_confidence": 0.8, "text": "ARY6", "confidence": 0.99,
          "method": "crnn_onnx", "accepted": True, "success": True, "error": None, "total_ms": 12.0,
          "cost_usd": 0.0, "detail": {"classifier_reasons": ["x"], "quality": {"range": 120}},
          "attempts": [{"method": "crnn_onnx", "text": "ARY6", "confidence": 0.99, "ms": 11.0,
                        "cost_usd": 0.0, "accepted": True, "error": None}]}


@pytest.fixture
def fake_api(monkeypatch):
    calls = {}

    def reg(name, result):
        def f(self, *a, **k):
            calls.setdefault(name, []).append((a, k))
            return result
        monkeypatch.setattr(ApiClient, name, f)

    reg("health", ApiResult(200, {"status": "ok"}))
    reg("stats", ApiResult(200, STATS))
    reg("attempts", ApiResult(200, {"total": 0, "items": []}))
    reg("verifications", ApiResult(200, {"total": 0, "items": []}))
    reg("solve", ApiResult(200, SOLVED))
    reg("classify", ApiResult(200, {}))
    reg("detect", ApiResult(200, {"provider": "hcaptcha", "variant": "checkbox", "sitekey": "k", "confidence": 0.85,
                                  "csv_label": "hcaptcha", "evidence": [{"kind": "widget_element", "weight": 4, "detail": ""}],
                                  "others": [], "needs_render": False}))
    reg("verify", ApiResult(200, {"ok": True, "reason": "ok", "latency_ms": 12.0, "token_fingerprint": "abc123def456",
                                  "provider_error_codes": [], "detail": None}))
    reg("solvers", ApiResult(200, {"solvers": [LOCAL, VISION_OFF]}))          # default: third-party solver off
    return calls


def run_page(page: str) -> AppTest:
    at = AppTest.from_file(APP, default_timeout=30).run()
    if page != "Overview":
        next(b for b in at.sidebar.button if b.label == page).click().run()
    return at


@pytest.mark.parametrize("page", ["Overview", "Solve", "Verify", "Detect", "Batch", "Sites"])
def test_every_page_renders_without_exceptions(fake_api, page):
    at = run_page(page)
    assert not at.exception, [e.value for e in at.exception]
    assert at.title[0].value


def test_overview_shows_metrics(fake_api):
    at = run_page("Overview")
    labels = {}
    for m in at.metric:
        labels.setdefault(m.label, m.value)          # "Total" appears for attempts, then verifications
    assert labels["Total"] == "10" and labels["Success rate"] == "90.0%" and labels["Accuracy (labelled)"] == "87.5%"
    assert labels["Latency p95"] == "9 ms" and labels["Cost / attempt"] == "$0"


def test_api_offline_is_explained_not_a_stack_trace(monkeypatch):
    down = ApiResult(0, {"ok": False, "reason": "api_unreachable"})
    for name in ("health", "stats", "attempts", "verifications"):
        monkeypatch.setattr(ApiClient, name, lambda self, *a, **k: down)
    at = run_page("Overview")
    assert not at.exception and any("not reachable" in e.value for e in at.error)


def test_bad_api_key_is_explained(monkeypatch, fake_api):
    monkeypatch.setattr(ApiClient, "stats", lambda self: ApiResult(401, {"detail": "bad key"}))
    at = run_page("Overview")
    assert any("rejected the key" in e.value for e in at.error)


@pytest.mark.skipif(not samples.test_samples(), reason="needs the generated dataset")
def test_solve_flow_sends_sample_with_expected_answer(fake_api):
    at = run_page("Solve")
    at.radio(key="solve_source").set_value("Held-out sample").run()
    at.button(key="solve_go").click().run()
    assert not at.exception
    (args, kwargs), = fake_api["solve"]
    assert isinstance(args[0], bytes) and args[1].endswith(".png") and args[2]       # filename + expected label
    assert any("ARY6" in s.value for s in at.success)


@pytest.mark.parametrize("status,body,expect", [
    (422, {**SOLVED, "accepted": False, "error": "unsupported_challenge", "challenge_type": "unsupported"}, "never sent to a solver"),
    (422, {**SOLVED, "accepted": False, "error": "poor_quality: too_blurry"}, "too_blurry"),
    (400, {**SOLVED, "error": "invalid_image: could not decode image bytes"}, "refused"),
    (413, {"ok": False, "reason": "file_too_large", "detail": "limit is 5 bytes"}, "refused"),
    (429, {"ok": False, "reason": "rate_limited"}, "refused"),
])
def test_solve_error_states_render(monkeypatch, fake_api, status, body, expect):
    monkeypatch.setattr(ApiClient, "solve", lambda self, *a, **k: ApiResult(status, body))
    at = run_page("Solve")
    at.radio(key="solve_source").set_value("Unsupported sample (grid)").run()
    at.button(key="solve_go").click().run()
    assert not at.exception
    shown = " ".join(x.value for x in list(at.warning) + list(at.error))
    assert expect in shown


def test_verify_flow_masks_token_and_sends_options(fake_api):
    at = run_page("Verify")
    assert at.text_input(key="vtoken").proto.type == TextInput.PASSWORD              # masked in the UI
    at.text_input(key="vtoken").set_value("tok_abc123").run()
    at.selectbox(key="vprovider").set_value("turnstile").run()
    at.selectbox(key="vscenario").set_value("fail").run()
    at.button(key="verify_go").click().run()
    assert not at.exception
    (args, _), = fake_api["verify"]
    assert args == ("turnstile", "tok_abc123", None, "fail")
    assert any("Token accepted" in s.value for s in at.success)
    assert "tok_abc123" not in " ".join(m.value for m in at.metric)                  # only the fingerprint is shown


def test_verify_page_never_echoes_the_token_in_page_text(fake_api):
    at = run_page("Verify")
    at.text_input(key="vtoken").set_value("SUPERSECRETTOKEN123").run()
    at.button(key="verify_go").click().run()
    texts = [*(m.value for m in at.markdown), *(s.value for s in at.success), *(c.value for c in at.caption)]
    assert not any("SUPERSECRETTOKEN123" in t for t in texts)


def test_detect_flow(fake_api):
    at = run_page("Detect")
    at.button(key="detect_go").click().run()
    assert not at.exception and any("hcaptcha" in s.value for s in at.success)
    assert fake_api["detect"][0][0][0].lstrip().startswith("<!doctype")


# ── sample generators ────────────────────────────────────────────────────────
def test_degradations_change_the_image_and_keep_it_decodable():
    import cv2
    import numpy as np

    img, _, _ = samples.math_sample(3)
    for mode in samples.DEGRADATIONS:
        out = samples.degrade(img, mode)
        assert cv2.imdecode(np.frombuffer(out, np.uint8), 0) is not None
        assert (out == img) == (mode == "none")


def test_grid_sample_is_unsupported_for_the_classifier():
    import cv2
    import numpy as np

    from app.recognition.image_classifier import classify_image

    gray = cv2.imdecode(np.frombuffer(samples.grid_sample(), np.uint8), 0)
    assert classify_image(gray).type == "unsupported"


def test_client_reports_unreachable_api_instead_of_raising():
    res = ApiClient(base="http://127.0.0.1:9", key="k").stats()
    assert res.status == 0 and not res.reachable and res.data["reason"] == "api_unreachable"


# ── Batch and Sites pages ────────────────────────────────────────────────────
BATCH = {"dataset": "test", "n": 100, "labelled": 100, "truncated": False, "total_seconds": 0.5, "exact_match": 0.97,
         "char_accuracy": 0.99, "mean_confidence": 0.98, "accepted": 99, "coverage": 0.99, "precision_on_accepted": 0.9697,
         "confident_wrong": 3, "latency_ms": {"p50": 3.5, "p95": 5.0, "mean": 3.9},
         "by_type": {"text": {"n": 100, "exact_match": 0.97}}, "by_error": {"poor_quality": 1},
         "worst": [{"name": "a.png", "expected": "ABCD", "predicted": "ABXD", "confidence": 0.99, "error": None}],
         "run_id": 1}
LABELS = ["recaptcha", "recaptcha_invisible", "hcaptcha", "turnstile", "none", "other"]
REPORT = {"total_sites": 200, "scanned": 150, "unscanned": 10, "coverage": 0.75, "scan_status": {"ok": 150, "timeout": 40},
          "exact_accuracy": 0.8, "family_accuracy": 0.85, "baseline_majority_class": {"label": "recaptcha", "accuracy": 0.5},
          "labels": LABELS, "confusion_matrix": {t: {p: 0 for p in LABELS} for t in LABELS},
          "per_class_exact": {"recaptcha": {"precision": 0.9, "recall": 0.8, "f1": 0.85, "support": 70}},
          "per_class_family": {"recaptcha": {"precision": 0.9, "recall": 0.85, "f1": 0.87, "support": 100}},
          "js_injected_suspected": 12,
          "mismatches": [{"id": 1, "company": "Acme", "contact_page": "https://a.example/c", "label": "recaptcha",
                          "detected": "none", "variant": None, "needs_render": True, "family_ok": False}]}


@pytest.fixture
def fake_m6(monkeypatch, fake_api):
    calls = {}

    def reg(name, result):
        def f(self, *a, **k):
            calls.setdefault(name, []).append((a, k))
            return result
        monkeypatch.setattr(ApiClient, name, f)

    reg("batch_datasets", ApiResult(200, {"datasets": [{"name": "test", "count": 1000, "note": None},
                                                       {"name": "train", "count": 20000, "note": "used for training: optimistic"}]}))
    reg("batch_run", ApiResult(200, BATCH))
    reg("batch_runs", ApiResult(200, {"items": []}))
    reg("sites_import", ApiResult(200, {"inserted": 200, "updated": 0, "unchanged": 0, "rows_valid": 200,
                                        "rows_rejected": 0, "errors": [], "warnings": []}))
    reg("sites_report", ApiResult(200, REPORT))
    reg("sites_list", ApiResult(200, {"total": 1, "items": [{"company": "Acme", "contact_page": "https://a.example/c",
                                                             "label": "recaptcha", "scan_status": "ok", "detected": "none"}]}))
    reg("sites_scan_status", ApiResult(200, {"status": "idle"}))
    reg("sites_scan_start", ApiResult(202, {"id": "x", "status": "running", "total": 200, "done": 0}))
    return calls


def test_batch_page_runs_and_shows_the_report(fake_m6):
    at = run_page("Batch")
    assert not at.exception
    at.button(key="batch_go").click().run()
    assert not at.exception and fake_m6["batch_run"][0][0] == ("test", 200)         # default: test split, 200 images
    labels = {m.label: m.value for m in at.metric}
    assert labels["Exact match"] == "97.0%" and labels["Character accuracy"] == "99.0%"
    assert labels["Confident but wrong"] == "3" and labels["Latency p95"] == "5 ms"


def test_batch_page_warns_when_the_training_split_is_selected(fake_m6):
    at = run_page("Batch")
    at.selectbox(key="batch_dataset").set_value("train (20000 images)").run()
    assert any("optimistic" in w.value for w in at.warning)


def test_batch_api_error_is_shown(monkeypatch, fake_m6):
    monkeypatch.setattr(ApiClient, "batch_run", lambda self, *a, **k: ApiResult(404, {"reason": "dataset_error", "detail": "unknown dataset"}))
    at = run_page("Batch")
    at.button(key="batch_go").click().run()
    assert not at.exception and any("unknown dataset" in e.value for e in at.error)


def test_sites_page_states_the_read_only_scope_up_front(fake_m6):
    at = run_page("Sites")
    assert not at.exception
    text = " ".join(i.value for i in at.info)
    for phrase in ("Read-only", "robots.txt", "never fills a field", "JavaScript"):
        assert phrase in text


def test_scan_cannot_start_without_explicit_confirmation(fake_m6):
    at = run_page("Sites")
    assert at.button(key="scan_go").disabled is True
    at.checkbox(key="scan_confirm").check().run()
    assert at.button(key="scan_go").disabled is False
    at.button(key="scan_go").click().run()
    assert not at.exception and fake_m6["sites_scan_start"][0][0] == (200, False)


def test_scan_start_rejection_is_shown(monkeypatch, fake_m6):
    monkeypatch.setattr(ApiClient, "sites_scan_start", lambda self, *a, **k: ApiResult(409, {"reason": "scan_running", "detail": "a scan is already running"}))
    at = run_page("Sites")
    at.checkbox(key="scan_confirm").check().run()
    at.button(key="scan_go").click().run()
    assert any("scan_running" in e.value for e in at.error)


def test_sites_report_renders_with_baseline_and_js_warning(fake_m6):
    at = run_page("Sites")
    assert not at.exception
    labels = {m.label: m.value for m in at.metric}
    assert labels["Analysed"] == "150 / 200" and labels["Exact accuracy"] == "80.0%"
    assert labels["Provider-level accuracy"] == "85.0%" and labels["Not analysed"] == "50"
    assert any("12 pages" in w.value and "JavaScript" in w.value for w in at.warning)


def test_sites_import_button(fake_m6):
    at = run_page("Sites")
    at.button(key="sites_import_bundled").click().run()
    assert not at.exception and any("200 new" in s.value for s in at.success)


def test_sites_page_before_any_scan_and_while_running(monkeypatch, fake_m6):
    monkeypatch.setattr(ApiClient, "sites_report", lambda self: ApiResult(200, {"total_sites": 200, "scanned": 0}))
    assert not run_page("Sites").exception
    monkeypatch.setattr(ApiClient, "sites_scan_status", lambda self: ApiResult(200, {
        "status": "running", "total": 200, "done": 40, "by_status": {"ok": 38, "timeout": 2}, "elapsed_s": 12.0, "error": None}))
    at = run_page("Sites")
    assert not at.exception


# ── M7: optional third-party solver ──────────────────────────────────────────
VISION_OFF = {"name": "vision_api", "kind": "external", "enabled": False, "provider": "anthropic",
              "model": "claude-haiku-5-5", "reason": "disabled: no VISION_API_KEY set (local solvers only)",
              "threshold": None, "advisory_only": True, "cost": None, "challenge_types": ["text"]}
VISION_ON = {**VISION_OFF, "enabled": True, "reason": None,
             "budget": {"limit_usd_per_day": 1.0, "spent_today_usd": 0.0004, "remaining_usd": 0.9996,
                        "calls_last_minute": 2, "max_calls_per_minute": 20}}
LOCAL = {"name": "crnn_onnx", "kind": "local", "enabled": True, "threshold": 0.6, "advisory_only": False,
         "cost": "free", "challenge_types": ["text"], "reason": None}


def solvers_result(vision):
    return ApiResult(200, {"solvers": [LOCAL, vision], "note": "x"})


def test_vision_off_is_explained_and_there_is_no_opt_in_box(monkeypatch, fake_api):
    monkeypatch.setattr(ApiClient, "solvers", lambda self: solvers_result(VISION_OFF))
    at = run_page("Solve")
    assert not at.exception and not at.checkbox
    assert any("Third-party vision solver: off" in c.value and "no VISION_API_KEY" in c.value for c in at.caption)


def test_vision_on_offers_a_per_request_opt_out_that_reaches_the_api(monkeypatch, fake_api):
    monkeypatch.setattr(ApiClient, "solvers", lambda self: solvers_result(VISION_ON))
    at = run_page("Solve")
    at.radio(key="solve_source").set_value("Unsupported sample (grid)").run()
    box = at.checkbox(key="solve_external")
    assert box.value is True and "image is sent to the provider" in box.label
    at.button(key="solve_go").click().run()
    assert fake_api["solve"][-1][0][4] is True                       # default: allowed
    at.checkbox(key="solve_external").uncheck().run()
    at.button(key="solve_go").click().run()
    assert fake_api["solve"][-1][0][4] is False                      # opted out: the image stays local


def test_external_answer_is_labelled_with_provider_model_and_tokens(monkeypatch, fake_api):
    external = {**SOLVED, "method": "vision_api", "cost_usd": 0.00015,
                "detail": {"external_call": True, "provider": "anthropic", "model": "claude-haiku-5-5",
                           "input_tokens": 1000, "output_tokens": 100}}
    monkeypatch.setattr(ApiClient, "solvers", lambda self: solvers_result(VISION_ON))
    monkeypatch.setattr(ApiClient, "solve", lambda self, *a, **k: ApiResult(200, external))
    at = run_page("Solve")
    at.radio(key="solve_source").set_value("Unsupported sample (grid)").run()
    at.button(key="solve_go").click().run()
    assert not at.exception
    text = " ".join(i.value for i in at.info)
    assert "third-party anthropic model claude-haiku-5-5" in text and "1000 input and 100 output tokens" in text
    assert any("$0.000150" in m.value for m in at.markdown)


def test_overview_shows_solver_status_budget_and_cost_by_method(monkeypatch, fake_api):
    stats = {**STATS, "attempts": {**STATS["attempts"], "cost_by_method": [
        {"key": "vision_api", "calls": 2, "cost_usd": 0.0004}, {"key": "crnn_onnx", "calls": 8, "cost_usd": 0.0}]}}
    monkeypatch.setattr(ApiClient, "stats", lambda self: ApiResult(200, stats))
    monkeypatch.setattr(ApiClient, "solvers", lambda self: solvers_result(VISION_ON))
    at = run_page("Overview")
    assert not at.exception
    labels = {m.label: m.value for m in at.metric}
    assert labels["Vision spend today"] == "$0.000400" and labels["Budget remaining"] == "$0.999600"
    assert labels["Calls in the last minute"] == "2 / 20"
    assert any("Cost by method" in c.value for c in at.caption)


def test_overview_without_the_solvers_endpoint_still_renders(monkeypatch, fake_api):
    monkeypatch.setattr(ApiClient, "solvers", lambda self: ApiResult(0, {"reason": "api_unreachable"}))
    assert not run_page("Overview").exception


# ── pagination ───────────────────────────────────────────────────────────────
def _attempt_rows(n):
    return [{"id": i, "challenge_type": "text", "method": "crnn_onnx", "predicted": f"P{i}", "expected": None,
             "success": True, "confidence": 0.9, "processing_ms": 5, "cost_usd": 0.0, "error": None}
            for i in range(n, 0, -1)]


def _rows_caption(at):
    return next((c.value for c in at.caption if c.value.startswith("Rows ")), None)


def test_long_tables_are_paged_with_navigation(monkeypatch, fake_api):
    monkeypatch.setattr(ApiClient, "attempts", lambda self, *a, **k: ApiResult(200, {"total": 35, "items": _attempt_rows(35)}))
    at = run_page("Overview")
    assert not at.exception
    assert _rows_caption(at) == "Rows 1-10 of 35"
    assert at.button(key="tbl_attempts_prev").disabled is True
    at.button(key="tbl_attempts_next").click().run()
    assert _rows_caption(at) == "Rows 11-20 of 35"
    at.button(key="tbl_attempts_next").click().run()
    at.button(key="tbl_attempts_next").click().run()
    assert _rows_caption(at) == "Rows 31-35 of 35"                 # last page is short
    assert at.button(key="tbl_attempts_next").disabled is True
    at.button(key="tbl_attempts_prev").click().run()
    assert _rows_caption(at) == "Rows 21-30 of 35"


def test_changing_rows_per_page_keeps_the_page_in_range(monkeypatch, fake_api):
    monkeypatch.setattr(ApiClient, "attempts", lambda self, *a, **k: ApiResult(200, {"total": 35, "items": _attempt_rows(35)}))
    at = run_page("Overview")
    for _ in range(3):
        at.button(key="tbl_attempts_next").click().run()           # page 4 of 4
    at.selectbox(key="tbl_attempts_size").set_value(25).run()      # only 2 pages now: must clamp, not break
    assert not at.exception and _rows_caption(at) == "Rows 26-35 of 35"
    at.selectbox(key="tbl_attempts_size").set_value(50).run()
    assert _rows_caption(at) == "Rows 1-35 of 35"
    assert at.button(key="tbl_attempts_prev").disabled and at.button(key="tbl_attempts_next").disabled


def test_short_tables_show_no_pager(monkeypatch, fake_api):
    monkeypatch.setattr(ApiClient, "attempts", lambda self, *a, **k: ApiResult(200, {"total": 4, "items": _attempt_rows(4)}))
    at = run_page("Overview")
    assert not at.exception and _rows_caption(at) is None


def test_reimport_says_nothing_changed(monkeypatch, fake_m6):
    # The reviewer clicked the import button and saw only "0 new, 0 updated": say what it means instead.
    monkeypatch.setattr(ApiClient, "sites_import", lambda self, *a, **k: ApiResult(200, {
        "inserted": 0, "updated": 0, "unchanged": 200, "rows_valid": 200, "rows_rejected": 0, "errors": [], "warnings": []}))
    at = run_page("Sites")
    at.button(key="sites_import_bundled").click().run()
    assert not at.exception
    assert any("Already imported: all 200 sites" in i.value for i in at.info)


def test_hard_sample_shows_true_text_and_the_fallback_chain(monkeypatch, fake_api):
    fell_back = {**SOLVED, "accepted": False, "text": "S8T976XX", "method": "crnn_onnx", "error": "implausible_length",
                 "attempts": [{"method": "crnn_onnx", "text": "S8T976XX", "confidence": 0.9, "ms": 3.0, "cost_usd": 0.0,
                               "accepted": False, "error": "implausible_length"},
                              {"method": "segment_knn", "text": "8TQ", "confidence": 0.4, "ms": 2.0, "cost_usd": 0.0,
                               "accepted": False, "error": "advisory_only"}]}
    monkeypatch.setattr(ApiClient, "solve", lambda self, *a, **k: (fake_api.setdefault("solve", []).append((a, k)),
                                                                   ApiResult(200, fell_back))[1])
    at = run_page("Solve")
    at.radio(key="solve_source").set_value("Hard sample").run()
    image, text = samples.hard_sample("Hard", 0)
    assert any(text in c.value for c in at.caption)
    at.button(key="solve_go").click().run()
    assert not at.exception
    (args, _), = fake_api["solve"]
    assert args[0] == image and args[2] == text
    page = " ".join(m.value for m in at.markdown) + " ".join(w.value for w in at.warning)
    assert "Fallback chain" in page and "segment_knn" in page and "answer length outside" in page


def test_confident_wrong_answer_is_shown_as_an_error_not_a_success(monkeypatch, fake_api):
    wrong = {**SOLVED, "accepted": True, "success": False, "text": "8WM6X", "confidence": 0.943}
    monkeypatch.setattr(ApiClient, "solve", lambda self, *a, **k: ApiResult(200, wrong))
    at = run_page("Solve")
    at.radio(key="solve_source").set_value("Hard sample").run()
    at.button(key="solve_go").click().run()
    assert not at.exception
    assert any("Wrong answer" in e.value and "94.3%" in e.value for e in at.error)
    assert not any("8WM6X" in s.value for s in at.success)
