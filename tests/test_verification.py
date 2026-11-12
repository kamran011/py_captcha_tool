"""M4 tests: token verification service and API against a mocked siteverify."""
from __future__ import annotations

import json
import logging
import threading
import time
from datetime import datetime, timedelta, timezone
from urllib.parse import parse_qs

import httpx
import pytest
from fastapi.testclient import TestClient
from sqlalchemy.orm import sessionmaker

from app import db as app_db
from app.api.verify import get_service
from app.config import Settings, settings
from app.main import app
from app.models import Base, TokenLedger, Verification
from app.verification.providers import ProviderClient
from app.verification.service import (
    HTTP_STATUS, MAX_TOKEN_LEN, STALE_PENDING_S, TURNSTILE_DUMMY_SECRETS,
    PUBLISHED_TEST_SECRETS, VerificationService, classify_codes, validate_token_shape,
)

TOKEN = "tok_ABCdef1234567890.live-looking_value"


def iso(delta_s: int = 0) -> str:
    return (datetime.now(timezone.utc) + timedelta(seconds=delta_s)).isoformat()


def body(success=True, codes=None, ts=None, **extra) -> httpx.Response:
    data = {"success": success, "challenge_ts": ts or iso(), "hostname": "localhost", **extra}
    if codes:
        data["error-codes"] = codes
    return httpx.Response(200, json=data)


class Fake:
    """Programmable siteverify. `script` is a list of responses/exceptions, consumed in order;
    the last one repeats. Records every request form."""

    def __init__(self, *script, delay=0.0):
        self.script = list(script) or [body()]
        self.calls: list[dict] = []
        self.delay = delay
        self._lock = threading.Lock()

    def __call__(self, request: httpx.Request) -> httpx.Response:
        form = {k: v[0] for k, v in parse_qs(request.content.decode()).items()}
        form["_url"] = str(request.url)
        with self._lock:
            self.calls.append(form)
            item = self.script.pop(0) if len(self.script) > 1 else self.script[0]
        if self.delay:
            time.sleep(self.delay)
        if isinstance(item, Exception):
            raise item
        return item


@pytest.fixture
def engine(tmp_path):
    original = settings.database_url
    settings.database_url = f"sqlite:///{tmp_path / 'v.db'}"
    try:
        eng = app_db._make_engine()
    finally:
        settings.database_url = original
    Base.metadata.create_all(eng)
    return eng


@pytest.fixture
def Session(engine):
    return sessionmaker(bind=engine)


@pytest.fixture
def cfg():
    return Settings(verify_backoff_s=0.0)


def make(fake: Fake, cfg: Settings) -> VerificationService:
    http = httpx.Client(transport=httpx.MockTransport(fake))
    return VerificationService(ProviderClient(http, cfg, sleep=lambda s: None), cfg)


def run(svc, Session, provider="recaptcha", token=TOKEN, **kw):
    with Session() as db:
        return svc.verify(db, provider, token, **kw)


# ── success ──────────────────────────────────────────────────────────────────
@pytest.mark.parametrize("provider", ["recaptcha", "hcaptcha", "turnstile"])
def test_valid_token_accepted(provider, Session, cfg):
    fake = Fake()
    r = run(make(fake, cfg), Session, provider)
    assert r.ok and r.reason == "ok" and r.http_status == 200
    assert r.hostname == "localhost" and r.token_fingerprint and len(r.token_fingerprint) == 12
    assert len(fake.calls) == 1


def test_secret_and_expected_sitekey_are_sent_to_the_right_endpoint(Session, cfg):
    fake = Fake()
    run(make(fake, cfg), Session, "hcaptcha")
    call = fake.calls[0]
    assert call["_url"] == cfg.hcaptcha_verify_url
    assert call["secret"] == cfg.hcaptcha_secret.get_secret_value()
    assert call["response"] == TOKEN and call["sitekey"] == cfg.hcaptcha_sitekey


def test_remote_ip_forwarded_only_when_valid(Session, cfg):
    fake = Fake()
    svc = make(fake, cfg)
    run(svc, Session, remote_ip="203.0.113.9")
    run(svc, Session, token=TOKEN + "2", remote_ip="not-an-ip; DROP")
    assert fake.calls[0]["remoteip"] == "203.0.113.9" and "remoteip" not in fake.calls[1]


# ── missing / malformed (no network) ─────────────────────────────────────────
@pytest.mark.parametrize("token", [None, "", "   ", "\t\n"])
def test_missing_token(token, Session, cfg):
    fake = Fake()
    r = run(make(fake, cfg), Session, token=token)
    assert (r.reason, r.http_status, r.ok) == ("missing_token", 400, False)
    assert fake.calls == []


@pytest.mark.parametrize("token", [
    "has space inside", "tab\tinside", "new\nline", "nul\x00byte", "unicodé", "x" * (MAX_TOKEN_LEN + 1), 12345,
])
def test_malformed_token_rejected_before_any_network_call(token, Session, cfg):
    fake = Fake()
    r = run(make(fake, cfg), Session, token=token)
    assert (r.reason, r.http_status) == ("malformed_token", 400)
    assert fake.calls == []


@pytest.mark.parametrize("token", ["XXXX.DUMMY.TOKEN.XXXX", "FAKE-TOKEN", "10000000-aaaa-bbbb-cccc-000000000001", "x"])
def test_official_test_tokens_pass_the_shape_check(token):
    assert validate_token_shape(token) is None


def test_unsupported_provider(Session, cfg):
    fake = Fake()
    for p in ("paypal", "", None, "RECAPTCHA ", {"a": 1}):
        r = run(make(fake, cfg), Session, provider=p)
        assert (r.reason, r.http_status) == ("unsupported_provider", 422)
    assert fake.calls == []


# ── invalid / expired ────────────────────────────────────────────────────────
def test_invalid_token(Session, cfg):
    r = run(make(Fake(body(False, ["invalid-input-response"])), cfg), Session)
    assert (r.reason, r.http_status) == ("invalid_token", 400)
    assert r.provider_error_codes == ["invalid-input-response"]


def test_unknown_error_code_is_invalid_not_a_crash(Session, cfg):
    r = run(make(Fake(body(False, ["some-new-code"])), cfg), Session)
    assert r.reason == "invalid_token" and r.provider_error_codes == ["some-new-code"]


@pytest.mark.parametrize("provider,code", [("hcaptcha", "expired-input-response"),
                                           ("recaptcha", "timeout-or-duplicate"),
                                           ("turnstile", "timeout-or-duplicate")])
def test_provider_reports_expired(provider, code, Session, cfg):
    r = run(make(Fake(body(False, [code])), cfg), Session, provider)
    assert (r.reason, r.http_status) == ("expired_token", 410)


def test_old_challenge_ts_is_expired_even_if_provider_accepts(Session, cfg):
    r = run(make(Fake(body(True, ts=iso(-200))), cfg), Session, "recaptcha")   # limit 120 s
    assert r.reason == "expired_token" and "limit is 120s" in r.detail


def test_turnstile_lifetime_is_longer_than_recaptcha(Session, cfg):
    assert run(make(Fake(body(True, ts=iso(-200))), cfg), Session, "turnstile").ok


def test_per_request_max_age_can_only_tighten(Session, cfg):
    svc = make(Fake(body(True, ts=iso(-30))), cfg)
    assert run(svc, Session, token=TOKEN + "a", max_age_s=10).reason == "expired_token"
    assert run(svc, Session, token=TOKEN + "b", max_age_s=3600).ok          # 30 s old is inside the 120 s lifetime
    old = make(Fake(body(True, ts=iso(-200))), cfg)
    assert run(old, Session, token=TOKEN + "c", max_age_s=86_400).reason == "expired_token"


def test_global_ceiling_applies(Session):
    c = Settings(verify_backoff_s=0.0, token_max_age_s=5)
    assert run(make(Fake(body(True, ts=iso(-30))), c), Session, "turnstile").reason == "expired_token"


# ── reuse ────────────────────────────────────────────────────────────────────
def test_replayed_token_rejected_locally_without_calling_the_provider(Session, cfg):
    fake = Fake()
    svc = make(fake, cfg)
    assert run(svc, Session).ok
    again = run(svc, Session)
    assert (again.reason, again.http_status) == ("reused_token", 409)
    assert len(fake.calls) == 1                                    # provider never saw the replay


def test_replay_is_caught_even_if_the_provider_would_accept_it(Session, cfg):
    # Turnstile documents eventually-consistent reuse detection; the ledger must not rely on it.
    svc = make(Fake(body(True)), cfg)
    assert run(svc, Session, "turnstile").ok
    assert run(svc, Session, "turnstile").reason == "reused_token"


def test_provider_already_seen_code(Session, cfg):
    r = run(make(Fake(body(False, ["already-seen-response"])), cfg), Session, "hcaptcha")
    assert (r.reason, r.http_status) == ("reused_token", 409)


def test_failed_verification_still_burns_the_token(Session, cfg):
    svc = make(Fake(body(False, ["invalid-input-response"])), cfg)
    assert run(svc, Session).reason == "invalid_token"
    assert run(svc, Session).reason == "reused_token"


def test_ledger_stores_hash_not_token(Session, engine, cfg):
    run(make(Fake(), cfg), Session)
    with Session() as db:
        row = db.query(TokenLedger).one()
        assert row.status == "consumed" and row.verdict == "ok"
        assert row.token_hash != TOKEN and TOKEN not in json.dumps(row.__dict__, default=str)


def test_concurrent_submissions_of_one_token_reach_the_provider_once(Session, cfg):
    fake = Fake(delay=0.08)
    svc = make(fake, cfg)
    results: list = []

    def worker():
        results.append(run(svc, Session))

    threads = [threading.Thread(target=worker) for _ in range(12)]
    [t.start() for t in threads]
    [t.join() for t in threads]
    reasons = sorted(r.reason for r in results)
    assert reasons == ["ok"] + ["reused_token"] * 11
    assert len(fake.calls) == 1


# ── published test tokens (identical on every widget run) ────────────────────
@pytest.mark.parametrize("provider,tok,secret", [
    ("turnstile", "XXXX.DUMMY.TOKEN.XXXX", "1x0000000000000000000000000000000AA"),
    ("hcaptcha", "10000000-aaaa-bbbb-cccc-000000000001", "0x" + "0" * 40),
])
def test_published_test_tokens_can_be_verified_repeatedly_in_test_mode(provider, tok, secret, Session):
    c = Settings(verify_backoff_s=0.0, **{f"{provider}_secret": secret})
    fake = Fake()
    svc = make(fake, c)
    for _ in range(3):
        r = run(svc, Session, provider, token=tok)
        assert r.ok and "replay tracking skipped" in r.detail
    assert len(fake.calls) == 3
    with Session() as db:
        assert db.query(TokenLedger).count() == 0


def test_published_test_tokens_get_no_exemption_with_a_real_secret(Session):
    real = Settings(verify_backoff_s=0.0, turnstile_secret="0x4AAAAAAA-real-looking-secret-value")
    svc = make(Fake(), real)
    assert run(svc, Session, "turnstile", token="XXXX.DUMMY.TOKEN.XXXX").ok
    assert run(svc, Session, "turnstile", token="XXXX.DUMMY.TOKEN.XXXX").reason == "reused_token"


def test_exemption_is_per_provider(Session):
    c = Settings(verify_backoff_s=0.0, recaptcha_secret="6LeIxAcTAAAAAGG-vFI1TnRWxMZNFuojJ4WifJWe")
    svc = make(Fake(), c)
    assert run(svc, Session, "recaptcha", token="XXXX.DUMMY.TOKEN.XXXX").ok
    assert run(svc, Session, "recaptcha", token="XXXX.DUMMY.TOKEN.XXXX").reason == "reused_token"


# ── provider failures ────────────────────────────────────────────────────────
def test_connect_failure_retried_once_then_503_and_token_stays_usable(Session, cfg):
    fake = Fake(httpx.ConnectError("boom"), httpx.ConnectError("boom"), body())
    svc = make(fake, cfg)
    r = run(svc, Session)
    assert (r.reason, r.http_status, r.retry_possible, r.ok) == ("provider_unavailable", 503, True, False)
    assert len(fake.calls) == 2 and "not consumed" in r.detail
    assert run(svc, Session).ok                                    # claim was released: retry works


def test_gateway_502_retried_then_succeeds(Session, cfg):
    fake = Fake(httpx.Response(502), body())
    r = run(make(fake, cfg), Session)
    assert r.ok and len(fake.calls) == 2


def test_read_timeout_not_retried_for_non_idempotent_provider(Session, cfg):
    fake = Fake(httpx.ReadTimeout("slow"), body())
    r = run(make(fake, cfg), Session, "recaptcha")
    assert (r.reason, r.retry_possible) == ("provider_unavailable", True)
    assert len(fake.calls) == 1 and "may or may not" in r.detail
    with Session() as db:
        assert db.query(TokenLedger).one().status == "indeterminate"


def test_indeterminate_token_allows_exactly_one_reattempt(Session, cfg):
    svc = make(Fake(httpx.ReadTimeout("slow"), body()), cfg)
    assert run(svc, Session).reason == "provider_unavailable"
    assert run(svc, Session).ok                                    # one re-attempt granted
    assert run(svc, Session).reason == "reused_token"              # now spent


def test_turnstile_read_timeout_is_retried_with_the_same_idempotency_key(Session, cfg):
    fake = Fake(httpx.ReadTimeout("slow"), body())
    r = run(make(fake, cfg), Session, "turnstile")
    assert r.ok and len(fake.calls) == 2
    assert fake.calls[0]["idempotency_key"] == fake.calls[1]["idempotency_key"]


def test_idempotency_key_only_sent_to_turnstile(Session, cfg):
    fake = Fake()
    run(make(fake, cfg), Session, "recaptcha")
    assert "idempotency_key" not in fake.calls[0]


@pytest.mark.parametrize("resp", [httpx.Response(500), httpx.Response(418), httpx.Response(200, text="<html>nope"),
                                  httpx.Response(200, json=["not", "a", "dict"]), httpx.Response(200, json={"no": "success"})])
def test_garbage_or_error_responses_are_503_never_fail_open(resp, Session, cfg):
    fake = Fake(resp)
    r = run(make(fake, cfg), Session)
    assert (r.ok, r.reason, r.http_status) == (False, "provider_unavailable", 503)
    assert len(fake.calls) == 1                                    # 500/other: not retried


def test_rate_limited_is_not_retried_and_not_consumed(Session, cfg):
    fake = Fake(httpx.Response(429), body())
    svc = make(fake, cfg)
    r = run(svc, Session)
    assert r.reason == "provider_unavailable" and len(fake.calls) == 1 and "not consumed" in r.detail
    assert run(svc, Session).ok


def test_provider_internal_error_code(Session, cfg):
    r = run(make(Fake(body(False, ["internal-error"])), cfg), Session, "turnstile")
    assert (r.reason, r.retry_possible) == ("provider_unavailable", True)


def test_unexpected_exception_releases_claim_as_indeterminate(Session, cfg):
    class Boom:
        def verify(self, *a, **k):
            raise RuntimeError("bug")

    svc = VerificationService(Boom(), cfg)          # type: ignore[arg-type]
    with pytest.raises(RuntimeError):
        run(svc, Session)
    with Session() as db:
        assert db.query(TokenLedger).one().status == "indeterminate"


def test_abandoned_pending_claim_is_recoverable_after_stale_window(Session, cfg):
    with Session() as db:
        db.add(TokenLedger(token_hash=__import__("hashlib").sha256(TOKEN.encode()).hexdigest(),
                           token_fingerprint="x", provider="recaptcha", status="pending",
                           updated_at=datetime.now(timezone.utc) - timedelta(seconds=STALE_PENDING_S + 5)))
        db.commit()
    assert run(make(Fake(), cfg), Session).ok


def test_fresh_pending_claim_is_not_stolen(Session, cfg):
    with Session() as db:
        db.add(TokenLedger(token_hash=__import__("hashlib").sha256(TOKEN.encode()).hexdigest(),
                           token_fingerprint="x", provider="recaptcha", status="pending",
                           updated_at=datetime.now(timezone.utc)))
        db.commit()
    assert run(make(Fake(), cfg), Session).reason == "reused_token"


# ── our configuration problems are not the caller's fault ────────────────────
@pytest.mark.parametrize("code", ["invalid-input-secret", "missing-input-secret",
                                  "sitekey-secret-mismatch", "not-using-dummy-passcode"])
def test_secret_problems_are_500_server_misconfigured(code, Session, cfg):
    r = run(make(Fake(body(False, [code])), cfg), Session, "hcaptcha")
    assert (r.reason, r.http_status) == ("server_misconfigured", 500)


def test_empty_secret_is_caught_before_calling_out(Session):
    c = Settings(verify_backoff_s=0.0, recaptcha_secret="")
    fake = Fake()
    r = run(make(fake, c), Session, "recaptcha")
    assert r.reason == "server_misconfigured" and fake.calls == []


def test_bad_request_code_is_502(Session, cfg):
    assert run(make(Fake(body(False, ["bad-request"])), cfg), Session).http_status == 502


def test_every_reason_has_an_http_status():
    for reason in ("ok", "missing_token", "malformed_token", "invalid_token", "expired_token", "reused_token",
                   "unsupported_provider", "server_misconfigured", "provider_unavailable", "provider_error",
                   "test_scenario_not_allowed"):
        assert reason in HTTP_STATUS
    assert classify_codes(["missing-input-secret", "invalid-input-response"])[0] == "server_misconfigured"


# ── Turnstile dummy-secret scenarios ─────────────────────────────────────────
def test_turnstile_scenarios_use_the_published_dummy_secrets(Session):
    cfg = Settings(verify_backoff_s=0.0, turnstile_secret=TURNSTILE_DUMMY_SECRETS["pass"])

    def cf(request):                          # behaves like Cloudflare's dummy secrets
        secret = parse_qs(request.content.decode())["secret"][0]
        if secret == TURNSTILE_DUMMY_SECRETS["fail"]:
            return body(False, ["invalid-input-response"])
        if secret == TURNSTILE_DUMMY_SECRETS["spent"]:
            return body(False, ["timeout-or-duplicate"])
        return body(True)

    svc = VerificationService(ProviderClient(httpx.Client(transport=httpx.MockTransport(cf)), cfg, lambda s: None), cfg)
    assert run(svc, Session, "turnstile", token="XXXX.DUMMY.TOKEN.XXXX.a", test_scenario="pass").ok
    assert run(svc, Session, "turnstile", token="XXXX.DUMMY.TOKEN.XXXX.b", test_scenario="fail").reason == "invalid_token"
    assert run(svc, Session, "turnstile", token="XXXX.DUMMY.TOKEN.XXXX.c", test_scenario="spent").reason == "expired_token"


def test_scenarios_refused_with_a_real_secret_or_other_provider(Session):
    real = Settings(verify_backoff_s=0.0, turnstile_secret="0x4AAAAAAA-real-looking-secret-value")
    fake = Fake()
    assert run(make(fake, real), Session, "turnstile", test_scenario="fail").reason == "test_scenario_not_allowed"
    dummy = Settings(verify_backoff_s=0.0, turnstile_secret=TURNSTILE_DUMMY_SECRETS["pass"])
    assert run(make(fake, dummy), Session, "recaptcha", test_scenario="pass").reason == "test_scenario_not_allowed"
    assert fake.calls == []


# ── secrets and tokens never leak ────────────────────────────────────────────
def test_no_secret_or_token_in_logs_responses_or_database(Session, engine, cfg, caplog):
    caplog.set_level(logging.DEBUG)
    fake = Fake(httpx.ConnectError("x"), httpx.ConnectError("x"))
    svc = make(fake, cfg)
    outputs = []
    for provider in ("recaptcha", "hcaptcha", "turnstile"):
        outputs.append(run(svc, Session, provider, token=f"{TOKEN}-{provider}").as_dict())
    ok = make(Fake(body(False, ["invalid-input-response"])), cfg)
    outputs.append(run(ok, Session, token=TOKEN + "-bad").as_dict())
    run(ok, Session, token=TOKEN + "-bad")                          # replay

    haystack = json.dumps(outputs) + "\n".join(r.getMessage() for r in caplog.records)
    with engine.connect() as c:
        for table in ("verifications", "token_ledger", "attempts"):
            haystack += json.dumps([list(map(str, row)) for row in c.exec_driver_sql(f"select * from {table}")])
    secrets = [cfg.recaptcha_secret.get_secret_value(), cfg.hcaptcha_secret.get_secret_value(),
               cfg.turnstile_secret.get_secret_value(), TOKEN]
    for s in secrets:
        assert s not in haystack, f"leaked {s[:8]}..."


def test_verification_rows_recorded_for_every_outcome(Session, cfg):
    svc = make(Fake(), cfg)
    run(svc, Session, token=None)
    run(svc, Session, token="bad token")
    run(svc, Session, token="good-token-1")
    run(svc, Session, token="good-token-1")
    with Session() as db:
        reasons = [v.reason for v in db.query(Verification).order_by(Verification.id)]
    assert reasons == ["missing_token", "malformed_token", "ok", "reused_token"]


# ── HTTP API ─────────────────────────────────────────────────────────────────
@pytest.fixture
def client(Session, cfg):
    fake = Fake()
    svc = make(fake, cfg)

    def db_override():
        with Session() as db:
            yield db

    app.dependency_overrides[app_db.get_db] = db_override
    app.dependency_overrides[get_service] = lambda: svc
    from app.api.deps import get_db as deps_get_db
    app.dependency_overrides[deps_get_db] = db_override
    c = TestClient(app)
    c.fake = fake
    yield c
    app.dependency_overrides.clear()


KEY = {"X-API-Key": settings.api_key.get_secret_value()}


def test_api_requires_key(client):
    assert client.post("/api/v1/verify", json={"provider": "recaptcha", "token": TOKEN}).status_code == 401


def test_api_success_and_replay_status_codes(client):
    ok = client.post("/api/v1/verify", headers=KEY, json={"provider": "recaptcha", "token": TOKEN})
    assert ok.status_code == 200 and ok.json()["ok"] is True
    again = client.post("/api/v1/verify", headers=KEY, json={"provider": "recaptcha", "token": TOKEN})
    assert again.status_code == 409 and again.json()["reason"] == "reused_token"


def test_api_missing_token_is_structured_400_not_framework_422(client):
    r = client.post("/api/v1/verify", headers=KEY, json={"provider": "recaptcha"})
    assert r.status_code == 400 and r.json()["reason"] == "missing_token"


def test_api_unsupported_provider(client):
    r = client.post("/api/v1/verify", headers=KEY, json={"provider": "nope", "token": TOKEN})
    assert r.status_code == 422 and r.json()["reason"] == "unsupported_provider"


def test_validation_errors_never_echo_the_submitted_token(client):
    secret_token = "SUPER-SECRET-TOKEN-VALUE-12345"
    r = client.post("/api/v1/verify", headers=KEY,
                    json={"provider": {"nested": secret_token}, "token": secret_token, "max_age_s": "abc"})
    assert r.status_code == 422 and r.json()["reason"] == "invalid_request"
    assert secret_token not in r.text


def test_demo_endpoint_needs_no_key_but_validates_the_same(client):
    r = client.post("/demo-api/verify", json={"provider": "turnstile", "token": "XXXX.DUMMY.TOKEN.XXXX"})
    assert r.status_code == 200 and r.json()["ok"] is True


def test_demo_endpoint_refuses_non_loopback_clients(Session, cfg):
    svc = make(Fake(), cfg)
    app.dependency_overrides[get_service] = lambda: svc
    try:
        c = TestClient(app, client=("203.0.113.7", 5000))
        assert c.post("/demo-api/verify", json={"provider": "recaptcha", "token": TOKEN}).status_code == 403
    finally:
        app.dependency_overrides.clear()


def test_openapi_documents_verify_but_hides_demo_endpoint(client):
    paths = client.get("/openapi.json").json()["paths"]
    assert "/api/v1/verify" in paths and "/demo-api/verify" not in paths
