"""
Opt-in tests against the REAL provider siteverify endpoints, using only each
provider's published test keys and test tokens (assignment section 7).

    RUN_LIVE=1 pytest tests/test_live_providers.py -v

Skipped by default so the normal suite stays offline and deterministic.
"""
from __future__ import annotations

import os

import pytest
from sqlalchemy.orm import sessionmaker

from app import db as app_db
from app.config import Settings, settings
from app.models import Base
from app.verification.service import TURNSTILE_DUMMY_SECRETS, VerificationService

pytestmark = pytest.mark.skipif(os.getenv("RUN_LIVE") != "1", reason="set RUN_LIVE=1 to call real provider endpoints")


@pytest.fixture
def Session(tmp_path):
    original = settings.database_url
    settings.database_url = f"sqlite:///{tmp_path / 'live.db'}"
    try:
        eng = app_db._make_engine()
    finally:
        settings.database_url = original
    Base.metadata.create_all(eng)
    return sessionmaker(bind=eng)


@pytest.fixture
def svc():
    return VerificationService(cfg=Settings(
        turnstile_secret=TURNSTILE_DUMMY_SECRETS["pass"],
        hcaptcha_secret="0x" + "0" * 40,
    ))


def run(svc, Session, provider, token, **kw):
    with Session() as db:
        return svc.verify(db, provider, token, **kw)


def test_turnstile_dummy_token_passes(svc, Session):
    r = run(svc, Session, "turnstile", "XXXX.DUMMY.TOKEN.XXXX")
    assert r.ok and r.http_status == 200


def test_turnstile_always_fail_secret(svc, Session):
    r = run(svc, Session, "turnstile", "XXXX.DUMMY.TOKEN.XXXX", test_scenario="fail")
    assert (r.reason, r.provider_error_codes) == ("invalid_token", ["invalid-input-response"])


def test_turnstile_duplicate_secret(svc, Session):
    r = run(svc, Session, "turnstile", "XXXX.DUMMY.TOKEN.XXXX", test_scenario="spent")
    assert (r.reason, r.http_status, r.provider_error_codes) == ("expired_token", 410, ["timeout-or-duplicate"])


def test_hcaptcha_published_test_token_passes(svc, Session):
    assert run(svc, Session, "hcaptcha", "10000000-aaaa-bbbb-cccc-000000000001").ok


def test_hcaptcha_fake_token_rejected(svc, Session):
    r = run(svc, Session, "hcaptcha", "FAKE-TOKEN")
    assert (r.reason, r.provider_error_codes) == ("invalid_token", ["invalid-input-response"])


def test_recaptcha_test_secret_accepts_any_token_and_replay_is_caught_locally(svc, Session):
    # Google documents that the test secret passes every request, so an "invalid"
    # reCAPTCHA token cannot be demonstrated live; our ledger still catches the replay.
    assert run(svc, Session, "recaptcha", "arbitrary-token-value-1").ok
    assert run(svc, Session, "recaptcha", "arbitrary-token-value-1").reason == "reused_token"
