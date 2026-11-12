"""
M0 tests — health, redaction and config sanity.

Run with:  pytest tests/test_m0.py -v
"""
from __future__ import annotations

import logging

import pytest
from fastapi.testclient import TestClient

from app.config import settings
from app.logging_config import RedactingFilter, fingerprint, register_secrets
from app.main import app


@pytest.fixture(scope="module")
def client():
    # Context manager runs the lifespan (DB init) like a real server start
    with TestClient(app) as c:
        yield c


# ── /health ───────────────────────────────────────────────────────────────────
class TestHealth:
    def test_returns_200(self, client):
        r = client.get("/health")
        assert r.status_code == 200

    def test_db_connected(self, client):
        data = client.get("/health").json()
        assert data["db"]["ok"] is True

    def test_status_ok(self, client):
        data = client.get("/health").json()
        assert data["status"] == "ok"

    def test_version_present(self, client):
        data = client.get("/health").json()
        assert "version" in data


# ── /api/v1/stats (requires API key) ─────────────────────────────────────────
class TestStats:
    def test_no_key_returns_401(self, client):
        r = client.get("/api/v1/stats")
        assert r.status_code == 401

    def test_wrong_key_returns_401(self, client):
        r = client.get("/api/v1/stats", headers={"X-API-Key": "wrong"})
        assert r.status_code == 401

    def test_valid_key_returns_200(self, client):
        key = settings.api_key.get_secret_value()
        r = client.get("/api/v1/stats", headers={"X-API-Key": key})
        assert r.status_code == 200

    def test_stats_shape(self, client):
        key = settings.api_key.get_secret_value()
        data = client.get("/api/v1/stats", headers={"X-API-Key": key}).json()
        assert "attempts" in data
        assert "verifications" in data
        assert "sites" in data


# ── Redacting log filter ──────────────────────────────────────────────────────
class TestRedactingFilter:
    def test_secret_scrubbed_from_message(self):
        filt = RedactingFilter(secrets=["supersecret123"])
        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="",
            lineno=0,
            msg="token=supersecret123 was used",
            args=(),
            exc_info=None,
        )
        filt.filter(record)
        assert "supersecret123" not in record.getMessage()
        assert "[REDACTED]" in record.getMessage()

    def test_harmless_message_unchanged(self):
        filt = RedactingFilter(secrets=["supersecret123"])
        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="",
            lineno=0,
            msg="Processed 10 images successfully",
            args=(),
            exc_info=None,
        )
        filt.filter(record)
        assert "Processed 10 images successfully" in record.getMessage()

    def test_multiple_secrets_all_scrubbed(self):
        filt = RedactingFilter(secrets=["secret_alpha", "secret_beta"])
        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="",
            lineno=0,
            msg="secret_alpha and secret_beta both present",
            args=(),
            exc_info=None,
        )
        filt.filter(record)
        msg = record.getMessage()
        assert "secret_alpha" not in msg
        assert "secret_beta" not in msg


# ── Token fingerprint ─────────────────────────────────────────────────────────
class TestFingerprint:
    def test_deterministic(self):
        assert fingerprint("token123") == fingerprint("token123")

    def test_length_12(self):
        assert len(fingerprint("anything")) == 12

    def test_hex_chars(self):
        fp = fingerprint("test")
        assert all(c in "0123456789abcdef" for c in fp)

    def test_different_inputs_differ(self):
        assert fingerprint("token_a") != fingerprint("token_b")


# ── Config: secrets never appear in string repr ───────────────────────────────
class TestConfigSecurity:
    def test_secret_not_in_repr(self):
        secret_val = settings.recaptcha_secret.get_secret_value()
        assert secret_val not in repr(settings)

    def test_secret_values_non_empty(self):
        vals = settings.secret_values()
        assert len(vals) > 0
        # Every value should be ≥8 characters (enforced in secret_values())
        assert all(len(v) >= 8 for v in vals)
