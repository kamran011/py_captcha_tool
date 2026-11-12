"""
Token verification service.

Every outcome is a VerifyResult with a machine-readable `reason`:

    ok                    200   provider accepted a fresh token
    missing_token         400   no token supplied
    malformed_token       400   not a plausible token (checked before any network call)
    invalid_token         400   provider rejected it
    expired_token         410   provider says expired, or older than our age limit
    reused_token          409   token already verified (our ledger, or provider)
    unsupported_provider  422
    server_misconfigured  500   our secret / sitekey is wrong (not the caller's fault)
    provider_unavailable  503   no verdict (timeout, 5xx, rate limit); never fail-open
    provider_error        502   provider said our request was malformed
    test_scenario_not_allowed 400

Replay protection is an atomic claim: the token's SHA-256 is INSERTed (UNIQUE)
*before* the provider is called, so concurrent submissions of one token cannot
both reach the provider. The provider's own reuse detection is not enough:
Cloudflare documents Turnstile's as eventually consistent, with repeats inside
a few seconds able to succeed.
"""
from __future__ import annotations

import hashlib
import ipaddress
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone

from sqlalchemy import delete, or_, update
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from app.config import Settings
from app.config import settings as default_settings
from app.logging_config import fingerprint, redact
from app.models import TokenLedger, Verification
from app.verification.providers import SPECS, ProviderClient, ProviderReply, ProviderUnavailable

logger = logging.getLogger(__name__)

MAX_TOKEN_LEN = 2048
STALE_PENDING_S = 60          # a claim older than this is treated as abandoned

HTTP_STATUS = {
    "ok": 200, "missing_token": 400, "malformed_token": 400, "invalid_token": 400,
    "expired_token": 410, "reused_token": 409, "unsupported_provider": 422,
    "test_scenario_not_allowed": 400, "server_misconfigured": 500,
    "provider_error": 502, "provider_unavailable": 503,
}

# Published Turnstile dummy secrets (Cloudflare "testing" docs)
TURNSTILE_DUMMY_SECRETS = {
    "pass": "1x0000000000000000000000000000000AA",
    "fail": "2x0000000000000000000000000000000AA",
    "spent": "3x0000000000000000000000000000000AA",    # -> timeout-or-duplicate
}

# Official test credentials. A test sitekey issues the SAME token string every time
# (Cloudflare: XXXX.DUMMY.TOKEN.XXXX; hCaptcha: fixed per sitekey), so replay tracking
# would reject every second legitimate demo run. Those exact tokens skip the ledger,
# and only while that provider's published test secret is configured: with a real
# secret they get no special treatment (and the provider rejects them anyway).
PUBLISHED_TEST_TOKENS = {
    "turnstile": {"XXXX.DUMMY.TOKEN.XXXX"},
    "hcaptcha": {"10000000-aaaa-bbbb-cccc-000000000001",
                 "20000000-aaaa-bbbb-cccc-000000000002",
                 "30000000-aaaa-bbbb-cccc-000000000003"},
}
PUBLISHED_TEST_SECRETS = {
    "turnstile": set(TURNSTILE_DUMMY_SECRETS.values()),
    "hcaptcha": {"0x0000000000000000000000000000000000000000"},
}

_MISCONFIG = {"missing-input-secret", "invalid-input-secret", "sitekey-secret-mismatch",
              "not-using-dummy-passcode"}


@dataclass
class VerifyResult:
    ok: bool
    provider: str | None
    reason: str
    http_status: int
    latency_ms: float = 0.0
    token_fingerprint: str | None = None
    provider_error_codes: list[str] = field(default_factory=list)
    challenge_ts: str | None = None
    hostname: str | None = None
    score: float | None = None
    action: str | None = None
    retry_possible: bool = False
    detail: str | None = None

    def as_dict(self) -> dict:
        return dict(self.__dict__)


def _now() -> datetime:
    return datetime.now(timezone.utc)


def validate_token_shape(token: object) -> str | None:
    """Return a reason if the token is unusable, else None. Deliberately lenient:
    real tokens are opaque and Turnstile's test token is `XXXX.DUMMY.TOKEN.XXXX`."""
    if token is None or (isinstance(token, str) and not token.strip()):
        return "missing_token"
    if not isinstance(token, str) or len(token) > MAX_TOKEN_LEN:
        return "malformed_token"
    if any(ord(c) < 33 or ord(c) > 126 for c in token):     # whitespace, control, non-ASCII
        return "malformed_token"
    return None


def classify_codes(codes: list[str]) -> tuple[str, str | None]:
    """Map provider error codes to (reason, detail). Most specific / most severe first."""
    s = set(codes)
    if s & _MISCONFIG:
        return "server_misconfigured", "verification secret or sitekey rejected by the provider"
    if "missing-input-response" in s:
        return "missing_token", None
    if "expired-input-response" in s:
        return "expired_token", None
    if "already-seen-response" in s:
        return "reused_token", "provider reports the token was already verified"
    if "timeout-or-duplicate" in s:
        # Our own replays were ruled out by the ledger, so this is expiry, or use
        # elsewhere; the provider uses a single code for both.
        return "expired_token", "provider reports expired or already used (single provider code)"
    if "internal-error" in s:
        return "provider_unavailable", "provider internal error"
    if "bad-request" in s:
        return "provider_error", "provider rejected the request as malformed"
    return "invalid_token", None


class VerificationService:
    def __init__(self, client: ProviderClient | None = None, cfg: Settings | None = None) -> None:
        self.cfg = cfg or default_settings
        self.client = client or ProviderClient(cfg=self.cfg)

    # ── public ──────────────────────────────────────────────────────────────
    def verify(
        self,
        db: Session,
        provider: str,
        token: str | None,
        *,
        remote_ip: str | None = None,
        max_age_s: int | None = None,
        test_scenario: str | None = None,
        now: datetime | None = None,
    ) -> VerifyResult:
        t0 = time.perf_counter()
        result = self._verify(db, provider, token, remote_ip, max_age_s, test_scenario, now or _now())
        result.latency_ms = round((time.perf_counter() - t0) * 1000, 2)
        result.http_status = HTTP_STATUS[result.reason]
        self._record(db, result)
        logger.info("verify provider=%s ok=%s reason=%s fp=%s latency_ms=%.0f",
                    result.provider, result.ok, result.reason, result.token_fingerprint, result.latency_ms)
        return result

    # ── internals ───────────────────────────────────────────────────────────
    def _fail(self, provider, reason, fp=None, **kw) -> VerifyResult:
        return VerifyResult(False, provider, reason, HTTP_STATUS[reason], token_fingerprint=fp, **kw)

    def _verify(self, db, provider, token, remote_ip, max_age_s, test_scenario, now) -> VerifyResult:
        spec = SPECS.get(provider) if isinstance(provider, str) else None
        if spec is None:
            return self._fail(None, "unsupported_provider",
                              detail=f"supported providers: {', '.join(sorted(SPECS))}")

        bad = validate_token_shape(token)
        if bad:
            return self._fail(provider, bad)
        token_hash = hashlib.sha256(token.encode()).hexdigest()
        fp = fingerprint(token)

        secret = None
        if test_scenario is not None:
            dummy = TURNSTILE_DUMMY_SECRETS
            configured = self.cfg.turnstile_secret.get_secret_value()
            if provider != "turnstile" or configured not in dummy.values() or test_scenario not in dummy:
                return self._fail(provider, "test_scenario_not_allowed", fp,
                                  detail="only available for Turnstile while its published dummy secret is configured")
            secret = dummy[test_scenario]
        elif not getattr(self.cfg, spec.secret_attr).get_secret_value():
            return self._fail(provider, "server_misconfigured", fp, detail="no secret configured")

        ip = None
        if remote_ip:
            try:
                ip = str(ipaddress.ip_address(remote_ip.strip()))
            except ValueError:
                ip = None                                # never forward junk upstream

        exempt = (token in PUBLISHED_TEST_TOKENS.get(provider, ())
                  and getattr(self.cfg, spec.secret_attr).get_secret_value()
                  in PUBLISHED_TEST_SECRETS.get(provider, ()))
        if not exempt and not self._claim(db, token_hash, fp, provider):
            return self._fail(provider, "reused_token", fp,
                              detail="this token was already submitted for verification")

        try:
            reply = self.client.verify(provider, token, secret=secret, remote_ip=ip,
                                       idempotency_seed=token_hash)
        except ProviderUnavailable as exc:
            if not exempt:
                self._release(db, token_hash, exc.may_have_consumed, exc.reason)
            return self._fail(provider, "provider_unavailable", fp, retry_possible=True,
                              detail=("no verdict (" + exc.reason + "); the token may or may not have been consumed"
                                      if exc.may_have_consumed else "no verdict (" + exc.reason + "); token not consumed"))
        except BaseException:
            self._release(db, token_hash, True, "unexpected_error")
            raise

        result = self._interpret(provider, reply, fp, now, max_age_s)
        if exempt:
            note = "published test token: replay tracking skipped"
            result.detail = f"{result.detail}; {note}" if result.detail else note
        else:
            self._settle(db, token_hash, result, reply)
        return result

    def _interpret(self, provider, reply: ProviderReply, fp, now, max_age_s) -> VerifyResult:
        base = dict(provider_error_codes=reply.error_codes, hostname=reply.hostname,
                    score=reply.score, action=reply.action,
                    challenge_ts=reply.challenge_ts.isoformat() if reply.challenge_ts else None)
        if not reply.success:
            reason, detail = classify_codes(reply.error_codes)
            return self._fail(provider, reason, fp, detail=detail,
                              retry_possible=reason == "provider_unavailable", **base)
        if reply.challenge_ts is not None:
            limit = min(SPECS[provider].lifetime_s, self.cfg.token_max_age_s,
                        max_age_s if max_age_s else 10**9)
            age = (now - reply.challenge_ts).total_seconds()
            if age > limit:
                return self._fail(provider, "expired_token", fp,
                                  detail=f"token is {int(age)}s old; limit is {limit}s", **base)
        return VerifyResult(True, provider, "ok", 200, token_fingerprint=fp, **base)

    # ── ledger (atomic claim) ───────────────────────────────────────────────
    def _claim(self, db: Session, token_hash: str, fp: str, provider: str) -> bool:
        db.add(TokenLedger(token_hash=token_hash, token_fingerprint=fp, provider=provider,
                           status="pending", updated_at=_now()))
        try:
            db.commit()
            return True
        except IntegrityError:
            db.rollback()
        # Already known. Re-attempt only if an earlier try ended without a verdict
        # (indeterminate) or its claimant vanished (stale pending); the UPDATE is a
        # compare-and-swap, so exactly one concurrent caller wins.
        stale = _now() - timedelta(seconds=STALE_PENDING_S)
        won = db.execute(
            update(TokenLedger)
            .where(TokenLedger.token_hash == token_hash)
            .where(or_(TokenLedger.status == "indeterminate",
                       (TokenLedger.status == "pending") & (TokenLedger.updated_at < stale)))
            .values(status="pending", updated_at=_now())
        ).rowcount
        db.commit()
        return won == 1

    def _release(self, db: Session, token_hash: str, may_have_consumed: bool, reason: str) -> None:
        if may_have_consumed:
            stmt = (update(TokenLedger).where(TokenLedger.token_hash == token_hash)
                    .values(status="indeterminate", verdict=reason[:64], updated_at=_now()))
        else:   # provably never consumed: forget the claim so the caller can retry cleanly
            stmt = delete(TokenLedger).where(TokenLedger.token_hash == token_hash)
        db.execute(stmt)
        db.commit()

    def _settle(self, db: Session, token_hash: str, result: VerifyResult, reply: ProviderReply) -> None:
        if result.reason in ("provider_unavailable",):
            self._release(db, token_hash, False, result.reason)
            return
        db.execute(update(TokenLedger).where(TokenLedger.token_hash == token_hash).values(
            status="consumed", verdict=result.reason[:64],
            challenge_ts=reply.challenge_ts, updated_at=_now()))
        db.commit()

    def _record(self, db: Session, r: VerifyResult) -> None:
        try:
            db.add(Verification(
                provider=r.provider, ok=r.ok, reason=r.reason, http_status=r.http_status,
                latency_ms=r.latency_ms, token_fingerprint=r.token_fingerprint,
                error=redact(r.detail) if r.detail else None))
            db.commit()
        except Exception:  # noqa: BLE001 - bookkeeping must never change the verdict
            db.rollback()
            logger.exception("could not record verification")
