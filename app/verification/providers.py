"""
Server-side `siteverify` clients for reCAPTCHA, hCaptcha and Cloudflare Turnstile.

Endpoints, parameters, error codes and token lifetimes follow each provider's
official documentation (checked while building M4):

    reCAPTCHA  POST https://www.google.com/recaptcha/api/siteverify     lifetime 2 min, single use
    hCaptcha   POST https://api.hcaptcha.com/siteverify                  lifetime 120 s, single use
    Turnstile  POST https://challenges.cloudflare.com/turnstile/v0/siteverify
               lifetime 300 s, single use, accepts an idempotency_key

Retry policy
------------
siteverify CONSUMES the token, so a blind retry after a lost response would be
reported by the provider as a replay. Therefore:

  * connect errors, pool timeouts, 502/503/504   retried once: the request either
    never reached the provider or failed at its gateway;
  * read timeouts and other transport errors     retried only for Turnstile, which
    accepts an idempotency_key (a retry with the same key is safe by design);
  * 429, 500, other statuses, unparsable bodies  not retried.

No exception text is ever propagated: library messages can echo URLs, and
nothing secret may reach logs or responses. Only a short reason code does.
"""
from __future__ import annotations

import logging
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone

import httpx

from app.config import Settings
from app.config import settings as default_settings

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ProviderSpec:
    name: str
    url_attr: str
    secret_attr: str
    sitekey_attr: str
    lifetime_s: int
    idempotent: bool        # accepts an idempotency key, so any failure is safe to retry
    sends_sitekey: bool     # provider can bind the expected sitekey to the secret


SPECS: dict[str, ProviderSpec] = {
    "recaptcha": ProviderSpec("recaptcha", "recaptcha_verify_url", "recaptcha_secret",
                              "recaptcha_sitekey", 120, False, False),
    "hcaptcha": ProviderSpec("hcaptcha", "hcaptcha_verify_url", "hcaptcha_secret",
                             "hcaptcha_sitekey", 120, False, True),
    "turnstile": ProviderSpec("turnstile", "turnstile_verify_url", "turnstile_secret",
                              "turnstile_sitekey", 300, True, False),
}
_GATEWAY_RETRYABLE = {502, 503, 504}


class ProviderUnavailable(Exception):
    """The provider gave no usable verdict."""

    def __init__(self, reason: str, *, may_have_consumed: bool) -> None:
        super().__init__(reason)
        self.reason = reason
        self.may_have_consumed = may_have_consumed


@dataclass
class ProviderReply:
    success: bool
    error_codes: list[str] = field(default_factory=list)
    challenge_ts: datetime | None = None
    hostname: str | None = None
    score: float | None = None
    action: str | None = None


def _parse_ts(value: object) -> datetime | None:
    if not isinstance(value, str):
        return None
    try:
        ts = datetime.fromisoformat(value.strip())
    except ValueError:
        return None
    return ts if ts.tzinfo else ts.replace(tzinfo=timezone.utc)


def _parse_reply(data: object) -> ProviderReply | None:
    if not isinstance(data, dict) or "success" not in data:
        return None
    codes = data.get("error-codes") or []
    codes = [str(c)[:64] for c in codes[:10]] if isinstance(codes, list) else []
    score = data.get("score")
    return ProviderReply(
        success=data["success"] is True,
        error_codes=codes,
        challenge_ts=_parse_ts(data.get("challenge_ts")),
        hostname=str(data["hostname"])[:255] if data.get("hostname") else None,
        score=float(score) if isinstance(score, (int, float)) else None,
        action=str(data["action"])[:100] if data.get("action") else None,
    )


class ProviderClient:
    def __init__(
        self,
        http: httpx.Client | None = None,
        cfg: Settings | None = None,
        sleep=time.sleep,
    ) -> None:
        self._cfg = cfg or default_settings
        self._http = http or httpx.Client()
        self._sleep = sleep

    def verify(
        self,
        provider: str,
        token: str,
        *,
        secret: str | None = None,
        remote_ip: str | None = None,
        idempotency_seed: str | None = None,
    ) -> ProviderReply:
        spec = SPECS[provider]
        cfg = self._cfg
        form = {
            "secret": secret if secret is not None else getattr(cfg, spec.secret_attr).get_secret_value(),
            "response": token,
        }
        if remote_ip:
            form["remoteip"] = remote_ip
        if spec.sends_sitekey and getattr(cfg, spec.sitekey_attr):
            form["sitekey"] = getattr(cfg, spec.sitekey_attr)
        if spec.idempotent and idempotency_seed:
            form["idempotency_key"] = str(uuid.uuid5(uuid.NAMESPACE_OID, idempotency_seed))
        url = getattr(cfg, spec.url_attr)
        timeout = httpx.Timeout(cfg.verify_timeout_s, connect=min(3.0, cfg.verify_timeout_s))

        attempts = 1 + max(0, cfg.verify_retries)
        reason, maybe_consumed = "unknown", False
        for attempt in range(1, attempts + 1):
            retryable = False
            try:
                resp = self._http.post(url, data=form, timeout=timeout)
            except (httpx.ConnectError, httpx.ConnectTimeout, httpx.PoolTimeout):
                reason, retryable = "connect_error", True          # never reached the provider
            except httpx.TransportError as exc:
                reason = "read_timeout" if isinstance(exc, httpx.ReadTimeout) else "transport_error"
                maybe_consumed = True                               # may have been processed
                retryable = spec.idempotent
            else:
                code = resp.status_code
                if code in _GATEWAY_RETRYABLE:
                    reason, retryable = f"http_{code}", True
                elif code == 429:
                    reason = "rate_limited"
                elif code != 200:
                    reason, maybe_consumed = f"http_{code}", True
                else:
                    try:
                        reply = _parse_reply(resp.json())
                    except ValueError:
                        reply = None
                    if reply is not None:
                        logger.info("siteverify provider=%s attempt=%d status=200 success=%s codes=%s",
                                    provider, attempt, reply.success, ",".join(reply.error_codes) or "-")
                        return reply
                    reason, maybe_consumed = "bad_response", True
            logger.warning("siteverify provider=%s attempt=%d failed reason=%s retry=%s",
                           provider, attempt, reason, retryable and attempt < attempts)
            if not retryable or attempt == attempts:
                break
            self._sleep(cfg.verify_backoff_s)
        raise ProviderUnavailable(reason, may_have_consumed=maybe_consumed)
