"""
Redacting log filter and token fingerprint helper.

Two layers of redaction, applied to every log message AND to formatted
exception tracebacks:

1. Registered secrets (settings.secret_values()) are replaced wherever they
   occur as plain text.
2. Key/value pairs whose key names a credential (token, secret, password,
   api_key, authorization, *-response ...) have their value replaced, in
   `key=value`, `key: value` and JSON `"key": "value"` forms. Tokens are
   dynamic and can never be pre-registered, so this layer is what protects
   them.

Callers should still log fingerprint(token), never the token.
`redact()` is also used before any error text is stored in the database.
"""
from __future__ import annotations

import hashlib
import logging
import re
from typing import Sequence

_KV = re.compile(
    r"""(?P<key>["']?[\w-]*?(?:secret|password|passwd|api[_-]?key|token|authorization|-response)["']?)
        (?P<sep>\s*[:=]\s*)
        (?P<val>"(?:[^"\\]|\\.)*"|'(?:[^'\\]|\\.)*'|(?:Bearer\s+)?[^\s,;&}\]]+)""",
    re.IGNORECASE | re.VERBOSE,
)
_REDACTED = "[REDACTED]"
_secrets: list[str] = []
_secret_pattern: re.Pattern[str] | None = None


def fingerprint(value: str) -> str:
    """Return the first 12 hex characters of the SHA-256 of *value*."""
    return hashlib.sha256(value.encode()).hexdigest()[:12]


def register_secrets(secrets: Sequence[str]) -> None:
    """Call once at startup with settings.secret_values()."""
    global _secret_pattern
    for s in secrets:
        if s and len(s) >= 8 and s not in _secrets:
            _secrets.append(s)
    _secret_pattern = (
        re.compile("|".join(re.escape(s) for s in sorted(_secrets, key=len, reverse=True)))
        if _secrets
        else None
    )


def _kv_sub(m: re.Match[str]) -> str:
    return f"{m.group('key')}{m.group('sep')}{_REDACTED}"


def redact(text: str) -> str:
    """Remove registered secrets and credential-looking key/value pairs."""
    if _secret_pattern is not None:
        text = _secret_pattern.sub(_REDACTED, text)
    return _KV.sub(_kv_sub, text)


class RedactingFilter(logging.Filter):
    """
    Attach to handlers (or loggers). Only rewrites a record when redaction
    actually changed something, so formatters that read record.args (uvicorn's
    access log) keep working for clean records.
    """

    def __init__(self, secrets: Sequence[str] | None = None) -> None:
        super().__init__()
        if secrets:
            register_secrets(secrets)

    def add_secrets(self, secrets: Sequence[str]) -> None:
        register_secrets(secrets)

    def filter(self, record: logging.LogRecord) -> bool:  # noqa: A003
        try:
            msg = record.getMessage()
        except Exception:  # noqa: BLE001 - malformed format args
            msg = str(record.msg)
        clean = redact(msg)
        if clean != msg:
            record.msg, record.args = clean, ()
        # Tracebacks are formatted later by the handler and can embed
        # exception text containing secrets: format now, redact, and replace.
        if record.exc_info and not record.exc_text:
            record.exc_text = logging.Formatter().formatException(record.exc_info)
        if record.exc_text:
            record.exc_text = redact(record.exc_text)
            record.exc_info = None
        if record.stack_info:
            record.stack_info = redact(record.stack_info)
        return True


_shared_filter = RedactingFilter()


def build_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    if _shared_filter not in logger.filters:
        logger.addFilter(_shared_filter)
    return logger


def configure_logging(level: int = logging.INFO) -> None:
    """
    Root logging setup. The redacting filter goes on every handler of the root
    logger and of uvicorn.error (unhandled-exception tracebacks are logged
    there through uvicorn's own handler, which root filters never see).
    uvicorn.access is left alone: its records carry only method, path and
    status, and tokens are never placed in URLs by this API.
    """
    logging.basicConfig(level=level, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    for name in ("", "uvicorn", "uvicorn.error"):
        for handler in logging.getLogger(name).handlers:
            if _shared_filter not in handler.filters:
                handler.addFilter(_shared_filter)
