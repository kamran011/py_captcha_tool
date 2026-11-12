"""
Abuse guards shared by the API routes.

  guard        API-key check + per-key sliding-window rate limit (429 + Retry-After)
  solve_slot   bounds CPU-bound solves running at once (503 `busy` when saturated)
  ApiError     structured error: {"ok": false, "reason": ..., "detail": ...}

In-process state: fine for one API process, which is how this tool runs. A
multi-process deployment would move both to a shared store.
"""
from __future__ import annotations

import hashlib
import threading
import time
from collections import deque
from contextlib import contextmanager

from fastapi import Depends, Request, Response

from app.api.deps import require_api_key
from app.config import settings


class ApiError(Exception):
    def __init__(self, status: int, reason: str, detail: str | None = None,
                 headers: dict[str, str] | None = None) -> None:
        super().__init__(reason)
        self.status, self.reason, self.detail, self.headers = status, reason, detail, headers or {}


class RateLimiter:
    """Sliding window: at most `limit()` hits per 60 s per key."""

    def __init__(self, limit=lambda: settings.rate_limit_per_min, window_s: float = 60.0) -> None:
        self._limit, self._window = limit, window_s
        self._hits: dict[str, deque[float]] = {}
        self._lock = threading.Lock()

    def check(self, key: str) -> tuple[bool, float, int]:
        """Return (allowed, retry_after_s, remaining)."""
        now, limit = time.monotonic(), self._limit()
        with self._lock:
            q = self._hits.setdefault(key, deque())
            while q and now - q[0] >= self._window:
                q.popleft()
            if len(q) >= limit:
                return False, max(self._window - (now - q[0]), 0.0), 0
            q.append(now)
            return True, 0.0, limit - len(q)

    def reset(self) -> None:
        with self._lock:
            self._hits.clear()


limiter = RateLimiter()


def guard(response: Response, key: str = Depends(require_api_key)) -> str:
    allowed, retry_after, remaining = limiter.check(hashlib.sha256(key.encode()).hexdigest())
    if not allowed:
        raise ApiError(429, "rate_limited", "too many requests; slow down",
                       {"Retry-After": str(int(retry_after) + 1)})
    response.headers["X-RateLimit-Remaining"] = str(remaining)
    return key


_slots: threading.BoundedSemaphore | None = None
_slots_size = 0
_slots_lock = threading.Lock()


def _semaphore() -> threading.BoundedSemaphore:
    global _slots, _slots_size
    with _slots_lock:
        if _slots is None or _slots_size != settings.max_concurrent_solves:
            _slots_size = settings.max_concurrent_solves
            _slots = threading.BoundedSemaphore(_slots_size)
        return _slots


@contextmanager
def solve_slot():
    sem = _semaphore()
    if not sem.acquire(timeout=settings.solve_queue_timeout_s):
        raise ApiError(503, "busy", "all solver slots are in use; retry shortly",
                       {"Retry-After": "1"})
    try:
        yield
    finally:
        sem.release()


def read_upload(file, max_bytes: int | None = None) -> bytes:
    """Read an UploadFile with a hard size cap (never trusts Content-Length)."""
    cap = max_bytes or settings.max_upload_bytes
    data = file.file.read(cap + 1)
    if len(data) > cap:
        raise ApiError(413, "file_too_large", f"limit is {cap} bytes")
    if not data:
        raise ApiError(400, "empty_file", "the uploaded file is empty")
    return data


def error_response_body(exc: ApiError) -> dict:
    body = {"ok": False, "reason": exc.reason}
    if exc.detail:
        body["detail"] = exc.detail
    return body


__all__ = ["ApiError", "RateLimiter", "limiter", "guard", "solve_slot", "read_upload",
           "error_response_body", "Request"]
