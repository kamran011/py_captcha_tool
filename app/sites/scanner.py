"""
Read-only provider-detection scan of the imported contact pages.

WHAT THIS DOES: for each page, one plain HTTP GET, then static analysis of the
returned HTML with app.recognition.provider_detector.

WHAT THIS NEVER DOES (assignment sections 1 and 7): fill a field, submit a form,
solve a challenge, request a token, run the page's JavaScript, or drive a
browser. A rendered visit would auto-run reCAPTCHA v3, invisible widgets and
Turnstile, i.e. request tokens from the providers; that is exactly what this
benchmark must not do. The price is that widgets injected by JavaScript are
invisible here; such pages are flagged `needs_render` and reported, not guessed.

Politeness and safety:
  * identifying User-Agent (config SCAN_USER_AGENT), no cookies, no Referer
  * robots.txt is fetched once per host and obeyed; per RFC 9309 a robots.txt
    that is unreachable (5xx / network error) means "do not fetch"
  * at most SCAN_CONCURRENCY pages at once, one request at a time per host with a
    minimum gap between requests to the same host
  * only http/https on ports 80/443; every redirect hop (max 3) is re-validated
  * hosts must resolve to public addresses only (no loopback / private /
    link-local), so the scanner cannot be aimed at internal services.
    Known limit: resolution is checked, then httpx resolves again to connect, so
    a malicious DNS server could still rebind between the two. The inputs here
    are a reviewed CSV, not arbitrary callers.
  * body capped at 2 MB (decompressed), per-site wall-clock deadline
"""
from __future__ import annotations

import asyncio
import ipaddress
import json
import logging
import socket
import ssl
import threading
import time
import urllib.robotparser
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from urllib.parse import urljoin, urlparse

import httpx

from app.config import settings
from app.models import Site
from app.recognition.provider_detector import detect
from app.sites.importer import validate_url

logger = logging.getLogger(__name__)
ROBOTS_TOKEN = "CaptchaRecognitionTool"
_HTML_TYPES = ("html", "xml", "text/plain")


@dataclass
class ScanConfig:
    concurrency: int = 5
    timeout_s: float = 20.0
    max_bytes: int = 2_000_000
    robots_max_bytes: int = 200_000
    max_redirects: int = 3
    host_delay_s: float = 1.0
    respect_robots: bool = True
    user_agent: str = ""

    def __post_init__(self) -> None:
        self.user_agent = self.user_agent or settings.scan_user_agent
        self.concurrency = max(1, min(self.concurrency, 10))

    @classmethod
    def from_settings(cls, **overrides) -> "ScanConfig":
        return cls(concurrency=settings.scan_concurrency, timeout_s=settings.scan_timeout_s, **overrides)


@dataclass
class Outcome:
    status: str                       # ok | blocked_robots | robots_unreachable | blocked_address | dns_error |
                                      # http_error | timeout | connect_error | tls_error | non_html |
                                      # redirect_error | invalid_url | request_error
    http_status: int | None = None
    final_url: str | None = None
    html: str | None = None
    ms: float = 0.0
    error: str | None = None
    truncated: bool = False


async def default_resolver(host: str) -> list[str]:
    infos = await asyncio.get_running_loop().getaddrinfo(host, None, type=socket.SOCK_STREAM)
    return sorted({i[4][0] for i in infos})


def make_client(cfg: ScanConfig) -> httpx.AsyncClient:
    return httpx.AsyncClient(
        headers={"User-Agent": cfg.user_agent,
                 "Accept": "text/html,application/xhtml+xml;q=0.9,*/*;q=0.1",
                 "Accept-Language": "en-US,en;q=0.7"},
        follow_redirects=False,
        timeout=httpx.Timeout(cfg.timeout_s, connect=min(10.0, cfg.timeout_s)),
        limits=httpx.Limits(max_connections=cfg.concurrency * 2, max_keepalive_connections=0),
    )


def classify_exception(exc: BaseException) -> str:
    if isinstance(exc, httpx.TimeoutException) or isinstance(exc, asyncio.TimeoutError):
        return "timeout"
    if isinstance(exc, httpx.ConnectError):
        cause = exc.__cause__ or exc.__context__
        while cause is not None:
            if isinstance(cause, ssl.SSLError):
                return "tls_error"
            cause = cause.__cause__ or cause.__context__
        return "connect_error"
    if isinstance(exc, (UnicodeError, httpx.InvalidURL)):      # e.g. a malformed Location header
        return "invalid_url"
    return "request_error"


class PageFetcher:
    def __init__(self, cfg: ScanConfig, client: httpx.AsyncClient, resolver=default_resolver) -> None:
        self.cfg, self.client, self.resolver = cfg, client, resolver
        self._robots: dict[str, urllib.robotparser.RobotFileParser | str] = {}
        self._robots_lock = asyncio.Lock()
        self._host_locks: dict[str, asyncio.Lock] = {}
        self._last_hit: dict[str, float] = {}

    # ── address safety ──────────────────────────────────────────────────────
    async def _host_status(self, host: str) -> str:
        try:
            ips = [host] if _is_ip_literal(host) else await self.resolver(host)
        except (OSError, UnicodeError):
            return "dns_error"
        if not ips:
            return "dns_error"
        try:
            return "ok" if all(ipaddress.ip_address(ip).is_global for ip in ips) else "blocked_address"
        except ValueError:
            return "blocked_address"

    # ── one polite request ──────────────────────────────────────────────────
    async def _get(self, url: str, max_bytes: int) -> tuple[httpx.Response, bytes, bool]:
        host = urlparse(url).hostname or ""
        lock = self._host_locks.setdefault(host, asyncio.Lock())
        async with lock:                                   # one request at a time per host
            gap = self._last_hit.get(host, 0.0) + self.cfg.host_delay_s - time.monotonic()
            if gap > 0:
                await asyncio.sleep(gap)
            try:
                async with self.client.stream("GET", url) as resp:
                    chunks, size, truncated = [], 0, False
                    redirect = 300 <= resp.status_code < 400
                    ctype = resp.headers.get("content-type", "").lower()
                    if not redirect and resp.status_code == 200 and (not ctype or any(t in ctype for t in _HTML_TYPES)):
                        async for chunk in resp.aiter_bytes():
                            size += len(chunk)
                            if size > max_bytes:
                                chunks.append(chunk[: len(chunk) - (size - max_bytes)])
                                truncated = True
                                break
                            chunks.append(chunk)
                    body = b"".join(chunks)
            finally:
                self._last_hit[host] = time.monotonic()
                self.client.cookies.clear()                # no cookie ever persists
        return resp, body, truncated

    # ── robots.txt ──────────────────────────────────────────────────────────
    async def _robots_allows(self, url: str) -> str:
        """'allowed' | 'blocked' | 'unreachable'"""
        u = urlparse(url)
        origin = f"{u.scheme}://{u.netloc}"
        async with self._robots_lock:
            entry = self._robots.get(origin)
            if entry is None:
                entry = await self._load_robots(origin)
                self._robots[origin] = entry
        if isinstance(entry, str):
            return entry
        path = u.path or "/"
        return "allowed" if entry.can_fetch(ROBOTS_TOKEN, path + (f"?{u.query}" if u.query else "")) else "blocked"

    async def _load_robots(self, origin: str):
        url = origin + "/robots.txt"
        try:
            for _ in range(self.cfg.max_redirects + 1):
                if validate_url(url) or await self._host_status(urlparse(url).hostname or "") != "ok":
                    return "allowed"                       # cannot fetch it safely: treat as absent
                resp, body, _ = await self._get(url, self.cfg.robots_max_bytes)
                if 300 <= resp.status_code < 400 and resp.headers.get("location"):
                    url = urljoin(url, resp.headers["location"])
                    continue
                if resp.status_code >= 500:
                    return "unreachable"                   # RFC 9309: server error -> assume disallow
                if resp.status_code != 200:
                    return "allowed"                       # 4xx: no robots.txt
                rp = urllib.robotparser.RobotFileParser()
                rp.parse(body.decode("utf-8", errors="replace").splitlines())
                return rp
            return "allowed"
        except Exception as exc:  # noqa: BLE001
            logger.info("robots.txt unreachable for %s: %s", origin, classify_exception(exc))
            return "unreachable"

    # ── public API ──────────────────────────────────────────────────────────
    async def fetch(self, url: str) -> Outcome:
        t0 = time.perf_counter()
        try:
            out = await asyncio.wait_for(self._fetch(url), timeout=self.cfg.timeout_s * 2)
        except Exception as exc:  # noqa: BLE001 - never let one site stop the scan
            out = Outcome(classify_exception(exc), error=type(exc).__name__)
        out.ms = round((time.perf_counter() - t0) * 1000, 1)
        return out

    async def _fetch(self, url: str) -> Outcome:
        for hop in range(self.cfg.max_redirects + 1):
            bad = validate_url(url)
            if bad:
                return Outcome("invalid_url", final_url=url[:2048], error=bad)
            host_state = await self._host_status(urlparse(url).hostname or "")
            if host_state != "ok":
                return Outcome(host_state, final_url=url[:2048], error=host_state)
            if self.cfg.respect_robots:
                verdict = await self._robots_allows(url)
                if verdict != "allowed":
                    return Outcome("blocked_robots" if verdict == "blocked" else "robots_unreachable",
                                   final_url=url[:2048], error=f"robots.txt: {verdict}")
            resp, body, truncated = await self._get(url, self.cfg.max_bytes)
            code = resp.status_code
            if 300 <= code < 400 and resp.headers.get("location"):
                url = urljoin(url, resp.headers["location"])
                continue
            if code != 200:
                return Outcome("http_error", code, url[:2048], error=f"http_{code}")
            ctype = resp.headers.get("content-type", "").lower()
            if ctype and not any(t in ctype for t in _HTML_TYPES):
                return Outcome("non_html", code, url[:2048], error="content_type")
            try:
                html = body.decode(resp.encoding or "utf-8", errors="replace")
            except LookupError:
                html = body.decode("utf-8", errors="replace")
            return Outcome("ok", code, url[:2048], html, truncated=truncated)
        return Outcome("redirect_error", final_url=url[:2048], error="too_many_redirects")


def _is_ip_literal(host: str) -> bool:
    try:
        ipaddress.ip_address(host.strip("[]"))
        return True
    except ValueError:
        return False


# ── persisting results ───────────────────────────────────────────────────────
def apply_outcome(site: Site, out: Outcome) -> None:
    site.scanned_at = datetime.now(timezone.utc)
    site.scan_status = out.status
    site.scan_error = out.error
    site.scan_http_status = out.http_status
    site.final_url = out.final_url
    site.scan_ms = out.ms
    site.detected_provider = site.detected_variant = site.detected_sitekey = None
    site.detected_evidence = None
    site.detected_confidence = None
    site.needs_render = None
    if out.status == "ok" and out.html is not None:
        det = detect(out.html)
        site.detected_provider = det.provider
        site.detected_variant = det.variant
        site.detected_sitekey = det.sitekey
        site.detected_evidence = json.dumps(det.evidence[:8])
        site.detected_confidence = det.confidence
        site.needs_render = det.needs_render


# ── background job ───────────────────────────────────────────────────────────
class ScanBusy(RuntimeError):
    pass


@dataclass
class ScanJob:
    id: str
    total: int
    done: int = 0
    status: str = "running"            # running | finished | cancelled | error
    started_at: float = field(default_factory=time.time)
    finished_at: float | None = None
    by_status: dict[str, int] = field(default_factory=dict)
    error: str | None = None
    cancel: threading.Event = field(default_factory=threading.Event, repr=False)

    def as_dict(self) -> dict:
        end = self.finished_at or time.time()
        return {"id": self.id, "status": self.status, "total": self.total, "done": self.done,
                "by_status": dict(self.by_status), "elapsed_s": round(end - self.started_at, 1),
                "error": self.error}


class ScanManager:
    """One scan at a time, run on a background thread with its own event loop."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._job: ScanJob | None = None

    def current(self) -> ScanJob | None:
        return self._job

    def cancel(self) -> bool:
        job = self._job
        if job and job.status == "running":
            job.cancel.set()
            return True
        return False

    def start(self, session_factory, pairs: list[tuple[int, str]], cfg: ScanConfig | None = None,
              client_factory=make_client, resolver=default_resolver) -> ScanJob:
        cfg = cfg or ScanConfig.from_settings()
        with self._lock:
            if self._job and self._job.status == "running":
                raise ScanBusy("a scan is already running")
            job = self._job = ScanJob(id=uuid.uuid4().hex[:12], total=len(pairs))
        threading.Thread(target=self._thread, args=(job, session_factory, pairs, cfg, client_factory, resolver),
                         name=f"scan-{job.id}", daemon=True).start()
        return job

    def _thread(self, job, session_factory, pairs, cfg, client_factory, resolver) -> None:
        try:
            asyncio.run(self._run(job, session_factory, pairs, cfg, client_factory, resolver))
            job.status = "cancelled" if job.cancel.is_set() else "finished"
        except Exception as exc:  # noqa: BLE001
            logger.exception("scan job failed")
            job.status, job.error = "error", type(exc).__name__
        finally:
            job.finished_at = time.time()

    async def _run(self, job, session_factory, pairs, cfg, client_factory, resolver) -> None:
        client = client_factory(cfg)
        fetcher = PageFetcher(cfg, client, resolver)
        sem = asyncio.Semaphore(cfg.concurrency)

        async def one(site_id: int, url: str) -> None:
            if job.cancel.is_set():
                return
            async with sem:
                if job.cancel.is_set():
                    return
                out = await fetcher.fetch(url)
            with session_factory() as db:
                site = db.get(Site, site_id)
                if site is not None:
                    apply_outcome(site, out)
                    db.commit()
            job.by_status[out.status] = job.by_status.get(out.status, 0) + 1
            job.done += 1

        try:
            await asyncio.gather(*(one(i, u) for i, u in pairs))
        finally:
            await client.aclose()


manager = ScanManager()
