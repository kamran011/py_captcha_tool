"""M6 tests: CSV import, the read-only scanner, the accuracy report and the sites API (no network)."""
from __future__ import annotations

import asyncio
import json
import socket
import time
from pathlib import Path

import httpx
import pytest
from fastapi.testclient import TestClient
from sqlalchemy.orm import sessionmaker

from app import db as app_db
from app.api import limits
from app.api import sites as sites_api
from app.config import settings
from app.main import app
from app.models import Base, Site
from app.sites import scanner
from app.sites.importer import CsvError, import_sites, parse_csv, validate_url
from app.sites.report import build_report, detected_label
from app.sites.scanner import Outcome, PageFetcher, ScanBusy, ScanConfig, ScanManager, apply_outcome

FIX = Path("tests/fixtures/detector")
KEY = {"X-API-Key": settings.api_key.get_secret_value()}
HEADER = "Company,Website,City,State,Category,Platform,Captcha,Contact Page\n"
PUBLIC = "93.184.216.34"
PROVIDER_HOSTS = ("google.com", "recaptcha.net", "hcaptcha.com", "challenges.cloudflare.com")


def fx(name: str) -> str:
    return (FIX / f"{name}.html").read_text(encoding="utf-8")


# ── CSV import ───────────────────────────────────────────────────────────────
class TestImport:
    def test_basic_and_header_case_and_bom(self):
        csv = "﻿" + "company,CAPTCHA,contact page\nAcme,reCAPTCHA,https://acme.com/contact\n"
        rep = parse_csv(csv.encode("utf-8"))
        assert rep.errors == [] and rep.rows[0]["contact_page"] == "https://acme.com/contact"
        assert rep.rows[0]["labelled_captcha"] == "reCAPTCHA"

    def test_bundled_csv_is_clean(self):
        rep = parse_csv(Path("data/sites.csv").read_bytes())
        assert len(rep.rows) == 200 and rep.errors == [] and rep.warnings == []

    @pytest.mark.parametrize("url,reason", [
        ("", "empty_url"), ("javascript:alert(1)", "scheme_not_http"), ("file:///etc/passwd", "scheme_not_http"),
        ("ftp://x.example.com/", "scheme_not_http"), ("https://localhost/x", "no_hostname"),
        ("https://user:pw@example.com/", "credentials_in_url"), ("https://example.com:8080/", "unusual_port"),
        ("https://exa mple.com/", "invalid_characters"), ("https://example.com/<script>", "invalid_characters"),
        ("https://" + "a" * 2050 + ".com", "url_too_long"), ("example.com/contact", "scheme_not_http"),
    ])
    def test_unsafe_or_bad_urls_rejected_with_reason(self, url, reason):
        assert validate_url(url) == reason

    def test_bad_rows_are_reported_not_fatal(self):
        csv = HEADER + (
            "A,,,,,,reCAPTCHA,https://a.example.com/c\n"
            "B,,,,,,none,javascript:alert(1)\n"
            "C,,,,,,none,\n"
            "D,,,,,,none,https://a.example.com/c\n"          # duplicate of the first
            ",,,,,,,\n"                                         # blank line
            "E,,,,,,Weird Label,https://e.example.com/c\n")
        rep = parse_csv(csv)
        assert [r["company"] for r in rep.rows] == ["A", "E"]
        assert {(e["line"], e["reason"]) for e in rep.errors} == {
            (3, "scheme_not_http"), (4, "empty_url"), (5, "duplicate_contact_page")}
        assert rep.warnings[0]["reason"].startswith("unknown_label")

    def test_city_repair_whitespace_and_extra_columns(self):
        csv = "Company,City,Notes,Contact Page\n  Acme   Inc ,Sioux 20Falls,ignored,https://acme.com/c\r\n"
        row = parse_csv(csv).rows[0]
        assert row["company"] == "Acme Inc" and row["city"] == "Sioux Falls" and "notes" not in row

    def test_quoted_commas(self):
        row = parse_csv(HEADER + '"Smith, Jones & Co",,,,,,none,https://s.example.com/c\n').rows[0]
        assert row["company"] == "Smith, Jones & Co"

    @pytest.mark.parametrize("content,msg", [("", "empty"), ("Company,City\nA,B\n", "Contact Page")])
    def test_structural_errors(self, content, msg):
        with pytest.raises(CsvError, match=msg):
            parse_csv(content)

    def test_size_and_row_limits(self, monkeypatch):
        with pytest.raises(CsvError, match="too large"):
            parse_csv(b"x" * 2_000_001)
        monkeypatch.setattr("app.sites.importer.MAX_ROWS", 3)
        rows = "".join(f"c{i},,,,,,none,https://s{i}.example.com/c\n" for i in range(5))
        with pytest.raises(CsvError, match="too many rows"):
            parse_csv(HEADER + rows)

    def test_import_is_idempotent_and_keeps_scan_results(self, Session):
        rows = parse_csv(HEADER + "A,,,,,,reCAPTCHA,https://a.example.com/c\n").rows
        with Session() as db:
            assert import_sites(db, rows) == {"inserted": 1, "updated": 0, "unchanged": 0}
            site = db.query(Site).one()
            site.scan_status, site.detected_provider = "ok", "recaptcha"
            db.commit()
            assert import_sites(db, rows) == {"inserted": 0, "updated": 0, "unchanged": 1}
            rows[0]["labelled_captcha"] = "hCaptcha"
            assert import_sites(db, rows)["updated"] == 1
            site = db.query(Site).one()
            assert site.labelled_captcha == "hCaptcha" and site.scan_status == "ok"      # scan kept


# ── scanner: fetcher ─────────────────────────────────────────────────────────
class Net:
    """Mock internet: records every request; routes by (host, path)."""

    def __init__(self, routes=None, default_robots=404):
        self.routes = routes or {}
        self.log: list[httpx.Request] = []
        self.default_robots = default_robots

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.log.append(request)
        key = (request.url.host, request.url.path)
        if key in self.routes:
            r = self.routes[key]
            if isinstance(r, Exception):
                raise r
            return r(request) if callable(r) else r
        if request.url.path == "/robots.txt":
            return httpx.Response(self.default_robots)
        return httpx.Response(404)

    @property
    def methods(self):
        return {r.method for r in self.log}

    def paths(self, host=None):
        return [r.url.path for r in self.log if host in (None, r.url.host)]


def html(body: str, ctype="text/html; charset=utf-8", **kw) -> httpx.Response:
    return httpx.Response(200, content=body.encode("utf-8"), headers={"content-type": ctype}, **kw)


async def stub_resolver(host: str):
    return {"internal.example": ["10.0.0.5"], "meta.example": ["169.254.169.254"],
            "mixed.example": [PUBLIC, "10.0.0.5"], "loop.example": ["127.0.0.1"]}.get(host, [PUBLIC])


def fetch(net: Net, url: str, resolver=stub_resolver, **cfgkw) -> Outcome:
    cfgkw.setdefault("host_delay_s", 0.0)
    cfg = ScanConfig(**cfgkw)

    async def go():
        async with httpx.AsyncClient(transport=httpx.MockTransport(net), follow_redirects=False,
                                     headers={"User-Agent": cfg.user_agent}) as client:
            return await PageFetcher(cfg, client, resolver).fetch(url)

    return asyncio.run(go())


class TestFetcher:
    def test_happy_path_exact_requests(self):
        net = Net({("shop.example", "/contact"): html(fx("recaptcha_v2_checkbox"))})
        out = fetch(net, "https://shop.example/contact")
        assert out.status == "ok" and out.http_status == 200 and "g-recaptcha" in out.html
        assert net.paths() == ["/robots.txt", "/contact"]                 # robots, then ONE page GET
        assert net.methods == {"GET"}

    def test_requests_are_plain_and_identify_the_tool(self):
        net = Net({("shop.example", "/contact"): html("<p>x</p>")})
        fetch(net, "https://shop.example/contact")
        for r in net.log:
            assert r.method == "GET" and r.content == b""
            assert "cookie" not in r.headers and "referer" not in r.headers and "authorization" not in r.headers
            assert "CaptchaRecognitionTool" in r.headers["user-agent"] and "no form interaction" in r.headers["user-agent"]

    def test_never_contacts_provider_endpoints_even_when_the_page_embeds_them(self):
        net = Net({("shop.example", "/contact"): html(fx("multi_hcaptcha_widget_plus_global_recaptcha_v3"))})
        out = fetch(net, "https://shop.example/contact")
        assert out.status == "ok"
        assert not [r for r in net.log if any(r.url.host.endswith(h) for h in PROVIDER_HOSTS)]

    def test_cookies_never_persist_between_requests(self):
        net = Net({("shop.example", "/a"): httpx.Response(
                       200, content=b"<p/>", headers={"content-type": "text/html", "set-cookie": "sid=abc; Path=/"}),
                   ("shop.example", "/b"): html("<p/>")})
        fetch(net, "https://shop.example/a")
        fetch(net, "https://shop.example/b")
        assert all("cookie" not in r.headers for r in net.log)

    # robots.txt
    def test_robots_disallow_blocks_the_page_request(self):
        net = Net({("shop.example", "/robots.txt"): httpx.Response(200, text="User-agent: *\nDisallow: /contact\n"),
                   ("shop.example", "/contact"): html("<p/>")})
        out = fetch(net, "https://shop.example/contact")
        assert out.status == "blocked_robots" and net.paths() == ["/robots.txt"]

    def test_robots_for_our_agent_and_unrelated_rules(self):
        robots = "User-agent: CaptchaRecognitionTool\nDisallow: /\n\nUser-agent: *\nAllow: /\n"
        net = Net({("shop.example", "/robots.txt"): httpx.Response(200, text=robots)})
        assert fetch(net, "https://shop.example/contact").status == "blocked_robots"
        other = Net({("shop.example", "/robots.txt"): httpx.Response(200, text="User-agent: *\nDisallow: /private\n"),
                     ("shop.example", "/contact"): html("<p/>")})
        assert fetch(other, "https://shop.example/contact").status == "ok"

    @pytest.mark.parametrize("robots,want", [
        (httpx.Response(404), "ok"), (httpx.Response(403), "ok"), (httpx.Response(410), "ok"),
        (httpx.Response(500), "robots_unreachable"), (httpx.Response(503), "robots_unreachable"),
        (httpx.ConnectTimeout("t"), "robots_unreachable"), (httpx.ReadTimeout("t"), "robots_unreachable"),
    ])
    def test_robots_availability_rules(self, robots, want):
        net = Net({("shop.example", "/robots.txt"): robots, ("shop.example", "/contact"): html("<p/>")})
        out = fetch(net, "https://shop.example/contact")
        assert out.status == want
        if want != "ok":
            assert "/contact" not in net.paths()             # unreachable robots means: do not fetch

    def test_robots_cached_per_host(self):
        net = Net({("shop.example", "/a"): html("<p/>"), ("shop.example", "/b"): html("<p/>")})
        cfg = ScanConfig(host_delay_s=0)

        async def go():
            async with httpx.AsyncClient(transport=httpx.MockTransport(net)) as c:
                f = PageFetcher(cfg, c, stub_resolver)
                await f.fetch("https://shop.example/a")
                await f.fetch("https://shop.example/b")

        asyncio.run(go())
        assert net.paths().count("/robots.txt") == 1

    def test_robots_can_be_turned_off(self):
        net = Net({("shop.example", "/contact"): html("<p/>")})
        fetch(net, "https://shop.example/contact", respect_robots=False)
        assert net.paths() == ["/contact"]

    # address safety
    @pytest.mark.parametrize("url", [
        "https://internal.example/contact", "https://meta.example/latest", "https://mixed.example/c",
        "https://loop.example/c", "http://127.0.0.1/admin", "http://169.254.169.254/latest/meta-data/",
        "http://10.1.2.3/x", "http://192.168.0.1/x", "http://172.16.5.5/x",
    ])
    def test_private_loopback_and_metadata_addresses_blocked_before_any_request(self, url):
        net = Net()
        out = fetch(net, url)
        assert out.status == "blocked_address" and net.log == []

    def test_dns_failure(self):
        async def failing(host):
            raise socket.gaierror("nope")

        out = fetch(Net(), "https://nonexistent.example/c", resolver=failing)
        assert out.status == "dns_error"

    @pytest.mark.parametrize("url", [
        "javascript:alert(1)", "file:///etc/passwd", "ftp://x.example.com/", "https://u:p@x.example.com/",
        "https://x.example.com:8080/", "not a url", "https://x.example.com/" + "a" * 2100,
    ])
    def test_invalid_urls_never_reach_the_network(self, url):
        net = Net()
        assert fetch(net, url).status == "invalid_url" and net.log == []

    # redirects
    def test_redirect_to_internal_host_is_blocked_and_not_followed(self):
        net = Net({("shop.example", "/c"): httpx.Response(302, headers={"location": "http://internal.example/secret"}),
                   ("internal.example", "/secret"): html("TOP SECRET")})
        out = fetch(net, "https://shop.example/c")
        assert out.status == "blocked_address"
        assert "internal.example" not in {r.url.host for r in net.log}

    def test_redirect_followed_and_robots_checked_for_the_new_host(self):
        net = Net({("shop.example", "/c"): httpx.Response(301, headers={"location": "https://www.shop.example/contact"}),
                   ("www.shop.example", "/robots.txt"): httpx.Response(200, text="User-agent: *\nDisallow: /contact\n"),
                   ("www.shop.example", "/contact"): html("<p/>")})
        out = fetch(net, "https://shop.example/c")
        assert out.status == "blocked_robots" and "/contact" not in net.paths("www.shop.example")

    def test_relative_redirect_and_final_url(self):
        net = Net({("shop.example", "/c"): httpx.Response(302, headers={"location": "/contact-us/"}),
                   ("shop.example", "/contact-us/"): html(fx("hcaptcha_checkbox"))})
        out = fetch(net, "https://shop.example/c")
        assert out.status == "ok" and out.final_url == "https://shop.example/contact-us/"

    def test_redirect_loop_and_bad_schemes(self):
        loop = Net({("shop.example", "/c"): httpx.Response(302, headers={"location": "/c"})})
        assert fetch(loop, "https://shop.example/c").status == "redirect_error"
        for loc in ("file:///etc/passwd", "ftp://x.example.com/", "javascript:alert(1)", "https://x.example.com:8080/"):
            n = Net({("shop.example", "/c"): httpx.Response(302, headers={"location": loc})})
            assert fetch(n, "https://shop.example/c").status == "invalid_url"

    # responses
    @pytest.mark.parametrize("code", [403, 404, 429, 500, 503])
    def test_error_pages_are_not_analysed(self, code):
        net = Net({("shop.example", "/c"): httpx.Response(code, content=fx("recaptcha_v2_checkbox").encode(),
                                                          headers={"content-type": "text/html"})})
        out = fetch(net, "https://shop.example/c")
        site = Site(contact_page="x")
        apply_outcome(site, out)
        assert out.status == "http_error" and out.http_status == code
        assert site.detected_provider is None and site.scan_status == "http_error"   # block pages never count

    def test_non_html_rejected(self):
        net = Net({("shop.example", "/c"): httpx.Response(200, content=b"%PDF-1.4", headers={"content-type": "application/pdf"})})
        assert fetch(net, "https://shop.example/c").status == "non_html"

    def test_body_is_capped_but_early_markup_is_still_detected(self):
        body = fx("turnstile_widget") + ("<!-- pad -->" * 5000)
        net = Net({("shop.example", "/c"): html(body)})
        out = fetch(net, "https://shop.example/c", max_bytes=2000)
        assert out.status == "ok" and out.truncated and len(out.html) <= 2000
        site = Site(contact_page="x")
        apply_outcome(site, out)
        assert site.detected_provider == "turnstile"

    def test_gzip_bomb_is_capped_after_decompression(self):
        import gzip
        bomb = gzip.compress(b"A" * 50_000_000)
        net = Net({("shop.example", "/c"): httpx.Response(200, content=bomb, headers={
            "content-type": "text/html", "content-encoding": "gzip"})})
        out = fetch(net, "https://shop.example/c", max_bytes=100_000)
        assert out.status == "ok" and out.truncated and len(out.html) <= 100_000

    def test_bad_charset_does_not_crash(self):
        net = Net({("shop.example", "/c"): httpx.Response(200, content="caf\xe9".encode("latin-1"), headers={
            "content-type": "text/html; charset=totally-bogus"})})
        assert fetch(net, "https://shop.example/c").status == "ok"

    @pytest.mark.parametrize("exc,want", [
        (httpx.ConnectTimeout("t"), "timeout"), (httpx.ReadTimeout("t"), "timeout"),
        (httpx.ConnectError("refused"), "connect_error"), (httpx.RemoteProtocolError("bad"), "request_error"),
    ])
    def test_network_failures_are_classified(self, exc, want):
        net = Net({("shop.example", "/c"): exc})
        assert fetch(net, "https://shop.example/c").status == want

    def test_tls_errors_are_recognised(self):
        import ssl
        err = httpx.ConnectError("tls")
        err.__cause__ = ssl.SSLCertVerificationError("bad cert")
        assert fetch(Net({("shop.example", "/c"): err}), "https://shop.example/c").status == "tls_error"

    def test_total_deadline_stops_a_slow_site(self):
        async def slow(request):
            await asyncio.sleep(2)
            return html("<p/>")

        cfg = ScanConfig(timeout_s=0.1, host_delay_s=0, respect_robots=False)

        async def go():
            async with httpx.AsyncClient(transport=httpx.MockTransport(slow)) as c:
                return await PageFetcher(cfg, c, stub_resolver).fetch("https://shop.example/c")

        t0 = time.perf_counter()
        assert asyncio.run(go()).status == "timeout" and time.perf_counter() - t0 < 1.5

    def test_same_host_requests_are_spaced(self):
        stamps = []
        net = Net({("shop.example", "/a"): lambda r: (stamps.append(time.perf_counter()) or html("<p/>")),
                   ("shop.example", "/b"): lambda r: (stamps.append(time.perf_counter()) or html("<p/>"))})
        cfg = ScanConfig(host_delay_s=0.3, respect_robots=False)

        async def go():
            async with httpx.AsyncClient(transport=httpx.MockTransport(net)) as c:
                f = PageFetcher(cfg, c, stub_resolver)
                await asyncio.gather(f.fetch("https://shop.example/a"), f.fetch("https://shop.example/b"))

        asyncio.run(go())
        assert len(stamps) == 2 and abs(stamps[1] - stamps[0]) >= 0.28


# ── scanner: background job ──────────────────────────────────────────────────
@pytest.fixture
def Session(tmp_path):
    original = settings.database_url
    settings.database_url = f"sqlite:///{tmp_path / 's.db'}"
    try:
        eng = app_db._make_engine()
    finally:
        settings.database_url = original
    Base.metadata.create_all(eng)
    return sessionmaker(bind=eng)


def seed_sites(Session, n, label="reCAPTCHA", host="site{i}.example"):
    with Session() as db:
        for i in range(n):
            db.add(Site(company=f"Co{i}", contact_page=f"https://{host.format(i=i)}/contact", labelled_captcha=label))
        db.commit()
        return [(s.id, s.contact_page) for s in db.query(Site).order_by(Site.id)]


def wait_done(manager, timeout=10):
    t0 = time.time()
    while manager.current().status == "running" and time.time() - t0 < timeout:
        time.sleep(0.02)
    return manager.current()


def runtime(net, **cfg):
    cfg.setdefault("host_delay_s", 0.0)
    return dict(cfg=ScanConfig(**cfg), resolver=stub_resolver,
                client_factory=lambda c: httpx.AsyncClient(transport=httpx.MockTransport(net), follow_redirects=False))


class TestJob:
    def test_scan_updates_rows_and_counts_statuses(self, Session):
        pairs = seed_sites(Session, 4)
        net = Net({("site0.example", "/contact"): html(fx("recaptcha_v2_checkbox")),
                   ("site1.example", "/contact"): html(fx("hcaptcha_checkbox")),
                   ("site2.example", "/contact"): httpx.Response(403),
                   ("site3.example", "/contact"): httpx.ConnectTimeout("t")})
        m = ScanManager()
        m.start(Session, pairs, **runtime(net))
        job = wait_done(m)
        assert job.status == "finished" and job.done == 4
        assert job.by_status == {"ok": 2, "http_error": 1, "timeout": 1}
        with Session() as db:
            rows = {s.company: s for s in db.query(Site)}
        assert (rows["Co0"].detected_provider, rows["Co0"].scan_status) == ("recaptcha", "ok")
        assert rows["Co1"].detected_provider == "hcaptcha" and rows["Co1"].scan_ms is not None
        assert rows["Co2"].detected_provider is None and rows["Co3"].scan_status == "timeout"
        assert all(r.scanned_at is not None for r in rows.values())

    def test_whole_job_is_get_only_and_never_touches_providers(self, Session):
        pairs = seed_sites(Session, 6)
        net = Net({(f"site{i}.example", "/contact"): html(fx("recaptcha_v2_checkbox")) for i in range(6)})
        m = ScanManager()
        m.start(Session, pairs, **runtime(net))
        wait_done(m)
        assert net.methods == {"GET"} and all(r.content == b"" for r in net.log)
        assert not [r for r in net.log if any(r.url.host.endswith(h) for h in PROVIDER_HOSTS)]
        assert len(net.log) == 12                                          # 6 robots + 6 pages, nothing else

    def test_concurrency_is_capped(self, Session):
        pairs = seed_sites(Session, 12)
        live = peak = 0

        async def handler(request):
            nonlocal live, peak
            if request.url.path == "/robots.txt":
                return httpx.Response(404)
            live += 1
            peak = max(peak, live)
            await asyncio.sleep(0.05)
            live -= 1
            return html("<p/>")

        rt = dict(cfg=ScanConfig(concurrency=3, host_delay_s=0), resolver=stub_resolver,
                  client_factory=lambda c: httpx.AsyncClient(transport=httpx.MockTransport(handler)))
        m = ScanManager()
        m.start(Session, pairs, **rt)
        assert wait_done(m).done == 12 and 1 < peak <= 3

    def test_one_failing_site_does_not_stop_the_scan(self, Session):
        pairs = seed_sites(Session, 3)
        net = Net({("site0.example", "/contact"): RuntimeError("boom")})
        m = ScanManager()
        m.start(Session, pairs, **runtime(net))
        job = wait_done(m)
        assert job.status == "finished" and job.done == 3 and job.by_status.get("request_error") == 1

    def test_only_one_scan_at_a_time_and_cancel(self, Session):
        pairs = seed_sites(Session, 8)

        async def slow(request):
            await asyncio.sleep(0.15)
            return html("<p/>") if request.url.path != "/robots.txt" else httpx.Response(404)

        rt = dict(cfg=ScanConfig(concurrency=1, host_delay_s=0), resolver=stub_resolver,
                  client_factory=lambda c: httpx.AsyncClient(transport=httpx.MockTransport(slow)))
        m = ScanManager()
        m.start(Session, pairs, **rt)
        with pytest.raises(ScanBusy):
            m.start(Session, pairs, **rt)
        time.sleep(0.4)
        assert m.cancel() is True
        job = wait_done(m)
        assert job.status == "cancelled" and 0 < job.done < 8
        assert m.cancel() is False                                         # nothing running now


# ── report ───────────────────────────────────────────────────────────────────
def add(db, label, provider, variant=None, status="ok", needs_render=False, n=1):
    for _ in range(n):
        db.add(Site(contact_page=f"https://x{db.query(Site).count()}.example.com/c", company="c",
                    labelled_captcha=label, scan_status=status, detected_provider=provider,
                    detected_variant=variant, needs_render=needs_render))
    db.commit()


class TestReport:
    def test_empty(self, Session):
        with Session() as db:
            r = build_report(db)
        assert r["total_sites"] == 0 and r["exact_accuracy"] is None and r["coverage"] is None

    def test_matrix_accuracies_and_coverage(self, Session):
        with Session() as db:
            add(db, "reCAPTCHA", "recaptcha", "v2_checkbox", n=5)                 # correct
            add(db, "reCAPTCHA", "none", needs_render=True, n=2)                  # missed (JS injected)
            add(db, "invisible reCAPTCHA", "recaptcha", "v2_invisible", n=2)      # correct exact
            add(db, "invisible reCAPTCHA", "recaptcha", "v2_checkbox", n=1)       # exact miss, family ok
            add(db, "hCaptcha", "hcaptcha", "checkbox", n=1)
            add(db, "none", "none", n=3)
            add(db, "none", "turnstile", "widget", n=1)                           # false positive
            add(db, "Turnstile", "turnstile", "widget", n=1)
            add(db, "reCAPTCHA", None, status="timeout", n=4)                     # not in accuracy
            add(db, "none", None, status=None, n=2)                               # unscanned
            r = build_report(db)
        assert r["total_sites"] == 22 and r["scanned"] == 16 and r["unscanned"] == 2
        assert r["scan_status"] == {"ok": 16, "timeout": 4} and r["coverage"] == round(16 / 22, 4)
        assert r["exact_accuracy"] == round(12 / 16, 4)             # 5+2+1+3+1 correct
        assert r["family_accuracy"] == round(13 / 16, 4)            # + the invisible/checkbox confusion
        m = r["confusion_matrix"]
        assert m["recaptcha"]["recaptcha"] == 5 and m["recaptcha"]["none"] == 2
        assert m["recaptcha_invisible"]["recaptcha_invisible"] == 2 and m["recaptcha_invisible"]["recaptcha"] == 1
        assert m["none"]["turnstile"] == 1
        assert r["js_injected_suspected"] == 2
        assert r["per_class_exact"]["turnstile"] == {"precision": 0.5, "recall": 1.0, "f1": 0.6667, "support": 1}
        assert r["per_class_family"]["recaptcha"]["recall"] == round(8 / 10, 4)
        assert r["baseline_majority_class"] == {"label": "recaptcha", "accuracy": round(7 / 16, 4)}
        assert {m_["label"] for m_ in r["mismatches"]} >= {"recaptcha", "recaptcha_invisible", "none"}
        assert next(m_ for m_ in r["mismatches"] if m_["label"] == "recaptcha_invisible")["family_ok"] is True

    def test_v3_as_invisible_view_is_separate_and_does_not_change_the_headline(self, Session):
        with Session() as db:
            add(db, "invisible reCAPTCHA", "recaptcha", "v3", n=3)
            add(db, "invisible reCAPTCHA", "recaptcha", "v2_invisible", n=1)
            add(db, "reCAPTCHA", "recaptcha", "v2_checkbox", n=2)
            r = build_report(db)
        assert r["exact_accuracy"] == round(3 / 6, 4)                    # strict: v3 is not v2_invisible
        assert r["exact_accuracy_v3_as_invisible"] == round(6 / 6, 4)
        assert r["family_accuracy"] == 1.0

    def test_detected_label_mapping(self):
        s = Site(contact_page="x")
        for prov, var, want in ((None, None, "none"), ("none", None, "none"), ("recaptcha", "v3", "recaptcha"),
                                ("recaptcha", "v2_invisible", "recaptcha_invisible"), ("hcaptcha", "invisible", "hcaptcha")):
            s.detected_provider, s.detected_variant = prov, var
            assert detected_label(s) == want


# ── API ──────────────────────────────────────────────────────────────────────
@pytest.fixture
def client(Session):
    net = Net({("site0.example", "/contact"): html(fx("recaptcha_v2_checkbox")),
               ("site1.example", "/contact"): html(fx("turnstile_widget")),
               ("site2.example", "/contact"): html(fx("none_plain_form"))})
    mgr = ScanManager()

    def db_override():
        with Session() as db:
            yield db

    limits.limiter.reset()
    app.dependency_overrides[app_db.get_db] = db_override
    app.dependency_overrides[sites_api.get_manager] = lambda: mgr
    app.dependency_overrides[sites_api.get_session_factory] = lambda: Session
    app.dependency_overrides[sites_api.get_scan_runtime] = lambda: runtime(net)
    c = TestClient(app)
    c.net, c.mgr = net, mgr
    yield c
    app.dependency_overrides.clear()
    limits.limiter.reset()


def upload(client, csv: str, name="s.csv"):
    return client.post("/api/v1/sites/import", headers=KEY, files={"file": (name, csv.encode(), "text/csv")})


class TestApi:
    def test_requires_key(self, client):
        for method, path in (("get", "/api/v1/sites"), ("post", "/api/v1/sites/import"), ("post", "/api/v1/sites/scan"),
                             ("get", "/api/v1/sites/scan"), ("get", "/api/v1/sites/report")):
            assert getattr(client, method)(path).status_code == 401

    def test_import_upload_bundled_and_errors(self, client):
        r = upload(client, HEADER + "A,,,,,,reCAPTCHA,https://site0.example/contact\nB,,,,,,none,javascript:x\n")
        assert r.status_code == 200 and (r.json()["inserted"], r.json()["rows_rejected"]) == (1, 1)
        assert r.json()["errors"][0]["reason"] == "scheme_not_http"
        r = client.post("/api/v1/sites/import", headers=KEY, data={"use_bundled": "true"})
        assert r.status_code == 200 and r.json()["rows_valid"] == 200
        again = client.post("/api/v1/sites/import", headers=KEY, data={"use_bundled": "true"}).json()
        assert again["inserted"] == 0 and again["unchanged"] == 200
        assert client.post("/api/v1/sites/import", headers=KEY).json()["reason"] == "no_input"
        assert upload(client, "Company,City\n").json()["reason"] == "invalid_csv"

    def test_import_size_limit(self, client):
        r = upload(client, "x" * 2_000_100)
        assert r.status_code == 413

    def test_list_filters_and_paging(self, client):
        upload(client, HEADER + "".join(f"C{i},,,,,,{'none' if i % 2 else 'reCAPTCHA'},https://site{i}.example/contact\n" for i in range(6)))
        j = client.get("/api/v1/sites?limit=2&offset=1", headers=KEY).json()
        assert j["total"] == 6 and [s["company"] for s in j["items"]] == ["C1", "C2"]
        assert client.get("/api/v1/sites?label=none", headers=KEY).json()["total"] == 3
        assert client.get("/api/v1/sites?scan_status=unscanned", headers=KEY).json()["total"] == 6
        for bad in ("limit=0", "limit=500", "offset=-1", "label=banana"):
            assert client.get(f"/api/v1/sites?{bad}", headers=KEY).status_code == 422

    def test_scan_report_roundtrip(self, client):
        upload(client, HEADER + "A,,,,,,reCAPTCHA,https://site0.example/contact\n"
                                "B,,,,,,Turnstile,https://site1.example/contact\n"
                                "C,,,,,,hCaptcha,https://site2.example/contact\n")
        r = client.post("/api/v1/sites/scan", headers=KEY, json={"limit": 10})
        assert r.status_code == 202 and r.json()["total"] == 3
        assert wait_done(client.mgr).status == "finished"
        st = client.get("/api/v1/sites/scan", headers=KEY).json()
        assert st["status"] == "finished" and st["done"] == 3
        rep = client.get("/api/v1/sites/report", headers=KEY).json()
        assert rep["scanned"] == 3 and rep["exact_accuracy"] == round(2 / 3, 4)
        assert rep["confusion_matrix"]["hcaptcha"]["none"] == 1
        mm = client.get("/api/v1/sites?mismatch_only=true", headers=KEY).json()
        assert [s["company"] for s in mm["items"]] == ["C"] and mm["items"][0]["detected"] == "none"
        assert client.net.methods == {"GET"}
        # a second scan has nothing new unless rescan is requested
        assert client.post("/api/v1/sites/scan", headers=KEY, json={}).json()["reason"] == "nothing_to_scan"
        assert client.post("/api/v1/sites/scan", headers=KEY, json={"rescan": True}).status_code == 202

    def test_scan_validation_and_busy(self, client):
        assert client.post("/api/v1/sites/scan", headers=KEY, json={"limit": 0}).status_code == 422
        assert client.post("/api/v1/sites/scan", headers=KEY, json={"limit": 501}).status_code == 422
        assert client.get("/api/v1/sites/scan", headers=KEY).json() == {"status": "idle"}
        assert client.post("/api/v1/sites/scan/cancel", headers=KEY).json() == {"cancelled": False}

    def test_scan_409_while_running(self, Session, client):
        upload(client, HEADER + "A,,,,,,none,https://site0.example/contact\nB,,,,,,none,https://site1.example/contact\n")

        async def slow(request):
            await asyncio.sleep(0.3)
            return httpx.Response(404)

        app.dependency_overrides[sites_api.get_scan_runtime] = lambda: dict(
            cfg=ScanConfig(host_delay_s=0), resolver=stub_resolver,
            client_factory=lambda c: httpx.AsyncClient(transport=httpx.MockTransport(slow)))
        assert client.post("/api/v1/sites/scan", headers=KEY, json={}).status_code == 202
        r = client.post("/api/v1/sites/scan", headers=KEY, json={"rescan": True})
        assert r.status_code == 409 and r.json()["reason"] == "scan_running"
        client.mgr.cancel()
        wait_done(client.mgr)
