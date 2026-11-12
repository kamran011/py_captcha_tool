"""
Obtain a real token from a provider's TEST widget on our local demo page and
verify it through the API, end to end.

    python scripts/acquire_token.py --provider turnstile
    python scripts/acquire_token.py --provider recaptcha --replay
    python scripts/acquire_token.py --provider turnstile --scenario fail

Scope (assignment section 7): this drives ONLY the local demo pages
(http://127.0.0.1:8000/demo/...), which use each provider's published test
keys. It will refuse any other base URL unless --allow-remote is given, and it
never prints the token itself, only a short fingerprint.

Requires the API running (uvicorn app.main:app) and Chromium installed
(python -m playwright install chromium).
"""
from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path
from urllib.parse import urlparse

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from app.config import settings  # noqa: E402
from app.verification.service import PUBLISHED_TEST_TOKENS  # noqa: E402

# provider -> (checkbox iframe selector, checkbox selector inside it, hidden response field)
WIDGETS = {
    "recaptcha": ("iframe[title='reCAPTCHA']", "#recaptcha-anchor", "textarea[name='g-recaptcha-response']"),
    "hcaptcha": ("iframe[title*='checkbox']", "#checkbox", "textarea[name='h-captcha-response']"),
    "turnstile": (None, None, "input[name='cf-turnstile-response']"),     # solves itself
}
LOCAL_HOSTS = {"127.0.0.1", "localhost", "::1"}


def fingerprint(token: str) -> str:
    return hashlib.sha256(token.encode()).hexdigest()[:12]


def acquire_token(provider: str, base: str, headed: bool, timeout_s: int) -> str:
    from playwright.sync_api import TimeoutError as PWTimeout
    from playwright.sync_api import sync_playwright

    frame_sel, box_sel, field_sel = WIDGETS[provider]
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=not headed)
        try:
            page = browser.new_page()
            page.goto(f"{base}/demo/{provider}.html", wait_until="domcontentloaded")
            if frame_sel:
                page.frame_locator(frame_sel).first.locator(box_sel).click(timeout=timeout_s * 1000)
            try:
                page.wait_for_function(
                    "sel => { const e = document.querySelector(sel); return e && e.value && e.value.length > 0; }",
                    arg=field_sel, timeout=timeout_s * 1000)
            except PWTimeout:
                raise SystemExit(f"timed out waiting for the {provider} widget to issue a token "
                                 "(try --headed, or check network access to the provider's CDN)")
            return page.eval_on_selector(field_sel, "e => e.value")
        finally:
            browser.close()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--provider", choices=sorted(WIDGETS), required=True)
    ap.add_argument("--base", default=f"http://127.0.0.1:{settings.port}")
    ap.add_argument("--headed", action="store_true", help="show the browser")
    ap.add_argument("--replay", action="store_true", help="submit the same token twice; the second must be rejected")
    ap.add_argument("--scenario", choices=["pass", "fail", "spent"], help="Turnstile dummy-secret scenario")
    ap.add_argument("--max-age", type=int, help="stricter token age limit (seconds)")
    ap.add_argument("--timeout", type=int, default=45)
    ap.add_argument("--allow-remote", action="store_true", help="permit a non-local base URL (not intended)")
    args = ap.parse_args()

    host = urlparse(args.base).hostname or ""
    if host not in LOCAL_HOSTS and not args.allow_remote:
        print(f"refusing {args.base}: this script only drives the local demo pages", file=sys.stderr)
        return 2

    token = acquire_token(args.provider, args.base, args.headed, args.timeout)
    print(f"{args.provider}: token acquired from the test widget (fingerprint {fingerprint(token)})")

    def submit(label: str) -> dict:
        body = {"provider": args.provider, "token": token}
        if args.scenario:
            body["test_scenario"] = args.scenario
        if args.max_age:
            body["max_age_s"] = args.max_age
        r = httpx.post(f"{args.base}/api/v1/verify", json=body, timeout=20,
                       headers={"X-API-Key": settings.api_key.get_secret_value()})
        d = r.json()
        print(f"  {label}: HTTP {r.status_code}  ok={d.get('ok')}  reason={d.get('reason')}  "
              f"codes={d.get('provider_error_codes') or '-'}  latency={d.get('latency_ms')} ms")
        return d

    first = submit("verify")
    if args.replay:
        second = submit("replay")
        if token in PUBLISHED_TEST_TOKENS.get(args.provider, ()):
            # Test sitekeys issue the same token string every run, so replay tracking
            # is skipped for it (service.PUBLISHED_TEST_TOKENS); both calls should pass.
            print("  note: published test token (identical on every run); replay tracking skipped by design")
            return 0 if first.get("ok") and second.get("ok") else 1
        return 0 if first.get("ok") and second.get("reason") == "reused_token" else 1
    expected_ok = args.scenario in (None, "pass")
    return 0 if bool(first.get("ok")) == expected_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
