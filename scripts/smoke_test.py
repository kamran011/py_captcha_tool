"""
End-to-end smoke test against a RUNNING API: exercises each main flow once and prints PASS / FAIL / SKIP.

    uvicorn app.main:app --port 8000          # in another terminal
    python scripts/smoke_test.py              # offline checks only
    python scripts/smoke_test.py --online     # also verifies provider TEST tokens (needs internet)

Exit code 0 when nothing failed. Skips are reported, never silently passed.
It sends nothing to third-party sites; --online only calls the providers' official test endpoints
with their published test tokens.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from app.config import settings  # noqa: E402
from dashboard import samples  # noqa: E402

results: list[tuple[str, str, str]] = []


def record(name: str, status: str, detail: str = "") -> None:
    results.append((name, status, detail))
    print(f"  [{status}] {name}" + (f": {detail}" if detail else ""), flush=True)


def check(name: str, ok: bool, detail: str = "") -> None:
    record(name, "PASS" if ok else "FAIL", detail)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base", default=f"http://127.0.0.1:{settings.port}")
    ap.add_argument("--online", action="store_true", help="also verify provider test tokens (internet needed)")
    args = ap.parse_args()
    http = httpx.Client(base_url=args.base, headers={"X-API-Key": settings.api_key.get_secret_value()}, timeout=60)

    print(f"Smoke test against {args.base}")
    try:
        h = http.get("/health").json()
    except httpx.HTTPError as exc:
        print(f"  [FAIL] API not reachable ({type(exc).__name__}). Start it with: uvicorn app.main:app --port 8000")
        return 1
    check("health", h.get("status") == "ok" and h["db"]["ok"], f"version {h.get('version')}")

    r = http.get("/api/v1/solvers")
    names = {s["name"]: s for s in r.json().get("solvers", [])} if r.status_code == 200 else {}
    check("solvers listed", {"crnn_onnx", "math_knn", "vision_api"} <= set(names),
          f"vision solver: {'on' if names.get('vision_api', {}).get('enabled') else 'off (' + str(names.get('vision_api', {}).get('reason')) + ')'}")

    # text solve on a held-out sample
    test = samples.test_samples(5)
    if test:
        fname, truth = test[0]
        d = http.post("/api/v1/solve", files={"file": (fname, samples.load_sample(fname))},
                      data={"expected": truth, "allow_external": "false"}).json()
        check("solve text (held-out sample)", d.get("success") is True and d.get("method") == "crnn_onnx",
              f"{truth} -> {d.get('text')} in {d.get('total_ms')} ms, ${d.get('cost_usd')}")
    else:
        record("solve text (held-out sample)", "SKIP", "no generated dataset: python ml/generate_dataset.py")

    img, expr, answer = samples.math_sample(7)
    d = http.post("/api/v1/solve", files={"file": ("math.png", img)}, data={"expected": str(answer)}).json()
    check("solve math", d.get("success") is True, f"{expr} = {d.get('text')}")

    d = http.post("/api/v1/solve", files={"file": ("grid.png", samples.grid_sample())})
    check("unsupported grid refused", d.status_code == 422 and d.json().get("error") == "unsupported_challenge",
          f"HTTP {d.status_code}")
    if test:
        blurred = samples.degrade(samples.load_sample(test[0][0]), "blur")
        d = http.post("/api/v1/solve", files={"file": ("blur.png", blurred)})
        check("blurry image refused by the quality gate", d.status_code == 422 and "too_blurry" in str(d.json().get("error")),
              str(d.json().get("error")))
    d = http.post("/api/v1/solve", files={"file": ("junk.png", b"not an image")})
    check("non-image rejected", d.status_code == 400, f"HTTP {d.status_code}")

    fixtures = samples.detector_fixtures()
    if "hcaptcha_checkbox" in fixtures:
        d = http.post("/api/v1/detect", json={"html": samples.load_fixture("hcaptcha_checkbox")}).json()
        check("detect hCaptcha in HTML", d.get("provider") == "hcaptcha", f"variant {d.get('variant')}, sitekey {d.get('sitekey')}")
    else:
        record("detect hCaptcha in HTML", "SKIP", "fixtures not found")

    d = http.post("/api/v1/verify", json={"provider": "recaptcha"})
    check("verify: missing token -> 400", d.status_code == 400 and d.json().get("reason") == "missing_token")
    d = http.post("/api/v1/verify", json={"provider": "recaptcha", "token": "has a space"})
    check("verify: malformed token -> 400", d.status_code == 400 and d.json().get("reason") == "malformed_token")
    if args.online:
        d = http.post("/api/v1/verify", json={"provider": "turnstile", "token": "XXXX.DUMMY.TOKEN.XXXX"})
        check("verify: Turnstile test token accepted (real siteverify)", d.status_code == 200 and d.json().get("ok") is True,
              f"{d.json().get('latency_ms')} ms")
        d = http.post("/api/v1/verify", json={"provider": "hcaptcha", "token": f"FAKE-{int(time.time())}-TOKEN"})
        check("verify: hCaptcha nonsense token rejected (real siteverify)", d.status_code == 400 and d.json().get("reason") == "invalid_token")
        token = f"smoke-{int(time.time())}-token"
        first = http.post("/api/v1/verify", json={"provider": "recaptcha", "token": token})
        again = http.post("/api/v1/verify", json={"provider": "recaptcha", "token": token})
        check("verify: replayed token -> 409 reused_token", first.status_code == 200 and again.status_code == 409,
              f"first {first.status_code}, replay {again.status_code}")
    else:
        record("verify against real provider test endpoints", "SKIP", "pass --online")

    if test:
        d = http.post("/api/v1/batch", json={"dataset": "test", "limit": 100}).json()
        check("batch accuracy (100 held-out images)", (d.get("exact_match") or 0) >= 0.9,
              f"exact {d.get('exact_match')}, char {d.get('char_accuracy')}, p95 {d.get('latency_ms', {}).get('p95')} ms")
    else:
        record("batch accuracy", "SKIP", "no generated dataset")

    d = http.post("/api/v1/sites/import", data={"use_bundled": "true"}).json()
    check("site CSV import", d.get("rows_valid") == 200 and d.get("rows_rejected") == 0,
          f"{d.get('rows_valid')} valid, {d.get('rows_rejected')} rejected (no scan is started here)")

    s = http.get("/api/v1/stats").json()
    check("stats", s["attempts"]["total"] >= 1, f"{s['attempts']['total']} attempts recorded, total cost ${s['attempts']['total_cost_usd']}")

    fails = [r for r in results if r[1] == "FAIL"]
    skips = [r for r in results if r[1] == "SKIP"]
    print(f"\n{len(results) - len(fails) - len(skips)} passed, {len(fails)} failed, {len(skips)} skipped")
    return 1 if fails else 0


if __name__ == "__main__":
    raise SystemExit(main())
