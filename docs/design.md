# Design ("The Planner")

## Stack

| Layer | Choice | Why |
|---|---|---|
| API | FastAPI + Uvicorn | Typed, OpenAPI docs for free, reusable endpoint (bonus) |
| Dashboard | **Streamlit** | File uploader, metric cards and charts with no frontend build; talks to the API over HTTP |
| Storage | SQLite via SQLAlchemy 2 | Zero setup for a reviewer |
| Solving | OpenCV, PyTorch (training), ONNX Runtime (inference) | Custom model, CPU-only inference |
| Outbound HTTP | httpx | Timeouts and retries |
| Browser automation | Playwright | Local demo pages and the read-only detection scan |
| Config | pydantic-settings | Environment only, secrets as `SecretStr` |
| Tests | pytest | |
| Environment | Python 3.12, `venv` | |

### Two processes

- `uvicorn app.main:app` on port 8000 — API, plus three small static demo pages.
- `streamlit run dashboard/streamlit_app.py` on port 8501 — the dashboard.

The provider demo pages stay as plain HTML served by FastAPI rather than living inside Streamlit. Each provider's widget script has to run on a real page with the test sitekey and post its token to our server; Streamlit renders custom HTML in a sandboxed iframe, which makes that flow unreliable. The dashboard links to the demo pages and shows the verification results they produce.

## Layout

```
app/
  main.py               FastAPI app, lifespan, routers, static mount for web/demo
  config.py             Settings (pydantic-settings), env only
  logging_config.py     Redacting log filter, token fingerprint helper
  db.py                 Engine, session factory, init_db
  models.py             Attempt, Verification, TokenLedger, Site
  api/
    deps.py             DB session and API-key dependencies
    health.py           /health
    stats.py            /api/v1/stats
    (M2+) solve.py classify.py verify.py batch.py sites.py
  recognition/
    image_classifier.py   text | math | unsupported
    provider_detector.py  HTML/DOM signatures per provider
  solvers/
    base.py             Solver protocol, SolveResult
    crnn_solver.py      primary: CNN + BiLSTM + CTC, ONNX
    segment_solver.py   fallback: OpenCV segmentation + kNN on glyphs
    vision_api_solver.py  optional third-party adapter with pricing table
    math_solver.py      reads the expression, AST-safe evaluation
    pipeline.py         ordering, thresholds, fallback, timeouts
  verification/
    providers.py        reCAPTCHA / hCaptcha / Turnstile siteverify clients
    service.py          token ledger, age checks, error mapping
  sites/
    importer.py         CSV parsing and validation
    scanner.py          detection-only fetch
dashboard/streamlit_app.py   Streamlit UI (pages: Overview, Solve, Verify, Batch, Sites)
web/demo/               recaptcha.html, hcaptcha.html, turnstile.html
ml/                     generate_dataset.py, train_crnn.py, evaluate.py
scripts/acquire_token.py   Playwright against the local demo pages only
data/                   synthetic/, models/, sites.csv, captcha_tool.db
tests/
docs/
```

## Data model

- **Attempt** — one solve: challenge type, method, methods tried, expected, predicted, success, confidence, processing ms, cost, error.
- **Verification** — one token check: provider, ok, reason, HTTP status, latency, token fingerprint.
- **TokenLedger** — SHA-256 of every token seen, provider, first-seen time. Detects reuse and local expiry.
- **Site** — one CSV row plus scan result: labelled provider, detected provider and variant, status, error.

Tokens are never stored or logged in clear; the ledger holds the hash and logs carry the first 12 hex characters.

## Flows

**Solve:** upload → `image_classifier` → `pipeline` (CRNN → segmentation → optional vision API) → `SolveResult` → `Attempt` row → dashboard.

**Verify:** demo page renders the widget with a test sitekey → browser posts the token to `POST /api/v1/verify` → service rejects missing or malformed tokens, checks the ledger for reuse and age, calls `siteverify` (5 s timeout, one retry) → provider error codes mapped to `{ok, provider, reason, http_status, latency_ms}` → ledger and `Verification` row written.

**Failure cases on test keys:** Turnstile's dummy secrets give always-pass, always-fail and "token already spent". reCAPTCHA and hCaptcha test secrets pass; a garbage token returns `invalid-input-response`. Expiry is shown by the ledger's age check (short configurable TTL for the demo) and by provider `timeout-or-duplicate`.

**Site scan (detection only):** CSV -> `Site` rows -> one plain GET per contact page (httpx) -> `provider_detector` -> compare with the label. It is **static HTML only**: an earlier draft rendered pages in Playwright when static HTML showed nothing, but a rendered visit auto-runs reCAPTCHA v3, invisible widgets and Turnstile, i.e. it requests tokens from the providers, which the agreed scope forbids. Playwright is used only against our own local demo pages. See "Batch and site benchmark (M6)" for the safeguards and measured results.

## Security

- Secrets only from environment or `.env`; `.env` is git-ignored; `.env.example` holds the providers' published test keys.
- A logging filter redacts registered secret values and any `token` / `secret` / `api_key` / `*-response` key-value pair on every handler.
- API endpoints under `/api/v1` require `X-API-Key`; `/health` is open.

## Cost reporting

Custom solvers report $0 and measured CPU time. The vision adapter reports tokens × price from a config table. The README carries a comparison against typical commercial solver pricing, with sources.

## Optional third-party solver (M7)

A vision-capable LLM can read the CAPTCHA text as the **last tier** of the text pipeline (`crnn_onnx` -> `segment_knn` -> `vision_api`). It is **off unless `VISION_API_KEY` is set**, and the image is sent to the provider only after the local models failed to give an accepted answer. Math challenges never go external.

| | |
|---|---|
| Providers | `anthropic` (Messages API, default) or `openai` (any OpenAI-compatible chat-completions endpoint, via `VISION_BASE_URL`) |
| Default model | `claude-haiku-5-5`: $0.10 in / $0.50 out per million tokens (prompts up to 100K tokens; ours are tiny). Request options keep the call cheap and are model-specific: thinking disabled and effort `low` on Haiku 5.5, `between_tools` + `low` on Sonnet 5.5, effort `low` on Opus 5.5 |
| Built-in prices (USD/Mtok in / out) | Haiku 5.5 0.10 / 0.50, Haiku 4.5 1.00 / 5.00, Sonnet 5.5 2.00 / 10.00, Opus 5.5 4.00 / 20.00 (Anthropic first-party rates, from the Claude API model table cached 2026-11-03) |
| Unknown model | needs `VISION_PRICE_IN_PER_MTOK` and `VISION_PRICE_OUT_PER_MTOK`; otherwise the solver refuses to run, because spend that cannot be priced is never allowed |

**Cost** = input tokens x input price + output tokens x output price, from the token usage the provider reports for that call, stored on the attempt (`attempts.cost_usd`) and shown per method in `/stats` and the dashboard. A call that was paid for but returned nothing usable still counts (`VisionError` carries its cost). Illustrative arithmetic only, not a measurement: 120 input + 20 output tokens on Haiku 5.5 = $0.000022.

**Spending guards** (in-process): a daily budget (`VISION_BUDGET_USD_PER_DAY`, default $1.00; every call first *reserves* a worst-case cost, so concurrent calls cannot overshoot, and the day's recorded spend is re-read from the database at startup so a restart cannot reset it) and a per-minute call cap (`VISION_MAX_CALLS_PER_MIN`, default 20). Batches never use the paid solver unless `allow_external` is set; `/solve` accepts `allow_external=false` to keep one image local.

**Failure handling:** 429 / 500 / 502 / 503 / 504 / 529 and network errors are retried (default 2 retries) with exponential backoff, `Retry-After` honoured up to 8 s, longer waits fail fast; 401 / 403 open a circuit breaker so a bad key is not hammered; other 4xx are not retried and only the error *type* is kept (never the provider's message, which could echo input); a malformed 200 is an error, never a crash. Every failure becomes an ordinary pipeline attempt and falls through.

**Treating the reply as data:** the model must return a JSON object whose `text` is 4-6 characters from the CAPTCHA alphabet; anything else (prose, other characters, wrong length, `{"text":""}` for illegible) is an error, so text inside an image that tries to give the model instructions cannot produce anything but a wrong answer.

**Uncalibrated by default:** an LLM's self-reported confidence means nothing until measured. The pipeline therefore treats the solver as **advisory** (reported, never accepted on its own) until `python ml/calibrate.py --vision N` measures it on N validation images and writes a threshold; that command spends money, prints the worst-case cost first, and refuses without a key. `GET /api/v1/solvers` lists every solver with its role, threshold and, for the external one, why it is off or what budget remains (never the key).

**Privacy:** an external call sends the image (resized, grayscale PNG, no metadata) to the configured provider. It is the user's configuration choice (a key), visible in the response (`detail.external_call`), the dashboard, and per-request opt-out.

**Not verified live:** no API key was available while building this, so the real provider path has not been exercised. The offline suite (101 tests against a mocked provider) covers the request format, parsing, cost arithmetic, retries, circuit breaker, budget and rate guards, key hygiene and pipeline integration; `tests/test_live_vision.py` (opt-in: `RUN_LIVE=1` plus a key, about three calls, hard-capped at 5 cents) checks the real endpoint. Real accuracy, token counts per image and therefore real cost per image are unmeasured until it is run.

## Batch and site benchmark (M6)

### Batch accuracy
- Datasets are **named folders** under `data/synthetic/` (`test`, `val`, `train`, `math_test`); callers never pass a path, names are matched against `^[A-Za-z0-9_-]{1,40}$` and the resolved folder must be a direct child of the root, and `labels.csv` entries that point outside the folder are skipped. `train` is flagged as optimistic.
- Metrics: exact match, **character accuracy (1 - edit distance / length, floored at 0)**, mean confidence, coverage (accepted answers), precision when accepted, confident-wrong count, latency p50/p95, per-type accuracy, unanswered reasons, and the most-confident wrong answers. Limits: 1000 images per dataset run, 100 per upload, 60 s time budget (partial results are flagged `truncated`). A batch holds one solver slot while it runs; dataset runs are stored in `batch_runs`, per-image rows are only returned for uploads.
- Measured: `test` (1000 images) exact 97.4 %, character accuracy 99.32 %, 23 confident-wrong, p50/p95 3.7/5.0 ms; `math_test` (300) exact 99.67 %.

### CSV import
Per-row validation with line numbers (never aborts the file): http/https only, real hostname, no credentials, no whitespace or control characters, port 80/443 only, <= 2048 chars, duplicates rejected; `Sioux 20Falls`-style cells repaired; idempotent upsert on the contact URL (scan results are kept). The bundled `data/sites.csv` imports 200 rows with 0 rejected.

### Read-only scanner: what it does and refuses to do
- One plain GET per page, plus that host's `robots.txt`, which is **obeyed**; per RFC 9309 an unreachable `robots.txt` (5xx / network error) means "do not fetch". Identifying User-Agent, no cookies (cleared after every response), no Referer, no JavaScript, no forms, no challenge, no token, no browser.
- Hosts must resolve to **public** addresses (loopback, private, link-local and the cloud metadata address are refused before any request); every redirect hop (max 3) is re-validated, including scheme and port. Known limit: the address is checked and then re-resolved by the HTTP client, so a hostile DNS server could rebind in between; the inputs are a reviewed CSV, not arbitrary callers.
- Body capped at 2 MB **after decompression** (gzip-bomb test), per-site wall-clock deadline, at most 5 pages in flight, one request at a time per host with a 1 s gap.
- Error pages (403/404/5xx) are **never analysed**: a block page could otherwise be mistaken for a CAPTCHA.
- One scan at a time (409 otherwise), cancellable, runs on a background thread; the dashboard requires an explicit confirmation before starting.
- Tests assert the behaviour, not just the output: only GET requests with empty bodies, no request ever reaches a provider endpoint even when the page embeds provider scripts, robots disallow/unreachable stops the page request, private addresses are blocked before any request, and each safeguard fails when disabled.

### Measured result on the 200 supplied sites (see `docs/site_scan_results.json`)
| | |
|---|---|
| Pages analysed | **87 of 200 (43.5 %)** |
| Not analysed | 112 HTTP errors (108 x 403, 2 x 405, 1 x 404, 1 x 503), 1 `robots.txt` unreachable |
| Exact accuracy (strict) | 55.2 % on the 87 analysed |
| ...counting v3 as invisible | 67.8 % (reported separately, not tuned on) |
| Provider-level accuracy | 77.0 % (baseline "always reCAPTCHA": 39.1 %) |
| Precision | 100 % for reCAPTCHA, hCaptcha, Turnstile; 0 of 18 pages labelled "none" were flagged |
| Recall (provider level) | reCAPTCHA 70.4 %, Turnstile 83.3 %, hCaptcha 33.3 % (n = 3) |

Reading the result honestly:
- **Coverage is the main limit, and it is the sites' choice.** 403 came back even for `robots.txt`: bot protection refuses automated clients wholesale (BigCommerce stores: 3 of 66 analysed; Volusion 0 of 9; WooCommerce 46 of 69). We do not spoof a browser User-Agent or retry around a refusal; assignment section 7 excludes bypassing anti-bot protection. Results therefore describe the sites that permit a plain client and may not generalise to the rest.
- **Misses are blind spots of static analysis, not false alarms.** 20 pages labelled with a CAPTCHA showed none: 16 mention a provider but render no widget (JavaScript injection), 4 show no trace at all (probably a tag manager or bundle).
- **"Invisible reCAPTCHA" is not separable statically.** Of 20 pages with that label, 12 were detected as v3, 6 as script-only, 1 as a checkbox: static markup only reveals `size=invisible` when it is written in HTML, so strict exact accuracy scores them 0 of 20. v3 shows only a badge, so counting it as invisible is plausible, but that mapping was chosen after seeing the data, hence the separate figure.
- n = 87 is small, and the origin of the CSV labels (possibly a rendered DOM) is unknown.

## API and dashboard (M5)

All routes are versioned under `/api/v1`, need `X-API-Key`, and appear in the OpenAPI docs at `/docs`.

| Route | Purpose | Notable statuses |
|---|---|---|
| `POST /classify` | image -> `text` / `math` / `unsupported`, plus quality metrics | 400 not an image, 413 too large |
| `POST /solve` | image (+ optional `expected`, `challenge_type`) -> answer, every method tried, cost; recorded as an attempt | 200 answered (check `accepted`), 400, 413, 422 unsupported or unreadable, 503 busy |
| `POST /detect` | page HTML -> provider, variant, sitekey, evidence | 413 |
| `POST /verify` | token verification (see "Token verification") | 200/400/409/410/5xx |
| `GET /attempts`, `/verifications` | paged history (`limit` 1-200, `offset`); verifications carry fingerprints only | 422 on bad paging |
| `GET /stats` | totals, success rate, labelled accuracy, latency p50/p95, cost, breakdowns by type, method, provider and verification outcome | |

- **Recording:** `/solve` records every result that reached recognition (including `unsupported` and `poor_quality`, which are meaningful outcomes) but **not** uploads that are not images (400/413), which would pollute volume and accuracy statistics. `success` = equals `expected` when supplied (normalised to upper case), else `accepted`.
- **Abuse guards (in-process, one API process):** 120 requests/min per key (429 + `Retry-After`, `X-RateLimit-Remaining` header); at most 4 solves run at once, others wait up to 2 s then get 503 `busy`; uploads capped at 5 MB, read with a hard limit rather than trusting `Content-Length`; filenames reduced to a sanitised base name. Verified with 8 parallel solves against a cap of 2.
- **Errors:** `{"ok": false, "reason": ..., "detail": ...}`; validation errors never echo submitted values.
- **Startup:** logs a warning while `API_KEY` is still the shipped default.
- **Dashboard (Streamlit, a separate process):** Overview (metrics, charts, history), Solve (upload, held-out sample, generated math, unsupported grid; optional degradation to show the quality gates), Verify (masked token input, published test tokens, Turnstile dummy-secret scenarios), Detect (fixtures, paste or upload HTML). Batch and Sites arrive in M6. It calls the API over HTTP and holds the API key server-side, so `.streamlit/config.toml` binds it to `127.0.0.1` and caps uploads at 5 MB; Streamlit's default would listen on all interfaces.

## Edge cases and failure modes

Status key: **built** (tested), **M5** (API layer, not yet built), **limit** (known, not fixable here).

### Solving (built)

| Input | Behaviour |
|---|---|
| Empty, random, truncated, audio bytes | `invalid_image: <fixed message>`; no library error text or paths returned |
| Oversized file (>10 MB) or dimensions (>16 MP) | Rejected from the header, before decoding (decompression-bomb guard) |
| Blank / flat / near-flat | `poor_quality: blank_or_flat` (intensity range < 20) |
| Pure noise | Rejected by the classifier ("too cluttered") or `poor_quality: excessive_noise` |
| Blurry (sigma >= ~1.2) | `poor_quality: too_blurry`. Gate: Laplacian variance / pixel variance < 0.09 |
| Low contrast | Contrast-stretched only when intensity range < 62; 25% -> 84% exact-match at half contrast. Stretching normal images costs 15 points, so it is conditional |
| Grid / photo / slider strip | `unsupported_challenge`; never reaches a solver. Size and aspect rules catch large images and strips; grids are found by long straight interior lines (morphological opening), which also works on textured photo backgrounds where edge-density tests fail. **Limits:** a 2x2 grid (one interior separator per direction) is not caught, and a plain frame around text is deliberately not treated as a grid. Audio challenges are not images and fail as `invalid_image` |
| Confident garbage output | `degenerate_output` (one repeated letter, the CRNN's sink mode) and `implausible_length` fall through |
| Any unexpected exception | `internal_error`; the pipeline never raises |

Thresholds are measured on generated data, so they describe this generator, not CAPTCHAs in general.

**Confidence is not a reliable error signal.** Wrong CRNN answers have median mean-confidence 0.97 versus 0.9998 for right ones; raising the threshold to 0.99 drops 16% of images and moves test precision only 97.4% -> 98.7%. So per-solver thresholds are derived from validation precision (`ml/calibrate.py`): the CRNN keeps the 0.6 floor; the segmentation solver never reaches 90% precision at any threshold and is **advisory-only** (reported, never self-accepted).

**limit:** out-of-distribution but normal-looking input is accepted with wrong text: 90-degree-rotated 23/60, mirrored 59/60 (inverted is caught, 0/60). No quality statistic distinguishes these.

### Token verification (built in M4; endpoints, error codes and lifetimes checked against each provider's docs)

| Outcome | HTTP | `reason` | Trigger |
|---|---|---|---|
| Accepted | 200 | `ok` | provider `success: true` and token younger than the limit |
| No token | 400 | `missing_token` | null / empty / whitespace-only |
| Not a plausible token | 400 | `malformed_token` | > 2048 chars, whitespace, control or non-ASCII characters; rejected before any network call |
| Rejected by provider | 400 | `invalid_token` | `invalid-input-response`, or any unrecognised code |
| Expired | 410 | `expired_token` | `expired-input-response`, `timeout-or-duplicate`, or `challenge_ts` older than the limit |
| Replayed | 409 | `reused_token` | our ledger, or `already-seen-response` |
| Provider gave no verdict | 503 | `provider_unavailable` | timeout, 5xx, 429, unparsable body, `internal-error`; never fail-open; `retry_possible: true` |
| Our secret/sitekey wrong | 500 | `server_misconfigured` | `invalid-input-secret`, `sitekey-secret-mismatch`, `not-using-dummy-passcode`, empty secret |
| Provider rejected our request | 502 | `provider_error` | `bad-request` |
| Unknown provider | 422 | `unsupported_provider` | |

- **Replay:** the token's SHA-256 is INSERTed (UNIQUE) *before* the provider is called; an `IntegrityError` means reuse. A 12-thread test submits one token concurrently: exactly one reaches the provider. The ledger is authoritative because Cloudflare documents Turnstile's own reuse detection as eventually consistent (repeats within seconds can still succeed). A failed verification also burns the token.
- **Ledger states:** `pending` -> `consumed`; a call that fails after the request may have been processed becomes `indeterminate` and allows exactly one re-attempt (compare-and-swap); a call that provably never reached the provider deletes the claim so the caller retries cleanly; an abandoned `pending` claim older than 60 s is recoverable.
- **Expiry:** limit = min(provider lifetime, `TOKEN_MAX_AGE_S`, optional per-request `max_age_s`). Lifetimes: reCAPTCHA 120 s, hCaptcha 120 s, Turnstile 300 s. Age comes from the provider's `challenge_ts`; `max_age_s` can only tighten. Observed live: with test keys `challenge_ts` is the verification time, so expiry by age cannot be triggered against test keys (hCaptcha's even lands ~2 minutes in the future, which is treated as not expired); it is covered by mocked tests. `timeout-or-duplicate` is one provider code for "expired" and "already used"; our ledger has already ruled out our own replays, so it maps to `expired_token` with an explanatory `detail`.
- **Retry policy** (`siteverify` consumes the token): retry once on connect errors, pool timeouts and 502/503/504 (request never reached the provider or failed at its gateway). Read timeouts and other transport errors are retried **only for Turnstile**, which accepts an `idempotency_key` (derived from the token hash, so retries reuse it). 429, 500, other statuses and bad bodies are not retried. Earlier drafts said "5 s timeout + one retry" for every provider; that would have misreported a lost-response retry as a replay.
- **Published test tokens:** Cloudflare's dummy sitekey issues the identical token `XXXX.DUMMY.TOKEN.XXXX` on every run and hCaptcha's test keys issue fixed tokens, so replay tracking would reject every second demo run. Exactly those tokens skip the ledger, and only while that provider's published test secret is configured. With a real secret they get no special treatment. Consequence: replay rejection is demonstrated with reCAPTCHA (unique tokens) and by unit tests, not with Turnstile/hCaptcha test keys.
- **Test keys cannot show every outcome:** Google documents that its test secret passes every request, so "invalid reCAPTCHA token" is only demonstrable with mocks. Turnstile's dummy secrets give pass / fail / `timeout-or-duplicate`; hCaptcha rejects a nonsense token.
- **Endpoints:** `POST /api/v1/verify` (API key) and `POST /demo-api/verify` (no key, loopback clients only, hidden from OpenAPI) so no API key is embedded in browser-visible demo pages. The 422 handler returns `{"ok": false, "reason": "invalid_request", "errors": [{loc, msg, type}]}` and never echoes submitted values.
- **`test_scenario`** (`pass|fail|spent`) selects Turnstile's published dummy secrets; refused with `test_scenario_not_allowed` unless the configured Turnstile secret is itself a published dummy.

### Concurrency (built unless noted)

- SQLite uses WAL, a 15 s busy timeout and `synchronous=NORMAL`; tested with 8 threads x 40 writes. Default journaling would let writers fail with "database is locked". WAL still allows a single writer at a time, which is adequate for one process, not for multi-process scale-out.
- Lazy singletons (ONNX session, glyph tables) are lock-guarded and the template cache is written atomically.
- Solvers run in a 4-thread pool with a per-solver timeout. A timed-out thread keeps running to completion (Python cannot kill it), so sustained timeouts could exhaust the pool. **M5:** bound in-flight solves with a semaphore and add per-key rate limiting on `/api/v1/*`; batch jobs process sequentially or with a small fixed worker count.

### Logging and secrets (built unless noted)

- The redaction filter scrubs registered secrets **and** `key=value` / `"key": "value"` pairs for token, secret, password, api_key, authorization and `*-response`, in messages **and** formatted tracebacks. (The first version only handled registered secrets and ignored tracebacks; both gaps were found in review and are fixed and tested.)
- The filter is attached to uvicorn's error handler (unhandled-exception tracebacks bypass root handlers). `uvicorn.access` is left alone because tokens never appear in URLs; the filter only rewrites records it changes, so uvicorn's access formatter keeps working.
- Exception text is passed through `redact()` before being stored in `attempts.error`; the DB holds token fingerprints only. `verifications.error` is redacted the same way.
- Demo pages no longer embed an API key (they use the loopback-only `/demo-api/verify`). **Open:** the default `API_KEY=changeme-local` must be changed for anything beyond localhost; M5 will warn at startup.
