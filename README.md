# CAPTCHA Solving & Verification Tool

Technology Brainz, Assignment 01. A standalone tool that **recognises** CAPTCHA challenge types, **solves** synthetic text and math CAPTCHAs with a model trained here, **verifies** provider tokens server side using the providers' official test keys, and **reports** accuracy, latency and cost through an API and a dashboard.

> **Scope (assignment sections 1 and 7).** Everything is exercised on synthetic CAPTCHAs, local demo pages and the providers' official test keys. Contact-form automation, bulk submission and solving CAPTCHAs on third-party sites are **not** built. The supplied 200-site CSV is used as a labelled recognition benchmark by a **read-only** scan: one plain GET per contact page plus that site's `robots.txt` (obeyed), static HTML analysis, nothing filled, submitted, solved or requested.

## Results at a glance

Measured on this project (synthetic data and test keys); every figure can be reproduced with the commands below. Details, caveats and limits are in `docs/architecture.md` and `docs/design.md`.

| | |
|---|---|
| Text CAPTCHA solver, 1,000 held-out images | **98.1% exact match**, 99.6% character accuracy, p50 / p95 3.3 / 4.7 ms, **$0** per image |
| Math CAPTCHA solver, 300 images | 99.7% |
| Token verification | real provider test endpoints: Turnstile, hCaptcha, reCAPTCHA; replay, missing, malformed, invalid, expired, provider-down each return a distinct structured result |
| Provider detection on the 200 supplied sites | **87 of 200 analysed** (the rest refused automated clients, mostly HTTP 403); provider-level accuracy 77.0% on those 87, precision 100% |
| Optional third-party vision solver | built, off by default, cost priced from token usage; **not yet run against the real provider** (no key was available) |
| Tests | 513 pass offline; 7 opt-in live tests |

## Quick start

Needs Python 3.10+ (developed and tested on 3.12) and internet for `pip` and, only for the token demos, the providers' test widgets.

```bash
# 1. Virtual environment
python -m venv .venv
.venv\Scripts\activate            # Windows
# source .venv/bin/activate       # macOS / Linux

# 2. Install (no PyTorch needed: the trained model ships as data/models/crnn.onnx)
pip install -r requirements.txt

# 3. Configuration (all provider keys in the example are the providers' PUBLISHED TEST keys)
copy .env.example .env            # Windows
# cp .env.example .env            # macOS / Linux

# 4. Generated datasets: samples for the dashboard, batch runs, calibration, evaluation, most tests (about 1 minute)
python ml/generate_dataset.py --train 0
python ml/generate_dataset.py --math 300

# 5. Browser for the token demos (once)
python -m playwright install chromium

# 6. Run: two terminals
uvicorn app.main:app --port 8000
streamlit run dashboard/streamlit_app.py       # http://127.0.0.1:8501, bound to loopback only

# 7. Confirm everything works (third terminal; --online also checks the providers' test endpoints)
python scripts/smoke_test.py --online
```

Open the dashboard at http://127.0.0.1:8501, the interactive API docs at http://127.0.0.1:8000/docs, and the local demo pages at `/demo/recaptcha.html`, `/demo/hcaptcha.html`, `/demo/turnstile.html`. The first solve after start-up is slower (about half a second: it loads the model).

`--train 0` skips the 20,000-image training split. The validation and test splits are generated from fixed seeds and come out byte-identical on every run, so the reported accuracy reproduces (two earlier, unseeded draws of the same generator scored 97.3-97.4%, so read it as roughly 97-98%; rendering may differ slightly across Pillow versions, which `requirements-lock.txt` pins). Without step 4 the tool still runs; the sample pickers, batch, calibration and the full test suite need the datasets.

## What it does

| Page / route | Purpose |
|---|---|
| Dashboard **Solve** / `POST /api/v1/solve` | upload an image or use a held-out, math or grid sample (optionally degraded: blur, noise, low contrast); see the answer, every method tried, confidence, time and cost |
| Dashboard **Verify** / `POST /api/v1/verify` | verify a provider token; see the verdict, latency and token fingerprint |
| Dashboard **Detect** / `POST /api/v1/detect` | which CAPTCHA provider, variant and sitekey a page's HTML embeds |
| Dashboard **Batch** / `POST /api/v1/batch` | accuracy over a named dataset or uploaded images: exact match, character accuracy, coverage, latency p50/p95, confident-wrong answers |
| Dashboard **Sites** / `/api/v1/sites/*` | import the CSV, run the read-only scan (asks for confirmation), accuracy vs the CSV labels, confusion matrix |
| Dashboard **Overview** / `GET /api/v1/stats`, `/solvers` | totals, success rate, latency, cost by method, which solvers are on |
| `POST /api/v1/classify` | image -> `text` / `math` / `unsupported` plus quality metrics |

All `/api/v1` routes need the `X-API-Key` header (the `API_KEY` value in `.env`) and are rate limited (120 requests/minute, at most 4 concurrent solves, 5 MB uploads). Example:

```bash
curl -s -H "X-API-Key: changeme-local" -F file=@data/synthetic/test/000000.png -F expected=ARY6 http://127.0.0.1:8000/api/v1/solve
```

## Token verification demo

```bash
python scripts/acquire_token.py --provider recaptcha --replay          # real widget token; the replay is rejected: 409 reused_token
python scripts/acquire_token.py --provider turnstile                   # published dummy token: 200 ok
python scripts/acquire_token.py --provider turnstile --scenario fail   # published always-fail secret
python scripts/acquire_token.py --provider turnstile --scenario spent  # published duplicate-token secret
```

The script drives only the local demo pages with the providers' test keys, never prints a token (a short fingerprint only) and refuses non-local URLs. With test keys, Google's secret accepts any token and the Cloudflare and hCaptcha dummy tokens are identical every run, so "invalid reCAPTCHA token", expiry by age and replay of a dummy token are covered by mocked tests rather than live runs (explained in `docs/design.md`).

## Batch accuracy and the site benchmark

```bash
curl -s -H "X-API-Key: changeme-local" -H "Content-Type: application/json" -d '{"dataset":"test","limit":1000}' http://127.0.0.1:8000/api/v1/batch
curl -s -H "X-API-Key: changeme-local" -F use_bundled=true http://127.0.0.1:8000/api/v1/sites/import
```

The scan makes **real requests to third-party sites**, so it is only started from the dashboard's Sites page (after a confirmation) or `POST /api/v1/sites/scan`. It is read-only and polite: identifying User-Agent, `robots.txt` obeyed, public addresses only, at most 5 pages in flight and one request at a time per host. Pages that answer 403 are recorded as not analysed, never retried around. The result of the run reported above is saved in `docs/site_scan_results.json`.

## Optional third-party vision solver (off by default)

Without `VISION_API_KEY` everything is local and free. With a key, a vision model becomes the **last resort** for text CAPTCHAs the local models fail on. It **sends the image to the provider**, so it can be refused per request (`allow_external=false`), is never used by batches unless asked, has a daily budget (default $1.00) and a per-minute cap, and prices each call from the provider's reported token usage. Until `python ml/calibrate.py --vision 50` measures it (this spends money), its answers are advisory and never accepted on their own. See `docs/design.md` and `docs/cost.md`. **Not verified against the real provider:** run `RUN_LIVE=1 pytest tests/test_live_vision.py` (about three calls, capped at 5 cents) after adding a key.

## Reproducing the measurements

```bash
python ml/evaluate.py                 # held-out accuracy and latency of the ONNX model -> data/models/eval_report.json
python ml/evaluate_pipeline.py        # whole pipeline: classifier, fallback, math -> data/models/pipeline_report.json
python ml/calibrate.py                # per-solver acceptance thresholds from validation precision
pip install -r requirements-train.txt && python ml/generate_dataset.py && python ml/train_crnn.py --epochs 20   # retrain (~45 min CPU)
```

`requirements-lock.txt` pins the exact versions used.

## Tests

```bash
pytest tests -q                                          # 513 tests, offline and deterministic (about 40 s with the datasets)
RUN_LIVE=1 pytest tests/test_live_providers.py -v        # also calls the providers' real test endpoints (test keys only)
RUN_LIVE=1 pytest tests/test_live_vision.py -v           # needs VISION_API_KEY; spends a few cents
```

The offline suite uses mocked providers to force every failure mode (timeouts, 5xx, replays, concurrent submissions, malformed responses, hostile redirects, private addresses). The main safeguards were mutation-checked: disabling replay protection, the scanner's address block, robots handling, the vision budget guard or log redaction makes tests fail.

## Configuration

Defaults live in `.env.example`; copy it to `.env` to override. The most useful settings:

| Variable | Default | Meaning |
|---|---|---|
| `API_KEY` | `changeme-local` | required for `/api/v1/*`; change it for anything beyond localhost (the API warns while it is the default) |
| `DATABASE_URL` | `sqlite:///data/captcha_tool.db` | created on first start |
| `RECAPTCHA_*`, `HCAPTCHA_*`, `TURNSTILE_*` | published test keys | sitekeys and secrets for verification |
| `CONFIDENCE_THRESHOLD` | `0.60` | floor for accepting a solver's answer (calibrated thresholds apply on top) |
| `TOKEN_MAX_AGE_S` | `300` | ceiling on token age (providers also have their own lifetime) |
| `SCAN_CONCURRENCY`, `SCAN_TIMEOUT_S` | `5`, `20` | site scan limits |
| `VISION_*` | off | optional third-party solver: key, provider, model, prices, budget |
| `API_BASE` | `http://127.0.0.1:8000` | where the dashboard finds the API |

## Documentation

| File | Contents |
|---|---|
| `docs/architecture.md` | architecture note, flows, decisions, measured results, requirements coverage, known limitations |
| `docs/cost.md` | cost breakdown per method, at scale, and sourced commercial-solver prices |
| `docs/design.md` | full design and decision log, status tables, edge cases and failure modes, token and scan details |
| `docs/requirements.md`, `docs/tasks.md` | acceptance criteria and the build plan |
| `docs/demo-script.md` | timed script for the 3-5 minute demo video |

## Project layout

```
app/          API, recognition, solvers, verification, batch, site scan, config, models
dashboard/    Streamlit UI and its API client
web/demo/     provider widget demo pages (reCAPTCHA, hCaptcha, Turnstile)
ml/           dataset generator, CRNN training, evaluation, calibration
scripts/      acquire_token.py (Playwright, local pages only), smoke_test.py
data/         trained model, calibration, bundled sites.csv; generated datasets and the database are not tracked
tests/        pytest suite and HTML fixtures
docs/         documentation
```

## Troubleshooting

| Symptom | Fix |
|---|---|
| `ModuleNotFoundError` | activate the virtual environment and run `pip install -r requirements.txt` |
| Dashboard says "API offline" | start the API first; if you changed `API_KEY` or the port, both processes read the same `.env` (or set `API_BASE`) |
| Dashboard stays on a grey loading skeleton; browser console says `WebSocket is closed before the connection is established` | open **http://127.0.0.1:8501**, not `localhost:8501`. On some machines `localhost` resolves to IPv6 (`::1`) first; the dashboard listens on IPv4 loopback only (deliberately, so it is not reachable from the network), and the roughly one second it takes to fall back exceeds Streamlit's connection timeout |
| `401` from the API | the `X-API-Key` header must equal `API_KEY` |
| "Database schema is out of date" | delete `data/captcha_tool.db` and restart; it is recreated |
| Sample pickers or batch say no dataset | run step 4 of the quick start |
| Playwright cannot find Chromium | `python -m playwright install chromium` |
| Token demo times out | the providers' widgets load from their CDNs; check internet access, or use `--headed` to watch |

## Known limitations

Accuracy figures describe the synthetic generator used here; other CAPTCHA styles are untested. The solver's confidence does not flag its own mistakes, and a normal-looking rotated or mirrored image can be accepted with wrong text. The site scan sees only static HTML and most pages refused automated clients, so its accuracy describes the cooperative minority. The vision solver is unverified live. Rate limits and budgets are in-process (single-process by design). Full list: `docs/architecture.md`.

## Assignment deliverables

| Deliverable | Where |
|---|---|
| Repo with setup and environment variables | this README, `.env.example`, `requirements.txt` |
| Working application | API + dashboard; `python scripts/smoke_test.py` |
| At least one solving workflow | custom CRNN pipeline (text), math solver, optional vision API |
| At least one official token verification workflow | reCAPTCHA, hCaptcha and Turnstile test flows |
| 3-5 minute demo video | script in `docs/demo-script.md` (the recording itself is not included) |
| Architecture note and cost breakdown | `docs/architecture.md`, `docs/cost.md` |
| Bonus | several challenge types and providers, fallback chain, batch and accuracy statistics, reusable API |
