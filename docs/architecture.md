# Architecture note

CAPTCHA Solving & Verification Tool, Technology Brainz Assignment 01. This is the short version; `docs/design.md` has the full decision log, status tables, edge cases and measurements, and `docs/requirements.md` the acceptance criteria.

## Scope

Per assignment sections 1 and 7 the tool is exercised only on **synthetic CAPTCHAs, local demo pages, and the providers' official test keys**. The supplied 200-site CSV is used as a **labelled recognition benchmark** by a read-only scan: one plain GET per contact page (plus `robots.txt`, obeyed), static HTML analysis, nothing filled, submitted, solved or requested. Contact-form automation and solving CAPTCHAs on third-party sites are out of scope and not built.

## System at a glance

```
                       +-------------------------- FastAPI  :8000 --------------------------+
 Browser ---> Streamlit  |  guards: X-API-Key, rate limit, solve slots, upload caps, redaction |
 dashboard    :8501 ---->|                                                                    |
 (loopback)   (HTTP)    |  /classify  /solve  /detect  /verify  /batch  /sites  /stats ...   |---> SQLite (WAL)
                        |                                                                    |     attempts, verifications,
 Local demo pages <-----|  recognition        solvers            verification     sites       |     token_ledger, sites,
 /demo/*.html           |  - classifier       - CRNN (ONNX)      - provider       - importer  |     batch_runs
   ^                    |  - quality gate     - segment kNN        clients        - scanner   |
   |                    |  - provider         - math kNN         - token ledger   - report    |
 scripts/               |    detector         - vision API (opt)                              |
 acquire_token.py       +----------------------------------------------------------------------+
 (Playwright, local only)        |                         |                          |
                                 v                         v                          v
                         ONNX model on CPU     provider siteverify (test keys)   public contact pages
                                               + optional vision API (off)       (read-only GET)
```

Two processes: the API and the dashboard. The dashboard holds the API key server-side and calls the API over HTTP, so it is bound to `127.0.0.1` (`.streamlit/config.toml`).

## Components

| Area | Module | Responsibility |
|---|---|---|
| Recognition | `app/recognition/image_classifier.py` | `text` / `math` / `unsupported` (size, aspect, long straight grid lines, glyph pattern) |
| | `app/recognition/quality.py` | rejects blank, noisy and blurry images before solving; stretches low contrast |
| | `app/recognition/provider_detector.py` | provider, variant, sitekey and evidence from HTML, via a real parser with strict host matching |
| Solving | `app/solvers/crnn_solver.py` | CNN + BiLSTM + CTC model, run with ONNX Runtime (primary) |
| | `app/solvers/segment_solver.py`, `glyphs.py` | OpenCV segmentation + NumPy kNN (fallback, advisory) |
| | `app/solvers/math_solver.py`, `safe_math.py` | reads `12+7`, evaluates with an AST whitelist (never `eval`) |
| | `app/solvers/vision_api_solver.py` | optional third-party LLM, last resort, priced from token usage, budget-guarded |
| | `app/solvers/pipeline.py` | order, calibrated per-solver thresholds, timeouts, fallback, cost, attempt recording |
| Verification | `app/verification/providers.py`, `service.py` | `siteverify` clients, atomic replay ledger, error mapping, retry policy |
| Benchmark | `app/batch.py` | accuracy over named datasets |
| | `app/sites/importer.py`, `scanner.py`, `report.py` | CSV validation, read-only scan, confusion matrix |
| Platform | `app/api/*`, `app/logging_config.py`, `app/db.py` | routes, guards, redaction, SQLite |
| UI | `dashboard/streamlit_app.py`, `web/demo/*.html` | dashboard; provider widget pages that post tokens to the server |
| ML | `ml/*.py` | dataset generator, training, evaluation, calibration |

## The four flows

**Solve.** upload -> size/dimension check from the header -> classifier -> (`unsupported`: stop) -> quality gate -> solvers in order -> first answer above its calibrated threshold wins -> attempt row with every method tried, latency, cost. Math goes to the math solver only. Text goes `crnn_onnx` -> `segment_knn` (never self-accepted: 30% accuracy on the held-out test split) -> `vision_api` (only if configured).

**Verify.** widget on a local demo page yields a token -> `POST /verify` -> shape check -> **insert the token's SHA-256 into a UNIQUE ledger before calling the provider** (so concurrent replays cannot both reach it) -> `siteverify` with a provider-specific retry policy -> provider error codes mapped to one structured verdict (`ok`, `missing_token`, `malformed_token`, `invalid_token`, `expired_token`, `reused_token`, `provider_unavailable`, ...) with a fitting HTTP status. Tokens and secrets never reach logs, responses or the database; only a 12-character fingerprint does.

**Detect.** page HTML -> weighted evidence (widget element, iframe, `<script src>` with host *and* path, inline API call) -> provider, variant, sitekey, confidence, `needs_render` when a provider is mentioned but no widget is visible.

**Site benchmark.** import CSV (per-row validation) -> scan (robots obeyed, public addresses only, redirects re-validated, body capped, one request at a time per host, error pages never analysed) -> accuracy vs the CSV label, exact and provider-level, with the confusion matrix and the unanalysed pages reported separately so failures cannot inflate the numbers.

## Decisions that shaped it

| Decision | Why | Cost of the choice |
|---|---|---|
| Train our own CRNN on a generated dataset, serve with ONNX Runtime | accuracy we can measure and reproduce; $0 per image; no PyTorch at run time | accuracy is only proven on our own generator |
| Thresholds from measured precision (`ml/calibrate.py`), not a constant | the CRNN's confidence does not flag its errors (wrong answers have median confidence 0.97), so a threshold cannot fix that; calibration decides which solver may answer alone | a solver that never reaches the target (segmentation, uncalibrated vision) is advisory |
| Quality gates before solving | the model answers confidently on noise (`MMMMM`) and heavy blur | rejects some images the model could partly read |
| Ledger-first replay protection | Cloudflare documents Turnstile's own reuse detection as eventually consistent | published test tokens (identical every run) skip the ledger, only while the published test secret is configured |
| Static-only site scan | a rendered visit would run reCAPTCHA v3 / Turnstile and request tokens, which the scope forbids | widgets injected by JavaScript are invisible; reported, not guessed |
| Do not spoof a browser User-Agent after a 403 | assignment section 7 excludes bypassing anti-bot protection | only 87 of 200 pages could be analysed |
| Optional vision solver off by default, advisory until calibrated, budget-guarded | a paid call that sends images off-machine must be a deliberate choice | the live path is unverified without a key |
| SQLite with WAL, in-process guards | one process, zero setup for a reviewer | not a multi-process design |

## Measured results (synthetic data and test keys; reproduce with the README)

| What | Result |
|---|---|
| Text solver, 1,000 held-out images | exact match **98.1%** (97.3% and 97.4% on two earlier, non-reproducible draws of the same generator, so read it as roughly 97-98%), character accuracy 99.6%, 18 confident-wrong answers, p50/p95 3.3 / 4.7 ms |
| Math solver, 300 images | 99.7% (templates come from the same renderer, so this shows consistency, not generality) |
| Classifier | text 1000/1000, math 299/300; grid, photo, strip and blank samples all `unsupported`; 2x2 grids are not caught |
| Quality gates | blank, noise and heavy blur rejected; accuracy unchanged on good images |
| Token verification (real provider test endpoints) | Turnstile pass / fail / duplicate, hCaptcha pass / invalid, reCAPTCHA pass and local replay (409) |
| Site benchmark, 200 supplied sites | 87 analysed (43.5%); provider-level accuracy 77.0%, exact 55.2% (67.8% counting v3 as invisible, reported separately); precision 100% |
| Tests | 517 passing offline, 7 opt-in (6 live-provider, 1 live-vision) |

## Requirements coverage

| Req | Status | Evidence |
|---|---|---|
| R1 classify image | met | `test_solvers.py` classifier tests; dashboard Solve page |
| R2 identify provider | met | 21 fixture pages + 3 demo pages in `test_detector.py`; real pages in the site report |
| R3 text solver >= 90% | met, **98.1% measured** | `ml/evaluate.py`, `data/models/eval_report.json`, batch page |
| R4 math | met | `math_test` 99.7%; AST-whitelist evaluation tested against injection |
| R5 fallback chain | met | `test_solvers.py`, `test_vision.py` (every method tried is recorded) |
| R6 third-party solver with cost | built and mock-verified; **not run against the real provider** | `test_vision.py` (101 tests); `tests/test_live_vision.py` is the opt-in live check |
| R7 official token flow | met | live runs with Playwright against the three test widgets; `test_live_providers.py` |
| R8 missing / malformed / invalid / expired / reused | met; *invalid reCAPTCHA* and *expiry by age* are mock-only | Google's test secret accepts any token and test `challenge_ts` is the verification time |
| R9 no secrets or tokens in logs | met | leak tests over logs, responses and the database; mutation-checked |
| R10 dashboard | met | Overview, Solve, Verify, Detect, Batch, Sites; AppTest suite and browser checks |
| R11 batch + accuracy | met | `app/batch.py`, API and dashboard |
| R12 CSV import + detection scan | met, **with 43.5% coverage** | `docs/site_scan_results.json` |
| R13 reusable API | met | versioned routes, OpenAPI at `/docs`, key, rate limits |
| R14 deliverables | this milestone | README, this note, `docs/cost.md`, `docs/demo-script.md` |

## Known limitations

- Accuracy figures describe **our synthetic generator**. Other CAPTCHA styles are untested, and a normal-looking but 90-degree-rotated or mirrored image is accepted with wrong text (19 of 60 and 58 of 60 in a test).
- Confidence is not an error signal for the CRNN; calibration cannot remove its confident mistakes.
- The site scan sees only what static HTML shows, and most pages refused automated clients, so its accuracy describes the cooperative minority.
- The optional vision solver has no measured accuracy or real per-image cost yet.
- Rate limits, the vision budget and the scan manager are in-process: single-process by design. One API key, no user accounts.
- Address checks in the scanner can be raced by a hostile DNS server (checked, then re-resolved by the HTTP client); inputs here are a reviewed CSV.
