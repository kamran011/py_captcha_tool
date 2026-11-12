# Requirements

CAPTCHA Solving & Verification Tool — Technology Brainz Assignment 01.

## Users

- **Reviewer** — runs the tool locally, watches success and failure cases, reads the code.
- **Integrator** — calls the JSON API from another authorised application.

## Scope boundary

The assignment (§1, §7) limits testing to synthetic CAPTCHAs, locally hosted demos, candidate-owned sites and official provider test environments, and excludes contact-form automation and bulk submissions.

The supplied 200-site CSV is therefore used as a **labelled recognition benchmark**: each contact page is loaded once, read-only, and the embedded provider is identified and compared with the CSV's `Captcha` column. No field is filled, no form is submitted, no challenge is solved and no token is requested on those sites.

## User stories and acceptance criteria

| ID | Story | Acceptance criteria |
|---|---|---|
| R1 | As a reviewer I upload a challenge image and the tool tells me what it is | Returns `text`, `math` or `unsupported` with a confidence. Grid, slider and puzzle images return `unsupported` and are never sent to a solver |
| R2 | As a reviewer I give the tool page HTML or a URL and it names the provider | Returns provider, variant (reCAPTCHA v2 checkbox / invisible / v3, hCaptcha, Turnstile, none), sitekey if present, and the evidence that matched |
| R3 | As a reviewer I see synthetic text CAPTCHAs solved by a custom model | Returns text and per-character confidence. Target: ≥ 90 % exact match on 1,000 held-out generated images. The measured figure is reported, whatever it is |
| R4 | As a reviewer I see a second challenge type solved | Synthetic arithmetic image is read and evaluated without `eval`; the answer is returned |
| R5 | As a reviewer I see fallback when the primary method is unsure | Below the confidence threshold or on error, the next solver runs. The response lists every method tried |
| R6 | As a reviewer I can enable a third-party solver and see its cost | Vision-model adapter, disabled unless an API key is set. Per-call cost computed from usage. Timeouts, retries and rate limits handled |
| R7 | As a reviewer I complete an official token flow | Local demo pages obtain tokens from reCAPTCHA, hCaptcha and Turnstile test keys; the server validates each with the provider's `siteverify` endpoint |
| R8 | As a reviewer I see bad tokens handled | Missing, malformed, invalid, expired and reused tokens each return a distinct structured error and a fitting HTTP status. Nothing crashes |
| R9 | As a reviewer I can confirm secrets are safe | Secrets come only from the environment. Tokens and secrets never appear in logs or responses; only a short hash fingerprint does. A test asserts this |
| R10 | As a reviewer I use a simple dashboard | Upload or select a challenge, run it, see the result. Shows total attempts, successes, failures, processing time, challenge type, method and estimated cost |
| R11 | As a reviewer I run a batch | A dataset folder is processed; exact-match accuracy, character accuracy, mean confidence and p50/p95 latency are reported |
| R12 | As a reviewer I import the site CSV | Rows are stored; a detection-only scan runs; accuracy and a confusion matrix against the labelled column are shown; request logs show GETs only |
| R13 | As an integrator I call a reusable API | Versioned JSON endpoints, OpenAPI docs, API-key header |
| R14 | As a reviewer I can run it from the README | Setup steps, environment variables, architecture note, cost table and a script for the 3–5 minute video |

## Non-functional

- Every outbound call has a timeout; provider calls get one retry.
- A failure in one solver or provider never takes down the request; it is recorded and reported.
- Modules are independent: recognition, solvers, verification, sites, API, dashboard.

## Out of scope

Solving or bypassing CAPTCHAs on third-party sites, harvesting tokens from them, filling or submitting any contact form.
