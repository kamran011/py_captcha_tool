# Cost breakdown

Each number is tagged by how it is known: **measured** (run on this project), **derived** (arithmetic on measured values), **estimate** (not measured, formula shown), or **quoted** (from a published price page, with source and date).

## Per-challenge cost by method

| Method | Marginal cost per challenge | Basis |
|---|---|---|
| Local text solver (CRNN on ONNX Runtime) | **$0** (about 3.3 ms of CPU) | measured: p50 3.3 ms, p95 4.7 ms over 1,000 images, CPU only |
| Segmentation + kNN fallback | **$0** | measured, same run |
| Math solver | **$0** (about 2.4 ms) | measured over 300 images |
| Optional vision API, `claude-haiku-5-5` | about **$0.00003** per call | **estimate**, see below; actual figure appears per call once enabled |
| Token verification | **$0** here | only the providers' published test keys were used; production pricing and quotas are each provider's own and are not priced in this project |
| Site detection scan | **$0** | plain GETs of public pages |

There are no per-call charges anywhere in the default configuration. The only recurring cost a user can switch on is the vision solver, and it is guarded.

## Optional vision solver

Cost per call = input tokens x input price + output tokens x output price, using the token counts the provider reports for that call. Built-in prices (Anthropic first-party rates from the Claude API model table cached 2026-11-03; Haiku 5.5's rate applies to prompts up to 100K tokens, ours are tiny):

| Model | Input $/Mtok | Output $/Mtok |
|---|---|---|
| `claude-haiku-5-5` (default) | 0.10 | 0.50 |
| `claude-haiku-4-5` | 1.00 | 5.00 |
| `claude-sonnet-5-5` | 2.00 | 10.00 |
| `claude-opus-5-5` | 4.00 | 20.00 |

Any other model needs `VISION_PRICE_IN_PER_MTOK` / `VISION_PRICE_OUT_PER_MTOK`; without them the solver refuses to run.

**Estimate for one call (not measured):** the image is resized to 320 px wide (about 320 x 96 = 30,720 px, roughly 40 image tokens at the usual width x height / 750 rule) plus a prompt of about 110 tokens, so about 150-200 input tokens, and about 20 output tokens for the JSON reply. At Haiku 5.5 rates: (180 x 0.10 + 20 x 0.50) / 1,000,000 = **$0.000028**. Real token counts have not been measured because no API key was available; `RUN_LIVE=1 pytest tests/test_live_vision.py` (three calls, capped at 5 cents) measures them.

**Worst case used for budget reservation:** 1,500 input + 200 output tokens = $0.00025 per call. With the default daily budget of $1.00 that allows at least 4,000 calls a day; the per-minute cap (default 20) bounds the rate.

The solver only runs after the local models fail to give an accepted answer (about 0.3% of the synthetic test images, rising for harder inputs), so the realistic blended cost is a small fraction of the per-call figure.

## At scale (derived)

| Volume | Local model | Vision API (estimate) | Commercial solving service (quoted, see below) |
|---|---|---|---|
| 1,000 images | about 3 s CPU, $0 | about $0.03 | $0.50 - $1.00 |
| 1,000,000 images | about 1 CPU-hour, $0 | about $28 | $500 - $1,000 |

CPU time is 3.3 ms x volume on one thread of the development machine; the vision column applies the estimate above. One-off cost to produce the local model: about 45 minutes of CPU for training (20 epochs of roughly 2 minutes) plus generating the dataset; the trained 4 MB model ships in the repository, so a reviewer does not repeat it.

## Commercial solving services, for context only

These services were **not used** and nothing in this project calls them. They are listed because the assignment asks for a cost view of the alternatives. Prices are as displayed on the vendor pages on **2026-11-12**, are ranges (volume discounts apply), and change.

| Service (source) | Image / text CAPTCHA | reCAPTCHA v2 | Turnstile | hCaptcha |
|---|---|---|---|---|
| 2Captcha, https://2captcha.com/pricing | $0.5 - $1 per 1,000 | $1 - $2.99 per 1,000 | $1.45 per 1,000 | not listed |
| Anti-Captcha, https://anti-captcha.com/ | $0.5 - $0.7 per 1,000 | $0.95 - $2 per 1,000 | $2 per 1,000 | not listed |

CapSolver was also checked but its pricing pages returned HTTP 404, so it is omitted rather than guessed. Using such services against sites that have not agreed to it would breach those sites' protections and is exactly what assignment section 7 rules out.

## Reading the table

- For the synthetic style this project is built for, the local model is both the cheapest (no marginal cost) and measurably accurate (98.1% exact match), but it only exists for that style: a different CAPTCHA design would need a new dataset and training run.
- The vision API is the flexible option and still cheap per call, but it sends images to a third party, its accuracy here is unmeasured, and its confidence is uncalibrated until `ml/calibrate.py --vision N` is run.
- For image CAPTCHAs the quoted commercial prices are roughly 18 to 36 times the vision estimate ($0.50 - $1.00 per 1,000 against about $0.028), and they are outside the project's scope.
