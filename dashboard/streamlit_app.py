"""
CAPTCHA Tool dashboard (Streamlit). Talks to the FastAPI backend; run both:

    uvicorn app.main:app --port 8000
    streamlit run dashboard/streamlit_app.py
"""
from __future__ import annotations

import sys
from pathlib import Path

# `streamlit run` puts only this file's folder on sys.path, not the project root.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import altair as alt
import pandas as pd
import streamlit as st

from dashboard import samples
from dashboard.client import ApiClient, ApiResult

st.set_page_config(page_title="CAPTCHA Tool", page_icon=":lock:", layout="wide", initial_sidebar_state="expanded",
                   menu_items={"Get help": None, "Report a bug": None,
                               "About": "CAPTCHA Solving & Verification Tool, Technology Brainz Assignment 01."})
api = ApiClient()
PAGES = {"Overview": ":material/dashboard:", "Solve": ":material/lock_open:", "Verify": ":material/verified:",
         "Detect": ":material/search:", "Batch": ":material/inventory_2:", "Sites": ":material/language:"}

# Presentation only. Hides the Streamlit menu, deploy button, footer and top decoration (this is a single-purpose
# tool, not a multi-page app builder), and gives metrics a card look that works in both light and dark themes.
STYLE = """
<style>
#MainMenu, footer, [data-testid="stMainMenu"], [data-testid="stAppDeployButton"],
[data-testid="stDecoration"], [data-testid="stStatusWidget"] { display: none !important; }
[data-testid="stVegaLiteChart"] ~ [data-testid="stElementToolbar"],
[data-testid="stElementContainer"]:has([data-testid="stVegaLiteChart"]) [data-testid="stElementToolbar"] { display: none !important; }
header[data-testid="stHeader"] { background: transparent; }
.block-container { padding-top: 1rem; padding-bottom: 3rem; max-width: 1320px; }
h1 { font-size: 1.85rem !important; font-weight: 650; margin-bottom: 0 !important; padding-bottom: .15rem !important; }
h3 { font-size: 1.1rem !important; font-weight: 600; margin-top: 1.4rem !important; padding-bottom: .3rem !important; }
[data-testid="stMetric"] { background: rgba(128,128,128,.07); border: 1px solid rgba(128,128,128,.22);
    border-radius: 10px; padding: .65rem .9rem; min-height: 5.6rem; }
[data-testid="stMetricLabel"], [data-testid="stMetricLabel"] * { white-space: normal !important; overflow: visible !important;
    text-overflow: clip !important; font-size: .74rem !important; letter-spacing: .04em; text-transform: uppercase; opacity: 1; }
[data-testid="stMetricLabel"] [data-testid="stTooltipHoverTarget"] svg { opacity: 1; }
[data-testid="stMetricValue"], [data-testid="stMetricValue"] * { font-size: 1.45rem !important; line-height: 1.3;
    white-space: normal !important; overflow-wrap: anywhere; }
[data-testid="stCaptionContainer"] { opacity: .8; }
[data-testid="stSidebarContent"] { overflow: hidden !important; }
[data-testid="stSidebarUserContent"] { padding-top: 0.1rem; margin-top: -2.75rem; height: 100%; }
[data-testid="stSidebarUserContent"] [data-testid="stVerticalBlock"] { gap: .2rem; display: flex; flex-direction: column; height: 100%; }

[data-testid="stSidebar"] [data-testid^="stBaseButton"] { justify-content: flex-start; min-height: 2.5rem; padding-left: .75rem; }
[data-testid="stSidebar"] [data-testid^="stBaseButton"] > div, [data-testid="stSidebar"] [data-testid^="stBaseButton"] p {
    justify-content: flex-start; text-align: left; }
[data-testid="stSidebar"] [data-testid^="stBaseButton"] { transition: background-color .15s ease; }
.nav-title { font-size: 1.1rem; font-weight: 650; padding: 0 .75rem 2.4rem; }

/* A checkbox is shorter than a dropdown; lift it so their midlines match when bottom-aligned. */
.st-key-sites_mismatch { margin-bottom: 0.875rem; }
</style>
"""
st.markdown(STYLE, unsafe_allow_html=True)

with st.sidebar:
    st.markdown('<div class="nav-title">CAPTCHA Tool</div>', unsafe_allow_html=True)
    st.session_state.setdefault("page", "Overview")
    for name, icon in PAGES.items():
        if st.button(name, icon=icon, key=f"nav_{name}", use_container_width=True,
                     type="primary" if st.session_state["page"] == name else "tertiary"):
            st.session_state["page"] = name
            st.rerun()
    page = st.session_state["page"]


def pct(x) -> str:
    return f"{x:.1%}" if x is not None else "n/a"


def usd(x: float | None) -> str:
    return "$0" if not x else f"${x:.6f}"


def confidence_badge(value: float) -> str:
    color = "#2e7d4f" if value >= 0.9 else "#b7791f" if value >= 0.6 else "#b83c4a"
    return (f"<span style='background:{color};color:#fff;border-radius:999px;padding:.15rem .6rem;"
            f"font-size:.9rem;font-weight:600'>{value:.1%}</span>")


def ms(x) -> str:
    return f"{x:,.0f} ms" if x is not None else "n/a"


def api_problem(res: ApiResult) -> bool:
    """Show a message and return True when the call did not reach a usable answer."""
    if res.status == 0:
        st.error("The API is not reachable. Start it with: uvicorn app.main:app --port 8000")
        return True
    if res.status == 401:
        st.error("The API rejected the key. Set API_KEY (same value for both processes).")
        return True
    return False


def _bars(df: pd.DataFrame, color: alt.Color | alt.value) -> alt.Chart:
    """Bars with horizontal category labels (st.bar_chart cannot set the label angle)."""
    return (alt.Chart(df).mark_bar()
            .encode(x=alt.X("key:N", sort=None, title=None, axis=alt.Axis(labelAngle=0, labelLimit=220, ticks=False)),
                    y=alt.Y("count:Q", title=None, axis=alt.Axis(tickMinStep=1, format="d")),
                    color=color, tooltip=["key", "series", "count"])
            .properties(height=260))


def outcome_chart(rows: list[dict]) -> None:
    """Stacked successful + failed, so the bar height is the total."""
    df = pd.DataFrame(rows)
    df["failed"] = df["total"] - df["successful"]
    long = df.melt(id_vars="key", value_vars=["successful", "failed"], var_name="series", value_name="count")
    scale = alt.Scale(domain=["successful", "failed"], range=["#2e9e5b", "#d6455d"])
    st.altair_chart(_bars(long, alt.Color("series:N", scale=scale, legend=alt.Legend(title=None, orient="bottom"))),
                    use_container_width=True)


def count_chart(rows: list[dict]) -> None:
    """One series of counts per key, e.g. verification outcomes."""
    df = pd.DataFrame(rows).rename(columns={"total": "count"})
    df["series"] = "total"
    st.altair_chart(_bars(df, alt.value("#7cc4ff")), use_container_width=True)


def header(title: str, subtitle: str) -> None:
    st.title(title)
    st.caption(subtitle)


def demo_links() -> None:
    """Links to the local provider demo pages that hand out real test tokens."""
    cols = st.columns(3)
    for col, (label, slug) in zip(cols, [("reCAPTCHA demo page", "recaptcha"), ("hCaptcha demo page", "hcaptcha"),
                                         ("Turnstile demo page", "turnstile")]):
        col.link_button(label, f"{api.base}/demo/{slug}.html", use_container_width=True)


PAGE_SIZES = [10, 25, 50]
HEADINGS = {"id": "ID", "ms": "ms", "cost_usd": "Cost (USD)", "http_status": "HTTP", "p50": "p50", "p95": "p95"}


def _heading(name: str) -> str:
    return HEADINGS.get(name) or name.replace("_", " ").capitalize()


@st.fragment
def _paged(df: pd.DataFrame, key: str) -> None:
    """A compact table with a rows-per-page selector and Previous/Next. A fragment, so changing the page re-runs only
    this table and does not call the API again."""
    size_key, page_key = f"tbl_{key}_size", f"tbl_{key}_page"
    n = len(df)
    size = int(st.session_state.get(size_key, PAGE_SIZES[0]))
    pages = max(1, -(-n // size))
    page = min(max(int(st.session_state.get(page_key, 1)), 1), pages)
    st.session_state[page_key] = page

    start = (page - 1) * size
    st.dataframe(df.iloc[start:start + size], hide_index=True)

    def go(delta: int) -> None:
        st.session_state[page_key] = min(max(st.session_state[page_key] + delta, 1), pages)

    info, per, prev, label, nxt = st.columns([2.4, 1.3, 0.9, 1.1, 0.9], vertical_alignment="center")
    info.caption(f"Rows {start + 1}-{min(start + size, n)} of {n}")
    per.selectbox("Rows per page", PAGE_SIZES, key=size_key, label_visibility="collapsed",
                  format_func=lambda s: f"{s} per page")
    prev.button("Previous", key=f"tbl_{key}_prev", on_click=go, args=(-1,), disabled=page <= 1, use_container_width=True)
    # Flex row the height of a button, so the text centres on the buttons' midline (markdown adds paragraph margins).
    label.markdown(f"<div style='display:flex;align-items:center;justify-content:center;height:2.5rem;margin:0 0 1rem'>"
                   f"Page {page} / {pages}</div>", unsafe_allow_html=True)
    nxt.button("Next", key=f"tbl_{key}_next", on_click=go, args=(1,), disabled=page >= pages, use_container_width=True)


def table(rows: list[dict], columns: list[str] | None = None, key: str | None = None) -> None:
    """Show rows as a table. Pass `key` for lists that can grow long: they are paged instead of scrolling forever."""
    if not rows:
        st.caption("Nothing recorded yet.")
        return
    df = pd.DataFrame(rows)
    if columns:
        df = df[[c for c in columns if c in df.columns]]          # tolerate absent columns
    df = df.rename(columns=_heading)
    if key and len(df) > PAGE_SIZES[0]:
        _paged(df, key)
    else:
        st.dataframe(df, hide_index=True)


# ── Overview ─────────────────────────────────────────────────────────────────
def overview() -> None:
    header("Overview", "Live totals from the API's database: solve attempts, token verifications and the site benchmark.")
    res = api.stats()
    if api_problem(res):
        return
    a, v, s = res.data["attempts"], res.data["verifications"], res.data["sites"]

    st.subheader("Solve attempts")
    c = st.columns(5)
    c[0].metric("Total", a["total"])
    c[1].metric("Successful", a["successful"])
    c[2].metric("Failed", a["failed"])
    c[3].metric("Success rate", pct(a["success_rate"]))
    c[4].metric("Accuracy (labelled)", pct(a["labelled_accuracy"]),
                help="Exact match on attempts where the true answer was supplied")
    c = st.columns(5)
    c[0].metric("Latency p50", ms(a["latency_ms"]["p50"]))
    c[1].metric("Latency p95", ms(a["latency_ms"]["p95"]))
    c[2].metric("Avg confidence", pct(a["avg_confidence"]))
    c[3].metric("Cost / attempt", usd(a["avg_cost_usd"]))
    c[4].metric("Total cost", usd(a["total_cost_usd"]))

    left, right = st.columns(2)
    with left, st.container(border=True):
        st.caption("Outcomes by challenge type")
        if a["by_type"]:
            outcome_chart(a["by_type"])
        else:
            st.caption("Nothing recorded yet.")
    with right, st.container(border=True):
        st.caption("Outcomes by method")
        if a["by_method"]:
            outcome_chart(a["by_method"])
        else:
            st.caption("Nothing recorded yet.")

    st.subheader("Solvers and cost")
    sv = api.solvers()
    if sv.ok:
        table(sv.data["solvers"], ["name", "kind", "enabled", "threshold", "advisory_only", "cost", "reason"])
        vis = next((s for s in sv.data["solvers"] if s["name"] == "vision_api"), None)
        if vis and vis.get("budget"):
            b = vis["budget"]
            c = st.columns(3)
            c[0].metric("Vision spend today", usd(b["spent_today_usd"]))
            c[1].metric("Budget remaining", usd(b["remaining_usd"]), help=f"Daily limit ${b['limit_usd_per_day']:.2f}")
            c[2].metric("Calls in the last minute", f"{b['calls_last_minute']} / {b['max_calls_per_minute']}")
    if a.get("cost_by_method"):
        st.caption("Cost by method")
        table(a["cost_by_method"], ["key", "calls", "cost_usd"])

    st.subheader("Token verifications")
    c = st.columns(4)
    c[0].metric("Total", v["total"])
    c[1].metric("Accepted", v["successful"])
    c[2].metric("Rejected", v["failed"])
    c[3].metric("Latency p50", ms(v["latency_ms"]["p50"]))
    if v["by_reason"]:
        st.caption("Outcomes")
        count_chart(v["by_reason"])

    st.subheader("Recent attempts")
    recent = api.attempts(200)
    if recent.ok:
        st.caption("Newest first, up to the latest 200.")
        table(recent.data["items"], ["id", "challenge_type", "method", "predicted", "expected",
                                     "success", "confidence", "processing_ms", "cost_usd", "error"], key="attempts")

    st.subheader("Site detection benchmark")
    st.caption(f"{s['total']} sites imported, {s['scanned']} analysed. The Sites page has the accuracy report.")
    st.subheader("Provider demo pages")
    st.caption("Obtain a real test token from a provider's widget, then verify it on the Verify page.")
    demo_links()


# ── Solve ────────────────────────────────────────────────────────────────────
def render_solve(res: ApiResult, expected: str | None) -> None:
    d = res.data
    if api_problem(res):
        return
    problem = d.get("error") or d.get("reason")
    if res.status in (400, 413, 429, 503):
        st.error(f"Request refused ({res.status}): {problem}. {d.get('detail') or ''}")
        return
    if res.status == 422 and d.get("challenge_type") == "unsupported":
        why = "; ".join((d.get("detail") or {}).get("classifier_reasons") or []) or problem
        st.warning(
            f"**Not solved: this is not a challenge the solver supports.** Classifier: {why}. "
            "The local solver reads generated text challenges (about 160×48 px) and generated arithmetic strips. "
            "Grids, photos and other sizes are never sent to a solver, because it would answer them with confidence "
            "and be wrong. Try a held-out or math sample, or a challenge from `python ml/generate_dataset.py`.")
    elif res.status == 422:
        st.warning(f"Not solved: {problem}. The image was recognised as **{d.get('challenge_type')}**; "
                   "unreadable challenges are never sent to a solver.")
    elif d.get("accepted"):
        if expected and not d.get("success"):
            st.error(f"Wrong answer: **{d.get('text')}** (expected {expected}). The solver accepted it at "
                     f"{d.get('confidence', 0):.1%} confidence: a confident mistake.")
        else:
            match = " (matches the expected answer)" if expected else ""
            st.success(f"Answer: **{d.get('text')}**{match}")
    else:
        st.warning(f"No answer accepted ({ATTEMPT_ERRORS.get(problem, problem)}). Best candidate: {d.get('text')!r}"
                   + (f", expected {expected}" if expected else ""))

    c = st.columns([1.7, 1.4, 1, 1, 1])
    with c[0].container(border=True):
        st.markdown("<span title=\"The classifier's guess, then how sure it is.\" style='font-size:.75rem;"
                    "letter-spacing:.05em;opacity:.7;cursor:help'>CHALLENGE TYPE <span style='color:#c9ced6;opacity:1'>ⓘ</span></span>", unsafe_allow_html=True)
        st.markdown(f"**{d.get('challenge_type', 'n/a')}** {confidence_badge(d.get('classifier_confidence', 0))}",
                    unsafe_allow_html=True)
    with c[1].container(border=True):
        st.caption("METHOD")
        st.markdown(f"**{d.get('method') or 'none'}**")
    with c[2].container(border=True):
        st.caption("CONFIDENCE")
        st.markdown(confidence_badge(d.get("confidence", 0)), unsafe_allow_html=True)
    with c[3].container(border=True):
        st.caption("TIME")
        st.markdown(f"**{ms(d.get('total_ms'))}**")
    with c[4].container(border=True):
        st.caption("COST")
        st.markdown(f"**{usd(d.get('cost_usd'))}**")
    if d.get("detail", {}).get("external_call"):
        det = d["detail"]
        st.info(f"This answer came from the third-party {det.get('provider')} model {det.get('model')}: the image was sent "
                f"to the provider ({det.get('input_tokens')} input and {det.get('output_tokens')} output tokens, "
                f"billed at the rates configured for the model).")
    if len(d.get("attempts") or []) > 1:
        st.markdown(f"Fallback chain: {solver_chain(d['attempts'])}")
    if d.get("attempts"):
        st.caption("Solvers tried, in order")
        table(d["attempts"], ["method", "text", "confidence", "ms", "cost_usd", "accepted", "error"])


HARD_NOTES = {
    "Medium": "Wave, tilt and speckle beyond the training data. Measured: the CRNN still reads 19 of 20 (15 falls back).",
    "Hard": "Adds a thick crossing curve. Measured: the CRNN is rejected on 19 of 20 and the chain falls back to "
            "segmentation; 16 is a confident wrong answer.",
    "Very hard": "Every letter skewed and tilted its own way, heavy warp, three curves and dense speckle: hard for a "
                 "person too. Measured: the CRNN is rejected on all 20 and the chain falls back to segmentation.",
    "Skewed": "Each letter sheared and tilted on its own, still readable for a person. Measured: 18 confidently wrong, "
              "2 fall back (9, 11): a known limit, the CRNN's confidence does not flag these mistakes.",
    "Hatched": "Crossed diagonal lines over the text. Measured: 12 read, 5 confidently wrong (8, 10, 13, 18, 20), "
               "2 fall back (1, 15), 1 refused by the quality gate as too noisy (2).",
    "Crowded": "Letters squeezed until they overlap. Measured: 15 read, 5 confidently wrong (6, 9, 12, 13, 15).",
    "Ink blob": "Dark blobs with the letters inverted inside. Measured: 6 fall back (3, 9, 11, 12, 13, 15), "
                "14 confidently wrong: a known limit, the CRNN's confidence does not flag these mistakes.",
}
ATTEMPT_ERRORS = {
    "low_confidence": "confidence below its threshold",
    "implausible_length": "answer length outside 4-6 characters",
    "degenerate_output": "one repeated character",
    "advisory_only": "advisory only, never accepted on its own",
    "timeout": "timed out",
}


def solver_chain(attempts: list[dict]) -> str:
    steps = []
    for a in attempts:
        err = a.get("error") or ""
        why = "accepted" if a.get("accepted") else ATTEMPT_ERRORS.get(err, err or "rejected")
        steps.append(f"**{a['method']}** ({why})")
    return " → ".join(steps)


def solve_page() -> None:
    header("Solve a CAPTCHA", "Classify an image, run the solver chain and see every step: type, method, confidence, time and cost.")
    image, name, expected, challenge = None, "challenge.png", "", None
    source = st.radio("Image source", ["Upload", "Held-out sample", "Hard sample", "Math sample", "Unsupported sample (grid)"],
                      horizontal=True, key="solve_source")
    controls, preview = st.columns([1.3, 1], gap="large")

    with controls:
        if source == "Upload":
            up = st.file_uploader("Challenge image", type=["png", "jpg", "jpeg", "bmp", "webp"], key="solve_upload")
            if up:
                image, name = up.getvalue(), up.name
            expected = st.text_input("Expected answer (optional, to measure accuracy)", key="solve_expected_up")
        elif source == "Held-out sample":
            options = samples.test_samples()
            if not options:
                st.info("No generated dataset found. Run `python ml/generate_dataset.py` first.")
            else:
                pick = st.selectbox("Sample (never seen in training)", options, format_func=lambda o: o[0], key="solve_pick")
                mode = st.selectbox("Degrade the image", samples.DEGRADATIONS, key="solve_degrade",
                                    help="Shows how the quality gates and the pipeline react to bad input")
                image, name, expected = samples.degrade(samples.load_sample(pick[0]), mode), pick[0], pick[1]
        elif source == "Hard sample":
            level = st.selectbox("Difficulty", list(samples.HARD_LEVELS), index=1, key="solve_hard_level")
            index = st.number_input("Sample", 1, samples.HARD_COUNT, 1, key="solve_hard_index")
            image, expected = samples.hard_sample(level, int(index) - 1)
            name = f"{level.lower().replace(' ', '_')}_{int(index):02d}.png"
            st.caption(f"True text: **{expected}**. {HARD_NOTES[level]}")
        elif source == "Math sample":
            seed = st.number_input("Seed", 0, 10_000, 1, key="solve_seed")
            image, expr, answer = samples.math_sample(int(seed))
            name, expected = "math.png", str(answer)
            st.caption(f"Generated expression: {expr}")
        else:
            image, name = samples.grid_sample(), "grid.png"
            st.caption("A 3x3 image grid: classified as unsupported and never sent to a solver.")

        override = st.selectbox("Challenge type", ["auto-detect", "text", "math"], key="solve_type")
        challenge = None if override == "auto-detect" else override

        allow_external = True
        sv = api.solvers()
        vision = next((s for s in sv.data.get("solvers", []) if s["name"] == "vision_api"), None) if sv.ok else None
        if vision and vision.get("enabled"):
            allow_external = st.checkbox(
                f"Allow the paid {vision['provider']} vision solver as a last resort (the image is sent to the provider)",
                value=True, key="solve_external")
        elif vision:
            st.caption(f"Third-party vision solver: off ({vision.get('reason')}). Everything stays local and free.")

        go = st.button("Solve", type="primary", disabled=image is None, key="solve_go")

    with preview:
        if image:
            with st.container(border=True):
                st.image(image, caption=name, width="stretch")
        else:
            st.caption("Choose or upload an image to preview it here.")

    if go:
        with st.spinner("Solving..."):
            st.session_state["solve_result"] = (
                api.solve(image, name, expected or None, challenge, allow_external), expected or None)
    if "solve_result" in st.session_state:
        st.divider()
        render_solve(*st.session_state["solve_result"])


# ── Verify ───────────────────────────────────────────────────────────────────
VERDICT_TEXT = {
    "ok": ("success", "Token accepted."),
    "reused_token": ("warning", "Replay rejected: this token was already verified."),
    "expired_token": ("warning", "Token expired (or already used elsewhere; providers use one code for both)."),
    "invalid_token": ("warning", "The provider rejected the token."),
    "missing_token": ("warning", "No token supplied."),
    "malformed_token": ("warning", "That does not look like a token."),
    "provider_unavailable": ("error", "No verdict from the provider; nothing was accepted."),
    "server_misconfigured": ("error", "Server-side secret or sitekey problem."),
}


def verify_page() -> None:
    header("Verify a provider token",
           "Server-side check of a token against the provider's siteverify endpoint. Test keys only; the token is never stored or logged.")
    st.caption("Get a token from a provider's test widget:")
    demo_links()
    st.caption("or fill in a published test value, or run `python scripts/acquire_token.py --provider recaptcha --replay` "
               "to do it end to end.")
    col = st.columns(3)
    col[0].button("Use Turnstile test token", on_click=lambda: st.session_state.update(
        vtoken="XXXX.DUMMY.TOKEN.XXXX", vprovider="turnstile"), use_container_width=True)
    col[1].button("Use hCaptcha test token", on_click=lambda: st.session_state.update(
        vtoken="10000000-aaaa-bbbb-cccc-000000000001", vprovider="hcaptcha"), use_container_width=True)
    col[2].button("Use a nonsense token", on_click=lambda: st.session_state.update(vtoken="FAKE-TOKEN"),
                  use_container_width=True)

    with st.container(border=True):
        provider = st.selectbox("Provider", ["recaptcha", "hcaptcha", "turnstile"], key="vprovider")
        token = st.text_input("Token", type="password", key="vtoken",
                              help="Sent to the API and verified server-side; only a fingerprint is stored")
        opt = st.columns(2)
        max_age = opt[0].number_input("Max token age (s), 0 = default", 0, 86_400, 0, key="vage")
        scenario = None
        if provider == "turnstile":
            label = opt[1].selectbox("Turnstile dummy secret", ["configured secret", "pass", "fail", "spent"], key="vscenario",
                                     help="Published Cloudflare test secrets: always pass / always fail / duplicate token")
            scenario = None if label == "configured secret" else label

    if st.button("Verify", type="primary", key="verify_go"):
        with st.spinner("Calling the provider's siteverify..."):
            st.session_state["verify_result"] = api.verify(provider, token, int(max_age) or None, scenario)
    res: ApiResult | None = st.session_state.get("verify_result")
    if res is not None and not api_problem(res):
        d = res.data
        if d.get("reason") == "invalid_request":
            st.error(f"Invalid request: {d.get('errors')}")
        else:
            kind, text = VERDICT_TEXT.get(d.get("reason"), ("warning", d.get("reason", "unknown")))
            getattr(st, kind)(f"{text}  (HTTP {res.status})")
            c = st.columns([1.3, 1, 1.6, 1.3])
            c[0].metric("Reason", d.get("reason", "n/a"))
            c[1].metric("Latency", ms(d.get("latency_ms")))
            c[2].metric("Token fingerprint", d.get("token_fingerprint") or "n/a")
            c[3].metric("Provider codes", ", ".join(d.get("provider_error_codes") or []) or "none")
            if d.get("detail"):
                st.caption(d["detail"])
            if d.get("retry_possible"):
                st.info("Retrying is possible.")

    st.subheader("Recent verifications")
    recent = api.verifications(200)
    if recent.ok:
        table(recent.data["items"], ["id", "provider", "ok", "reason", "http_status", "latency_ms", "token_fingerprint"],
              key="verifications")


# ── Detect ───────────────────────────────────────────────────────────────────
def detect_page() -> None:
    header("Detect the CAPTCHA provider on a page",
           "Paste page HTML or pick a fixture. Static analysis only: nothing is fetched or interacted with.")
    mode = st.radio("Input", ["Fixture", "Paste HTML", "Upload file"], horizontal=True, key="detect_mode")
    html = ""
    if mode == "Fixture":
        names = samples.detector_fixtures()
        if names:
            pick = st.selectbox("Fixture", names, key="detect_pick")
            html = samples.load_fixture(pick)
            with st.expander("Show HTML"):
                st.code(html, language="html")
    elif mode == "Paste HTML":
        html = st.text_area("HTML", height=220, key="detect_html")
    else:
        up = st.file_uploader("HTML file", type=["html", "htm"], key="detect_upload")
        html = up.getvalue().decode("utf-8", errors="replace") if up else ""

    if st.button("Detect", type="primary", disabled=not html, key="detect_go"):
        st.session_state["detect_result"] = api.detect(html)
    res: ApiResult | None = st.session_state.get("detect_result")
    if res is not None and not api_problem(res):
        d = res.data
        if res.status != 200:
            st.error(f"{d.get('reason')}: {d.get('detail') or d.get('errors')}")
            return
        if d["provider"] == "none":
            st.info("No CAPTCHA provider found." + (" The page mentions one, so a browser-rendered check may find more."
                                                    if d.get("needs_render") else ""))
        else:
            st.success(f"{d['provider']} ({d['variant']})")
        c = st.columns(4)
        c[0].metric("Provider", d["provider"])
        c[1].metric("Variant", d.get("variant") or "n/a")
        c[2].metric("Confidence", f"{d['confidence']:.0%}")
        c[3].metric("Benchmark label", d["csv_label"])
        st.caption(f"Sitekey: {d.get('sitekey') or 'not found'}")
        if d.get("evidence"):
            st.caption("Evidence")
            table(d["evidence"], ["kind", "weight", "detail"], key="evidence")
        if d.get("others"):
            st.caption("Other providers on the same page")
            table(d["others"])


# ── Batch ────────────────────────────────────────────────────────────────────
def render_batch(d: dict) -> None:
    c = st.columns(6)
    c[0].metric("Images", d["n"])
    c[1].metric("Exact match", pct(d["exact_match"]))
    c[2].metric("Character accuracy", pct(d["char_accuracy"]), help="1 - edit distance / answer length")
    c[3].metric("Coverage", pct(d["coverage"]), help="Share of images where a solver's answer was accepted")
    c[4].metric("Latency p50", ms(d["latency_ms"]["p50"]))
    c[5].metric("Latency p95", ms(d["latency_ms"]["p95"]))
    c = st.columns(4)
    c[0].metric("Mean confidence", pct(d["mean_confidence"]))
    c[1].metric("Precision when accepted", pct(d["precision_on_accepted"]))
    c[2].metric("Confident but wrong", d["confident_wrong"],
                help="Accepted answers that were wrong: confidence does not flag these")
    c[3].metric("Total time", f"{d['total_seconds']} s")
    if d.get("truncated"):
        st.warning("The time budget ran out; only part of the dataset was processed.")
    if d["by_type"]:
        st.caption("By challenge type")
        table([{"type": k, **v} for k, v in d["by_type"].items()])
    if d["by_error"]:
        st.caption("Images not answered, by reason")
        table([{"reason": k, "count": v} for k, v in d["by_error"].items()])
    if d["worst"]:
        st.caption("Wrong answers, most confident first")
        table(d["worst"], key="worst")


def batch_page() -> None:
    header("Batch accuracy", "Run the solver over a labelled dataset or your own images and measure accuracy, coverage and latency.")
    ds = api.batch_datasets()
    if api_problem(ds):
        return
    datasets = ds.data.get("datasets", [])
    st.subheader("Run a dataset")
    if not datasets:
        st.info("No datasets found. Run `python ml/generate_dataset.py` first.")
    else:
        names = {f"{d['name']} ({d['count']} images)": d for d in datasets}
        pick = names[st.selectbox("Dataset", list(names), key="batch_dataset",
                                  index=next((i for i, k in enumerate(names) if k.startswith("test ")), 0))]
        if pick.get("note"):
            st.warning(pick["note"])
        limit = st.number_input("Images to run", 1, 1000, min(200, pick["count"]) or 1, key="batch_limit")
        if st.button("Run batch", type="primary", key="batch_go"):
            with st.spinner("Solving..."):
                st.session_state["batch_result"] = api.batch_run(pick["name"], int(limit))
        res: ApiResult | None = st.session_state.get("batch_result")
        if res is not None and not api_problem(res):
            if res.ok:
                render_batch(res.data)
            else:
                st.error(f"{res.data.get('reason')}: {res.data.get('detail') or res.data.get('errors')}")

    st.subheader("Or upload your own images")
    images = st.file_uploader("Images (up to 100)", type=["png", "jpg", "jpeg", "bmp", "webp"],
                              accept_multiple_files=True, key="batch_images")
    labels = st.file_uploader("Optional labels CSV (columns: filename,text)", type=["csv"], key="batch_labels")
    if st.button("Run uploaded images", disabled=not images, key="batch_up_go"):
        with st.spinner("Solving..."):
            st.session_state["batch_upload"] = api.batch_upload(
                [(f.name, f.getvalue()) for f in images], labels.getvalue() if labels else None)
    up: ApiResult | None = st.session_state.get("batch_upload")
    if up is not None and not api_problem(up):
        if up.ok:
            render_batch(up.data)
            st.caption("Per-image results")
            table(up.data["results"], ["name", "expected", "predicted", "type", "accepted", "confidence", "ms", "error"],
                  key="batch_results")
        else:
            st.error(f"{up.data.get('reason')}: {up.data.get('detail') or up.data.get('errors')}")

    st.subheader("History")
    runs = api.batch_runs(100)
    if runs.ok:
        table(runs.data["items"], ["id", "created_at", "dataset", "n", "exact_match", "char_accuracy",
                                   "mean_confidence", "latency_p50_ms", "latency_p95_ms"], key="batch_history")


# ── Sites ────────────────────────────────────────────────────────────────────
def render_scan_status() -> None:
    st.session_state.setdefault("scan_was_running", False)
    res = api.sites_scan_status()
    d = res.data
    running = d.get("status") == "running"
    if d.get("status") == "idle" or not res.ok:
        st.caption("No scan has been run in this API session.")
    else:
        st.progress(d["done"] / d["total"] if d["total"] else 0.0,
                    text=f"{d['status']}: {d['done']} / {d['total']} pages, {d['elapsed_s']} s")
        if d.get("by_status"):
            st.caption("Results so far: " + ", ".join(f"{k} {v}" for k, v in sorted(d["by_status"].items())))
        if d.get("error"):
            st.error(f"Scan stopped with an error: {d['error']}")
    if running:
        st.button("Cancel scan", key="scan_cancel", on_click=api.sites_scan_cancel)
    if st.session_state["scan_was_running"] and not running:
        st.session_state["scan_was_running"] = False
        st.rerun()                                     # scan just finished: refresh the report below
    st.session_state["scan_was_running"] = running


def import_message(res: ApiResult) -> tuple[str, str]:
    """Plain-language outcome of a CSV import: (st method name, text)."""
    if not res.ok:
        return "error", f"Import failed. {res.data.get('reason')}: {res.data.get('detail')}"
    d = res.data
    changed = d["inserted"] + d["updated"]
    rejected = f" {d['rows_rejected']} rows were rejected." if d["rows_rejected"] else ""
    if changed:
        return "success", (f"Imported {d['inserted']} new and updated {d['updated']} sites; "
                           f"{d['unchanged']} were already there.{rejected}")
    if d["unchanged"]:
        return "info", (f"Already imported: all {d['unchanged']} sites are in the database and nothing changed."
                        f"{rejected}")
    return "warning", f"No sites found in the file.{rejected}"


def sites_page() -> None:
    header("Site detection benchmark",
           "Read-only detection of the CAPTCHA provider on the supplied contact pages, scored against the CSV labels.")
    st.info(
        "**Read-only.** The scan makes one plain GET of each contact page (plus that site's robots.txt, which it obeys) "
        "and analyses the returned HTML. It never fills a field, submits a form, solves a challenge, requests a token, "
        "runs the page's JavaScript or drives a browser, so widgets injected by JavaScript are not seen. "
        "Results are compared with the CSV's labelled `Captcha` column.")

    st.subheader("1. Import")
    cols = st.columns(2, vertical_alignment="top")
    cols[0].markdown('<div style="margin:0;height:3rem;line-height:3rem"><b>Bundled dataset</b></div>',
                     unsafe_allow_html=True)   # same label row height as the uploader's label, so the controls line up
    if cols[0].button("Import the bundled CSV (200 sites)", key="sites_import_bundled", use_container_width=True):
        with st.spinner("Importing..."):
            st.session_state["sites_import"] = api.sites_import()
        st.toast(import_message(st.session_state["sites_import"])[1])
    up = cols[1].file_uploader("...or upload a CSV with a 'Contact Page' column", type=["csv"], key="sites_upload")
    if up is not None and st.button("Import uploaded CSV", key="sites_import_upload"):
        with st.spinner("Importing..."):
            st.session_state["sites_import"] = api.sites_import(up.getvalue())
        st.toast(import_message(st.session_state["sites_import"])[1])
    imp: ApiResult | None = st.session_state.get("sites_import")
    if imp is not None and not api_problem(imp):
        kind, text = import_message(imp)
        getattr(st, kind)(text)
        if imp.ok and imp.data["errors"]:
            st.caption("Rejected rows and why:")
            table(imp.data["errors"], key="import_errors")

    st.subheader("2. Scan")
    c = st.columns(3, vertical_alignment="bottom")
    limit = c[0].number_input("Pages to scan", 1, 500, 200, key="scan_limit")
    rescan = c[1].checkbox("Include pages already scanned", key="scan_rescan")
    ok = st.checkbox("I understand this sends real GET requests to the listed third-party sites "
                     "(one per page, plus robots.txt)", key="scan_confirm")
    if st.button("Start read-only scan", type="primary", disabled=not ok, key="scan_go"):
        r = api.sites_scan_start(int(limit), rescan)
        if r.status != 202:
            st.error(f"{r.data.get('reason')}: {r.data.get('detail')}")
        else:
            st.session_state["scan_was_running"] = True
    status = api.sites_scan_status()
    fragment = st.fragment(run_every=2 if status.data.get("status") == "running" else None)(render_scan_status)
    fragment()

    st.subheader("3. Accuracy")
    rep = api.sites_report()
    if api_problem(rep):
        return
    r = rep.data
    if not r.get("scanned"):
        st.caption(f"{r.get('total_sites', 0)} sites imported, none scanned yet.")
        return
    c = st.columns(6)
    c[0].metric("Analysed", f"{r['scanned']} / {r['total_sites']}", help="Pages fetched and analysed; the rest were refused or failed")
    c[1].metric("Exact accuracy", pct(r["exact_accuracy"]), help="invisible reCAPTCHA only when the markup says size=invisible")
    c[2].metric("...with v3 as invisible", pct(r.get("exact_accuracy_v3_as_invisible")),
                help="reCAPTCHA v3 shows only a badge, so a labeller may call it invisible. Reported separately, not tuned on.")
    c[3].metric("Provider-level accuracy", pct(r["family_accuracy"]), help="reCAPTCHA variants count as one provider")
    c[4].metric("Baseline", pct(r["baseline_majority_class"]["accuracy"]), help="Accuracy of always answering the most common label")
    c[5].metric("Not analysed", r["total_sites"] - r["scanned"])
    if r["scan_status"]:
        st.caption("Pages by scan outcome")
        table([{"outcome": k, "pages": v} for k, v in sorted(r["scan_status"].items(), key=lambda kv: -kv[1])])
    if r["js_injected_suspected"]:
        st.warning(f"{r['js_injected_suspected']} pages labelled with a CAPTCHA showed none in static HTML but mention a "
                   "provider: likely injected by JavaScript, which this read-only scan does not execute.")

    st.caption("Confusion matrix: rows are the CSV label, columns are what the scan detected")
    labels = r["labels"]
    st.dataframe(pd.DataFrame([[r["confusion_matrix"][t][p] for p in labels] for t in labels],
                              index=pd.Index(labels, name="label / detected"), columns=labels))
    st.caption("Per class (provider level)")
    table([{"class": k, **v} for k, v in r["per_class_family"].items()])
    st.caption("Per class (exact)")
    table([{"class": k, **v} for k, v in r["per_class_exact"].items()])
    if r["mismatches"]:
        st.caption(f"Mismatches (first {len(r['mismatches'])})")
        table(r["mismatches"], ["company", "contact_page", "label", "detected", "variant", "needs_render", "family_ok"],
              key="mismatches")

    st.subheader("All sites")
    f = st.columns(3, vertical_alignment="bottom")
    status_f = f[0].selectbox("Scan outcome", ["any", "ok", "unscanned", *sorted(r["scan_status"])], key="sites_status")
    label_f = f[1].selectbox("Label", ["any", *labels[:-1]], key="sites_label")
    mism = f[2].checkbox("Mismatches only", key="sites_mismatch")
    lst = api.sites_list(200, None if status_f == "any" else status_f, None if label_f == "any" else label_f, mism)
    if lst.ok:
        st.caption(f"{lst.data['total']} matching sites")
        table(lst.data["items"], ["company", "contact_page", "label", "scan_status", "detected", "variant",
                                  "needs_render", "http_status", "scan_ms", "error"], key="sites_list")


{
    "Overview": overview,
    "Solve": solve_page,
    "Verify": verify_page,
    "Detect": detect_page,
    "Batch": batch_page,
    "Sites": sites_page,
}[page]()
