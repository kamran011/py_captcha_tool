"""M3 tests: provider detector against fixture HTML, including false positives."""
from __future__ import annotations

from pathlib import Path

import pytest

from app.recognition.provider_detector import detect, normalise_label

FIX = Path("tests/fixtures/detector")
DEMO = Path("web/demo")

RK = "6LeIxAcTAAAAAJcZVRqyHh71UMIEGNQ_MXjiZKhI"
V3 = "6LdAbCdEfGhIjKlMnOpQrStUvWxYz0123456789_"
HK = "10000000-ffff-ffff-ffff-000000000001"
TK = "1x00000000000000000000AA"
TL = "0x4AAAAAAADemoLiveFormatKey01"


def load(name: str) -> str:
    return (FIX / f"{name}.html").read_text(encoding="utf-8")


# fixture, provider, variant, sitekey, csv_label
POSITIVE = [
    ("recaptcha_v2_checkbox", "recaptcha", "v2_checkbox", RK, "recaptcha"),
    ("recaptcha_v2_invisible", "recaptcha", "v2_invisible", RK, "recaptcha_invisible"),
    ("recaptcha_v2_explicit_invisible_js", "recaptcha", "v2_invisible", RK, "recaptcha_invisible"),
    ("recaptcha_v3", "recaptcha", "v3", V3, "recaptcha"),
    ("recaptcha_enterprise_v3", "recaptcha", "v3", V3, "recaptcha"),
    ("recaptcha_rendered_dom_iframe", "recaptcha", "v2_checkbox", RK, "recaptcha"),
    ("recaptcha_net_domain", "recaptcha", "v2_checkbox", RK, "recaptcha"),
    ("recaptcha_script_only", "recaptcha", "v2_unknown", None, "recaptcha"),
    ("recaptcha_v2_plus_v3_fallback", "recaptcha", "v2_checkbox", RK, "recaptcha"),
    ("recaptcha_dynamic_injection", "recaptcha", "v3", V3, "recaptcha"),
    ("hcaptcha_checkbox", "hcaptcha", "checkbox", HK, "hcaptcha"),
    ("hcaptcha_invisible", "hcaptcha", "invisible", HK, "hcaptcha"),
    ("hcaptcha_rendered_dom_iframe", "hcaptcha", "checkbox", HK, "hcaptcha"),
    ("turnstile_widget", "turnstile", "widget", TK, "turnstile"),
    ("turnstile_explicit_js", "turnstile", "widget", TL, "turnstile"),
]


@pytest.mark.parametrize("name,provider,variant,sitekey,label", POSITIVE)
def test_positive_fixtures(name, provider, variant, sitekey, label):
    d = detect(load(name))
    assert (d.provider, d.variant, d.sitekey, d.csv_label) == (provider, variant, sitekey, label)
    assert d.evidence and d.confidence >= 0.7 and d.error is None
    assert not d.needs_render


@pytest.mark.parametrize("name", [
    "none_plain_form", "none_text_mention", "none_commented_out", "none_lookalike_domains",
])
def test_no_false_positives(name):
    d = detect(load(name))
    assert d.provider == "none" and d.csv_label == "none" and d.sitekey is None


def test_needs_render_only_when_page_mentions_a_provider():
    assert detect(load("none_plain_form")).needs_render is False
    assert detect(load("none_text_mention")).needs_render is True


def test_enterprise_flag():
    assert detect(load("recaptcha_enterprise_v3")).extra.get("enterprise") is True
    assert not detect(load("recaptcha_v3")).extra.get("enterprise")


def test_multiple_providers_visible_widget_wins_and_others_reported():
    d = detect(load("multi_hcaptcha_widget_plus_global_recaptcha_v3"))
    assert d.provider == "hcaptcha" and d.sitekey == HK
    assert [o["provider"] for o in d.others] == ["recaptcha"]
    assert d.others[0]["variant"] == "v3" and d.others[0]["sitekey"] == V3


def test_visible_v2_widget_beats_global_v3_script_and_keeps_its_own_key():
    d = detect(load("recaptcha_v2_plus_v3_fallback"))
    assert d.variant == "v2_checkbox" and d.sitekey == RK


def test_lookalike_hosts_and_wrong_paths_are_rejected():
    html = load("none_lookalike_domains")
    assert detect(html).provider == "none"
    # right host, wrong path (Google Maps) must not count as reCAPTCHA
    assert detect('<script src="https://www.google.com/maps/api.js"></script>').provider == "none"


def test_commented_out_markup_ignored_but_live_markup_alongside_it_counts():
    live = load("none_commented_out").replace("</body>", f'<div class="h-captcha" data-sitekey="{HK}"></div><script src="https://js.hcaptcha.com/1/api.js"></script></body>')
    assert detect(live).provider == "hcaptcha"


def test_class_token_must_match_exactly():
    # the hidden response textarea uses a g-recaptcha-* class; it is not a widget
    d = detect('<textarea class="g-recaptcha-response"></textarea>')
    assert d.provider == "none"


def test_sitekey_must_have_the_provider_format():
    d = detect('<div class="g-recaptcha" data-sitekey="not-a-real-key"></div><script src="https://www.google.com/recaptcha/api.js"></script>')
    assert d.provider == "recaptcha" and d.sitekey is None


# ── robustness ───────────────────────────────────────────────────────────────
def test_truncated_markup_does_not_crash():
    d = detect(load("malformed_truncated"))
    assert d.provider in ("none", "recaptcha") and d.error is None


@pytest.mark.parametrize("bad", [None, "", b"", b"\xff\xfe\x00garbage", 12345, ["x"], "<" * 10_000])
def test_never_raises_on_odd_input(bad):
    d = detect(bad)  # type: ignore[arg-type]
    assert d.provider == "none"


def test_bytes_input_decoded():
    assert detect(load("hcaptcha_checkbox").encode("utf-8")).provider == "hcaptcha"


def test_huge_page_is_truncated_not_slow():
    big = "<p>x</p>" * 2_000_000 + load("hcaptcha_checkbox")   # real markup past the cap
    d = detect(big)
    assert d.provider == "none"                                  # beyond 5 MB: ignored by design


def test_deeply_nested_and_many_tags_stay_fast():
    import time

    html = "<div>" * 50_000 + load("turnstile_widget") + "</div>" * 50_000
    t0 = time.perf_counter()
    assert detect(html).provider == "turnstile"
    assert time.perf_counter() - t0 < 3


# ── our own demo pages must be detected as what they are ─────────────────────
@pytest.mark.parametrize("page,provider", [
    ("recaptcha.html", "recaptcha"), ("hcaptcha.html", "hcaptcha"), ("turnstile.html", "turnstile"),
])
def test_demo_pages(page, provider):
    d = detect((DEMO / page).read_text(encoding="utf-8"))
    assert d.provider == provider and d.sitekey


# ── label vocabulary shared with the CSV benchmark ───────────────────────────
@pytest.mark.parametrize("raw,want", [
    ("reCAPTCHA", "recaptcha"), ("invisible reCAPTCHA", "recaptcha_invisible"),
    ("hCaptcha", "hcaptcha"), ("Turnstile", "turnstile"), ("none", "none"),
    ("", "none"), (None, "none"), ("  None ", "none"), ("Something else", "something else"),
])
def test_normalise_label(raw, want):
    assert normalise_label(raw) == want


def test_as_dict_includes_csv_label():
    assert detect(load("recaptcha_v2_invisible")).as_dict()["csv_label"] == "recaptcha_invisible"
