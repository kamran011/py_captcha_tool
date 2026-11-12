"""
Identify which CAPTCHA provider (if any) a page embeds, from its HTML.

Pure function of the markup: no network, no page interaction. The M6 scanner
fetches pages (read-only GET) and feeds the HTML, or a browser-rendered DOM,
to `detect()`.

How it decides
--------------
The page is parsed with a real HTML parser, so text mentions, HTML comments
and escaped code samples never count. Evidence is weighted:

    widget element (class g-recaptcha / h-captcha / cf-turnstile)  4
    provider iframe                                                4
    provider <script src>  (host AND path checked)                 3
    inline JS call  grecaptcha.render / hcaptcha.execute / ...     3
    hidden response field, reCAPTCHA badge, provider URL in JS     2

A provider is reported when its evidence sums to >= 3. Hosts are matched on a
dot boundary, so `google.com.evil.example` or `evil-hcaptcha.com` do not count.
Static HTML cannot see widgets injected later by JavaScript; when nothing is
found but the page still mentions a provider, `needs_render` is set so the
caller can retry on a rendered DOM.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from html.parser import HTMLParser
from urllib.parse import parse_qs, urlparse

MAX_HTML_BYTES = 5_000_000
DETECT_THRESHOLD = 3
TIE_ORDER = ("turnstile", "hcaptcha", "recaptcha")

SITEKEY_RE = {
    "recaptcha": re.compile(r"^6L[\w-]{38}$"),
    "hcaptcha": re.compile(r"^[0-9a-f]{8}-(?:[0-9a-f]{4}-){3}[0-9a-f]{12}$", re.I),
    "turnstile": re.compile(r"^(?:0x[\w-]{16,}|[123]x\w{20,})$"),
}
_HOSTS = {
    "recaptcha": ("google.com", "recaptcha.net"),
    "hcaptcha": ("hcaptcha.com",),
    "turnstile": ("challenges.cloudflare.com",),
}
_PATH_PREFIX = {"recaptcha": "/recaptcha/", "hcaptcha": "/", "turnstile": "/turnstile/"}
_WIDGET_CLASS = {"recaptcha": "g-recaptcha", "hcaptcha": "h-captcha", "turnstile": "cf-turnstile"}
_RESPONSE_FIELD = {
    "recaptcha": "g-recaptcha-response",
    "hcaptcha": "h-captcha-response",
    "turnstile": "cf-turnstile-response",
}
_INLINE_CALL = {
    "recaptcha": re.compile(r"\bgrecaptcha(?:\.enterprise)?\.(?:render|execute)\s*\("),
    "hcaptcha": re.compile(r"\bhcaptcha\.(?:render|execute)\s*\("),
    "turnstile": re.compile(r"\bturnstile\.(?:render|execute)\s*\("),
}
_INLINE_URL = {
    "recaptcha": re.compile(r"(?<![\w.-])(?:www\.)?(?:google\.com|recaptcha\.net)/recaptcha/"),
    "hcaptcha": re.compile(r"(?<![\w.-])(?:[\w-]+\.)?hcaptcha\.com/\d*/?api\.js"),
    "turnstile": re.compile(r"(?<![\w.-])challenges\.cloudflare\.com/turnstile/"),
}
_INLINE_SITEKEY = re.compile(r"""sitekey['"]?\s*[:=]\s*['"]([^'"]{8,80})['"]""", re.I)
_INLINE_INVISIBLE = re.compile(r"""['"]?size['"]?\s*[:=]\s*['"]invisible['"]""", re.I)
_RECAPTCHA_EXECUTE_KEY = re.compile(r"""execute\(\s*['"](6L[\w-]{38})['"]""")
_MENTION = re.compile(r"recaptcha|hcaptcha|turnstile", re.I)

# CSV label vocabulary used by the benchmark: recaptcha | recaptcha_invisible |
# hcaptcha | turnstile | none
CSV_LABEL = {"v2_invisible": "recaptcha_invisible"}


@dataclass
class Detection:
    provider: str                       # recaptcha | hcaptcha | turnstile | none
    variant: str | None = None          # see module notes
    sitekey: str | None = None
    confidence: float = 0.0
    evidence: list[dict] = field(default_factory=list)
    others: list[dict] = field(default_factory=list)   # other providers on the page
    extra: dict = field(default_factory=dict)
    needs_render: bool = False
    error: str | None = None

    @property
    def csv_label(self) -> str:
        if self.provider == "none":
            return "none"
        if self.provider == "recaptcha" and self.variant == "v2_invisible":
            return "recaptcha_invisible"
        return self.provider

    def as_dict(self) -> dict:
        d = dict(self.__dict__)
        d["csv_label"] = self.csv_label
        return d


def normalise_label(raw: str | None) -> str:
    """Map a CSV 'Captcha' label onto the detector's vocabulary."""
    s = (raw or "").lower().strip()
    if "recaptcha" in s:
        return "recaptcha_invisible" if "invisible" in s else "recaptcha"
    if "hcaptcha" in s:
        return "hcaptcha"
    if "turnstile" in s:
        return "turnstile"
    if s in ("", "none", "no", "no captcha", "n/a"):
        return "none"
    return s


class _Collector(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.script_srcs: list[str] = []
        self.inline_js: list[str] = []
        self.iframes: list[str] = []
        self.widgets: list[tuple[str, dict]] = []     # (provider, attrs)
        self.fields: set[str] = set()
        self.badge = False
        self._in_script = False
        self._buf: list[str] = []

    def handle_starttag(self, tag, attrs):
        a = {k.lower(): (v or "") for k, v in attrs}
        classes = a.get("class", "").split()
        if tag == "script":
            if a.get("src"):
                self.script_srcs.append(a["src"].strip())
            self._in_script, self._buf = True, []
        elif tag == "iframe" and a.get("src"):
            self.iframes.append(a["src"].strip())
        for provider, cls in _WIDGET_CLASS.items():
            if cls in classes:
                self.widgets.append((provider, a))
        if "grecaptcha-badge" in classes:
            self.badge = True
        for key in ("name", "id"):
            if a.get(key) in _RESPONSE_FIELD.values():
                self.fields.add(a[key])

    def handle_data(self, data):
        if self._in_script:
            self._buf.append(data)

    def handle_endtag(self, tag):
        if tag == "script" and self._in_script:
            text = "".join(self._buf).strip()
            if text:
                self.inline_js.append(text)
            self._in_script = False


def _host_ok(provider: str, url: str) -> tuple[bool, object]:
    try:
        u = urlparse(url)
    except ValueError:
        return False, None
    host = (u.hostname or "").lower().rstrip(".")
    ok = any(host == d or host.endswith("." + d) for d in _HOSTS[provider])
    return (ok and u.path.startswith(_PATH_PREFIX[provider])), u


def _analyse(provider: str, page: _Collector) -> dict:
    ev: list[dict] = []
    keys: list[str] = []        # widget / iframe / explicit-render keys
    v3_keys: list[str] = []     # keys from a v3 script URL or grecaptcha.execute('KEY')
    hints: set[str] = set()
    extra: dict = {}

    def add(kind: str, weight: int, detail: str = "") -> None:
        ev.append({"kind": kind, "weight": weight, "detail": detail[:160]})

    for src in page.script_srcs:
        ok, u = _host_ok(provider, src)
        if not ok:
            continue
        add("script_src", 3, src)
        if provider == "recaptcha":
            if "enterprise" in u.path:
                extra["enterprise"] = True
            render = parse_qs(u.query).get("render", [None])[0]
            if render and render not in ("explicit", "onload") and SITEKEY_RE["recaptcha"].match(render):
                hints.add("v3")
                v3_keys.append(render)

    for prov, attrs in page.widgets:
        if prov != provider:
            continue
        add("widget_element", 4, f"data-sitekey={attrs.get('data-sitekey', '')}")
        if attrs.get("data-sitekey"):
            keys.append(attrs["data-sitekey"].strip())
        hints.add("invisible" if attrs.get("data-size", "").lower() == "invisible" else "visible")
        if provider == "turnstile" and attrs.get("data-appearance"):
            extra["appearance"] = attrs["data-appearance"]

    for src in page.iframes:
        ok, u = _host_ok(provider, src)
        if not ok or (provider == "hcaptcha" and "/captcha" not in u.path):
            continue
        add("iframe", 4, src)
        q = parse_qs(u.query)
        frag = parse_qs(u.fragment)
        for k in (q.get("k", []) + frag.get("sitekey", []) + q.get("sitekey", [])):
            keys.append(k)
        size = (q.get("size", [""])[0] or frag.get("size", [""])[0]).lower()
        hints.add("invisible" if size == "invisible" else "visible")

    if _RESPONSE_FIELD[provider] in page.fields:
        add("response_field", 2, _RESPONSE_FIELD[provider])
    if provider == "recaptcha" and page.badge:
        add("badge", 2, "grecaptcha-badge")
        hints.add("badge")

    for js in page.inline_js:
        if _INLINE_CALL[provider].search(js):
            add("inline_api_call", 3)
            if provider == "recaptcha":
                if "grecaptcha.enterprise" in js:
                    extra["enterprise"] = True
                m = _RECAPTCHA_EXECUTE_KEY.search(js)
                if m:
                    hints.add("v3")
                    v3_keys.append(m.group(1))
            if _INLINE_INVISIBLE.search(js):
                hints.add("invisible")
            for m in _INLINE_SITEKEY.finditer(js):
                keys.append(m.group(1))
        if _INLINE_URL[provider].search(js):
            add("provider_url_in_js", 2)

    # Report the key that belongs to the variant we will report: a visible v2
    # widget outranks a global v3 script, so its key does too.
    visible = bool(hints & {"visible", "invisible"})
    ordered = keys + v3_keys if visible else v3_keys + keys
    sitekey = next((k for k in ordered if SITEKEY_RE[provider].match(k)), None)
    return {"score": sum(e["weight"] for e in ev), "evidence": ev, "sitekey": sitekey,
            "hints": hints, "extra": extra}


def _variant(provider: str, hints: set[str]) -> str:
    if provider == "recaptcha":
        # A visible widget beats a global v3 script: sites often load both.
        if "invisible" in hints:
            return "v2_invisible"
        if "visible" in hints:
            return "v2_checkbox"
        if "v3" in hints:
            return "v3"
        return "v2_unknown"
    if provider == "hcaptcha":
        if "invisible" in hints:
            return "invisible"
        return "checkbox" if "visible" in hints else "unknown"
    return "widget"


def detect(html: str | bytes | None) -> Detection:
    """Detect the embedded CAPTCHA provider. Never raises."""
    try:
        if html is None:
            return Detection("none", error="no_html")
        if isinstance(html, (bytes, bytearray)):
            html = bytes(html[:MAX_HTML_BYTES]).decode("utf-8", errors="replace")
        elif not isinstance(html, str):
            return Detection("none", error="invalid_input")
        html = html[:MAX_HTML_BYTES]

        page = _Collector()
        page.feed(html)
        page.close()

        found = {p: _analyse(p, page) for p in TIE_ORDER}
        hits = {p: r for p, r in found.items() if r["score"] >= DETECT_THRESHOLD}
        if not hits:
            return Detection(
                "none",
                confidence=0.6,
                needs_render=bool(_MENTION.search(html)),
            )
        order = sorted(hits, key=lambda p: (-hits[p]["score"], TIE_ORDER.index(p)))
        primary = order[0]
        r = hits[primary]
        score = r["score"]
        return Detection(
            provider=primary,
            variant=_variant(primary, r["hints"]),
            sitekey=r["sitekey"],
            confidence=0.95 if score >= 8 else 0.85 if score >= 5 else 0.7,
            evidence=r["evidence"],
            others=[
                {"provider": p, "variant": _variant(p, hits[p]["hints"]),
                 "sitekey": hits[p]["sitekey"], "score": hits[p]["score"]}
                for p in order[1:]
            ],
            extra=r["extra"],
        )
    except Exception as exc:  # noqa: BLE001 - malformed markup must not crash a scan
        return Detection("none", error=f"parse_error: {type(exc).__name__}")
