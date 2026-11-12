"""
Accuracy of the static detector against the CSV's labelled `Captcha` column.

Three views, because the label vocabulary is finer than static HTML can always
support:
  * exact   - recaptcha / recaptcha_invisible / hcaptcha / turnstile / none must match
            (invisible only when the markup says size=invisible)
  * family  - invisible reCAPTCHA counts as reCAPTCHA
  * exact with v3 counted as invisible - v3 shows only a badge, no checkbox

Only pages that were actually scanned (`scan_status == "ok"`) enter accuracy;
everything else (robots-blocked, errors, unscanned) is reported as coverage so
failures cannot silently inflate or deflate the numbers.
"""
from __future__ import annotations

from sqlalchemy import func
from sqlalchemy.orm import Session

from app.models import Site

LABELS = ["recaptcha", "recaptcha_invisible", "hcaptcha", "turnstile", "none"]
FAMILY = {"recaptcha": "recaptcha", "recaptcha_invisible": "recaptcha", "hcaptcha": "hcaptcha",
          "turnstile": "turnstile", "none": "none"}
FAMILIES = ["recaptcha", "hcaptcha", "turnstile", "none"]
OTHER = "other"
MISMATCH_LIMIT = 60


def detected_label(site: Site) -> str:
    """The detector's answer in the CSV label vocabulary."""
    if site.detected_provider in (None, "none"):
        return "none"
    if site.detected_provider == "recaptcha" and site.detected_variant == "v2_invisible":
        return "recaptcha_invisible"
    return site.detected_provider


def detected_label_visibility(site: Site) -> str:
    """Like detected_label, but reCAPTCHA v3 counts as invisible (it only ever shows a badge)."""
    if site.detected_provider == "recaptcha" and site.detected_variant in ("v2_invisible", "v3"):
        return "recaptcha_invisible"
    return detected_label(site)


def _prf(tp: int, fp: int, fn: int) -> dict:
    p = tp / (tp + fp) if tp + fp else None
    r = tp / (tp + fn) if tp + fn else None
    f = 2 * p * r / (p + r) if p and r else (0.0 if p is not None and r is not None else None)
    rnd = lambda x: round(x, 4) if x is not None else None  # noqa: E731
    return {"precision": rnd(p), "recall": rnd(r), "f1": rnd(f), "support": tp + fn}


def build_report(db: Session) -> dict:
    total = db.query(func.count(Site.id)).scalar() or 0
    statuses = dict(db.query(Site.scan_status, func.count(Site.id)).group_by(Site.scan_status).all())
    unscanned = statuses.pop(None, 0)
    scanned_ok = [s for s in db.query(Site).filter(Site.scan_status == "ok")]

    truth_counts = {k: 0 for k in LABELS}
    matrix = {t: {p: 0 for p in LABELS + [OTHER]} for t in LABELS + [OTHER]}
    exact = family = exact_vis = 0
    mismatches = []
    missed_static = 0           # labelled with a CAPTCHA, detected none, page mentions a provider
    for s in scanned_ok:
        truth = s.label_normalised
        t = truth if truth in LABELS else OTHER
        pred = detected_label(s)
        truth_counts[t] = truth_counts.get(t, 0) + 1
        matrix[t][pred] += 1
        ok_exact = truth == pred
        ok_family = truth in FAMILY and FAMILY.get(truth) == FAMILY.get(pred)
        exact += ok_exact
        exact_vis += truth == detected_label_visibility(s)
        family += ok_family
        if not ok_exact:
            if truth not in ("none", "") and pred == "none" and s.needs_render:
                missed_static += 1
            if len(mismatches) < MISMATCH_LIMIT:
                mismatches.append({"id": s.id, "company": s.company, "contact_page": s.contact_page,
                                   "label": truth, "detected": pred, "variant": s.detected_variant,
                                   "needs_render": bool(s.needs_render), "family_ok": ok_family})

    n = len(scanned_ok)

    def per_class(vocab: list[str], key) -> dict:
        out = {}
        for c in vocab:
            tp = sum(1 for s in scanned_ok if key(s.label_normalised) == c and key(detected_label(s)) == c)
            fp = sum(1 for s in scanned_ok if key(s.label_normalised) != c and key(detected_label(s)) == c)
            fn = sum(1 for s in scanned_ok if key(s.label_normalised) == c and key(detected_label(s)) != c)
            out[c] = _prf(tp, fp, fn)
        return out

    majority = max(truth_counts.items(), key=lambda kv: kv[1]) if n else (None, 0)
    return {
        "total_sites": total,
        "scanned": n,
        "unscanned": unscanned,
        "coverage": round(n / total, 4) if total else None,
        "scan_status": statuses,
        "exact_accuracy": round(exact / n, 4) if n else None,
        "family_accuracy": round(family / n, 4) if n else None,
        # Reported separately and NOT tuned on: it follows from the data that v3 (badge only) is what
        # a labeller sees as "invisible". The strict figure above stays the headline.
        "exact_accuracy_v3_as_invisible": round(exact_vis / n, 4) if n else None,
        "baseline_majority_class": {"label": majority[0],
                                    "accuracy": round(majority[1] / n, 4) if n else None},
        "labels": LABELS + [OTHER],
        "confusion_matrix": matrix,
        "per_class_exact": per_class(LABELS, lambda x: x),
        "per_class_family": per_class(FAMILIES, lambda x: FAMILY.get(x, x)),
        "js_injected_suspected": missed_static,
        "mismatches": mismatches,
    }
