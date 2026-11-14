"""Alert data and plain-English reasons (moved out of alerts.py so export_dashboard.py can reuse them).

Publication-time data only: scores_final (Step 8d detectors + caution flags) and price_events without outcome_*
columns. The proxy labels in eval_events are never read.
"""
import math

import numpy as np
import pandas as pd

from families import assign_families

CONF_ORDER = {"high": 2, "medium": 1, "none": 0}
EVENT_COLS = ["drug_group", "published_week", "old_price", "new_price", "pct_change", "is_brand", "classification",
              "pricing_unit", "n_ndcs_priced", "ndc_confirmed_share", "matched_ndc_median_change", "mix_only",
              "artefact_suspect"]


def load_scored_events(weeks=None):
    """Real events (scores_final = not mix_only, not artefact_suspect) with their price-event details."""
    filters = [("published_week", "in", list(weeks))] if weeks is not None else None
    sf = pd.read_parquet("data/eval/scores_final", filters=filters)
    pe = pd.read_parquet("data/tables/price_events", columns=EVENT_COLS, filters=filters)
    df = sf.merge(pe, on=["drug_group", "published_week"], how="left", validate="one_to_one")
    df["abs_log_change"] = np.abs(np.log(df.new_price / df.old_price))
    df["description"] = df.drug_group.str.split("|").str[0]
    df["tier"] = df.confidence.map(CONF_ORDER)
    return df


def with_families(df, prefixes):
    """Families are formed among ALL real events of a week, so alerts linked through other moves stay together."""
    return assign_families(df, "published_week", "description", "pct_change", prefixes)


def size_text(pct_size):
    if pct_size >= 1:
        return "bigger than any price move in 2022–24"
    share = math.floor(pct_size * 1000) / 10   # round DOWN, so we never overstate
    return f"bigger than {share:.1f}% of price moves in 2022–24"


def history_text(r):
    usual = (math.exp(r["typical_past_move"]) - 1) * 100
    usual_txt = "<0.1%" if usual < 0.05 else f"~{usual:.1f}%" if usual < 10 else f"~{usual:.0f}%"
    if r["typical_from_class"]:
        kind = "brand" if r["is_brand"] else "generic"
        return f"little price history for this drug; {kind} drugs normally move {usual_txt}"
    ratio = r["abs_log_change"] / max(r["typical_past_move"], 1e-6)
    level = "very unusual" if ratio >= 10 else "unusual" if ratio >= 3 else "not unusual"
    return f"{level} for this drug (normally moves {usual_txt})"


def products_text(r):
    share, med = r["ndc_confirmed_share"], r["matched_ndc_median_change"]
    if share is None or (isinstance(share, float) and math.isnan(share)):
        return f"{int(r['n_ndcs_priced'])} product(s) priced; none were priced the previous week"
    n = int(r["n_ndcs_priced"])
    if n == 1:
        return f"its only product changed its own price ({med:+.1f}%)"
    who = "all products" if share >= 1 else f"{share:.0%} of products"
    return f"{n} products; {who} priced last week changed their own price (median {med:+.1f}%)"


def agreement_text(r):
    if r["confidence"] == "high":
        return "both detectors agree"
    return ("flagged by the main score only" if r["medium_source"] == "hybrid_only"
            else "flagged by the second opinion (IF-small) only")


def cautions(r):
    out = []
    if r["caution_class_changed"]:
        out.append("caution: a product changed classification this week (possible repricing artefact)")
    if r["caution_after_gap"]:
        w = int(r["gap_weeks_missing"])
        out.append(f"caution: group was missing for {w} week{'s' if w != 1 else ''}")
    med = r["matched_ndc_median_change"]
    if med is not None and not math.isnan(med) and abs(math.log1p(med / 100)) < 0.5 * r["abs_log_change"]:
        # same idea as artefact_suspect, but relative: the products present both weeks moved far less than the group
        out.append(f"caution: products present in both weeks moved only {med:+.1f}% vs the group's "
                   f"{r['pct_change']:+.1f}%, so products entering/leaving drive much of the change")
    if r["caution_jan_brand_round"]:
        out.append("caution: round January brand increase (usually a scheduled list-price change)")
    if r["caution_jan_brand_not_round"]:
        out.append("caution: January brand increase (often a list-price change, even if not a round %)")
    return out


def reason(r):
    return "; ".join([size_text(r["pct_size"]), history_text(r), products_text(r), agreement_text(r)]
                     + cautions(r))
