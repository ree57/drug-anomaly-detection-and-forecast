"""Step 8d: analyst alert list for one published week.

    python alerts.py 2026-06-03 [--top 20] [--save reports/sample_alerts.md]

Shows the week's alerts (events with confidence high or medium in data/eval/scores_final), grouped into families
(data/eval/families: same week + same name stem or identical big % change), best family first.
Every line is built from publication-time data only: price_events (no outcome_* columns), the detector scores,
and the 2022-24 reference used by the hybrid. eval_events (which holds the proxy labels) is NOT read.

Already left out by the detectors: mix_only and artefact_suspect events.
Cautions are shown, never used to drop an alert.
"""
import argparse
import math
import os
import sys

os.environ["PYSPARK_PYTHON"] = sys.executable
from pyspark.sql import SparkSession

CONF_ORDER = {"high": 2, "medium": 1}
MAX_ROWS_PER_FAMILY = 8

parser = argparse.ArgumentParser(description="Alert list for one published week")
parser.add_argument("week", help="published week, YYYY-MM-DD")
parser.add_argument("--top", type=int, default=20, help="max families to show (default 20)")
parser.add_argument("--save", help="also write the list to this markdown file")
args = parser.parse_args()

spark = (SparkSession.builder.master("local[*]").appName("alerts")
         .config("spark.driver.memory", "4g")
         .config("spark.sql.execution.arrow.pyspark.enabled", "true").getOrCreate())
spark.sparkContext.setLogLevel("ERROR")
spark.read.parquet("data/eval/scores_final").createOrReplaceTempView("sf")
spark.read.parquet("data/tables/price_events").createOrReplaceTempView("pe")
spark.read.parquet("data/eval/families").createOrReplaceTempView("fam")

week_info = spark.sql(f"""
    SELECT COUNT(*) AS events, SUM(CAST(NOT mix_only AND NOT artefact_suspect AS INT)) AS real_events
    FROM pe WHERE published_week = DATE'{args.week}'""").first()
if not week_info.events:
    near = spark.sql(f"""SELECT published_week FROM pe GROUP BY published_week
                         ORDER BY ABS(DATEDIFF(published_week, DATE'{args.week}')) LIMIT 3""").collect()
    sys.exit(f"{args.week} is not a published week. Nearest: {', '.join(str(r[0]) for r in near)}")

df = spark.sql(f"""
    SELECT s.*, COALESCE(f.family_id, CONCAT(s.drug_group, '|', s.published_week)) AS family_id,
           p.old_price, p.new_price, p.pct_change, p.is_brand, p.classification, p.pricing_unit,
           p.n_ndcs_priced, p.ndc_confirmed_share, p.matched_ndc_median_change,
           ABS(LN(p.new_price / p.old_price)) AS abs_log_change
    FROM sf AS s
    JOIN pe AS p USING (drug_group, published_week)
    LEFT JOIN fam AS f USING (drug_group, published_week)
    WHERE s.published_week = DATE'{args.week}' AND s.confidence IN ('high', 'medium')""").toPandas()
spark.stop()


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


lines = [f"# Price alerts for the week published {args.week}", "",
         f"{int(week_info.real_events or 0):,} real price changes this week "
         f"({int(week_info.events):,} incl. mix-only/artefact-suspect, which are never alerted).", ""]
if df.empty:
    lines.append("**No alerts this week.** No price change reached the top 1% of either detector.")
else:
    df["tier"] = df["confidence"].map(CONF_ORDER)
    fams = (df.groupby("family_id")
              .agg(best=("score_final", "max"), tier=("tier", "max"), n=("drug_group", "size"))
              .sort_values("best", ascending=False))
    n_high, n_med = int((fams.tier == 2).sum()), int((fams.tier == 1).sum())
    lines += [f"**{len(fams)} alert famil{'y' if len(fams) == 1 else 'ies'}** "
              f"({n_high} high confidence, {n_med} medium)"
              + (f"; showing the top {args.top}." if len(fams) > args.top else "."), ""]
    for i, (fid, fam_row) in enumerate(fams.head(args.top).iterrows(), 1):
        members = df[df.family_id == fid].sort_values("score_final", ascending=False)
        top = members.iloc[0]
        names = members["drug_group"].str.split("|").str[0].tolist()
        title = " / ".join(names[:3]) + (f" (+{len(names) - 3} more)" if len(names) > 3 else "")
        conf = "HIGH" if fam_row.tier == 2 else "MEDIUM"
        lines += [f"## {i}. {title} — {conf}", "",
                  "| Drug | Unit | Brand/generic | Old → new price | Change | Confidence |",
                  "|---|---|---|---|---|---|"]
        for _, r in members.head(MAX_ROWS_PER_FAMILY).iterrows():
            kind = f"brand ({r['classification']})" if r["is_brand"] else "generic"
            lines.append(f"| {r['drug_group'].split('|')[0]} | {r['pricing_unit']} | {kind} | "
                         f"${r['old_price']:.5g} → ${r['new_price']:.5g} | {r['pct_change']:+.1f}% | "
                         f"{r['confidence']}{' (' + r['medium_source'].replace('_', ' ') + ')' if r['medium_source'] else ''} |")
        if len(members) > MAX_ROWS_PER_FAMILY:
            lines.append(f"| … {len(members) - MAX_ROWS_PER_FAMILY} more | | | | | |")
        lines += ["", f"**Why:** {reason(top)}."]
        if len(members) > 1:
            pcts = members["pct_change"]
            lines.append(f"Family of {len(members)} drug groups moving the same week "
                         f"({pcts.min():+.1f}% to {pcts.max():+.1f}%); the reason describes the top one.")
        lines.append("")

text = "\n".join(lines) + "\n"
print(text)
if args.save:
    os.makedirs(os.path.dirname(args.save) or ".", exist_ok=True)
    with open(args.save, "w", encoding="utf-8") as f:
        f.write(text)
    print(f"saved {args.save}")
