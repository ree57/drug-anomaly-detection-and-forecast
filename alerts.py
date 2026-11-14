"""Step 8d (updated in Step 11): analyst alert list for one published week.

    python alerts.py 2026-06-03 [--top 20] [--save reports/sample_alerts.md]

Shows the week's alerts (events with confidence high or medium in data/eval/scores_final), grouped into families
(families.py: same name stem after removing store-brand prefixes, or the same big % change), best family first.
Every line is built from publication-time data only (see alert_lib.py). Pandas only, no Spark.

Already left out by the detectors: mix_only and artefact_suspect events.
Cautions are shown, never used to drop an alert.
"""
import argparse
import datetime as dt
import os
import sys

import pandas as pd

from alert_lib import load_scored_events, reason, with_families
from families import load_store_prefixes

MAX_ROWS_PER_FAMILY = 8
sys.stdout.reconfigure(encoding="utf-8")   # the Windows console default (cp1252) can't print "→"

parser = argparse.ArgumentParser(description="Alert list for one published week")
parser.add_argument("week", help="published week, YYYY-MM-DD")
parser.add_argument("--top", type=int, default=20, help="max families to show (default 20)")
parser.add_argument("--save", help="also write the list to this markdown file")
args = parser.parse_args()
week = dt.date.fromisoformat(args.week)

pe = pd.read_parquet("data/tables/price_events", columns=["published_week", "mix_only", "artefact_suspect"])
in_week = pe[pe.published_week == week]
if in_week.empty:
    weeks = pd.Series(sorted(pe.published_week.unique()))
    near = weeks.iloc[(weeks.map(lambda w: abs((w - week).days))).argsort()[:3]]
    sys.exit(f"{args.week} is not a published week. Nearest: {', '.join(str(w) for w in near)}")
n_events, n_real = len(in_week), int((~in_week.mix_only & ~in_week.artefact_suspect).sum())

df = with_families(load_scored_events([week]), load_store_prefixes())
df = df[df.confidence.isin(["high", "medium"])]

lines = [f"# Price alerts for the week published {args.week}", "",
         f"{n_real:,} real price changes this week "
         f"({n_events:,} incl. mix-only/artefact-suspect, which are never alerted).", ""]
if df.empty:
    lines.append("**No alerts this week.** No price change reached the top 1% of either detector.")
else:
    fams = (df.groupby("family")
              .agg(best=("score_final", "max"), tier=("tier", "max"), n=("drug_group", "size"))
              .sort_values("best", ascending=False))
    n_high, n_med = int((fams.tier == 2).sum()), int((fams.tier == 1).sum())
    lines += [f"**{len(fams)} alert famil{'y' if len(fams) == 1 else 'ies'}** "
              f"({n_high} high confidence, {n_med} medium)"
              + (f"; showing the top {args.top}." if len(fams) > args.top else "."), ""]
    for i, (fid, fam_row) in enumerate(fams.head(args.top).iterrows(), 1):
        members = df[df.family == fid].sort_values("score_final", ascending=False)
        top = members.iloc[0]
        names = members.description.tolist()
        title = " / ".join(names[:3]) + (f" (+{len(names) - 3} more)" if len(names) > 3 else "")
        conf = "HIGH" if fam_row.tier == 2 else "MEDIUM"
        lines += [f"## {i}. {title} — {conf}", "",
                  "| Drug | Unit | Brand/generic | Old → new price | Change | Confidence |",
                  "|---|---|---|---|---|---|"]
        for _, r in members.head(MAX_ROWS_PER_FAMILY).iterrows():
            kind = f"brand ({r['classification']})" if r["is_brand"] else "generic"
            lines.append(f"| {r['description']} | {r['pricing_unit']} | {kind} | "
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
