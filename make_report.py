"""Step 7: Day 1 data quality report and charts.

Every number in reports/day1_data_report.md is computed here from data/clean and data/tables
(also saved to reports/day1_numbers.json). Rerun this script to refresh the report.
"""
import json
import os
import sys
from datetime import date
from pathlib import Path

os.environ["PYSPARK_PYTHON"] = sys.executable
import matplotlib

matplotlib.use("Agg")  # draw to files, no window
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from pyspark.sql import SparkSession

YEARS = [2022, 2023, 2024, 2025, 2026]
REPORTS = Path("reports")
FIG = REPORTS / "figures"
FIG.mkdir(parents=True, exist_ok=True)
GROUPS = {
    "isosorbide": "ISOSORBIDE MONONIT 20 MG TAB|EA|G",
    "prucalopride": "PRUCALOPRIDE 2 MG TABLET|EA|G",
    "otezla": "OTEZLA 30 MG TABLET|EA|B",
    "phenobarbital": "PHENOBARBITAL 16.2 MG TABLET|EA|G",
}

spark = (SparkSession.builder.master("local[*]").appName("make_report")
         .config("spark.driver.memory", "4g").getOrCreate())
spark.sparkContext.setLogLevel("ERROR")
spark.read.parquet("data/clean/nadac").createOrReplaceTempView("nadac")
for name in ["ndc_dim", "group_dim", "group_weekly", "price_events"]:
    spark.read.parquet(f"data/tables/{name}").createOrReplaceTempView(name)


def one(sql):
    return spark.sql(sql).first().asDict()


def rows(sql):
    return [r.asDict() for r in spark.sql(sql).collect()]


# =====================================================================================
# 1. Numbers
# =====================================================================================
N = {}

# ---- data ----
N["csv_rows"] = {y: spark.read.csv(f"data/raw/nadac_{y}.csv", header=True).count() for y in YEARS}
N["clean_rows"] = {r["source_year"]: r["n"] for r in rows("SELECT source_year, COUNT(*) AS n FROM nadac GROUP BY 1")}
N["rejects"] = sum(N["csv_rows"].values()) - sum(N["clean_rows"].values())
N["overview"] = one("""SELECT COUNT(*) AS rows, COUNT(DISTINCT ndc) AS ndcs, COUNT(DISTINCT as_of_date) AS weeks,
                              MIN(as_of_date) AS first_week, MAX(as_of_date) AS last_week FROM nadac""")
N["groups"] = one("SELECT COUNT(*) AS n FROM group_dim")["n"]
N["group_weekly_rows"] = one("SELECT COUNT(*) AS n FROM group_weekly")["n"]

# ---- trust ----
weeks = rows("SELECT as_of_date AS d, COUNT(DISTINCT source_year) AS files FROM nadac GROUP BY 1 ORDER BY 1")
N["week_gaps"] = sum(1 for a, b in zip(weeks, weeks[1:]) if (b["d"] - a["d"]).days != 7)
N["weeks_in_two_files"] = sum(1 for w in weeks if w["files"] > 1)
N["not_wednesday"] = sum(1 for w in weeks if w["d"].weekday() != 2)
N["duplicate_ndc_weeks"] = one("""SELECT COUNT(*) AS n FROM
                                  (SELECT ndc, as_of_date FROM nadac GROUP BY 1, 2 HAVING COUNT(*) > 1)""")["n"]
N["rows_in_group_weekly"] = one("SELECT SUM(n_ndcs_priced) AS n FROM group_weekly")["n"]
N["group_key"] = one("""
    WITH r AS (SELECT d.drug_group, n.as_of_date, n.nadac_per_unit, ARRAY_CONTAINS(n.explanation_codes, 5) AS code5
               FROM nadac n JOIN ndc_dim d ON d.ndc = n.ndc),
    all_rows AS (SELECT drug_group, as_of_date, COUNT(*) AS ndcs, COUNT(DISTINCT nadac_per_unit) AS prices
                 FROM r GROUP BY 1, 2),
    no5 AS (SELECT drug_group, as_of_date, COUNT(*) AS ndcs, COUNT(DISTINCT nadac_per_unit) AS prices
            FROM r WHERE NOT code5 GROUP BY 1, 2)
    SELECT (SELECT COUNT(*) FROM all_rows WHERE ndcs > 1)                                      AS multi_ndc_group_weeks,
           (SELECT AVG(CASE WHEN prices = 1 THEN 1.0 ELSE 0 END) FROM all_rows WHERE ndcs > 1) AS share_one_price,
           (SELECT AVG(CASE WHEN prices = 1 THEN 1.0 ELSE 0 END) FROM no5 WHERE ndcs > 1)      AS share_one_price_no_code5,
           (SELECT AVG(CAST(code5 AS INT)) FROM r)                                             AS share_rows_code5""")
N["reworded_ndcs"] = one("SELECT SUM(CASE WHEN groups_seen > 1 THEN 1 ELSE 0 END) AS n FROM ndc_dim")["n"]
N["tier_share"] = one("SELECT AVG(CASE WHEN n_distinct_prices > 1 THEN 1.0 ELSE 0 END) AS s FROM group_weekly")["s"]
N["events"] = one("""SELECT COUNT(*) AS n, AVG(CAST(mix_only AS INT)) AS mix_only, AVG(CAST(after_gap AS INT)) AS after_gap,
                            SUM(CAST(mix_only AS INT)) AS n_mix_only, SUM(CAST(after_gap AS INT)) AS n_after_gap
                     FROM price_events""")
N["lag"] = one("""SELECT AVG(CASE WHEN publication_lag_days = 0 THEN 1.0 ELSE 0 END)              AS share_0,
                         AVG(CASE WHEN publication_lag_days BETWEEN 1 AND 14 THEN 1.0 ELSE 0 END) AS share_1_14,
                         AVG(CASE WHEN publication_lag_days > 14 THEN 1.0 ELSE 0 END)             AS share_over_14,
                         AVG(CASE WHEN publication_lag_days < 0 THEN 1.0 ELSE 0 END)              AS share_negative,
                         SUM(CASE WHEN publication_lag_days > 90 THEN 1 ELSE 0 END)               AS n_over_90,
                         SUM(CASE WHEN publication_lag_days > 90 AND ARRAY_CONTAINS(explanation_codes, 4)
                                   AND ARRAY_CONTAINS(explanation_codes, 5) THEN 1 ELSE 0 END)    AS n_over_90_codes_4_5
                  FROM price_events WHERE NOT mix_only""")
N["lag_brand_increases"] = one("""
    SELECT PERCENTILE_DISC(0.5) WITHIN GROUP (ORDER BY publication_lag_days) AS median_days,
           AVG(CASE WHEN publication_lag_days > 0 THEN 1.0 ELSE 0 END)       AS share_backdated,
           COUNT(*)                                                          AS n
    FROM price_events WHERE is_brand AND pct_change >= 1 AND NOT mix_only""")

# ---- behaviour ----
N["share_group_weeks_changed"] = one("""SELECT AVG(CAST(price_changed AS INT)) AS s FROM group_weekly
                                        WHERE price_changed IS NOT NULL""")["s"]
N["by_kind"] = {r["kind"]: r for r in rows("""
    SELECT CASE WHEN is_brand THEN 'brand' ELSE 'generic' END AS kind,
           COUNT(*)                                                           AS events,
           COUNT(DISTINCT drug_group)                                         AS groups_with_events,
           PERCENTILE_DISC(0.5) WITHIN GROUP (ORDER BY ABS(pct_change))       AS median_abs_pct,
           AVG(CASE WHEN ABS(pct_change) <= 5 THEN 1.0 ELSE 0 END)            AS share_within_5pct,
           AVG(CASE WHEN direction = 'up' THEN 1.0 ELSE 0 END)                AS share_up,
           PERCENTILE_DISC(0.5) WITHIN GROUP (ORDER BY weeks_since_last_event) AS median_weeks_between
    FROM price_events WHERE NOT mix_only GROUP BY 1""")}
N["january"] = {r["kind"]: r for r in rows("""
    SELECT CASE WHEN is_brand THEN 'brand' ELSE 'generic' END AS kind, COUNT(*) AS increases,
           AVG(CAST(is_january AS INT)) AS share_january
    FROM price_events WHERE pct_change >= 1 AND NOT mix_only GROUP BY 1""")}
N["big_moves"] = one("""
    SELECT COUNT(*)                                                                         AS n,
           SUM(CASE WHEN outcome_reverted THEN 1 ELSE 0 END)                                AS reverted,
           SUM(CASE WHEN outcome_persisted THEN 1 ELSE 0 END)                               AS persisted_2pct,
           SUM(CASE WHEN outcome_price_8w IS NOT NULL THEN 1 ELSE 0 END)                    AS with_8w_price,
           -- looser, relative check: 8 weeks later, is the price closer to the NEW level than the OLD one?
           SUM(CASE WHEN ABS(outcome_price_8w - new_price) < ABS(outcome_price_8w - old_price) THEN 1 ELSE 0 END)
                                                                                            AS closer_to_new_8w
    FROM price_events WHERE ABS(pct_change) > 50 AND NOT mix_only""")
N["big_moves_mix_only"] = one("""SELECT SUM(CAST(mix_only AS INT)) AS n, COUNT(*) AS total
                                 FROM price_events WHERE ABS(pct_change) > 50""")
N["otezla_increases"] = rows(f"""
    SELECT effective_date, published_week, old_price, new_price, pct_change, is_january, publication_lag_days
    FROM price_events WHERE drug_group = '{GROUPS["otezla"]}' AND pct_change >= 1 ORDER BY published_week""")
N["example_events"] = {k: one(f"""SELECT published_week, old_price, new_price, pct_change, outcome_price_4w, outcome_price_8w
                                  FROM price_events WHERE drug_group = '{g}' ORDER BY ABS(pct_change) DESC LIMIT 1""")
                       for k, g in GROUPS.items() if k != "otezla"}

# =====================================================================================
# 2. Charts
# =====================================================================================
def series(group):
    r = rows(f"SELECT as_of_date, median_price FROM group_weekly WHERE drug_group = '{group}' ORDER BY as_of_date")
    return [x["as_of_date"] for x in r], [x["median_price"] for x in r]


def style(ax, title, ylabel):
    ax.set_title(title, fontsize=14, loc="left", pad=12)
    ax.set_ylabel(ylabel)
    ax.set_xlabel("Week the price was published (NADAC weekly file)")
    ax.grid(alpha=0.3)
    locator = mdates.AutoDateLocator()
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator))
    for side in ["top", "right"]:
        ax.spines[side].set_visible(False)


def note(ax, text):
    ax.text(0.01, -0.2, text, transform=ax.transAxes, fontsize=9, color="dimgray", va="top")


def level_shift_chart(key, filename, title, colour):
    x, y = series(GROUPS[key])
    ev = N["example_events"][key]
    # how many weeks did the new price hold (within ±10% of the new level)?
    after = [p for d, p in zip(x, y) if d >= ev["published_week"]]
    held = next((i for i, p in enumerate(after) if abs(p / ev["new_price"] - 1) > 0.10), len(after))
    N["example_events"][key]["held_weeks"] = held
    until = " (to the end of the data)" if held == len(after) else ""
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.step(x, y, where="post", color=colour, linewidth=2)
    style(ax, title, "Average pharmacy cost, $ per tablet")
    # "\$" = a literal dollar sign (matplotlib treats text between two $ as a maths formula)
    ax.annotate(f"{ev['pct_change']:+,.0f}% on {ev['published_week']:%d %b %Y}: "
                f"\\${ev['old_price']:.2f} → \\${ev['new_price']:.2f}\n"
                f"then held within ±10% of the new price for {held} weeks{until}",
                xy=(ev["published_week"], ev["new_price"]),
                xytext=(0.05, 0.55), textcoords="axes fraction", fontsize=11,
                arrowprops=dict(arrowstyle="->", color="black"))
    ax.set_ylim(bottom=0)
    note(ax, "Source: CMS NADAC weekly files 2022-2026 (data.medicaid.gov). Price = median across the drug group's NDCs.")
    fig.tight_layout()
    fig.savefig(FIG / filename, dpi=150)
    plt.close(fig)


level_shift_chart("isosorbide", "a_level_shift_up_isosorbide.png",
                  "A level shift up: isosorbide mononitrate 20 mg tablet", "firebrick")
level_shift_chart("prucalopride", "b_level_shift_down_prucalopride.png",
                  "A level shift down: prucalopride 2 mg tablet", "steelblue")

# ---- c. Otezla staircase ----
x, y = series(GROUPS["otezla"])
fig, ax = plt.subplots(figsize=(10, 5.5))
ax.step(x, y, where="post", color="dimgray", linewidth=2, label="Weekly price")
for i, ev in enumerate(N["otezla_increases"]):
    jan = ev["is_january"]
    ax.scatter(ev["published_week"], ev["new_price"], s=70, zorder=3,
               color="crimson" if jan else "darkorange",
               label=("January increase" if jan else "Mid-year increase") if i < 2 or (not jan and i < 3) else None)
    ax.annotate(f"{ev['pct_change']:+.1f}%\n{ev['effective_date']:%b %Y}", (ev["published_week"], ev["new_price"]),
                xytext=(-10, 12), textcoords="offset points", fontsize=9, ha="right",
                color="crimson" if jan else "darkorange")
handles, labels = ax.get_legend_handles_labels()
uniq = dict(zip(labels, handles))
ax.legend(uniq.values(), uniq.keys(), loc="lower right", frameon=False)
style(ax, "Otezla 30 mg: brand prices climb in yearly steps, mostly in January", "Average pharmacy cost, $ per tablet")
ax.set_ylim(55, max(y) * 1.08)
ax.set_xlim(left=date(2021, 9, 1))  # room for the first label
note(ax, "Labels show each increase of 1% or more and the month it took effect. Source: CMS NADAC 2022-2026.")
fig.tight_layout()
fig.savefig(FIG / "c_otezla_january_staircase.png", dpi=150)
plt.close(fig)

# ---- d. phenobarbital reverted spike ----
x, y = series(GROUPS["phenobarbital"])
ev = N["example_events"]["phenobarbital"]
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5), gridspec_kw={"width_ratios": [2, 1]})
ax1.step(x, y, where="post", color="seagreen", linewidth=2)
style(ax1, "A spike that reverted: phenobarbital 16.2 mg tablet (a jumpy drug)", "Average pharmacy cost, $ per tablet")
ax1.set_ylim(bottom=0)
zoom = [(d, p) for d, p in zip(x, y) if abs((d - ev["published_week"]).days) <= 70]
ax2.step([d for d, _ in zoom], [p for _, p in zoom], where="post", color="seagreen", linewidth=2, marker="o", markersize=4)
ax2.set_title("Zoom: 10 weeks either side", fontsize=12, loc="left")
ax2.xaxis.set_major_formatter(mdates.DateFormatter("%d %b\n%Y"))
ax2.xaxis.set_major_locator(mdates.WeekdayLocator(byweekday=mdates.WE, interval=7))
ax2.grid(alpha=0.3)
ax2.set_ylim(bottom=0)
for side in ["top", "right"]:
    ax2.spines[side].set_visible(False)
ax2.annotate(f"{ev['pct_change']:+.0f}% for 4 weeks\n\\${ev['old_price']:.3f} → \\${ev['new_price']:.3f} → "
             f"\\${ev['outcome_price_4w']:.3f}", xy=(ev["published_week"], ev["new_price"]),
             xytext=(0.03, 0.12), textcoords="axes fraction", fontsize=10,
             arrowprops=dict(arrowstyle="->", color="black"))
# note on the figure (not the left axes), so it doesn't stretch the left panel and push the zoom away
fig.text(0.01, 0.01, f"The {ev['pct_change']:+.0f}% jump in {ev['published_week']:%B %Y} was back to the old price "
         "4 weeks later. But this drug flips between price levels again and again,\n"
         "so a reverted spike is not always a one-off data error.", fontsize=9, color="dimgray")
fig.tight_layout(rect=(0, 0.07, 1, 1))
fig.savefig(FIG / "d_reverted_spike_phenobarbital.png", dpi=150)
plt.close(fig)

# ---- e. distribution of price changes ----
dist = rows("""SELECT CASE WHEN is_brand THEN 'brand' ELSE 'generic' END AS kind,
                      CAST(FLOOR(GREATEST(LEAST(pct_change, 30), -30)) AS INT) AS bin, COUNT(*) AS n
               FROM price_events WHERE NOT mix_only GROUP BY 1, 2""")
fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
for ax, kind, colour in [(axes[0], "generic", "steelblue"), (axes[1], "brand", "darkorange")]:
    d = sorted((r["bin"], r["n"]) for r in dist if r["kind"] == kind)
    ax.bar([b + 0.5 for b, _ in d], [n for _, n in d], width=0.9, color=colour)
    ax.set_yscale("log")
    ax.set_ylabel("Number of price changes\n(log scale)")
    k = N["by_kind"][kind]
    ax.set_title(f"{kind.capitalize()} drugs: {k['events']:,} changes, median size {k['median_abs_pct']:.2f}%, "
                 f"{k['share_within_5pct']:.0%} within ±5%", fontsize=11, loc="left")
    ax.grid(alpha=0.3, axis="y")
    for side in ["top", "right"]:
        ax.spines[side].set_visible(False)
axes[1].set_xlabel("Size of one price change, % vs previous published price  "
                   "(changes beyond ±30% are piled into the end bars)")
axes[1].set_xticks(range(-30, 31, 5))
axes[1].set_xticklabels([("≤" if t == -30 else "≥" if t == 30 else "") + f"{t}%" for t in range(-30, 31, 5)])
fig.suptitle("How big is a typical price change?", fontsize=14, x=0.01, ha="left")
fig.tight_layout()
fig.savefig(FIG / "e_price_change_distribution.png", dpi=150)
plt.close(fig)

# ---- f. January effect ----
months = rows("""SELECT CASE WHEN is_brand THEN 'brand' ELSE 'generic' END AS kind,
                        MONTH(effective_date) AS m, COUNT(*) AS n
                 FROM price_events WHERE pct_change >= 1 AND NOT mix_only GROUP BY 1, 2""")
fig, ax = plt.subplots(figsize=(10, 5))
names = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
for offset, kind, colour in [(-0.2, "brand", "darkorange"), (0.2, "generic", "steelblue")]:
    counts = {r["m"]: r["n"] for r in months if r["kind"] == kind}
    total = sum(counts.values())
    ax.bar([m + offset for m in range(1, 13)], [counts.get(m, 0) / total * 100 for m in range(1, 13)],
           width=0.4, color=colour, label=f"{kind.capitalize()} ({total:,} increases)")
ax.set_xticks(range(1, 13))
ax.set_xticklabels(names)
ax.set_ylabel("Share of that group's price increases, %")
ax.set_xlabel("Month the increase took effect (2022-2026)")
ax.set_title("The January effect: brand price increases cluster in January", fontsize=14, loc="left", pad=12)
ax.legend(frameon=False)
ax.grid(alpha=0.3, axis="y")
for side in ["top", "right"]:
    ax.spines[side].set_visible(False)
note(ax, "Increases of 1% or more, excluding mix-only changes. Each colour adds up to 100%. "
         "2026 covers January to early November only.")
fig.tight_layout()
fig.savefig(FIG / "f_january_effect.png", dpi=150)
plt.close(fig)

spark.stop()
# saved after the charts, which add a few numbers (e.g. held_weeks)
(REPORTS / "day1_numbers.json").write_text(json.dumps(N, indent=2, default=str), encoding="utf-8")


# =====================================================================================
# 3. Report
# =====================================================================================
def pct(x, digits=1):
    return f"{x * 100:.{digits}f}%"


o, g, b, e, lg = N["overview"], N["by_kind"]["generic"], N["by_kind"]["brand"], N["events"], N["lag"]
bm, gk = N["big_moves"], N["group_key"]
csv_total, clean_total = sum(N["csv_rows"].values()), sum(N["clean_rows"].values())
otezla_list = ", ".join(f"{r['pct_change']:+.1f}% ({r['effective_date']:%b %Y})" for r in N["otezla_increases"])
rows_by_year = " · ".join(f"{y}: {N['clean_rows'][y]:,}" for y in YEARS)

report = f"""# Day 1 data report: NADAC drug prices 2022–2026

*Generated by `make_report.py` on {date.today():%d %B %Y}. Every number below is computed from the project's
tables (`data/clean/nadac`, `data/tables/`); the raw values are in `reports/day1_numbers.json`.*

## 1. The data

**Source.** CMS's *National Average Drug Acquisition Cost* (NADAC): the average price retail pharmacies pay to
buy a drug, from a monthly invoice survey, republished every week. One CSV per year from data.medicaid.gov
(dataset IDs listed in `download.py`).

| | |
|---|---|
| Years | 2022–2026 (2026 is partial: up to {o['last_week']:%d %b %Y}) |
| Rows (one NDC in one weekly file) | {o['rows']:,} ({rows_by_year}) |
| NDCs (individual product packages) | {o['ndcs']:,} |
| Drug groups ("same product, priced together") | {N['groups']:,} |
| Weeks | {o['weeks']} (every Wednesday from {o['first_week']:%d %b %Y} to {o['last_week']:%d %b %Y}) |
| Group-week rows / price-change events | {N['group_weekly_rows']:,} / {e['n']:,} |

**Cleaning.** Every file is read as text first (so 11-digit NDCs keep their leading zeros), then typed with
"try" conversions and checked against 12 rules (valid NDC, price > 0, real dates, known codes, week inside its
year...). Failing rows would go to a rejects folder with a reason instead of being dropped: **{N['rejects']} rows were
rejected** out of {csv_total:,}. Two format differences between years were fixed in the cleaning code: 2022–2023
column names use underscores, and one 2022 week is written `1/12/2022` instead of `01/12/2022`.

## 2. Can we trust it?

**Checks that passed**

- **Complete timeline:** {o['weeks']} weekly files, {N['week_gaps']} gaps, {N['weeks_in_two_files']} weeks appearing in two yearly files, {N['not_wednesday']} dates that are not a Wednesday.
- **No duplicates:** {N['duplicate_ndc_weeks']} repeated (NDC, week) pairs across all years.
- **Rows reconcile at every step:** {csv_total:,} rows in the CSVs = {clean_total:,} clean rows = {N['rows_in_group_weekly']:,} rows counted in the weekly group table.
- **The drug-group key works:** in {gk['multi_ndc_group_weeks']:,} group-weeks with 2+ NDCs, all NDCs share one identical price {pct(gk['share_one_price'])} of the time, and {pct(gk['share_one_price_no_code5'])} once we leave out NDCs that CMS prices by package size (explanation code 5, {pct(gk['share_rows_code5'], 0)} of rows).
- **Reworded names don't break histories:** {N['reworded_ndcs']} NDCs had their description reworded by CMS (e.g. `TESTOSTERON` → `TESTOSTERONE`); each NDC is filed under its most recent name, so it keeps one continuous history.

**Known limitations**

- **Price tiers:** in {pct(N['tier_share'])} of group-weeks, the group has more than one price (mostly package sizes). We use the median NDC price, which is always a real price, but it can move when the NDC mix changes.
- **Mix-only moves:** {e['n_mix_only']:,} events ({pct(e['mix_only'])}) changed the median only because NDCs entered or left the group. Rare overall, but {N['big_moves_mix_only']['n']} of the {N['big_moves_mix_only']['total']:,} moves larger than 50%. They are flagged, not deleted.
- **Gaps:** {e['n_after_gap']:,} events ({pct(e['after_gap'])}) come right after weeks where the group was missing, so their exact timing is uncertain (flagged `after_gap`).
- **Backdating:** {pct(lg['share_0'])} of changes take effect the week they are published, but brand list-price increases are usually published later ({pct(N['lag_brand_increases']['share_backdated'], 0)} backdated, median {N['lag_brand_increases']['median_days']:.0f} days). The {lg['n_over_90']} "lags" over 90 days are not backdating: {lg['n_over_90_codes_4_5']} of them are carried-forward package-size prices (codes 4 + 5).
- **2026 is partial,** and the newest events have no "what happened next" yet.

## 3. How prices behave

- **Generics move often, in small steps.** {g['events']:,} changes across {g['groups_with_events']:,} groups, typically every {g['median_weeks_between']:.0f} weeks (CMS's monthly survey). The median change is {g['median_abs_pct']:.2f}%, {pct(g['share_within_5pct'], 0)} are within ±5%, and they go up and down about equally ({pct(g['share_up'], 0)} up). Overall, {pct(N['share_group_weeks_changed'], 0)} of group-weeks show a change. *(Chart e)*
- **Brands are flat, then step up in January.** Brand changes are tiny (median {b['median_abs_pct']:.2f}%), but of the {N['january']['brand']['increases']:,} brand increases of 1% or more, **{pct(N['january']['brand']['share_january'], 0)} took effect in January**, against {pct(N['january']['generic']['share_january'], 0)} for generics. Otezla 30 mg is a textbook case: {otezla_list}. *(Charts c, f)*
- **Big moves are rare, and the clear ones stick.** {bm['n']:,} real moves exceeded ±50%. Only **{bm['reverted']}** returned to the old price within 4 weeks (like phenobarbital, chart d, although that drug flips between price levels repeatedly, so a reversal is not always a one-off error). Eight weeks later, {bm['closer_to_new_8w']:,} of the {bm['with_8w_price']:,} with an 8-week price were still closer to the new level than the old one: level shifts such as isosorbide mononitrate (chart a, held for {N['example_events']['isosorbide']['held_weeks']} weeks before easing) and prucalopride (chart b). Our strict "within ±2% of the new price" test confirms only {bm['persisted_2pct']}, because generic prices keep drifting month to month.

![a](figures/a_level_shift_up_isosorbide.png)
![b](figures/b_level_shift_down_prucalopride.png)
![c](figures/c_otezla_january_staircase.png)
![d](figures/d_reverted_spike_phenobarbital.png)
![e](figures/e_price_change_distribution.png)
![f](figures/f_january_effect.png)

## 4. What this means for Day 2

- **Define an anomaly relative to the drug's own normal.** A 5% move is routine for a generic and unusual for a brand outside January. Judge each change against its group's history and class, and treat January brand increases as expected events.
- **Separate two kinds of anomaly:** level shifts (a new price that sticks, the common big event) and spikes that revert (rare; some may be data blips, others are drugs that genuinely jump back and forth). They need different detectors and different actions.
- **Model drug groups, not NDCs.** Many NDCs share one price; per-NDC models would count one event dozens of times.
- **Use only what was known at the time:** order everything by the *published* week, never the effective date (it can be backdated), so no model peeks ahead.
- **Skip `mix_only` events by default and handle `after_gap` with care.**
- **Outcome columns are for evaluation only.** `outcome_*` in `price_events` describes the future; it is how we will score detectors, never an input to them.
"""
(REPORTS / "day1_data_report.md").write_text(report, encoding="utf-8")
print("wrote", REPORTS / "day1_data_report.md", "and", len(list(FIG.glob("*.png"))), "charts")
