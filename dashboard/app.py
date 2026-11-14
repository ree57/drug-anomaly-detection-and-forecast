"""Drug price watch: NADAC anomaly alerts, a "likely to move" watch list and a drug lookup.

Run locally:   streamlit run dashboard/app.py
Data:          dashboard_data/ (made by export_dashboard.py). No Spark, no model training in the app.
"""
import json
from datetime import timedelta
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

GITHUB_URL = "https://github.com/ree57/drug-anomaly-detection-and-forecast"
DATA = Path(__file__).resolve().parent.parent / "dashboard_data"
MOVE_LABEL = "Chance of moving ≥ 5% (up or down) in the next 12 weeks"
CONF_TEXT = {"high": "High confidence", "medium": "Medium confidence"}

st.set_page_config(page_title="Drug price watch (NADAC)", page_icon="💊", layout="wide")


@st.cache_data
def load(name):
    return pd.read_parquet(DATA / name)


@st.cache_data
def load_summary():
    return json.loads((DATA / "summary.json").read_text(encoding="utf-8"))


def kind(is_brand):
    return "Brand" if is_brand else "Generic"


def money(x):
    return f"${x:,.4f}" if x < 1 else f"${x:,.2f}"


def chance_band(p):
    return "likely" if p >= 0.6 else "possible" if p >= 0.3 else "unlikely"


def family_title(names, n_show=3):
    names = list(dict.fromkeys(names))
    return " / ".join(names[:n_show]) + (f" (+{len(names) - n_show} more)" if len(names) > n_show else "")


# ------------------------------------------------------------------ weekly alerts
def page_alerts():
    st.title("Weekly price alerts")
    st.markdown(
        "Each week CMS publishes the **NADAC**: the average price pharmacies pay for each drug. "
        "This page lists the price changes that were **unusual**: much bigger than normal for the market *and* "
        "for that drug's own history. Expected changes (such as January brand list-price increases) and changes "
        "caused only by products entering or leaving a group are filtered out.\n\n"
        "- **High confidence:** both detectors flag the change.\n"
        "- **Medium confidence:** one of the two detectors flags it.\n"
        "- Related drugs that moved together in the same week are shown as one **family**.")
    alerts = load("alerts.parquet")
    weeks_all = sorted(load("prices.parquet").week.unique(), reverse=True)
    counts = alerts.groupby("published_week").family.nunique()
    default = weeks_all.index(max(counts.index))
    week = st.selectbox("Published week", weeks_all, index=default,
                        format_func=lambda w: f"{w}  ({counts.get(w, 0)} alert famil{'y' if counts.get(w, 0) == 1 else 'ies'})")
    wk = alerts[alerts.published_week == week]
    if wk.empty:
        st.info("No alerts this week: no price change was unusual enough for either detector.")
        return
    if wk.period.iloc[0].startswith("2022"):
        st.caption("Weeks before 2025 were also used to set the detectors' thresholds, so their alerts are "
                   "in-sample. Alerts from 2025 on are fully out of sample.")
    fams = wk.groupby("family", sort=False).agg(rank=("family_rank", "first"), tier=("family_tier", "first"))
    c1, c2, c3 = st.columns(3)
    c1.metric("Alert families", len(fams))
    c2.metric("High confidence", int((fams.tier == 2).sum()))
    c3.metric("Medium confidence", int((fams.tier == 1).sum()))
    for fid, f in fams.sort_values("rank").iterrows():
        members = wk[wk.family == fid].sort_values("score_final", ascending=False)
        top = members.iloc[0]
        conf = CONF_TEXT["high" if f.tier == 2 else "medium"]
        with st.container(border=True):
            st.subheader(f"{int(f['rank'])}. {family_title(members.description)}")
            st.markdown(f"**{conf}** · {kind(top.is_brand)} · {len(members)} drug group"
                        f"{'s' if len(members) > 1 else ''} · change {top["pct_change"]:+.1f}%")
            table = pd.DataFrame({"Drug": members.description, "Unit": members.pricing_unit,
                                  "Brand / generic": members.is_brand.map(kind),
                                  "Old price": members.old_price.map(money), "New price": members.new_price.map(money),
                                  "Change": members["pct_change"].map(lambda x: f"{x:+.1f}%"),
                                  "Confidence": members.confidence.map(CONF_TEXT)})
            st.dataframe(table, hide_index=True, width="stretch")
            why = top.reason.split("; caution:")[0]
            st.markdown(f"**Why it was flagged:** {why[0].upper() + why[1:]}.")
            if top.caution_text:
                for c in top.caution_text.split("; "):
                    st.warning(f"Caution: {c}.")


# ------------------------------------------------------------------ watch list
def page_watchlist():
    st.title("Watch list: drugs likely to move")
    st.markdown(
        f"For each date, a model estimates the **chance that a drug's price moves by 5% or more, up or down, over "
        f"the next 12 weeks**. It does **not** predict the direction or the new price: in our tests, predicting the "
        f"size of a change did not beat \"no change\", while ranking *which* drugs will move worked well.\n\n"
        f"Every date is scored only by a model trained on data from before that date. Related products are grouped "
        f"into one family.")
    wl = load("watchlist.parquet")
    origins = sorted(wl.origin_week.unique(), reverse=True)
    known = wl.groupby("origin_week").outcome_known.first()
    origin = st.selectbox("Forecast date", origins, format_func=lambda o: f"{pd.Timestamp(o).date()}  "
                          f"({'outcome known' if known[o] else 'outcome not known yet'})")
    o = wl[wl.origin_week == origin]
    fam = (o.groupby("family", sort=False)
             .agg(rank=("family_rank", "first"), p=("family_p", "first"), n=("gid", "size"),
                  drugs=("description", lambda s: family_title(s, 2)), brand=("is_brand", "max"),
                  recent=("pct_12w_before", "median"), moved=("moved_5pct", "sum"),
                  lo=("target_pct_change", "min"), hi=("target_pct_change", "max"))
             .sort_values("rank").head(25))
    end = pd.Timestamp(origin).date() + timedelta(weeks=12)
    c1, c2, c3 = st.columns(3)
    c1.metric("Drugs scored", f"{len(o):,}")
    c2.metric("Families", f"{o.family.nunique():,}")
    if known[origin]:
        top_ids = set(fam.index)
        c3.metric("Top 25 families with a member that moved ≥ 5%", f"{int((fam.moved > 0).sum())} of 25",
                  help=f"Across all drugs at this date, {o.moved_5pct.mean():.0%} moved ≥ 5% by {end}.")
        st.caption(f"Outcome period: {pd.Timestamp(origin).date()} → {end}. Across all drugs, "
                   f"{o.moved_5pct.mean():.0%} moved ≥ 5%; in the top 25 families, "
                   f"{int(o[o.family.isin(top_ids)].moved_5pct.sum())} of {int(fam.n.sum())} drug groups did.")
    else:
        c3.metric("Outcome", "not known yet", help=f"The 12-week window ends {end}.")
    chance = "Chance to move ≥ 5%"
    table = pd.DataFrame({"Rank": fam["rank"].astype(int), chance: fam.p})
    if known[origin]:
        table["Moved ≥ 5%?"] = [f"yes ({int(m)} of {n})" if m > 0 else "no" for m, n in zip(fam.moved, fam.n)]
        table["Actual change"] = [f"{lo:+.1f}%" if abs(hi - lo) < 0.05 else f"{lo:+.1f}% to {hi:+.1f}%"
                                  for lo, hi in zip(fam.lo, fam.hi)]
    table["Drug(s)"] = fam.drugs.to_numpy()
    table["Drug groups"] = fam.n.to_numpy()
    table["Brand / generic"] = fam.brand.map(lambda b: "Brand" if b else "Generic").to_numpy()
    table["Change in the 12 weeks before"] = fam.recent.map(lambda x: f"{x:+.1f}%").to_numpy()
    st.dataframe(table, hide_index=True, width="stretch",
                 column_config={chance: st.column_config.ProgressColumn(chance, help=MOVE_LABEL, format="percent",
                                                                        min_value=0, max_value=1)})
    st.caption("How to read the chance: about 60% or more = likely, 30–60% = possible, under 30% = unlikely. "
               "In testing, the highest chances were somewhat optimistic (predicted ~83%, actual ~71% for generics).")


# ------------------------------------------------------------------ drug lookup
def page_lookup():
    st.title("Drug lookup")
    groups = load("groups.parquet")
    q = st.text_input("Search by drug name", value="dapagliflozin", placeholder="e.g. insulin, amoxicillin")
    hits = groups[groups.description.str.contains(q.strip(), case=False, regex=False)] if q.strip() else groups.head(0)
    if hits.empty:
        st.info("No drug matches that name. Try a shorter search, e.g. the first word of the drug name.")
        return
    label = {r.gid: f"{r.description}  ({r.pricing_unit}, {kind(r.is_brand).lower()})" for r in hits.itertuples()}
    gid = st.selectbox(f"{len(hits):,} match{'es' if len(hits) != 1 else ''}", list(label), format_func=label.get)
    g = groups[groups.gid == gid].iloc[0]
    st.subheader(g.description)
    st.caption(f"Priced per {g.pricing_unit} · {kind(g.is_brand)} ({g.classification}) · "
               f"in NADAC {g.first_seen} → {g.last_seen}")

    prices = load("prices.parquet").query("gid == @gid")
    events = load("events.parquet").query("gid == @gid").copy()
    events["Marker"] = events.alert.map({"high": "Alert: high confidence", "medium": "Alert: medium confidence",
                                         "not scored: mix change": "Not scored (product mix / data artefact)",
                                         "not scored: artefact": "Not scored (product mix / data artefact)"}
                                        ).fillna("Other price change")
    domain = ["Alert: high confidence", "Alert: medium confidence", "Not scored (product mix / data artefact)",
              "Other price change"]
    line = alt.Chart(prices).mark_line(interpolate="step-after").encode(
        x=alt.X("week:T", title="Published week"), y=alt.Y("price:Q", title=f"Price per {g.pricing_unit} ($)"))
    marks = alt.Chart(events[events.Marker != "Other price change"]).mark_point(size=120, filled=True).encode(
        x="week:T", y=alt.Y("new_price:Q"),
        color=alt.Color("Marker:N", scale=alt.Scale(domain=domain[:3], range=["#d62728", "#ff7f0e", "#7f7f7f"]),
                        legend=alt.Legend(title=None, orient="top")),
        shape=alt.Shape("Marker:N", scale=alt.Scale(domain=domain[:3], range=["circle", "circle", "cross"]),
                        legend=None),
        tooltip=[alt.Tooltip("week:T", title="Week"), alt.Tooltip("old_price:Q", title="Old price", format="$.4f"),
                 alt.Tooltip("new_price:Q", title="New price", format="$.4f"),
                 alt.Tooltip("pct_change:Q", title="Change %", format="+.1f"), alt.Tooltip("Marker:N", title="")])
    st.altair_chart((line + marks).properties(height=360), width="stretch")

    wl = load("watchlist.parquet")
    latest = wl.origin_week.max()
    mine = wl[(wl.gid == gid) & (wl.origin_week == latest)]
    if mine.empty:
        st.info(f"No watch-list estimate for {pd.Timestamp(latest).date()}: the drug needs at least 26 weeks of "
                "price history and a price that week.")
    else:
        p = float(mine.p_move.iloc[0])
        end = pd.Timestamp(latest).date() + timedelta(weeks=12)
        st.metric(f"Latest watch list ({pd.Timestamp(latest).date()} → {end})",
                  f"{chance_band(p).capitalize()} to move ≥ 5% (up or down)", f"estimated chance {p:.0%}",
                  delta_color="off")

    st.markdown("**Price changes**")
    tbl = events.sort_values("week", ascending=False)
    st.dataframe(pd.DataFrame({
        "Week": tbl.week, "Old price": tbl.old_price.map(money), "New price": tbl.new_price.map(money),
        "Change": tbl["pct_change"].map(lambda x: f"{x:+.1f}%"),
        "Alert": tbl.alert.map({"high": "High confidence", "medium": "Medium confidence", "none": "—",
                                "not scored: mix change": "Not scored: only the product mix changed",
                                "not scored: artefact": "Not scored: likely data artefact"})}),
        hide_index=True, width="stretch")


# ------------------------------------------------------------------ about
def page_about():
    s = load_summary()
    d = s["data"]
    st.title("About this dashboard")
    st.markdown(f"""
**Data.** The CMS *National Average Drug Acquisition Cost* (NADAC) weekly files, {d['first_week']} to
{d['last_week']} ({d['weeks']} weeks). NDCs (individual packages) are grouped into **{d['drug_groups']:,} drug groups**
(same description, pricing unit and brand/generic class). A group's weekly price is the median of its NDCs.
There are **{d['price_events']:,} price changes** in total.

**Alerts (anomaly detection).** No labelled "true anomalies" exist, so detectors were compared on proxy labels:
big moves that persisted, moves far outside a drug's own history, and expected events that should *not* be flagged
(January brand list-price increases, normal moves). The final detector combines a **hybrid score** (how big the
change is compared with all past changes, and how unusual it is for that drug) with an **Isolation Forest** second
opinion. Changes caused only by products entering or leaving a group are never scored.

**Watch list (forecasting).** Predicting *how much* a price will change in 12 weeks did not beat "no change" in
2025–26. Predicting *whether* it will move by 5% or more did work, so the watch list ranks drugs by that chance.
""")
    st.subheader("Alert detectors (test period 2025–26)")
    det = pd.DataFrame(s["anomaly_detectors"]).rename(columns={
        "detector": "Detector", "big_shift_recall_top100": "Big lasting moves found in top 100",
        "relative_recall_top100": "Drug-unusual moves found in top 100",
        "known_positive_share_top100": "Share of top 100 that are known anomalies",
        "hard_volatile_flagged_top1000": "Normal-for-this-drug big moves wrongly in top 1,000",
        "january_list_increases_flagged_top1000": "January list increases wrongly in top 1,000",
        "normal_moves_flagged_top1000": "Small normal moves wrongly in top 1,000"})
    st.dataframe(det, hide_index=True, width="stretch",
                 column_config={c: st.column_config.NumberColumn(format="%.2f") for c in det.columns[1:4]})
    a = s["alerts_test_period"]
    st.caption(f"2025–26: {a['alert_events']} alerts in {a['families']} families over {a['weeks']} published weeks; "
               f"{a['weeks_with_alerts']} weeks had any alert, at most {a['max_families_per_week']} families in one week.")

    st.subheader("Forecasting the size of the 12-week change (4 backtest folds, 2025–26)")
    fr = pd.DataFrame(s["forecast_regression"])
    fr = fr.pivot(index="model", columns="segment", values="skill").reindex(
        ["Predict no change", "Drift", "Seasonal (brands, January)", "LightGBM"])[["all", "generic", "brand"]]
    st.dataframe(fr.rename(columns=str.capitalize), width="stretch",
                 column_config={c: st.column_config.NumberColumn(format="%+.3f") for c in ["All", "Generic", "Brand"]})
    st.caption("Skill = how much smaller the error is than \"predict no change\" (0 = same, negative = worse). "
               f"A later attempt with a recent-data window and shrinking still scored "
               f"{s['forecast_regression_fix_skill']['all']:+.3f}.")

    st.subheader("Will it move ≥ 5%? (4 backtest folds, 2025–26)")
    mc = pd.DataFrame(s["move_classifier"])
    mc = mc[mc.segment.isin(["all", "generic", "brand"])].rename(columns={
        "model": "Method", "segment": "Drugs", "roc_auc": "Ranking quality (ROC AUC)",
        "precision_top100": "Share of top 100 that moved", "precision_top500": "Share of top 500 that moved",
        "base_rate": "Share of all drugs that moved"})
    st.dataframe(mc, hide_index=True, width="stretch",
                 column_config={c: st.column_config.NumberColumn(format="%.3f") for c in mc.columns[2:]})
    st.caption("ROC AUC: pick one drug that moved and one that didn't; how often does the method rank the mover "
               "higher? 0.5 = coin flip, 1.0 = perfect.")

    st.subheader("Limitations")
    st.markdown("""
- **No ground truth.** "Anomaly" labels are proxies built from what happened later; a person should review alerts.
- **Data artefacts.** Some alerts come from package-mix or classification changes rather than real price moves;
  they carry a caution note. Two known cases from 2025 still appear as alerts.
- **Calm 2025–26.** Prices moved about half as much as in 2022–24, and 2024 had market-wide swings, so models
  trained on older data can be over-confident.
- **Families are approximate.** Related products are grouped by name and identical changes; store-brand
  prefixes are ignored, but some unrelated products can still be linked.
- **The watch list predicts movement, not direction.** It never says a price will go up.
""")
    if GITHUB_URL:
        st.markdown(f"Code and full reports: [{GITHUB_URL}]({GITHUB_URL})")
    else:
        st.markdown("Code and full reports: GitHub link coming soon.")


nav = st.navigation([st.Page(page_alerts, title="Weekly alerts", icon="🚨", default=True),
                     st.Page(page_watchlist, title="Watch list", icon="👀", url_path="watchlist"),
                     st.Page(page_lookup, title="Drug lookup", icon="🔎", url_path="lookup"),
                     st.Page(page_about, title="About", icon="ℹ️", url_path="about")])
nav.run()
