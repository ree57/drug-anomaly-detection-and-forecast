# Step 11: dashboard notes

## Running it

```bash
python export_dashboard.py          # pipeline outputs -> dashboard_data/ (pandas + LightGBM, ~70 s)
streamlit run dashboard/app.py      # http://localhost:8501
```

- **What the app reads:** only `dashboard_data/`, with no Spark and no model training.
- **Deployment requirements:** `dashboard/requirements.txt` sits next to `app.py`, so Streamlit Community Cloud
  installs that file rather than the full pipeline requirements.

## Family grouping fix (`families.py`)

**How prefixes are found.** A store-brand / repackager prefix is a first word that passes both tests:
- **The rest of the name exists elsewhere:** for at least 10 groups, the name without that word is itself another
  group's name.
- **It labels unrelated products:** it is followed by at least 20 different next words.

**Found in the data:**

| Prefix | Groups | Rest of name also exists | Different next words |
|---|---|---|---|
| SM | 178 | 75 | 92 |
| FT | 191 | 74 | 90 |
| HM | 175 | 71 | 94 |
| QC | 179 | 60 | 93 |
| GNP | 171 | 57 | 95 |
| GS | 120 | 45 | 56 |

**Rejected:** ARMOUR, NP and NIVA. Each is always followed by THYROID, so they are product names, not stores.
CVS does not appear in NADAC descriptions.

**Before vs after:**

| Families | Before | After |
|---|---|---|
| Event families (per week, all real events), largest | 608 | **85** |
| Next largest | 514, 469 | 71, 69 |
| Watch-list families (per origin), largest | 595 | **81** |
| Next largest | 586, 574 | 69, 66 |

- **Before:** the giant families were store-brand OTC lines chained together (FT, SM, GNP, HM, GS, …).
- **After:** the largest families are mostly one molecule in many strengths, such as methylphenidate (41) and
  amlodipine (29–31). Some are joined to other drugs through an identical big % change.
- **Remaining gap:** one generic first word, "NASAL" (22 groups at one origin), still links a few products.

**Where it applies:** `alerts.py`, the watch list and the export. `evaluate.py` and the Step 8 family metrics are
unchanged, because they are the committed record. With the new grouping, 2025–26 has 113 alert families (Step 8d:
112), and the busiest week has 12 (Step 8d: 13).

## Exported data (`dashboard_data/`, 10.75 MB)

| File | Size | Content |
|---|---|---|
| alerts.parquet | 0.19 MB | 2,181 high/medium alerts in 1,498 families, 97 weeks with alerts, reason text and cautions |
| watchlist.parquet | 4.44 MB | 144,706 rows, 25 origins: 22 test origins scored walk-forward by their own fold's model, plus 3 newest origins (17,394 rows) scored by a "current" model; outcome attached where known |
| prices.parquet | 2.69 MB | weekly group median price |
| events.parquet | 3.15 MB | every price event with its alert status |
| groups.parquet | 0.27 MB | drug group id, name, unit, class |
| summary.json | 0.01 MB | About-page metrics, read from `reports/metrics_*.csv` |

**Newest origins:**
- **Features:** `forecast_setup.py` now also saves the newest origins (no 12-week outcome yet) as
  `data/tables/forecast_latest`, and `competition_features.py` covers them.
- **Leakage checks:** rerun on all 296,919 rows, 0 violations and 0 truncation mismatches. The re-run reproduced
  Step 9 exactly.
- **Models:** the fold models reproduce Step 10c's trees (277 / 146 / 242 / 245). The current model uses 217 trees,
  trained on origins up to 2026-08-12.
