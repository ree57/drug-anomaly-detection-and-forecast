-- forecast_dataset: one row per (drug group, forecast origin week t), origins every 4th published week.
-- Built by forecast_setup.py, which keeps only rows with target_status = 'ok'.
--
-- wk = index of the published week (0 = 2022-01-05; the publication calendar has no missing weeks).
-- Every feature uses window frames that END at the current row (week t) or before: past-only.
-- !!! Columns prefixed target_ / tgt_ look INTO THE FUTURE (weeks t+1 .. t+12). Never use them as features.
--
-- Price series = the raw published group median (what NADAC shows). Count-type features use REAL price changes
-- only (not mix_only, not artefact_suspect), like Step 8. Both flags are known when the week is published.

WITH weeks AS (
    SELECT as_of_date, CAST(ROW_NUMBER() OVER (ORDER BY as_of_date) - 1 AS INT) AS wk
    FROM (SELECT DISTINCT as_of_date FROM group_weekly)
),

gw AS (
    SELECT g.drug_group, g.as_of_date, w.wk, g.median_price, g.n_ndcs_priced, g.n_distinct_prices,
           LN(g.median_price)                                                             AS lp,
           -- log change since this group's previous PRICED week (across a gap if there was one)
           LN(g.median_price) - LAG(LN(g.median_price)) OVER (PARTITION BY g.drug_group ORDER BY w.wk) AS dl,
           COALESCE(g.price_changed AND NOT e.mix_only AND NOT e.artefact_suspect, FALSE) AS real_change,
           COALESCE(e.artefact_suspect, FALSE)                                            AS is_artefact,
           COALESCE(e.mix_only, FALSE)                                                    AS is_mix_only,
           COUNT(*) OVER (PARTITION BY g.drug_group ORDER BY w.wk ROWS UNBOUNDED PRECEDING) AS n_priced_to_date
    FROM group_weekly AS g
    JOIN weeks AS w USING (as_of_date)
    LEFT JOIN price_events AS e ON e.drug_group = g.drug_group AND e.published_week = g.as_of_date
),

feat AS (
    SELECT gw.*,
        -- price "as of" k weeks ago = last priced week at or before t-k (LAST ... IGNORE NULLS over a RANGE frame)
        LAST(lp, TRUE) OVER (PARTITION BY drug_group ORDER BY wk RANGE BETWEEN UNBOUNDED PRECEDING AND 4 PRECEDING)  AS lp_4,
        LAST(wk, TRUE) OVER (PARTITION BY drug_group ORDER BY wk RANGE BETWEEN UNBOUNDED PRECEDING AND 4 PRECEDING)  AS src_wk_4,
        LAST(lp, TRUE) OVER (PARTITION BY drug_group ORDER BY wk RANGE BETWEEN UNBOUNDED PRECEDING AND 12 PRECEDING) AS lp_12,
        LAST(wk, TRUE) OVER (PARTITION BY drug_group ORDER BY wk RANGE BETWEEN UNBOUNDED PRECEDING AND 12 PRECEDING) AS src_wk_12,
        LAST(lp, TRUE) OVER (PARTITION BY drug_group ORDER BY wk RANGE BETWEEN UNBOUNDED PRECEDING AND 26 PRECEDING) AS lp_26,
        LAST(wk, TRUE) OVER (PARTITION BY drug_group ORDER BY wk RANGE BETWEEN UNBOUNDED PRECEDING AND 26 PRECEDING) AS src_wk_26,
        LAST(lp, TRUE) OVER (PARTITION BY drug_group ORDER BY wk RANGE BETWEEN UNBOUNDED PRECEDING AND 52 PRECEDING) AS lp_52,
        LAST(wk, TRUE) OVER (PARTITION BY drug_group ORDER BY wk RANGE BETWEEN UNBOUNDED PRECEDING AND 52 PRECEDING) AS src_wk_52,
        -- last 26 published weeks = weeks t-25 .. t
        STDDEV_SAMP(dl)               OVER (PARTITION BY drug_group ORDER BY wk RANGE BETWEEN 25 PRECEDING AND CURRENT ROW) AS vol_26w,
        SUM(CAST(real_change AS INT)) OVER (PARTITION BY drug_group ORDER BY wk RANGE BETWEEN 25 PRECEDING AND CURRENT ROW) AS n_changes_26w,
        MIN(wk)                       OVER (PARTITION BY drug_group ORDER BY wk RANGE BETWEEN 25 PRECEDING AND CURRENT ROW) AS src_wk_26w_window,
        LAST(IF(real_change, wk, NULL), TRUE) OVER (PARTITION BY drug_group ORDER BY wk ROWS UNBOUNDED PRECEDING) AS last_change_wk,
        LAST(IF(real_change, dl, NULL), TRUE) OVER (PARTITION BY drug_group ORDER BY wk ROWS UNBOUNDED PRECEDING) AS last_change_dl,
        MIN(wk) OVER (PARTITION BY drug_group)                                                       AS first_wk,
        -- TARGET (future): the price exactly 12 published weeks later, and what happened in between
        FIRST(lp)  OVER (PARTITION BY drug_group ORDER BY wk RANGE BETWEEN 12 FOLLOWING AND 12 FOLLOWING)   AS target_lp_12,
        SUM(CAST(is_artefact AS INT)) OVER (PARTITION BY drug_group ORDER BY wk RANGE BETWEEN 1 FOLLOWING AND 12 FOLLOWING) AS tgt_n_artefact,
        SUM(CAST(is_mix_only AS INT)) OVER (PARTITION BY drug_group ORDER BY wk RANGE BETWEEN 1 FOLLOWING AND 12 FOLLOWING) AS tgt_n_mix_only
    FROM gw
),

market AS (
    -- median 4-week log change of all generic groups priced at week t (past-only: uses weeks <= t)
    SELECT f.wk, PERCENTILE(f.lp - f.lp_4, 0.5) AS market_move_4w,
           AVG(CAST(f.lp <> f.lp_4 AS INT))     AS market_share_moved_4w,
           COUNT(*)                             AS market_n_generics
    FROM feat AS f JOIN group_dim AS d ON d.drug_group = f.drug_group
    WHERE d.classification = 'G' AND f.lp_4 IS NOT NULL
    GROUP BY f.wk
),

max_week AS (SELECT MAX(wk) AS last_wk FROM weeks)

SELECT
    f.drug_group,
    f.as_of_date                                                      AS origin_week,
    f.wk                                                              AS origin_wk,
    DATE_ADD(f.as_of_date, 84)                                        AS target_week,
    d.classification,
    d.classification <> 'G'                                           AS is_brand,
    d.pricing_unit,
    -- features (weeks <= t)
    f.lp - f.lp_4                                                     AS chg_4w,
    f.lp - f.lp_12                                                    AS chg_12w,
    f.lp - f.lp_26                                                    AS chg_26w,
    f.lp - f.lp_52                                                    AS chg_52w,
    f.lp_52 IS NULL                                                   AS chg_52w_missing,
    f.vol_26w,
    f.n_changes_26w,
    -- never a real change since the group appeared: count from its first week, last change size 0
    f.wk - COALESCE(f.last_change_wk, f.first_wk)                     AS weeks_since_change,
    COALESCE(f.last_change_dl, 0)                                     AS last_change_size,
    f.last_change_wk IS NULL                                          AS no_change_seen,
    f.lp                                                              AS log_price,
    f.n_ndcs_priced                                                   AS n_ndcs,
    f.n_distinct_prices,
    YEAR(DATE_ADD(f.as_of_date, 84)) > YEAR(f.as_of_date) OR MONTH(DATE_ADD(f.as_of_date, 7)) = 1
                                                                      AS next_12w_includes_january,
    MONTH(f.as_of_date)                                               AS month,
    m.market_move_4w,
    m.market_share_moved_4w,
    -- provenance for the leakage check: latest week index each feature read (must be <= origin_wk)
    GREATEST(f.wk, COALESCE(f.src_wk_4, -1), COALESCE(f.src_wk_12, -1), COALESCE(f.src_wk_26, -1),
             COALESCE(f.src_wk_52, -1), COALESCE(f.last_change_wk, -1))  AS feat_max_src_wk,
    f.n_priced_to_date                                                AS history_weeks,
    -- TARGET (future, evaluation/training label only)
    f.target_lp_12 - f.lp                                             AS target_log_change,
    (EXP(f.target_lp_12 - f.lp) - 1) * 100                            AS target_pct_change,
    CASE WHEN f.target_lp_12 IS NULL THEN NULL
         WHEN EXP(f.target_lp_12 - f.lp) - 1 >  0.02 THEN 'up'
         WHEN EXP(f.target_lp_12 - f.lp) - 1 < -0.02 THEN 'down' ELSE 'flat' END AS target_direction,
    COALESCE(f.tgt_n_artefact, 0) > 0                                 AS tgt_has_artefact,
    COALESCE(f.tgt_n_mix_only, 0) > 0                                 AS tgt_has_mix_only,
    CASE WHEN f.target_lp_12 IS NOT NULL THEN 'ok'
         WHEN f.wk + 12 > mw.last_wk     THEN 'beyond_data_end'
         ELSE 'not_priced_at_t_plus_12' END                           AS target_status,
    -- time splits (dates in reports/forecast_setup.md)
    CASE WHEN DATE_ADD(f.as_of_date, 84) <  DATE'2025-01-01' THEN 'train'
         WHEN f.as_of_date               <  DATE'2025-01-01' THEN 'purged'
         ELSE 'test' END                                              AS split,
    CASE WHEN f.as_of_date BETWEEN DATE'2025-01-01' AND DATE'2025-05-21' THEN 1
         WHEN f.as_of_date BETWEEN DATE'2025-06-18' AND DATE'2025-11-05' THEN 2
         WHEN f.as_of_date BETWEEN DATE'2025-12-03' AND DATE'2026-03-25' THEN 3
         WHEN f.as_of_date BETWEEN DATE'2026-04-22' AND DATE'2026-08-12' THEN 4 END AS fold
FROM feat AS f
JOIN group_dim AS d ON d.drug_group = f.drug_group
LEFT JOIN market AS m ON m.wk = f.wk
CROSS JOIN max_week AS mw
WHERE f.wk % 4 = 0                -- origins: every 4th published week
  AND f.n_priced_to_date >= 26    -- at least 26 priced weeks at or before t
