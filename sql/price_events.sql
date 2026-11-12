-- price_events: one row per price change of a drug group
-- (a week in group_weekly where price_changed is true).
--
-- !!! WARNING: columns prefixed outcome_ look INTO THE FUTURE (what happened in the weeks AFTER the event).
-- !!! They exist only to evaluate events and detectors after the fact.
-- !!! They must NEVER be used as model inputs / features: at prediction time nobody knows them.

WITH gw AS (
    -- ALL weeks of every group (not just events), with look-back and look-ahead values.
    -- The look-ahead (LEAD) must happen here, BEFORE filtering to events: otherwise
    -- "4 rows ahead" would mean "4 events later" instead of "4 weeks later".
    SELECT
        *,
        LAG(as_of_date)        OVER w AS prev_published_week,
        LEAD(median_price, 4)  OVER w AS price_4_rows_ahead,
        LEAD(as_of_date, 4)    OVER w AS date_4_rows_ahead,
        LEAD(median_price, 8)  OVER w AS price_8_rows_ahead,
        LEAD(as_of_date, 8)    OVER w AS date_8_rows_ahead,
        LEAD(median_price, 12) OVER w AS price_12_rows_ahead,
        LEAD(as_of_date, 12)   OVER w AS date_12_rows_ahead,
        -- the next 1-4 rows of this group, as a list of (week, price), for outcome_reverted
        COLLECT_LIST(STRUCT(as_of_date, median_price)) OVER (
            PARTITION BY drug_group ORDER BY as_of_date
            ROWS BETWEEN 1 FOLLOWING AND 4 FOLLOWING) AS next_4_rows
    FROM group_weekly
    WINDOW w AS (PARTITION BY drug_group ORDER BY as_of_date)
),

events AS (
    -- keep only the weeks where the price changed. Window functions run AFTER the WHERE,
    -- so these two only see events: they number them and look back to the previous EVENT.
    SELECT
        *,
        ROW_NUMBER()    OVER e AS event_number,
        LAG(as_of_date) OVER e AS last_event_week
    FROM gw
    WHERE price_changed
    WINDOW e AS (PARTITION BY drug_group ORDER BY as_of_date)
),

ndc_rows AS (
    -- every clean NDC row, labelled with its canonical group, plus that NDC's OWN previous row
    SELECT
        d.drug_group, n.ndc, n.as_of_date, n.nadac_per_unit, n.effective_date, n.explanation_codes,
        LAG(n.nadac_per_unit) OVER (PARTITION BY n.ndc ORDER BY n.as_of_date) AS ndc_prev_price,
        LAG(n.as_of_date)     OVER (PARTITION BY n.ndc ORDER BY n.as_of_date) AS ndc_prev_as_of
    FROM nadac AS n
    JOIN ndc_dim AS d ON d.ndc = n.ndc
),

new_price_ndcs AS (
    -- the NDCs that carry the new price: same group, same week, own price = the new median
    SELECT ev.drug_group, ev.as_of_date, r.effective_date, r.explanation_codes
    FROM events AS ev
    JOIN ndc_rows AS r
      ON r.drug_group = ev.drug_group AND r.as_of_date = ev.as_of_date AND r.nadac_per_unit = ev.median_price
),

effective_dates AS (
    SELECT drug_group, as_of_date, effective_date, COUNT(*) AS n_ndcs
    FROM new_price_ndcs
    GROUP BY drug_group, as_of_date, effective_date
),

picked_date AS (
    -- if those NDCs disagree on the effective date: take the most common one, and the latest on a tie
    SELECT
        drug_group, as_of_date,
        MAX_BY(effective_date, STRUCT(n_ndcs, effective_date)) AS effective_date,
        COUNT(*)                                                AS n_effective_dates
    FROM effective_dates
    GROUP BY drug_group, as_of_date
),

codes AS (
    -- every explanation code seen on the new price, as one sorted list without repeats
    SELECT drug_group, as_of_date,
           ARRAY_SORT(ARRAY_DISTINCT(FLATTEN(COLLECT_LIST(explanation_codes)))) AS explanation_codes
    FROM new_price_ndcs
    GROUP BY drug_group, as_of_date
),

mix AS (
    -- NDCs present in BOTH the event week and the group's previous published week:
    -- how many of them changed their OWN price? If none did, the median moved only because
    -- NDCs entered or left the group (a mix-only change).
    SELECT
        ev.drug_group, ev.as_of_date,
        COUNT(*)                                                         AS n_ndcs_in_both_weeks,
        SUM(CASE WHEN r.nadac_per_unit <> r.ndc_prev_price THEN 1 ELSE 0 END) AS n_ndcs_changed
    FROM events AS ev
    JOIN ndc_rows AS r
      ON r.drug_group = ev.drug_group AND r.as_of_date = ev.as_of_date
     AND r.ndc_prev_as_of = ev.prev_published_week          -- this NDC was also there last time
    GROUP BY ev.drug_group, ev.as_of_date
)

SELECT
    ev.drug_group,
    g.classification,
    g.pricing_unit,
    ev.event_number,
    ev.as_of_date                                           AS published_week,
    ev.prev_published_week,
    ev.prev_price                                           AS old_price,
    ev.median_price                                         AS new_price,
    ev.pct_change,
    CASE WHEN ev.median_price > ev.prev_price THEN 'up' ELSE 'down' END AS direction,
    p.effective_date,
    p.n_effective_dates,
    DATEDIFF(ev.as_of_date, p.effective_date)               AS publication_lag_days,
    c.explanation_codes,
    MONTH(p.effective_date) = 1                             AS is_january,
    g.classification <> 'G'                                 AS is_brand,
    CAST(DATEDIFF(ev.as_of_date, ev.last_event_week) / 7 AS INT) AS weeks_since_last_event,

    -- quality flags
    COALESCE(m.n_ndcs_changed, 0) = 0                       AS mix_only,   -- also true if no NDC is in both weeks
    ROUND(m.n_ndcs_changed / m.n_ndcs_in_both_weeks, 4)     AS ndc_confirmed_share,
    ev.weeks_since_prev > 1                                 AS after_gap,
    ev.n_ndcs_priced,
    ev.n_distinct_prices,

    -- OUTCOMES: future information, for evaluation only. NEVER model inputs (see warning at top).
    -- NULL when that week isn't in the data (the group had a gap, or the data ends too soon).
    CASE WHEN ev.date_4_rows_ahead  = DATE_ADD(ev.as_of_date, 28) THEN ev.price_4_rows_ahead  END AS outcome_price_4w,
    CASE WHEN ev.date_8_rows_ahead  = DATE_ADD(ev.as_of_date, 56) THEN ev.price_8_rows_ahead  END AS outcome_price_8w,
    CASE WHEN ev.date_12_rows_ahead = DATE_ADD(ev.as_of_date, 84) THEN ev.price_12_rows_ahead END AS outcome_price_12w,
    CASE
        -- back within ±2% of the old price in any of the next 4 weeks
        WHEN EXISTS(ev.next_4_rows, x -> x.as_of_date <= DATE_ADD(ev.as_of_date, 28)
                                         AND ABS(x.median_price / ev.prev_price - 1) <= 0.02) THEN TRUE
        -- all 4 following weeks were observed and none went back: did not revert
        WHEN ev.date_4_rows_ahead = DATE_ADD(ev.as_of_date, 28) THEN FALSE
    END                                                     AS outcome_reverted,
    CASE WHEN ev.date_8_rows_ahead = DATE_ADD(ev.as_of_date, 56)
         THEN ABS(ev.price_8_rows_ahead / ev.median_price - 1) <= 0.02
    END                                                     AS outcome_persisted
FROM events AS ev
JOIN group_dim   AS g ON g.drug_group = ev.drug_group
JOIN picked_date AS p ON p.drug_group = ev.drug_group AND p.as_of_date = ev.as_of_date
JOIN codes       AS c ON c.drug_group = ev.drug_group AND c.as_of_date = ev.as_of_date
LEFT JOIN mix    AS m ON m.drug_group = ev.drug_group AND m.as_of_date = ev.as_of_date
