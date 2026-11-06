-- group_weekly: one row per canonical drug group per week. This is the time series we analyse.
WITH priced AS (
    -- every clean row, labelled with its NDC's canonical group (not the raw, possibly reworded one)
    SELECT d.drug_group, n.as_of_date, n.ndc, n.nadac_per_unit
    FROM nadac AS n
    JOIN ndc_dim AS d ON d.ndc = n.ndc
),
weekly AS (
    -- collapse to one row per group per week
    SELECT
        drug_group,
        as_of_date,
        -- "lower median": always one of the real NDC prices, never an average of two tiers
        PERCENTILE_DISC(0.5) WITHIN GROUP (ORDER BY nadac_per_unit) AS median_price,
        COUNT(*)                       AS n_ndcs_priced,      -- one row per NDC per week
        COUNT(DISTINCT nadac_per_unit) AS n_distinct_prices   -- > 1 = the group has price tiers that week
    FROM priced
    GROUP BY drug_group, as_of_date
),
with_prev AS (
    -- window: for each group separately, in week order, look one row back
    SELECT
        *,
        LAG(median_price) OVER w AS prev_price,
        LAG(as_of_date)   OVER w AS prev_as_of_date
    FROM weekly
    WINDOW w AS (PARTITION BY drug_group ORDER BY as_of_date)
)
SELECT
    drug_group,
    as_of_date,
    median_price,
    n_ndcs_priced,
    n_distinct_prices,
    prev_price,
    ROUND((median_price / prev_price - 1) * 100, 4)        AS pct_change,      -- NULL in a group's first week
    median_price <> prev_price                             AS price_changed,   -- NULL in a group's first week
    CAST(DATEDIFF(as_of_date, prev_as_of_date) / 7 AS INT) AS weeks_since_prev -- 1 = the week right before
FROM with_prev
