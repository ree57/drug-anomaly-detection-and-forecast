-- ndc_dim: one row per NDC (one product package from one manufacturer).
-- The NDC's "canonical" group comes from its MOST RECENT week, so a description
-- that CMS reworded between years doesn't split the NDC's history into two groups.
SELECT
    ndc,
    MAX_BY(drug_group,     as_of_date) AS drug_group,      -- value from the row with the latest as_of_date
    MAX_BY(description,    as_of_date) AS description,
    MAX_BY(pricing_unit,   as_of_date) AS pricing_unit,
    MAX_BY(classification, as_of_date) AS classification,
    MIN(as_of_date)                    AS first_seen,
    MAX(as_of_date)                    AS last_seen,
    COUNT(*)                           AS weeks_present,   -- one row per NDC per week (checked in step 5)
    COUNT(DISTINCT classification) > 1 AS class_changed,
    COUNT(DISTINCT drug_group)         AS groups_seen      -- more than 1 = its raw group changed (reworded)
FROM nadac
GROUP BY ndc
