-- group_dim: one row per canonical drug group ("the same drug product, priced together").
-- Built from ndc_dim, so every NDC counts in exactly one group: its canonical one.
SELECT
    drug_group,
    MIN(description)    AS description,     -- NDCs in a group can differ only in case/spacing; pick one
    MIN(pricing_unit)   AS pricing_unit,    -- identical for every NDC in a group (part of the key)
    MIN(classification) AS classification,  -- identical too (part of the key)
    COUNT(*)            AS n_ndcs,
    MIN(first_seen)     AS first_seen,
    MAX(last_seen)      AS last_seen
FROM ndc_dim
GROUP BY drug_group
