-- eval_events: every row of price_events, plus past-only context and evaluation labels.
--
-- Two kinds of columns:
--   prior_*, is_round_pct, is_brand_january_increase : use only information published up to this event's
--       week. A detector MAY see these.
--   closer_to_new_8w, is_pos_*, category              : built from outcome_* (the FUTURE).
--       !!! Evaluation only. A detector must NEVER see these.
-- is_neg_* use only past information; they are labels, but no future leaks into them.

WITH base AS (
    SELECT
        e.*,
        ABS(e.pct_change)                                               AS abs_pct,
        -- a "real" change: mix_only events moved the median without any NDC changing its price, and
        -- artefact_suspect events moved it mostly through the NDC mix (both known at publication time)
        CASE WHEN NOT e.mix_only AND NOT e.artefact_suspect THEN ABS(e.pct_change) END AS real_abs_pct,
        ABS(e.pct_change - ROUND(e.pct_change)) <= 0.1                  AS is_round_pct,
        e.is_brand AND e.is_january AND e.pct_change > 0                AS is_brand_january_increase
    FROM price_events AS e
),

history AS (
    -- the group's EARLIER real changes only: the frame stops one row before the current event
    SELECT
        *,
        COUNT(real_abs_pct)             OVER past AS prior_real_changes,
        PERCENTILE(real_abs_pct, 0.75)  OVER past AS prior_p75_abs_pct,
        PERCENTILE(real_abs_pct, 0.95)  OVER past AS prior_p95_abs_pct
    FROM base
    WINDOW past AS (PARTITION BY drug_group ORDER BY published_week
                    ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING)
),

manual_artefacts AS (
    -- events judged artefacts in the top-20 review (reports/top20_review.md)
    SELECT drug_group, CAST(week AS DATE) AS published_week, reason FROM VALUES
        ('DILANTIN 100 MG CAPSULE|EA|B-ANDA', '2025-03-19',
         'manual review: new NDCs reclassified G to B-ANDA and repriced to brand level; original NDCs flat'),
        ('SODIUM CHLORIDE 0.9% SOLUTION|ML|G', '2025-03-19',
         'manual review: 12x pricier package sizes (code 5) entered the group; remaining NDC +6.8%'),
        ('SODIUM CHLORIDE 0.9% SOLUTION|ML|G', '2026-07-22',
         'manual review: 12x pricier package sizes (code 5) left the group; remaining NDC +0.5%')
        AS t(drug_group, week, reason)
),

flags AS (
    SELECT
        h.*,
        COALESCE(ma.reason, CASE WHEN h.artefact_suspect
            THEN 'auto: group moved >= 30% while NDCs present in both weeks moved < 5% (median)' END) AS artefact_reason,
        -- FUTURE: 8 weeks later, is the price nearer the new level than the old one? NULL if not observed.
        CASE WHEN outcome_price_8w IS NOT NULL
             THEN ABS(outcome_price_8w - new_price) < ABS(outcome_price_8w - old_price) END AS closer_to_new_8w
    FROM history AS h
    LEFT JOIN manual_artefacts AS ma USING (drug_group, published_week)
),

flags2 AS (
    -- excluded = not a real price change (mix_only) or a known/suspected data artefact
    SELECT *, artefact_reason IS NOT NULL AS is_artefact, mix_only OR artefact_reason IS NOT NULL AS excluded
    FROM flags
)

SELECT
    drug_group, classification, pricing_unit, event_number, published_week, effective_date,
    old_price, new_price, pct_change, direction, is_brand, is_january, mix_only, after_gap,
    matched_ndc_median_change, artefact_suspect, class_changed, is_artefact, artefact_reason,

    -- past-only context (allowed as detector inputs)
    is_round_pct, is_brand_january_increase,
    prior_real_changes, ROUND(prior_p75_abs_pct, 4) AS prior_p75_abs_pct, ROUND(prior_p95_abs_pct, 4) AS prior_p95_abs_pct,

    -- future outcomes (evaluation only)
    outcome_price_8w, outcome_reverted, closer_to_new_8w,

    -- POSITIVES (evaluation only)
    COALESCE(NOT excluded AND abs_pct > 50 AND outcome_reverted, FALSE)                       AS is_pos_big_spike,
    COALESCE(NOT excluded AND abs_pct > 50 AND NOT outcome_reverted AND closer_to_new_8w, FALSE) AS is_pos_big_shift,
    COALESCE(NOT excluded AND NOT is_brand_january_increase AND prior_real_changes >= 6
             AND abs_pct >= 10 AND abs_pct >= 5 * prior_p95_abs_pct AND closer_to_new_8w, FALSE) AS is_pos_relative,

    -- EXPECTED / NEGATIVES (past information only)
    COALESCE(NOT excluded AND is_brand AND is_january AND pct_change BETWEEN 1 AND 20 AND is_round_pct, FALSE)
                                                                                              AS is_neg_january_list_increase,
    COALESCE(NOT excluded AND prior_real_changes >= 6 AND abs_pct <= prior_p75_abs_pct AND abs_pct < 10, FALSE)
                                                                                              AS is_neg_normal_move,

    -- HARD negatives (past information only): big in absolute terms, but expected.
    -- A size-only detector should flag them; a good detector shouldn't.
    COALESCE(NOT excluded AND is_brand AND is_january AND pct_change BETWEEN 7 AND 20 AND is_round_pct, FALSE)
                                                                                              AS is_neg_hard_january,
    -- big for a generic, but no bigger than this drug's own usual changes (not a positive, not > 50%)
    COALESCE(NOT excluded AND prior_real_changes >= 6 AND abs_pct >= 20 AND abs_pct <= prior_p75_abs_pct
             AND abs_pct <= 50
             AND NOT (NOT is_brand_january_increase AND abs_pct >= 10 AND abs_pct >= 5 * prior_p95_abs_pct
                      AND closer_to_new_8w), FALSE)                                           AS is_neg_hard_volatile,

    -- one primary category per event (first matching rule wins)
    CASE
        WHEN mix_only                                                                         THEN 'excluded_mix_only'
        WHEN is_artefact                                                                      THEN 'excluded_artefact'
        WHEN abs_pct > 50 AND outcome_reverted                                                THEN 'pos_big_spike'
        WHEN abs_pct > 50 AND NOT outcome_reverted AND closer_to_new_8w                       THEN 'pos_big_shift'
        WHEN NOT is_brand_january_increase AND prior_real_changes >= 6 AND abs_pct >= 10
             AND abs_pct >= 5 * prior_p95_abs_pct AND closer_to_new_8w                        THEN 'pos_relative'
        WHEN abs_pct > 50                                                                     THEN 'big_unclear'
        WHEN is_brand AND is_january AND pct_change BETWEEN 7 AND 20 AND is_round_pct          THEN 'neg_hard_january'
        WHEN prior_real_changes >= 6 AND abs_pct >= 20 AND abs_pct <= prior_p75_abs_pct       THEN 'neg_hard_volatile'
        WHEN is_brand AND is_january AND pct_change BETWEEN 1 AND 20 AND is_round_pct          THEN 'neg_january_list_increase'
        WHEN prior_real_changes >= 6 AND abs_pct <= prior_p75_abs_pct AND abs_pct < 10        THEN 'neg_normal_move'
        ELSE 'unlabelled'
    END                                                                                       AS category,

    -- named test cases: what a good detector should do with them
    CASE drug_group
        WHEN 'ISOSORBIDE MONONIT 20 MG TAB|EA|G'  THEN 'isosorbide_20mg'
        WHEN 'PRUCALOPRIDE 2 MG TABLET|EA|G'      THEN 'prucalopride_2mg'
        WHEN 'PHENOBARBITAL 16.2 MG TABLET|EA|G'  THEN 'phenobarbital_16_2mg'
        WHEN 'OTEZLA 30 MG TABLET|EA|B'           THEN 'otezla_30mg'
    END                                                                                       AS test_case,
    CASE
        WHEN drug_group IN ('ISOSORBIDE MONONIT 20 MG TAB|EA|G', 'PRUCALOPRIDE 2 MG TABLET|EA|G',
                            'PHENOBARBITAL 16.2 MG TABLET|EA|G') AND abs_pct > 50 AND NOT excluded THEN 'flag'
        WHEN drug_group = 'OTEZLA 30 MG TABLET|EA|B' AND is_brand_january_increase              THEN 'no_flag'
    END                                                                                       AS test_case_expect,

    -- time split: tune on the past, score on the later period
    CASE WHEN published_week < DATE'2025-01-01' THEN 'train' ELSE 'test' END                  AS split
FROM flags2
