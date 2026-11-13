# Drug price anomaly detection: summary (Step 8)

**Goal.** Flag NADAC price changes that are unusual for that drug, and leave out expected events (January brand
list-price increases, normal moves). There are no ground-truth labels, so we used proxy labels
(`reports/anomaly_eval_plan.md`). They were built with future prices, for grading only. Detectors are trained or
tuned on events published in 2022–24 and tested on 2025–26. All numbers below are from `reports/metrics_8d.csv`
(test split, single events unless noted).

## Methods tried

| Detector | Idea |
|---|---|
| Baseline | Rank by \|% change\|. Every detector must beat it. |
| Robust z, Variant B (8b) | ln(new/old) vs the median and MAD of the drug's previous 12 real changes; brand January moves are compared only with earlier January moves. Variant A (no seasonal split) was dropped for B. |
| **Hybrid (8b-2)** | Average of two percentiles against 2022–24: the size of the move and Variant B's \|z\|. High only if the move is both big and unusual for the drug. |
| IF-small (8c) | Isolation Forest on 3 features: size, z, typical past move. |
| IF-full (8c) | Isolation Forest on 13 publication-time features (adds brand, January, gap, codes, price level, ...). |
| **Final (8d)** | Hybrid ranking with IF-small as a second opinion. **High** = both in their top 1% (cut-off = 99th percentile on 2022–24), **medium** = one of them. |

## Results (test split 2025–26)

| | Baseline | Variant B | Hybrid | IF-small | IF-full | **Final** |
|---|---|---|---|---|---|---|
| Big-shift recall, top 100 (of 65) | 1.00 | 0.66 | 0.82 | 0.85 | 0.51 | 0.82 |
| Relative recall, top 100 (of 90) | 0.50 | 0.80 | 0.70 | 0.74 | 0.63 | 0.70 |
| Known-positive share, top 100 | 0.76 | 0.83 | 0.84 | 0.88 | 0.65 | 0.84 |
| Hard volatile wrongly flagged, top 500 / 1000 | 38 / 82 | 1 / 3 | 8 / 27 | 5 / 10 | 0 / 0 | 8 / 27 |
| January list increases flagged, top 1000 | 0 | 7 | 3 | 12 | 554 | 3 |
| Normal moves flagged, top 1000 | 0 | 158 | 0 | 245 | 64 | 0 |
| Named test cases passed (of 14) | 14 | 7 | 8 | 14 | 1 | **14** |
| Families: relative recall, top 100 (of 40) | 0.55 | 0.88 | 0.63 | 0.75 | 0.45 | 0.65 |

**What the final detector alerts on (test split):**

| Confidence | Events | Known positives | Unlabelled or unclear | Known negatives |
|---|---|---|---|---|
| High | 129 | 98 (76%) | 31 | 0 |
| Medium | 47 | 7 | 40 | 0 |

- **Load:** 112 alert families in 84 weeks. The median week has 0 alerts and 51 weeks have none. The busiest week
  has 13 families, and the mean is 1.33. No stricter threshold is needed.
- **Missed:** 16 known positives are in neither top 1% (1 big shift, 15 relative). They are still ranked by the
  hybrid, below the alerts.

## Why the hybrid is the main score

- **No false alarms from normal moves.** It is the only learned detector with 0 normal moves and 3 January list
  increases in its top 1000.
- **Fewer volatile false flags.** It cuts hard-volatile false flags from 82 (baseline) to 27.
- **Average precision ≥ 0.99** against both easy and hard negatives.
- **Easy to explain:** "bigger than X% of past moves, and N times this drug's usual move".
- **Its gap:** large but repeated flips (phenobarbital) look normal for that drug, so it fails those test cases.
  IF-small catches them, which is why the second opinion lifts the final detector to 14/14 test cases. IF-small
  alone is noisier deeper down (245 normal moves in its top 1000), so it is not the main score.

## Why IF-full failed

Isolation Forest flags what is **rare**, not what is **suspicious**. Context features make routine events rare:
- code 3 (CMS aligning brand prices to list prices) is on only 1.8% of training events
- a brand moving in January is also an unusual combination

As a result, 261 of its top 500 events are January list increases. It fails every Otezla January test case and
13 of the 14 named cases. Context belongs in caution flags and explanations, not in the anomaly score.

## Known limitations

- **Proxy labels.** Positives are defined partly by "big compared with the drug's own past", which favours
  detectors that measure exactly that. The manual top-20 review is the independent check.
- **Artefacts.** Two judged artefacts still alert: Dilantin (Mar 2025) and sodium chloride (Mar 2025).
  - The auto rule (`artefact_suspect`) needs the products present in both weeks to move < 5%. Sodium chloride's
    moved +6.8%.
  - The manual relabels came from a review that used later weeks, so they are not applied live. Both alerts carry a
    caution instead: Dilantin has `class_changed`; sodium chloride has "products present in both weeks moved far
    less than the group". That second caution fires on 2 of 178 test alerts.
- **In-sample cut-off.** IF-small's top-1% cut-off uses training-period scores, which are in-sample.
- **Train period is noisier.** About 1.5% of 2022–24 events alert, compared with 0.16% in 2025–26.
- **Families are approximate.** Store-brand OTC lines merge into one family, and same-company actions under
  different names stay separate (Fiasp and Tresiba are two families).
- **Unconfirmed alerts.** Most medium alerts are unlabelled (40 of 47). Their real precision is unknown until an
  analyst reviews them.
