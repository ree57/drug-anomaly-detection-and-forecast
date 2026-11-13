# Step 8b: review of the z-score top-20 flags (test split 2025–26)

The top 20 of both variants (A plain, B seasonal) are all `pos_big_shift` events. Together they cover about 9
real-world events. Evidence for the three open cases comes from `investigate_events.py` (event row, group
weekly series, every NDC's own prices, description/unit/class history) plus a few one-off follow-up queries.

## 1. Owner's review (external check)

| Event | Owner's finding |
|---|---|
| Dapagliflozin 5 mg and 10 mg (G), Jun 2026, −98% | **Confirmed real:** generic launch |
| Dapagliflozin-metformin ER 5/1000 and 10/1000 (G), Jul 2026, −92% / −93% | **Confirmed real:** generic launch |
| Isoniazid 100 mg (G), Nov 2025, +1,451% | **Confirmed real:** manufacturer change |
| Dilantin 100 mg (B-ANDA), Mar 2025, +957% | No external explanation found → investigated below |
| Clemastine fumarate 2.68 mg (G), Jan 2026, +563% | No external explanation found → investigated below |
| Sodium chloride 0.9% solution (ML, G), Jul 2026, −92% | No external explanation found → investigated below |

Not reviewed externally: the Fiasp/Tresiba (Jan 14, 2026) and Lyumjev/Humalog (Jan 28, 2026) brand insulin cuts,
Chantix starting box, flurbiprofen 100 mg and the Minivelle patches.

## 2. Verdicts from our own data

### Dilantin 100 mg capsule (B-ANDA): +957% on 2025-03-19 → **classification artefact (not a market move)**

- The original NDCs (labeler code 00071) `00071-0369-24/-32/-40` were priced at **$1.66071 every week** around the event (−4w, −1w,
  event, +4w, +8w). Their price never moved.
- On 2025-02-12, three **new NDCs** (`58151-0110-01/-32/-88`) appeared, classified **G** and priced at
  **$0.15711**, close to the corresponding generic NADAC ($0.15306). Our canonical group uses each NDC's *latest*
  classification (B-ANDA), so these 5 generic-classified weeks were filed under the brand group. The group's median
  fell: −88.4% (2025-02-12, correctly `mix_only`), then −18.8% (2025-02-19, unlabelled).
- On 2025-03-19 the three new NDCs were **reclassified G → B-ANDA** and repriced to exactly **$1.66071**, the original NDCs' price (codes [2, 6], effective date backdated to 2025-02-12, lag 35 days). The median returned to $1.66071 and
  stayed there (4w and 8w $1.66071, 12w $1.65668).
- `mix_only` = false and `ndc_confirmed_share` = 0.5: three of the six NDCs really did change their own price, so the
  mix rule couldn't catch it. A fourth new NDC (`58151-0110-10`) appeared in June 2025 as G and was reclassified
  B-ANDA in Sep 2026.
- **Net effect for Dilantin buyers: none.** The +957% only undoes the February drop caused by the temporary generic
  classification.

### Clemastine fumarate 2.68 mg tablet (G): +563% on 2026-01-21 → **real market move (timing uncertain)**

- The group **disappeared from NADAC for 17 weeks**. Last seen 2025-09-24 at $0.50086, back on 2026-01-21 at
  $3.32279 (`after_gap` = true).
- **Both NDCs**, from two different labelers (`00093-0308-01` and `64950-0268-10`), left at the same $0.50086 and
  came back at the same $3.32279. `ndc_confirmed_share` = 1.0, `mix_only` = false, n_distinct_prices = 1.
- Code [1] only (no package-size or adjustment code), and the effective date equals the published date.
- The new level **held**: $3.34925 at +4w, $3.35596 at +8w and +12w, then a slow survey-like drift down to $3.17751
  by Sep 2026. That's the behaviour of a real survey price, not a one-week glitch.
- Description changed from `...TAB` to `...TABLET` in Dec 2024 (a rewording already handled by the canonical group).
  Unit (EA) and class (G) never changed.
- Caveat: because of the 17-week gap we can't tell *when* during Oct 2025 – Jan 2026 the price moved, or whether
  it went up in steps.

### Sodium chloride 0.9% solution (ML, G): −92.3% on 2026-07-22 → **package/unit artefact (NDC mix by package size)**

- The group mixes products of very different package sizes, priced **per mL** with **explanation code 5** (package
  size) on every row.
  - `00338-0049-04`: about **$0.002/mL**, present all along (2024-08 to 2026-11).
  - `00264-1800-32` and `00487-4301-10`: **$0.0257–0.0262/mL**, about 12× more, present only **2025-03-19 →
    2026-07-15**.
- When the two expensive NDCs **left** the file after 2026-07-15, the median fell to the remaining NDC's price:
  $0.02623 → $0.00203. That NDC's own price moved only **$0.00202 → $0.00203 (+0.5%)**. n_ndcs_priced went 3 → 1.
- The same mechanism caused the earlier **+1,152% on 2025-03-19**. The two expensive NDCs entered the same week
  `00338-0049-04` moved $0.00205 → $0.00219.
- `mix_only` = false both times, because the one NDC present in both weeks happened to change its own price that same
  week (`ndc_confirmed_share` = 1.0 means "1 of 1"). **No one's price fell 92%.**

## 3. Summary

| Event | Verdict | Key evidence |
|---|---|---|
| Dapagliflozin, dapagliflozin-metformin | Real market move (owner, external) | Generic launch |
| Isoniazid 100 mg | Real market move (owner, external) | Manufacturer change |
| Dilantin 100 mg +957% | **Artefact: classification change** | Original 00071 NDCs flat at $1.66; new NDCs reclassified G → B-ANDA and repriced to brand level |
| Clemastine 2.68 mg +563% | **Real market move** (timing uncertain) | 2 labelers, same new price, survey code 1, held ≥ 8 months, after a 17-week gap |
| Sodium chloride 0.9% −92% | **Package/unit artefact** | Code 5; 12×-pricier NDCs left the group; remaining NDC +0.5% |

Of the 6 externally checked or investigated top-20 events, **4 are real and 2 are artefacts**, and both artefacts
passed every existing quality flag.

## 4. What this means for the pipeline

1. **`mix_only` is too narrow.** It only fires when *no* NDC changed its own price. A composition change plus one
   small coincidental own-price change slips through. Proposed flag: **matched-NDC change**, the median own-price
   change of NDCs present in both weeks. If the group moved −92% but matched NDCs moved +0.5%, the move is
   composition-driven.
2. **Classification changes need a flag.** Proposed: `class_changed`, true if any NDC of the group has a different
   classification than in its previous row. Dilantin's G → B-ANDA switch would have been caught.
3. **The canonical-group rule (latest classification) has a side effect:** an NDC's early weeks under a different
   class get filed under its later class's group. That's probably still the right default, but it should be visible
   through the flag above.
4. **`after_gap` matters for timing, not truth:** clemastine is real, but its date is uncertain.
5. These two artefacts were labelled `pos_big_shift` in the evaluation set. See the decisions below.

## 5. Decisions (owner) and what was applied

- Dilantin (2025-03-19) and sodium chloride (2026-07-22 and its 2025-03-19 mirror) are relabelled
  **`excluded_artefact`** in `data/eval/eval_events`, with the reason in `artefact_reason`
  (`sql/eval_events.sql`, `manual_artefacts`).
- New columns on `price_events` (`sql/price_events.sql`), all known at publication time:
  - `matched_ndc_median_change`: median own-price change (%) of NDCs present in both weeks
  - `artefact_suspect`: |pct_change| ≥ 30% and |matched_ndc_median_change| < 5%
  - `class_changed`: an NDC in the group has a different classification from its own previous row
- `artefact_suspect` events are also `excluded_artefact` (reason "auto: ...") and no longer count as real changes
  in a group's `prior_*` history.
- Result on the 1,206 big real moves (> 50%): **37 become `artefact_suspect`** (17 in the test split), and 1 has
  `class_changed` (Dilantin). Of the three reviewed artefacts, the rule catches only the sodium chloride July 2026
  event. Its March 2025 mirror has a matched change of +6.8% (above 5%), and Dilantin's is +478.5% (half its NDCs
  repriced), so those two stay manual.
