# Changes from the submitted version

Manuscript JATM-D-26-00779, submitted 24 July 2026; revision prepared September 2026.

This document records every difference between the analysis in the submitted manuscript and
the analysis in this repository, including the errors found while responding to the referee
and the editor. Nothing here is hidden in a footnote: where a published number was wrong,
weaker than claimed, or unreproducible, it is named.

---

## 1. Errors in the submitted analysis

### 1.1 A target leak in the time-of-day feature

The submitted model used the hour of the **actual** next departure. That time is displaced
by the very delay being predicted, so the feature carried information about the outcome.
Replaced by the **scheduled** departure hour.

*Effect:* AUC falls by 0.018. This is the main reason the headline moved from 0.896 to 0.887.

### 1.2 The scheduled turnaround was wrapped unsigned

Scheduled ground time was computed as `(CRSDepTime_next − CRSArrTime) mod 1440`. Where the
next departure was scheduled **before** the inbound's scheduled arrival — a rotation already
infeasible as scheduled, 1.95% of pairs — the modulo maps a small negative value onto
roughly +1400 minutes, i.e. onto a large imaginary buffer, in precisely the most stressed
cases. Replaced by a signed wrap into [−720, +720).

*Effect:* correlation between scheduled and actual ground time rises from 0.712 to 0.851.

### 1.3 The rule-based baseline was crippled by 1.2

The baseline flags when the inbound delay exceeds the scheduled ground time. With a
spuriously large scheduled value it cannot fire. Corrected, the baseline achieves recall
0.305, not the 0.24 reported.

*Effect:* the submitted claim that the model "nearly triples" the baseline recall becomes
**roughly doubles it** (0.631 against 0.305).

### 1.4 The feature ablation was invalid

`buffer_pressure` is exactly `inbound_arr_delay − sched_turnaround_min`. Removing
`inbound_arr_delay` while retaining `buffer_pressure` and `sched_turnaround_min` leaves the
removed signal exactly reconstructible. Measured directly: that ablation costs **0.000**
AUC (0.887 → 0.887). The correct information-source ablation, which removes derivatives with
their parents and refits, gives 0.731.

The submitted Table 2 reported 0.775 for this row, which matches neither figure.

### 1.5 The categorical encoder was fitted on the full dataset

Airport and aircraft-type codes were assigned over train and test together. Now fitted on
the training split only, with unseen levels mapped to −1. The effect is small — codes are
arbitrary integers to a tree model — but the protocol is now clean.

## 2. Claims that were overstated

### 2.1 "Recall is essentially unchanged under noise"

True only because the alert rate is free to inflate from 19.0% to 30.1%. Held to a matched
alarm budget, additive noise of sd = 40 min costs recall 0.631 → 0.441.

### 2.2 "Missing information collapses recall at every operating threshold"

At a fixed 0.5 threshold, recall falls to 0.146 and 80.2% of propagated delay goes
unflagged. At a **matched alarm budget** those become 0.382 and 57.9%. Roughly a third of
the published collapse is an operating-point artefact, because imputation shrinks predicted
probabilities toward the base rate.

The threshold-free loss is nonetheless real and large: AUC 0.887 → 0.731.

### 2.3 "The operating point requires local recalibration rather than retraining"

Tested head-to-head on held-out target data across eight hubs, comparing no adaptation,
threshold adjustment, isotonic recalibration and local retraining:

| remedy | mean recall | mean precision | mean ECE |
|---|---|---|---|
| none | 0.591 | 0.934 | 0.026 |
| threshold only | 0.575 | 0.910 | — |
| recalibration | 0.598 | 0.926 | 0.018 |
| local retraining | 0.610 | 0.916 | 0.021 |

Local retraining buys **nothing** in discrimination (AUC 0.866 retrained against 0.868
transferred). All four strategies land within recall 0.575–0.610. The real benefit of
recalibration is calibration, not catch rate. The submitted claim is corrected accordingly.

## 3. European claims withdrawn or not supported

### 3.1 Ground-time-state propagation (submitted: r = 0.31) — **withdrawn**

The figure reproduces exactly (r = 0.312; 41.8 against 21.3 minutes). It is an artefact:

| restriction | n | pooled r | within-route r |
|---|---|---|---|
| none (as submitted) | 25,363 | 0.210 | 0.066 |
| gap < 24 h | 24,217 | 0.146 | 0.041 |
| **gap < 12 h** | 21,332 | **−0.004** | **−0.024** |
| gap < 3 h | 20,834 | −0.029 | −0.031 |

69% of the association is between-route composition — aircraft flying long-turn routes have
long turns at both ends — and the remainder rides on pairs whose "next leg" is more than 12
hours later, where the aircraft left the eight sampled hubs and returned. Restricted to
genuine consecutive legs and with route means removed, within-route r = −0.024
(95% bootstrap CI −0.038 to −0.009).

### 3.2 European temporal stability (submitted: AUC 0.79–0.85) — **not supported**

| window | label threshold fitted within-window | fitted across all five windows |
|---|---|---|
| 2022-01 | 0.606 | 0.750 |
| 2024-01 | 0.630 | 0.798 |
| 2024-06 (in-distribution) | 0.833 | 0.850 |
| 2025-06 | 0.650 | 0.832 |
| 2026-01 | 0.611 | 0.764 |

The published range corresponds to the right-hand column, which estimates the label
threshold using the test windows. Under a clean per-window protocol the model does not
transfer.

### 3.3 The degradation experiments cannot be repeated in Europe

OpenSky carries no schedules, so the only candidate inbound-status signal is the aircraft's
own previous ground stop. It is observable for 10.5% of turnarounds, because only eight
hubs are sampled, and its association with the outcome is −0.024. There is no signal to
degrade.

### 3.4 "Five months" of European data

The European sample is five **14-day windows** (13 days spanned each), not five months. The
total of 203,103 turnarounds reproduces exactly.

### 3.5 What survives

| claim | submitted | reproduced |
|---|---|---|
| turnarounds, 8 hubs | 203,103 | 203,103 |
| June 2024 window | 51,803 | 51,803 |
| aircraft-type match | 88.6% | 88.6% |
| cross-airport classifier | 0.78 → 0.81 | 0.783 → 0.814 |
| duration regression | R² 0.45 → 0.55 | R² 0.480 → 0.558 |

## 4. Numbers that could not be reproduced at all

The code for the following was not preserved and does not exist in any form. A full search
of the author's working disk confirmed it. These analyses have been rebuilt from scratch
and the numbers restated from the rebuilt pipeline.

- Table 2 in its entirety, including the full-model row (AUC 0.900, recall 0.636), which
  differs from Table 1's full model (0.896, 0.625) without explanation — the referee's
  question 4. Both tables now come from one script, one split and one fitted model.
- The coarse-status row of Table 1.
- The US aircraft-type join (submitted: 89.3% matched; rebuilt: 87.4% for June 2024).
- The European drift and aircraft-type analyses.
- The rotation-accumulation analysis referenced in §5.4.

## 5. Sample and pipeline changes

| | submitted | revised |
|---|---|---|
| turnarounds, June 2024 | 547,300 | 564,144 |
| event-time reconstruction | clock arithmetic with a +1-day rule | scheduled time + reported delay; duration modulo 24 h |
| train/test split | first 75% of rows | calendar-day boundary, 1–23 vs 24–30 June |
| aircraft type, US arm | absent from the pipeline | joined, 87.4% matched |
| causal delay fields | not carried | all five carried per outbound leg |
| decision points | one (inbound in-block) | three (inbound off-block, wheels-on, in-block) |

The retained-turnaround range of 20–720 minutes conditions on a quantity that a late
outbound mechanically inflates. No referee raised this; it is raised here. The filter
excludes 3.9% of pairs, which are slightly more delay-prone, and AUC across five alternative
ranges including 1–1440 minutes spans 0.879–0.896.

## 6. Analyses added in response to the review

- Validation of the outcome against the BTS causal fields, schedule buffers and operational
  conditions (referee item 1).
- A propagation-specific secondary outcome, `y_prop`.
- A decision-point ladder quantifying the discrimination/lead-time trade-off (item 2).
- Matched-alarm-budget evaluation of every information condition (items 2 and 6).
- Decision-curve analysis over false-alarm/miss cost ratios (item 2, editor point b).
- Grouped information-source ablation with retraining (item 4).
- Per-airport and per-period prevalence, AUPRC, calibration and uncertainty, and the
  four-way remediation comparison (item 6).
- Sensitivity to turnaround bounds, unmatched aircraft types, hyperparameters and seeds
  (item 7).
- Mutually exclusive four-band coarse-status categories including the 15–60 minute band
  (item 8.1).
- Bin-level sample sizes and Wilson intervals on every conditional-mean figure (item 8.4).
