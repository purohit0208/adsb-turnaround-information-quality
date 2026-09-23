# Availability before precision: how the quality of inbound information governs the aircraft turnaround departure-delay decision

## Abstract

Aircraft turnaround sits on the critical path between an inbound arrival and the next departure, and a late inbound consumes the scheduled ground time that would otherwise absorb it. A large literature predicts turnaround duration and delay, and a parallel literature optimises ground operations and models network delay propagation, but the two are rarely joined: prediction studies report accuracy, while optimisation and recovery studies assume clean inputs. What is missing is a measurement, on real data, of how the *quality* of operational information changes a constrained turnaround decision. We provide one. From United States Bureau of Transportation Statistics on-time records we derive 2.52 million turnarounds across five months spanning 2022 to 2026, and we frame a transparent decision: flag the next departure as likely to leave more than fifteen minutes late so that an operator can act. We validate what that outcome measures against the carrier-reported causal delay fields, against the scheduled buffer, and across operational conditions: the outcome occurs in 93.5% of turnarounds where the inbound delay exceeds the scheduled ground time against 18.6% where it fits inside, and a propagation-specific variant confirmed by the carrier's own attribution behaves the same way. We then degrade the inbound-status signal and measure the decision, holding the alarm budget fixed so that operating-point shifts are not mistaken for information loss. The central result is an asymmetry between availability and precision. Coarse inbound status — four buckets instead of an exact figure — costs 0.009 of area under the ROC curve, while losing the signal entirely costs 0.156 and leaves 57.9% of the delay unflagged at an equal alarm budget. Additive noise degrades precision at a fixed threshold and recall at a fixed budget, but never as much as absence. A ladder of decision points shows the alert can be raised 144 minutes earlier, at the inbound aircraft's own off-block, for 0.032 of discrimination. Decision-curve analysis shows the signal is worth most precisely where false alarms are costly. The predictor transfers across airports and four years, but local retraining buys nothing in discrimination and recalibration improves calibration rather than catch rate, which corrects a claim we previously made. A European ADS-B arm is reported as descriptive evidence only; re-auditing it caused us to withdraw a propagation result we had reported. The study shows that what governs this decision is whether the inbound signal is present, not how precise it is.

**Keywords:** aircraft turnaround; departure delay; information quality; value of information; decision support; calibration; reproducibility

---

## 1. Introduction

Aircraft turnaround is the set of coordinated ground processes that prepares an inbound aircraft for its next departure: deboarding, cabin cleaning, catering, fuelling, baggage and cargo handling, boarding, and the stand, crew and information coordination that surround them. Because these activities lie on the critical path between an arrival and the next departure, a late inbound aircraft consumes the scheduled ground time that would otherwise absorb it, and what is not absorbed is pushed into the next departure and onward through the day's rotation. Delay that originates or is absorbed in the turnaround is a recognised driver of system-wide punctuality, and its economic cost is large, which is why the turnaround attracts both airlines and airports as a lever.

Reflecting this importance, a substantial body of work applies machine learning, optimisation, simulation and digital-twin methods to the turnaround and its surrounding operations. That body of work is fragmented in a specific and consequential way. Prediction studies typically report accuracy on a forecasting target — turnaround time, an operational milestone, a delay — and stop there. Optimisation and recovery studies typically treat the relevant information as a clean input and evaluate scheduling, apron-planning or recovery performance under that assumption. Network studies model how delay propagates between flights and across airports but treat the turnaround largely as a buffer parameter. The connective question that sits between these literatures is rarely posed: given that operational information about an inbound aircraft arrives late, noisily, or not at all, how does the quality of that information change a constrained turnaround decision, and how much is the difference worth?

A second limitation is evidential. Much turnaround research is developed and validated on synthetic or simulated data, because proprietary airline data describing sub-process timings and resource assignments is rarely accessible to researchers. Synthetic data is legitimate for controlled method development, but on its own it cannot establish that a relationship holds in real operations. The two limitations compound: the decision-value of information quality is exactly the kind of claim that most needs real-world evidence and least often receives it.

This paper addresses both, using only public data and a fully reproducible pipeline. We make four contributions.

First, we derive aircraft turnarounds at scale from public United States on-time records, and — unusually for this literature — we establish what our outcome actually measures rather than asserting it. The outcome is validated against the carrier-reported causal delay fields, against the scheduled buffer, and across operational conditions.

Second, and centrally, we measure how information quality governs the decision. We degrade the inbound-status signal in three operationally meaningful ways — additive noise, coarsening to categories, and complete absence — and measure the effect on the decision rather than on a single accuracy number. Crucially, every condition is evaluated both at a fixed decision threshold and at a **matched alarm budget**, because a degraded model's probabilities compress toward the base rate, so part of any apparent collapse is an operating-point shift rather than a loss of information. Separating the two is what makes the result interpretable.

Third, we specify *when* the decision can be made. We construct a ladder of three decision points, each using only information that genuinely exists at that moment, and quantify the discrimination surrendered in exchange for lead time.

Fourth, we test robustness: across airports, across four years, and against the choices in our own pipeline. Where our own earlier claims did not survive that testing, we say so and correct them.

The methodological contribution is the evaluation framework — propagating degraded information into a real operational decision and measuring its value in operational units, at a controlled alarm budget — rather than a new predictor. The predictor is deliberately standard, so that the result is attributable to the information available, not to the model.

Section 2 positions the work. Section 3 describes the data and the derivation. Section 4 formalises the decision, the predictor, and the experiments. Section 5 reports results. Section 6 discusses implications, Section 7 states limitations, Section 8 outlines future work, and Section 9 concludes.

## 2. Related work

The study connects four strands: turnaround and delay prediction, ground-operations optimisation and recovery, network delay propagation, and operational data derived from public surveillance.

### 2.1 Turnaround and delay prediction

A consistent finding across the prediction literature is that turnaround duration and its milestones are predictable from operational data to an operationally useful degree. Process-aware models represent the turnaround as a network of partly sequential, partly concurrent activities and predict total duration under uncertainty (Schmidt, 2017; Asadi and Fricke, 2022); machine-learning models predict A-CDM milestones and taxi times from historical operational records (Okwir et al., 2025; Tang et al., 2025; Cui et al., 2024), including under missing data (Liu et al., 2023) and for ground-handling times directly (Volt et al., 2026); and arrival-time predictions are used to drive ground-resource allocation (Sahadevan et al., 2023). Passenger boarding, a principal driver of the critical path, has itself been modelled with machine learning (Schultz and Reitmann, 2019). These studies establish predictability but generally evaluate against a forecasting metric and do not carry the prediction into a decision or test its sensitivity to input quality. Our predictor is deliberately of this established kind — infrastructure for the study, not its novelty.

### 2.2 Ground-operations optimisation and recovery

A parallel literature treats the turnaround and its surrounding operations as constrained optimisation: schedule-recovery models that re-time and re-sequence operations after disruption (Evler et al., 2021a, 2021b), robust apron and stand planning (Gök et al., 2022), stochastic ground-resource and specialised-vehicle scheduling (Guimarans and Padrón, 2022; Zhu et al., 2022), baggage-handling reoptimisation (Ruf et al., 2022), and integrated aircraft-assignment-and-turnaround formulations (Glomb et al., 2023); the field has recently been surveyed comprehensively (Dahanayaka et al., 2026). Schedule-buffer analyses characterise how scheduled ground time absorbs propagated delay and how buffers are distributed across a day and a network (Brueckner et al., 2021). These studies provide the decision layer that prediction work lacks, but they typically assume the quality of their inputs; the interaction between input information quality and decision quality is generally outside their scope.

### 2.3 Network delay propagation

A third literature models how delay propagates between connected flights and through airport networks. Early analyses quantified the potential for delay to propagate along aircraft and crew connections (AhmadBeygi et al., 2008) and the role of schedule buffers in damping it, while integrated turnaround-and-aircraft recovery has been proposed to damp propagation in airline networks (Evler et al., 2022); subsequent work modelled propagation within an airport (Pyrgiotis et al., 2013), across the network with analytical-econometric methods (Kafle and Zou, 2016), and as a systemic phenomenon (Fleurquin et al., 2013). This literature establishes that the turnaround is a principal propagation mechanism — a late inbound eats the ground buffer and pushes the next departure — which is the relationship our decision targets and against which we validate our outcome in Section 5.1. Our contribution is at a different level: rather than modelling propagation at network scale, we study how the *quality of the information* about it changes the operational decision to mitigate it.

### 2.4 Operational data from public surveillance

Finally, a growing literature derives operational information from public ADS-B rather than proprietary feeds. Operational milestones and A-CDM-like ground events can be reconstructed from ADS-B messages (Schultz et al., 2022) and from computer vision at the stand (Xu et al., 2023), and open ADS-B from the OpenSky Network (Schäfer et al., 2014; Strohmeier et al., 2021), processed with community tools (Olive, 2019), has become a standard substrate for reproducible air-transport research. We build on this, and Section 5.6 reports what such data can and cannot support for the present question.

### 2.5 Evaluation under shifting prevalence and asymmetric costs

Two methodological strands outside air transport bear directly on our design. Where outcome prevalence varies across deployment settings, the area under the ROC curve can remain stable while performance at a usable operating point does not, and precision-recall summaries are the more informative companion (Saito and Rehmsmeier, 2015). And where the costs of false alarms and missed events are asymmetric and not precisely known, decision-curve analysis evaluates a model across the whole plausible range of that trade-off rather than at a single assumed operating point (Vickers and Elkin, 2006). We adopt both, because the turnaround decision has exactly these two properties.

In summary: prediction tells us turnaround delay is forecastable, optimisation tells us it can be mitigated given clean inputs, network studies tell us it propagates, and public-data work tells us we can observe it reproducibly. What is missing, and what we provide, is a real-data measurement of how the quality of the information bridging prediction and mitigation governs the decision itself.

## 3. Data and turnaround derivation

### 3.1 Source

The primary analysis uses the United States Bureau of Transportation Statistics (BTS) Reporting Carrier On-Time Performance database, which records, for every domestic flight of the reporting carriers, the flight date, aircraft tail number, origin and destination, scheduled and actual gate departure and arrival times, departure and arrival delays in minutes, taxi times, cancellation and diversion flags, and — for flights arriving fifteen or more minutes late — a five-way causal decomposition of the arrival delay into carrier, weather, national-airspace-system, security and late-aircraft components. Because BTS records both scheduled and actual gate times, it supports a schedule-based notion of ground time and delay; because it records the causal decomposition, it permits the outcome validation of Section 5.1, which is the element most conspicuously absent from comparable studies.

We use five complete months spanning four years: January 2022, January 2024, June 2024, June 2025 and January 2026. Aircraft type is joined from the public OpenSky aircraft database by registration.

A European arm derived from OpenSky ADS-B surveillance is reported separately in Section 5.6 as descriptive evidence; its scope and its limits are stated there.

### 3.2 Turnaround pairing

A turnaround is reconstructed by chaining the flights of a single aircraft at a common airport. For each aircraft, identified by tail number, we order its movements in time and pair an arrival into airport A with that same aircraft's next departure from A; the turnaround is the elapsed time between the inbound gate arrival and the outbound gate departure. Only clean same-airport turns are retained — the arrival's destination must equal the next departure's origin — which excludes ferry repositioning and gaps caused by intermediate movements. Cancelled and diverted flights are removed before pairing.

Two construction details matter for reproducibility and were handled explicitly.

**Event times.** Rather than reconstructing timestamps from local clock times and a day-shift rule, each event is built as its scheduled time plus its reported delay. This reproduces the reported clock times in 100.000% of rows and is immune to the midnight wrap. Turnaround duration is then computed modulo twenty-four hours, which is additionally immune to timezone differences between origin and destination; it agrees with a naive date-shifted construction in 100.00% of retained pairs.

**Scheduled ground time.** The scheduled turnaround is the difference between the next flight's scheduled departure and this flight's scheduled arrival, wrapped into a **signed** window of ±720 minutes. An unsigned wrap into [0, 1440) — which we used previously — maps genuinely negative scheduled turnarounds onto values near +1440, that is, onto a large imaginary buffer. Such rotations are real: in June 2024, 1.95% of pairs were scheduled so that the next departure was due to leave before the inbound was due to arrive, and these are precisely the most stressed cases. The signed form raises the correlation between scheduled and actual ground time from 0.712 to 0.851.

Turnarounds between 20 and 720 minutes are retained. This range conditions on a quantity that a late outbound mechanically inflates, so Section 5.5 reports sensitivity to it.

Table 1 summarises the resulting dataset.

*Table 1. Derived United States turnarounds. Prevalence is the share of turnarounds whose next departure left more than fifteen minutes late; the propagation-specific share additionally requires the carrier to have attributed at least fifteen minutes of the resulting arrival delay to a late inbound aircraft.*

| Month | Flights | Turnarounds | Median actual (min) | Median scheduled (min) | Aircraft type matched | Prevalence | Propagation-specific |
|---|---|---|---|---|---|---|---|
| Jan 2022 | 537,902 | 454,290 | 73 | 61 | 98.4% | 18.37% | 6.85% |
| Jan 2024 | 547,271 | 479,131 | 72 | 62 | 90.6% | 22.02% | 10.04% |
| Jun 2024 | 611,132 | 564,144 | 69 | 60 | 87.4% | 24.07% | 10.98% |
| Jun 2025 | 611,575 | 561,216 | 69 | 61 | 82.7% | 26.33% | 12.43% |
| Jan 2026 | 544,003 | 464,429 | 73 | 63 | 80.1% | 19.49% | 7.90% |
| **Total** | **2,851,883** | **2,523,210** | | | | | |

The aircraft-type match rate declines over time because the public aircraft database is a point-in-time snapshot: registrations entering service after it was downloaded cannot be matched. Section 5.5 shows that no conclusion depends on how unmatched registrations are treated.

## 4. Methods

### 4.1 The decision, and when it is made

We study a transparent operational decision: flag a next-leg departure as likely to leave **more than fifteen minutes late**, so that an operator may act — for example by protecting a tight connection, adding buffer, re-sequencing ground resources, or in the limit swapping the aircraft. Fifteen minutes is the conventional on-time boundary in the United States reporting system and provides an externally defined cutoff.

We are explicit about two things the submitted version of this work left implicit.

**What the outcome is called.** We do not label this outcome "delay propagation". It is the delay of the next departure, which may arise from the inbound aircraft or from local causes. Section 5.1 establishes how much of it is attributable to the inbound aircraft, and defines a **propagation-specific secondary outcome** which additionally requires the carrier to have attributed at least fifteen minutes of the resulting arrival delay to a late inbound aircraft. All principal analyses are reported for both.

**When the alert can be raised.** The inbound aircraft's arrival delay is only known at its gate arrival, which fixes the decision at the start of the turnaround. Because the feasible interventions depend on how much time remains, we construct a ladder of three decision points, each using only information that exists at that moment:

- **DP1, inbound off-block.** The inbound aircraft's own departure delay from its previous airport is known.
- **DP2, inbound wheels-on.** The landing time is known; gate arrival is estimated by adding the airport's median taxi-in time, because the actual taxi time is not yet known.
- **DP3, inbound in-block.** The actual gate-arrival delay is known. This is the decision point used in the submitted version.

For each, we report both the discrimination available and the lead time to the outbound gate departure.

### 4.2 The predictor

The predictor is a histogram-based gradient-boosted classifier estimating the probability that the next leg departs more than fifteen minutes late. Its inputs are restricted to information available before that departure: the inbound status appropriate to the decision point, the scheduled turnaround, the difference between them (the *buffer pressure*, capturing how much of the scheduled ground time the inbound delay has consumed), the **scheduled** departure hour and day of week, the airport, and the aircraft type. The actual turnaround duration is an outcome of the process being predicted and is excluded.

Time-of-day features are taken from the scheduled departure rather than the actual departure. The actual departure time is displaced by the very delay being predicted, so using its hour leaks the outcome into the feature set; the submitted version did this, and correcting it costs 0.018 of area under the ROC curve.

That the predictor leans on the inbound status is intentional. The aim is not to discover a novel predictor but to make the decision's dependence on this information explicit, so that its sensitivity to that information's quality can be measured. As a reference point we also evaluate a transparent rule that flags whenever the inbound delay exceeds the scheduled turnaround.

**Protocol.** Models are trained on 1–23 June 2024 (435,694 turnarounds) and evaluated on 24–30 June 2024 (128,450 turnarounds). The split falls on a calendar-day boundary, so no day straddles it. There is **no validation split, because no hyperparameter search was performed**: the settings (250 boosting iterations, learning rate 0.1, maximum depth 6, seed 0) were fixed a priori and never tuned against any split; Section 5.5 reports sensitivity to them. Categorical variables are ordinally encoded with the encoder fitted on the training split only, unseen levels mapping to a distinct code. Confidence intervals are percentile bootstraps over test rows, with 2,000 replications for headline figures and 300–400 for per-stratum figures. Expected calibration error uses ten equal-width bins on the unit interval.

**Reported quantities.** We report the area under the ROC curve as a threshold-free discrimination measure; the area under the precision-recall curve, because outcome prevalence varies substantially across airports and periods and the ROC summary is insensitive to that variation (Saito and Rehmsmeier, 2015); recall and precision at the operating point; the realised alert rate; calibration as a reliability curve and expected calibration error; and the share of delay-minutes among true positives that the decision did not flag.

That last quantity requires care. It is an **upper bound on addressable delay, not preventable delay**. Detecting a delay does not imply it can be removed: that would require an explicit intervention mechanism, resource constraints and treatment effectiveness, none of which we model or claim. We therefore call it *unflagged delay* and use it only as a relative measure between information conditions.

### 4.3 The information-quality experiment

Holding the trained model fixed, we degrade the inbound-status input on the test set in three operationally meaningful ways.

**Additive noise** of increasing standard deviation (5, 10, 20 and 40 minutes), representing an uncertain estimate of how late the inbound will ultimately be. Buffer pressure, being derived from the inbound status, is recomputed consistently from the degraded value.

**Coarsening** to a categorical status. The categories are mutually exclusive and exhaustive: early or on time (≤ 0 min), minor (0–15 min], moderate (15–60 min], and major (> 60 min). Each is represented by the training-set median of its band (−12, 6, 30 and 103 minutes respectively). This represents the common operational case in which only a rough indication of the inbound's lateness is available.

**Absence.** Two variants are reported, because they answer different questions. In the *fallback* variant the deployed model persists and the missing input is imputed by the training mean — what a system does if it simply fills in a default. In the *retrained* variant the model is refitted without the inbound source at all — what a well-engineered system would do if the signal were known to be unavailable.

Every condition is evaluated twice: at a fixed 0.5 threshold, and at the threshold that issues the **same number of alerts** as the clean model. The second evaluation is essential. Imputation compresses predicted probabilities toward the base rate, so fewer cross any fixed threshold; without an alarm-budget control, that compression is indistinguishable from a genuine loss of information, and would be reported as a far larger effect than it is.

### 4.4 Information-source ablation

To value each input we remove whole **information sources** and refit. An information source is what an operator either has or does not have; features derived deterministically from a source belong to that source and are removed with it. Buffer pressure is exactly the inbound status minus the scheduled turnaround, so it belongs to both parents and is removed whenever either is removed.

This matters more than it may appear. Removing the inbound delay while retaining buffer pressure and the scheduled turnaround leaves the removed signal exactly reconstructible from the two survivors, and we measure directly that such an ablation costs **nothing at all**. The submitted version of this work performed single-feature ablations and consequently misstated the value of the inbound signal.

### 4.5 Transfer, calibration and remediation

We test cross-airport transfer by leave-one-hub-out over the eight busiest hubs, and temporal transfer by training on June 2024 and evaluating out of time across 2022–2026. For both, airport identity is excluded from the feature set, so the evaluation isolates transfer of the mechanism rather than memorisation.

Because stable discrimination does not establish stable performance at an operating point, we compare four remedies on held-out target data. For each hub, the source model is trained on all other airports; the hub's own data is split in time, the first 70% forming an adaptation set and the last 30% held out for evaluation. The remedies are: no adaptation; threshold adjustment on the adaptation set; isotonic recalibration on the adaptation set with the threshold kept at 0.5; and retraining on the adaptation set. Every remedy is judged on data none of them saw.

### 4.6 Cost sensitivity

The relative cost of a false alarm and a missed departure is not known precisely and varies by operator. We therefore report decision-curve analysis (Vickers and Elkin, 2006): for a threshold probability *p*, at which an operator is indifferent between acting and not, a false alarm costs *p*/(1−*p*) times a miss, and the net benefit of a strategy is the true-positive rate minus the false-positive rate weighted by that ratio. Plotting net benefit against *p* compares the model against the two default strategies — alert on everything, alert on nothing — across the whole plausible range.

## 5. Results

### 5.1 What the outcome measures

Before measuring how information quality changes the decision, we establish what the decision is about. The BTS causal fields provide the carrier's own attribution of delay, and are the closest thing to ground truth for propagation available in public data. Three properties of that attribution must be stated rather than assumed. The five fields decompose **arrival** delay, not departure delay; they sum exactly to the reported arrival delay in 100.00% of populated rows; and they exist only for flights arriving fifteen or more minutes late. Consequently, 83.0% of our flagged cases carry an attribution. The remaining 17.0% are, by construction, departures that left late but recovered en route and arrived on time — the cases with no downstream consequence.

Among flagged cases that carry an attribution, late-aircraft delay is non-zero in 65.1% and at least fifteen minutes in 54.9%. It is the largest single share of attributed delay-minutes at 47.8%, ahead of carrier delay at 32.4%, national-airspace-system delay at 13.8%, weather at 5.8% and security at 0.2%, and it is the dominant component for 51.9% of flagged cases. The association is graded in the way a propagation account predicts: late-aircraft delay is non-zero in 83.5% of cases where the inbound arrived more than fifteen minutes late, 58.6% where it arrived up to fifteen minutes late, and 24.9% where it arrived on time or early.

The outcome therefore is not synonymous with propagation — roughly a third to a half of flagged cases are attributable to other causes — which is why we do not name it as such. We accordingly define the propagation-specific secondary outcome and report all principal analyses for both.

The scheduled buffer tells the same story more sharply. Figure 2 shows the outcome rate across inbound delay and the scheduled ground time available to absorb it. The rate rises monotonically with inbound delay and falls monotonically with the available buffer. The direct test is decisive: where the inbound delay **exceeds** the scheduled ground time, 93.5% of next departures leave more than fifteen minutes late and 77.1% are propagation-specific; where it **fits inside**, the figures are 18.6% and 5.7%.

Two objections must be met. First, this might merely track disrupted days, on which everything is late. It does not: constructing a daily disruption index from the causal fields and splitting into tertiles, the ratio between the two regimes is 4.77, 4.75 and 4.25 from the least to the most disrupted third, and discrimination is stable across them (0.889, 0.877, 0.882). Second, if the outcome is close to deterministic once the buffer is exhausted, the prediction task might be arithmetic rather than inference. It is not, because that regime is a small minority: on the **91.0%** of test turnarounds where the inbound delay still fits inside the scheduled ground time, the model reaches an area under the ROC curve of 0.838 against a prevalence of 21.5%.

### 5.2 Information quality governs the decision

On the full test set the predictor reaches an area under the ROC curve of **0.887** (95% bootstrap interval 0.884–0.889) on 128,450 held-out turnarounds, with an area under the precision-recall curve of 0.851 against a prevalence of 28.08%. It is well calibrated, with an expected calibration error of 0.015 and a reliability curve lying essentially on the diagonal (Figure 6a). At a 0.5 threshold it achieves 63.1% recall at 93.2% precision, alerting on 19.0% of turnarounds. The transparent rule baseline — flag when the inbound delay exceeds the scheduled turnaround — fires on 9.0% of turnarounds and achieves 30.5% recall at 94.7% precision. The model therefore roughly **doubles** the recall of the rule at a comparable precision, which is a more modest advantage than we previously reported and is stated here as corrected.

Degrading the inbound signal produces the pattern in Table 2 and Figure 3, and the two panels of that figure carry the paper's central message.

*Table 2. Effect of degrading the inbound-status signal (BTS, test set 24–30 June 2024, n = 128,450). Left block: at a fixed 0.5 threshold, where the alert rate is free to move. Right block: at a matched alarm budget of 19.0%, where it is not. Unflagged delay is the share of delay-minutes among true positives that the decision did not flag; it bounds addressable delay and does not measure preventable delay.*

| Inbound-status condition | AUC | AUPRC | ECE | Recall | Precision | Alert rate | Decision flips | Unflagged delay | Recall @ budget | Precision @ budget | Unflagged @ budget |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Clean | 0.887 | 0.851 | 0.015 | 0.631 | 0.932 | 19.0% | — | 27.4% | 0.631 | 0.932 | 27.4% |
| Coarse, 4 buckets | 0.878 | 0.828 | 0.032 | 0.602 | 0.898 | 18.8% | 3.3% | 30.7% | 0.606 | 0.894 | 30.4% |
| Noise, sd 5 min | 0.882 | 0.843 | 0.013 | 0.631 | 0.905 | 19.6% | 2.2% | 27.5% | 0.620 | 0.916 | 28.2% |
| Noise, sd 10 min | 0.871 | 0.820 | 0.026 | 0.623 | 0.850 | 20.6% | 4.8% | 27.7% | 0.597 | 0.881 | 29.5% |
| Noise, sd 20 min | 0.836 | 0.746 | 0.072 | 0.610 | 0.711 | 24.1% | 10.5% | 28.3% | 0.535 | 0.789 | 34.2% |
| Noise, sd 40 min | 0.769 | 0.580 | 0.152 | 0.591 | 0.552 | 30.1% | 19.5% | 29.5% | 0.441 | 0.651 | 43.2% |
| Missing, mean fallback | 0.700 | 0.512 | 0.064 | 0.190 | 0.618 | 8.6% | 18.1% | 76.5% | 0.342 | 0.505 | 62.6% |
| Missing, model retrained | 0.731 | 0.555 | 0.045 | 0.146 | 0.778 | 5.3% | 16.4% | 80.2% | 0.382 | 0.563 | 57.9% |

**Coarse information is nearly free.** Replacing the exact inbound delay by one of four buckets costs 0.009 of discrimination, 0.026 of recall at a matched budget, and changes only 3.3% of decisions. An operator who knows only that the inbound is "moderately late" retains almost the entire value of knowing that it is 34 minutes late.

**Absence is expensive.** Losing the signal costs 0.156 of discrimination if the model is retrained without it, and 0.186 if a deployed model imputes a default. This is a threshold-free loss and cannot be recovered by tuning.

**Noise sits between, and its cost depends on what is held fixed.** At a fixed threshold, noise leaves recall almost unchanged and collapses precision from 0.932 to 0.552 — but only because the alert rate is free to inflate from 19.0% to 30.1%. Held to the same alarm budget, noise costs recall as well: 0.631 falls to 0.441 at a 40-minute standard deviation. Our earlier characterisation of noise as leaving recall intact was an artefact of not controlling the alarm budget, and is corrected here.

**Part of the apparent collapse under absence is an operating-point effect.** At a fixed threshold, recall under a retrained no-inbound model is 0.146 and 80.2% of delay goes unflagged. At a matched alarm budget, the same model achieves 0.382 recall with 57.9% unflagged. Re-setting the threshold alone therefore recovers about half of the recall drop under the retrained model (0.236 of 0.485) and about a third under the mean-fallback variant (0.152 of 0.441); the remainder is genuine. Both numbers are reported because both are operationally meaningful: the first is what happens to a deployed system that loses an input and keeps its threshold, the second is the irreducible information loss.

The propagation-specific outcome reproduces the same pattern more sharply. Coarse status costs 0.014 of discrimination; absence costs 0.213 with retraining. Propagation-specific decisions depend *more* on inbound information than next-departure decisions do, which is what one would expect if the signal is carrying genuinely causal content. We note explicitly that the absolute discrimination on this outcome (0.985) is not a headline accuracy claim: the label is partly endogenous to the inbound aircraft's lateness, because the carrier attributes on that basis. It is not tautological — the inbound delay alone yields 0.871, and the outcome occurs in only 53.7% of cases where the inbound arrived more than an hour late — but it is not an independent prediction target, and we use it only for validation and robustness.

Taken together: **what governs this decision is whether the inbound signal is present, not how precise it is.** A rough but available signal is nearly as good as an exact one; a missing signal is expensive in a way no amount of tuning recovers.

### 5.3 When the decision can be made

Table 3 reports the decision-point ladder.

*Table 3. Discrimination and lead time at three decision points. Recall and precision are at a matched alarm budget of 19.0%. Lead time is measured to the outbound gate departure.*

| Decision point | Inbound signal available | AUC | AUPRC | Recall | Precision | Median lead (min) | 25th pct | 10th pct |
|---|---|---|---|---|---|---|---|---|
| DP3, inbound in-block | actual gate-arrival delay | 0.887 | 0.851 | 0.631 | 0.932 | 68 | 51 | 41 |
| DP2, inbound wheels-on | landing time plus median taxi-in | 0.878 | 0.838 | 0.617 | 0.912 | 77 | 59 | 48 |
| DP1, inbound off-block | inbound's own departure delay | 0.855 | 0.800 | 0.580 | 0.857 | 212 | 156 | 125 |

The trade-off is unexpectedly favourable. Moving the decision from the inbound's gate arrival back to the inbound's own gate departure buys **144 minutes** of median lead time — from 68 to 212 minutes — for 0.032 of discrimination and 0.051 of recall at the same alarm budget. The tenth percentile of lead time rises from 41 to 125 minutes.

This changes what the alert is for. At DP3 the median lead time is a little over an hour and 38.7% of cases have under an hour; the feasible responses are largely local — protecting a connection, prioritising a cleaning or fuelling crew, holding a gate. At DP1 the median is three and a half hours and essentially no case has under an hour; responses that require planning, such as re-assigning ground-handling capacity across a bank or swapping the aircraft, become feasible. The information-quality result of Section 5.2 applies at every rung: it is the presence of an inbound signal, at whatever resolution and however early, that carries the decision.

### 5.4 What each information source is worth

Table 4 reports the ablation, with each information source removed together with its derivatives and the model refitted.

*Table 4. Value of each information source, measured by refitting without it (test set n = 128,450, prevalence 28.08%). Recall at budget is at the full model's 19.0% alert rate.*

| Source removed | AUC | 95% interval | ΔAUC | AUPRC | Recall | Recall @ budget | Unflagged delay |
|---|---|---|---|---|---|---|---|
| — (full model) | 0.887 | 0.884–0.889 | — | 0.851 | 0.631 | 0.631 | 27.4% |
| Inbound status | 0.731 | 0.728–0.734 | 0.156 | 0.555 | 0.146 | 0.382 | 80.2% |
| Schedule | 0.842 | 0.840–0.845 | 0.044 | 0.759 | 0.527 | 0.554 | 40.7% |
| Time of day | 0.878 | 0.876–0.880 | 0.009 | 0.841 | 0.624 | 0.629 | 28.6% |
| Aircraft type | 0.880 | 0.877–0.883 | 0.007 | 0.841 | 0.618 | 0.620 | 28.3% |
| Airport | 0.882 | 0.880–0.885 | 0.005 | 0.846 | 0.624 | 0.629 | 28.0% |

The ordering is stark and the gap is an order of magnitude: the inbound status is worth 0.156 of discrimination, the schedule 0.044, and everything else 0.009 or less. On the propagation-specific outcome the gap is wider still, at 0.213 against 0.040.

The grouping is what makes this measurement valid. Removing the inbound delay while retaining buffer pressure and the scheduled turnaround leaves the signal exactly reconstructible, and we measure that such an ablation yields an area under the ROC curve of **0.887 — identical to the full model, to three decimal places**. A single-feature ablation of this design cannot detect the value of an input that has a surviving derivative, and any conclusion drawn from one is unsafe.

Read as a value-of-information statement, the decision is governed overwhelmingly by whether the inbound signal is present, secondarily by the schedule, and barely at all by the remaining context. This both explains the cost of absence in Section 5.2 and tells an operator where investment in data quality pays.

### 5.5 Transfer, calibration, and what to do about it

**Across airports.** Leave-one-hub-out over the eight busiest hubs gives areas under the ROC curve of 0.837 to 0.878, with a mean of 0.861. Discrimination transfers. The operating point does not: at a fixed 0.5 threshold, recall ranges from 0.483 at Atlanta to 0.610 at Chicago O'Hare, and prevalence ranges from 20.0% to 34.2%. Expected calibration error ranges from 0.008 to 0.037.

**Across time.** A model trained on June 2024 and applied out of time retains areas under the ROC curve of 0.830 to 0.876 against an in-distribution 0.882. But the precision-recall summary, which is sensitive to prevalence, moves far more: 0.726 to 0.829 against an in-distribution 0.846, tracking prevalence from 18.4% to 28.1%. The realised alert rate at a fixed threshold moves correspondingly, from 11.4% to 18.8%. Reporting discrimination alone would have concealed this entirely, which is precisely the reason to report both (Saito and Rehmsmeier, 2015). Figure 6 shows both panels.

**Does local adaptation help?** Table 5 compares four remedies on held-out target data. The answer corrects a claim we previously made.

*Table 5. Four remedies for transferring a model to a new airport, evaluated on held-out target data. Means across the eight busiest hubs.*

| Remedy | Recall | Precision | Alert rate | ECE |
|---|---|---|---|---|
| No adaptation | 0.591 | 0.934 | 18.6% | 0.026 |
| Threshold adjustment | 0.575 | 0.910 | 18.6% | — |
| Isotonic recalibration | 0.598 | 0.926 | 19.1% | 0.018 |
| Local retraining | 0.610 | 0.916 | 19.6% | 0.021 |

Local retraining buys **nothing** in discrimination: a locally retrained model achieves a mean area under the ROC curve of 0.866 against 0.868 for the transferred model. All four remedies land within a recall band of 0.575 to 0.610. Recalibration's real benefit is calibration — expected calibration error falls from 0.026 to 0.018 — rather than catch rate, and naive threshold transplanting is the weakest of the four.

The submitted version of this work concluded that a transferred model "needs local threshold recalibration to reach a target catch-rate". Tested directly, that is overstated: the model transfers robustly, local adaptation yields modest gains, and its value lies in restoring calibrated probabilities rather than in recovering missed detections.

**Robustness of the pipeline itself.** Three checks bound the influence of our own design choices. Varying the retained-turnaround range across five settings, including 1–1440 minutes, gives areas under the ROC curve from 0.879 to 0.896; the 20–720 filter excludes 3.9% of pairs, which are slightly more delay-prone, and does not drive the result. Varying hyperparameters across six settings gives 0.883 to 0.887, and five random seeds give results identical to three decimal places. Treating unmatched aircraft registrations three different ways — coding them as a distinct level, excluding them, or dropping the type feature entirely — gives 0.887, 0.888 and 0.880.

### 5.6 The European arm, and two claims withdrawn

We derived 203,103 turnarounds at eight European hubs from OpenSky ADS-B surveillance, across five 14-day windows beginning 1 January 2022, 1 January 2024, 1 June 2024, 1 June 2025 and 1 January 2026. We note that a previous version of this work described these as five *months*; they are 14-day windows.

The descriptive structure is sound and is reported as such. Median turnaround durations range from 76.8 minutes at Zurich to 122.3 minutes at London Heathrow (Figure 7a, with per-hub counts). These sit above the United States gate-to-gate figures because ADS-B first-seen and last-seen timestamps are airborne, so the European durations include taxi-in and taxi-out time. A cross-airport classifier for long turnarounds transfers between hub groups with an area under the ROC curve of 0.783 on generic features, rising to 0.814 with aircraft type joined from the public database; aircraft type likewise lifts turnaround-duration regression from an *R*² of 0.480 to 0.558.

Two claims made in the submitted version do not survive re-examination, and we withdraw them.

**Ground-time-state propagation.** We had reported that an aircraft's excess ground time correlates 0.31 with the excess on its next leg. That figure reproduces exactly, but it is an artefact of two things. First, route composition: aircraft operating long-turn routes have long turns at both ends, which is a fixed effect, not propagation. Removing route-level means reduces the association from 0.210 to 0.066. Second, the pairs are not consecutive operations: a substantial share have a gap of more than twelve hours, meaning the aircraft left the eight sampled hubs and returned later. Restricting to genuine consecutive legs within twelve hours and removing route means leaves a within-route correlation of **−0.024** (95% bootstrap interval −0.038 to −0.009). Figure 7b shows the full sequence. There is no ground-time-state propagation detectable in this data.

**Temporal stability.** We had reported that a European model retains areas under the ROC curve of 0.79–0.85 across the five windows. That result holds only when the label threshold and the categorical frequency maps are estimated across all five windows, including the test windows. Estimated within each window, as a clean temporal protocol requires, discrimination falls to 0.606–0.650 against an in-distribution 0.833. The claim is not supported and we do not make it.

**Why the degradation experiments have no European counterpart.** OpenSky carries no schedules, so the only candidate inbound-status signal is the aircraft's own previous ground stop. It is observable for 10.5% of turnarounds, because only eight hubs are sampled, and its within-route association with the outcome is −0.024. There is no signal of any strength to degrade, so the central experiment of this paper cannot be replicated on this dataset. We therefore present the European arm as **descriptive supplementary evidence** and make no cross-regional replication claim.

## 6. Discussion

The results reframe a familiar message into an actionable one. That delay propagates through the turnaround, and that turnaround delay is predictable, are both established; neither is the contribution here. Indeed, that a late inbound predicts a late next departure is close to arithmetic once the buffer is exhausted, and we show exactly where that regime begins and that it covers only 9% of turnarounds.

The contribution is the asymmetry. **Availability dominates precision.** Coarse inbound status — four buckets — costs 0.009 of discrimination, while absence costs 0.156. Between them, noise degrades the decision in a mode that depends entirely on what is held fixed: at a constant threshold it floods the operator with false alarms while preserving recall, and at a constant alarm budget it costs recall instead. These are different operational failures and they call for different remedies. The noisy case calls for uncertainty handling, abstention and alert suppression; the absent case calls for data availability, redundancy and explicit fallback logic. A system designed without distinguishing them, or evaluated only on aggregate accuracy at a single threshold, will mis-prioritise its engineering effort.

The practical corollary is encouraging for operators. Because it is presence rather than precision that carries the decision, the investment that pays is not a better inbound-delay estimator but a more *available* one. A categorical status that is always there beats a minute-accurate figure that is sometimes missing.

The decision-point ladder sharpens this. Because the inbound aircraft's own departure delay is available roughly three and a half hours before the outbound gate departure, and costs only 0.032 of discrimination, the practically useful decision point is much earlier than the one this literature usually assumes. A model evaluated only at the inbound's gate arrival understates what such a system could do, because at that point most of the interventions worth making are already infeasible.

Decision-curve analysis makes the value of information conditional in an intuitive way. Where false alarms are nearly free, alerting on everything is close to optimal and the inbound signal is worth almost nothing (a net-benefit gap of 0.0008 at a threshold probability of 0.05). Where false alarms are costly, the gap widens to 0.135 at a threshold probability of 0.5. **Information is worth most precisely when one cannot afford to alert on everything** — which is the regime real operators are in, and the regime in which alert fatigue is a genuine risk rather than a rhetorical one.

On deployment, our own corrected finding is the useful one. The model's ranking transfers across airports and four years; local retraining adds nothing; and what local data buys is calibration rather than catch rate. That is a cheaper deployment story than the one we previously told, and a better-evidenced one. What must be monitored is prevalence and the realised alert rate, not discrimination: across months, the area under the ROC curve moved by 0.05 while the precision-recall summary moved by 0.12 and the alert rate at a fixed threshold by nearly two-thirds of its own value.

Finally, the study speaks to the evidential problem that motivates much synthetic turnaround research. The information-quality-to-decision-value relationship is exactly the kind of claim that simulation-based studies assert but cannot establish alone. Demonstrating it on real, public data provides an empirical anchor that complements rather than replaces model-based work. That anchor is at block level — turnaround durations, delays and their propagation — and does not extend to internal sub-process resource mechanics, which remain visible only through proprietary data.

## 7. Limitations

The analysis is block-level throughout. It concerns turnaround durations, delays and their propagation, not the internal sub-process timings — cleaning, catering, fuelling, resource assignment — which public data does not expose.

The causal validation inherits the limits of its source. The BTS causal fields are **carrier-reported attributions**, not independently audited measurements; they decompose arrival rather than departure delay; and they exist only for flights arriving fifteen or more minutes late, so 17.0% of flagged cases carry no attribution. The propagation-specific outcome built from them is partly endogenous to the inbound aircraft's lateness and is therefore used for validation and robustness rather than as an independent prediction target.

The information degradation is a controlled perturbation of a known input, not a measurement of a specific real sensing or reporting pipeline. It bounds the decision's sensitivity to information quality rather than reproducing a particular operational fault. The decision-point ladder mitigates this by using genuinely earlier real signals, but the noise and coarsening conditions remain synthetic perturbations.

The modelled decision is a transparent proxy for operator action, not a deployed system evaluated against operator outcomes. Unflagged delay bounds addressable delay and does not measure preventable delay; establishing the latter would require an intervention model, resource constraints and treatment effectiveness.

The primary analysis is United States domestic operations of the reporting carriers. The European arm is descriptive, is taxi-inclusive, lacks schedules, and — as Section 5.6 documents — does not support propagation or stability claims at the eight hubs sampled. We make no cross-regional generalisation.

Finally, the aircraft-type join relies on a point-in-time public database whose coverage of our sample falls from 98.4% to 80.1% across the four years; Section 5.5 shows the conclusions are insensitive to this, but the join is not complete.

## 8. Future work

Three extensions follow. First, the decision-point ladder can be extended below DP1 by reconstructing inbound-status estimates as they would have been available at successive times before the inbound even departs, quantifying the value of information as a continuous function of how early it is known. Second, a European replication of the central experiment requires a schedule-bearing European source; the present OpenSky sample cannot provide one, and a denser airport sample would address the chaining sparsity but not the absence of schedules. Third, the decision model can be connected to an explicit mitigation and evaluated against realised outcomes, moving from a decision proxy toward an operational pilot — the step that would convert unflagged delay from a bound on addressable delay into a measurement of preventable delay.

## 9. Conclusion

Using public United States on-time records we derived 2.52 million real aircraft turnarounds and measured how the quality of inbound information governs the decision to flag a late next departure. We first established what that outcome measures, validating it against the carrier-reported causal fields, against the scheduled buffer, and across operational conditions.

The central finding is an asymmetry between availability and precision. Coarsening the inbound signal to four categories costs 0.009 of discrimination; losing it costs 0.156 and leaves 57.9% of the delay unflagged at an equal alarm budget. Noise costs precision at a fixed threshold and recall at a fixed budget, but never as much as absence. The alert can be raised 144 minutes earlier than the turnaround itself for 0.032 of discrimination, which changes which interventions are feasible. The signal is worth most where false alarms are costly. The predictor transfers across airports and four years; local retraining adds nothing, and recalibration restores calibration rather than catch rate.

Where our own earlier claims did not survive testing — the baseline comparison, the behaviour of noise, the magnitude of the collapse under absence, the necessity of local recalibration, and two European results — we have corrected or withdrawn them. What remains is a reproducible, real-data measurement of a single operationally useful proposition: for this decision, it is the presence of the inbound signal, not its precision, that matters.

## Data and code availability

This study uses only public data: the United States Bureau of Transportation Statistics Reporting Carrier On-Time Performance database, OpenSky Network historical ADS-B, and the public OpenSky aircraft database. The complete derivation and analysis pipeline is released in a public repository under the MIT licence and archived with a permanent digital object identifier; the repository contains every script needed to reproduce every number, table and figure in this paper, pinned dependency versions, a documented run order, the scripts used for the previously submitted version of this work, a record of every difference between the two analyses, and a verification script that checks each quantitative claim in this manuscript against the stored results. The identifier is given on the title page and is withheld here so as not to identify the author during anonymous review.

## References

AhmadBeygi, S., Cohn, A., Guan, Y., Belobaba, P., 2008. Analysis of the potential for delay propagation in passenger airline networks. Journal of Air Transport Management 14, 221–236. https://doi.org/10.1016/j.jairtraman.2008.04.010

Asadi, E., Fricke, H., 2022. Aircraft total turnaround time estimation using fuzzy critical path method. Journal of Project Management 7, 241–254. https://doi.org/10.5267/j.jpm.2022.4.001

Brueckner, J.K., Czerny, A.I., Gaggero, A.A., 2021. Airline mitigation of propagated delays via schedule buffers: Theory and empirics. Transportation Research Part E: Logistics and Transportation Review 150, 102333. https://doi.org/10.1016/j.tre.2021.102333

Cui, Y., Ma, L., Ding, Q., He, X., Xiao, F., Cheng, B., 2024. Aircraft turnaround time dynamic prediction based on Time Transition Petri Net. PLOS ONE 19, e0305237. https://doi.org/10.1371/journal.pone.0305237

Dahanayaka, M., Prak, D., Mes, M., 2026. From gate to runway: A systematic review of airport ground operations optimization. Journal of Air Transport Management 135, 103013. https://doi.org/10.1016/j.jairtraman.2026.103013

Evler, J., Asadi, E., Preis, H., Fricke, H., 2021a. Airline ground operations: Optimal schedule recovery with uncertain arrival times. Journal of Air Transport Management 92, 102021. https://doi.org/10.1016/j.jairtraman.2021.102021

Evler, J., Asadi, E., Preis, H., Fricke, H., 2021b. Airline ground operations: Schedule recovery optimization approach with constrained resources. Transportation Research Part C: Emerging Technologies 128, 103129. https://doi.org/10.1016/j.trc.2021.103129

Evler, J., Lindner, M., Fricke, H., Schultz, M., 2022. Integration of turnaround and aircraft recovery to mitigate delay propagation in airline networks. Computers & Operations Research 138, 105602. https://doi.org/10.1016/j.cor.2021.105602

Fleurquin, P., Ramasco, J.J., Eguíluz, V.M., 2013. Systemic delay propagation in the US airport network. Scientific Reports 3, 1159. https://doi.org/10.1038/srep01159

Glomb, L., Liers, F., Rösel, F., 2023. Optimizing integrated aircraft assignment and turnaround handling. European Journal of Operational Research 310, 1051–1071. https://doi.org/10.1016/j.ejor.2023.03.036

Gök, Y.S., Padrón, S., Tomasella, M., Guimarans, D., Ozturk, C., 2022. Constraint-based robust planning and scheduling of airport apron operations through simheuristics. Annals of Operations Research 320, 795–830. https://doi.org/10.1007/s10479-022-04547-0

Guimarans, D., Padrón, S., 2022. A stochastic approach for planning airport ground support resources. International Transactions in Operational Research 29, 3316–3345. https://doi.org/10.1111/itor.13104

Kafle, N., Zou, B., 2016. Modeling flight delay propagation: A new analytical-econometric approach. Transportation Research Part B: Methodological 93, 520–542. https://doi.org/10.1016/j.trb.2016.08.012

Liu, C., Chen, Y., Wang, H., Zhang, Y., Dai, X., Luo, Q., Chen, L., 2023. Airport flight ground service time prediction with missing data using graph convolutional neural network imputation and bidirectional sliding mechanism. Applied Soft Computing 133, 109941. https://doi.org/10.1016/j.asoc.2022.109941

Okwir, S., Amouzgar, K., Ng, A.H., 2025. Exploring prediction accuracy for optimal taxi times in airport operations using various machine learning models. Journal of Air Transport Management 122, 102684. https://doi.org/10.1016/j.jairtraman.2024.102684

Olive, X., 2019. traffic, a toolbox for processing and analysing air traffic data. Journal of Open Source Software 4, 1518. https://doi.org/10.21105/joss.01518

Pyrgiotis, N., Malone, K.M., Odoni, A., 2013. Modelling delay propagation within an airport network. Transportation Research Part C: Emerging Technologies 27, 60–75. https://doi.org/10.1016/j.trc.2011.05.017

Ruf, C., Schiffels, S., Kolisch, R., Frey, M.M., 2022. A data-driven approach for baggage handling operations at airports. Transportation Science 56, 1179–1195. https://doi.org/10.1287/trsc.2022.1127

Sahadevan, D., Al Ali, H., Notman, D., Mukandavire, Z., 2023. Optimising airport ground resource allocation for multiple aircraft using machine learning-based arrival time prediction. Aerospace 10, 509. https://doi.org/10.3390/aerospace10060509

Saito, T., Rehmsmeier, M., 2015. The precision-recall plot is more informative than the ROC plot when evaluating binary classifiers on imbalanced datasets. PLOS ONE 10, e0118432. https://doi.org/10.1371/journal.pone.0118432

Schäfer, M., Strohmeier, M., Lenders, V., Martinovic, I., Wilhelm, M., 2014. Bringing up OpenSky: A large-scale ADS-B sensor network for research. In: Proceedings of the 13th IEEE/ACM International Symposium on Information Processing in Sensor Networks (IPSN), pp. 83–94.

Schmidt, M., 2017. A review of aircraft turnaround operations and simulations. Progress in Aerospace Sciences 92, 25–38. https://doi.org/10.1016/j.paerosci.2017.05.002

Schultz, M., Reitmann, S., 2019. Machine learning approach to predict aircraft boarding. Transportation Research Part C: Emerging Technologies 98, 391–408. https://doi.org/10.1016/j.trc.2018.09.007

Schultz, M., Rosenow, J., Olive, X., 2022. Data-driven airport management enabled by operational milestones derived from ADS-B messages. Journal of Air Transport Management 99, 102164. https://doi.org/10.1016/j.jairtraman.2021.102164

Strohmeier, M., Olive, X., Lübbe, J., Schäfer, M., Lenders, V., 2021. Crowdsourced air traffic data from the OpenSky Network 2019–2020. Earth System Science Data 13, 357–372. https://doi.org/10.5194/essd-13-357-2021

Tang, X., Wu, J., Wu, C.-L., Ding, Y., Zhang, S., 2025. Dynamic prediction of aircraft turnaround milestone times using a cascaded gradient boosting model for improved airport collaborative decision-making. Journal of Air Transport Management 128, 102842. https://doi.org/10.1016/j.jairtraman.2025.102842

United States Bureau of Transportation Statistics, 2026. Reporting Carrier On-Time Performance (1987–present). TranStats, U.S. Department of Transportation. https://www.transtats.bts.gov

Vickers, A.J., Elkin, E.B., 2006. Decision curve analysis: A novel method for evaluating prediction models. Medical Decision Making 26, 565–574. https://doi.org/10.1177/0272989X06295361

Volt, J., Had, P., Stojić, S., Delahaye, D., 2026. Improving aircraft ground handling times prediction using machine learning approaches. Transport Policy 184, 104196. https://doi.org/10.1016/j.tranpol.2026.104196

Xu, J., Ding, M., Zhang, Z.-Z., Xu, Y.-B., Wang, X.-H., Zhao, F., 2023. Vision-based automatic collection of nodes of in/off block and docking/undocking in aircraft turnaround. Applied Sciences 13, 7832. https://doi.org/10.3390/app13137832

Zhu, S., Sun, H., Guo, X., 2022. Cooperative scheduling optimization for ground-handling vehicles by considering flights' uncertainty. Computers & Industrial Engineering 169, 108092. https://doi.org/10.1016/j.cie.2022.108092

---

## Figure captions

**Figure 1.** United States turnaround structure, June 2024 (n = 564,144). (a) Distribution of derived gate-to-gate ground time, median 69 minutes. (b) Share of turnarounds whose next departure left more than fifteen minutes late, and the propagation-specific share, as a function of inbound arrival delay. Shaded bands are 95% Wilson intervals; the count in each bin is printed above its point.

**Figure 2.** Share of turnarounds whose next departure left more than fifteen minutes late, by inbound arrival delay and by the scheduled ground time available to absorb it (BTS, June 2024). Cell counts are printed beneath each percentage. The outcome rises monotonically with inbound delay and falls monotonically with available buffer, which is the behaviour a buffer-absorption account predicts.

**Figure 3.** Effect of degrading the inbound-status signal. (a) At a fixed 0.5 decision threshold, as in the submitted version: the alert rate is free to move, rising from 19.0% to 30.1% under noise, so recall is preserved and precision collapses. (b) At a matched alarm budget of 19.0%: noise now costs recall as well, and missing information costs most of all. The dotted line marks clean-information recall.

**Figure 4.** Decision-point ladder. Discrimination against median lead time to the outbound gate departure; the horizontal bar extends to the tenth percentile of lead time. Moving the decision from the inbound's gate arrival to its own gate departure buys 144 minutes for 0.032 of area under the ROC curve.

**Figure 5.** Decision-curve analysis. Net benefit against threshold probability *p*, at which a false alarm costs *p*/(1−*p*) times a missed late departure. The shaded region is the value of having the inbound signal. It is negligible where false alarms are nearly free and largest where they are costly.

**Figure 6.** Calibration and transfer. (a) Reliability curve with 95% Wilson intervals and bin counts (expected calibration error 0.015). (b) Held-out hubs: discrimination is stable while the precision-recall summary tracks prevalence. (c) Out-of-time months: the same pattern. Bin and group counts are printed on each panel.

**Figure 7.** European arm (OpenSky ADS-B, eight hubs, five 14-day windows, 203,103 turnarounds). (a) Median and interquartile range of taxi-inclusive turnaround duration per hub, with the number of turnarounds behind each median. (b) Re-audit of the previously reported ground-time-state propagation result: the pooled correlation falls to zero once pairs separated by more than twelve hours are excluded, and the within-route correlation is near zero throughout.
