# adsb-turnaround-information-quality

Reproducible pipeline and analysis for a study of how the **quality of inbound-status
information** governs an aircraft-turnaround departure-delay decision, using public US
Bureau of Transportation Statistics (BTS) on-time records and public European OpenSky
ADS-B data.

This repository accompanies the manuscript *"Availability before precision: how the quality
of inbound information governs the aircraft turnaround departure-delay decision"* (Journal
of Air Transport Management, manuscript JATM-D-26-00779, revised version). It contains
everything needed to regenerate every number, table and figure in the paper from the public
source data.

`code/verify_manuscript.py` re-checks all 290 quantitative claims in the manuscript against
the stored results in `out/`, and runs without re-deriving the data.

---

## 1. What the study measures

A turnaround is reconstructed by pairing an aircraft's arrival at an airport with that
same aircraft's next departure from it. The modelled decision is: **flag the next
departure as likely to leave more than 15 minutes late**, so that an operator can act.
The experiment then degrades the single most operationally uncertain input — the inbound
aircraft's status — and measures what happens to the *decision*, not just to accuracy.

Two outcomes are used throughout:

| Symbol | Definition | Role |
|---|---|---|
| `y` | the next departure leaves more than 15 min late | primary |
| `y_prop` | `y`, **and** the carrier attributed at least 15 min of the resulting arrival delay to a late inbound aircraft (`LateAircraftDelay`) | propagation-specific secondary |

`y_prop` is partly endogenous to the inbound aircraft's lateness, because the carrier
assigns `LateAircraftDelay` on that basis. It is therefore used to validate the outcome
and as a robustness target, never as a headline accuracy figure.

## 2. Installation

```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Verified with Python 3.10.12, numpy 2.2.6, pandas 2.3.3, scikit-learn 1.7.2,
scipy 1.15.3, matplotlib 3.10.9.

## 3. Obtaining the source data

Neither dataset is redistributed here; both are public and free.

**United States — BTS Reporting Carrier On-Time Performance.**
From <https://www.transtats.bts.gov/DL_SelectFields.aspx?gnoyr_VQ=FGJ> (TranStats →
Aviation → *Reporting Carrier On-Time Performance (1987–present)*), download one CSV per
month and unzip into `data_raw/`. The analysis uses five months: **2022-01, 2024-01,
2024-06, 2025-06, 2026-01**. Required fields:

```
FlightDate, Tail_Number, Reporting_Airline, Origin, Dest,
CRSDepTime, DepTime, DepDelay, CRSArrTime, ArrTime, ArrDelay, ArrDel15,
Cancelled, Diverted, WheelsOn, TaxiIn, Distance,
CRSElapsedTime, ActualElapsedTime,
CarrierDelay, WeatherDelay, NASDelay, SecurityDelay, LateAircraftDelay
```

**Aircraft types — OpenSky aircraft database.** Download `aircraftDatabase.csv` from
<https://opensky-network.org/datasets/metadata/> into `data_raw/`. It is a point-in-time
snapshot, so the match rate falls for later months (98.4% for 2022-01 down to 80.1% for
2026-01); `exp_robustness.py --part types` shows no conclusion depends on this.

**Europe — OpenSky historical ADS-B.** Derived turnarounds for eight hubs (EHAM, EDDF,
EGLL, LFPG, LEMD, EDDM, LIRF, LSZH) over five 14-day windows beginning 2022-01-01,
2024-01-01, 2024-06-01, 2025-06-01 and 2026-01-01, pulled through the Trino interface with
`pyopensky`. OpenSky access requires a free account and agreement to the network's
efficient-use policy. The pull scripts are in `code/legacy/`; the derived per-window
turnaround files are what the European analysis consumes.

## 4. Running everything

```bash
# 1. build turnarounds (about 80 s per month)
for m in 2022_01 2024_01 2024_06 2025_06 2026_01; do
  python code/build_turnarounds_us.py --only $m
done
python code/build_turnarounds_us.py --only 2024_06 --bounds 1,1440 --suffix _wide

# 2. analyses
python code/validate_outcome.py
python code/exp_information_quality.py
python code/exp_ablation.py --outcome y
python code/exp_ablation.py --outcome y_prop
python code/exp_decision_timing.py
python code/exp_transfer.py --stage drift
python code/exp_transfer.py --stage airports
python code/exp_transfer.py --stage remediation
python code/exp_robustness.py --part buffer
python code/exp_robustness.py --part bounds
python code/exp_robustness.py --part types
python code/exp_robustness.py --part protocol
python code/exp_robustness.py --part conditions
python code/exp_europe.py

# 3. figures
python code/make_figures.py --fig all
```

Results land in `out/` (CSV plus a plain-text log per experiment) and `figures/`
(600 dpi PNG and vector PDF).

The source and derived data are large, so every path can be redirected without moving the
checkout:

| variable | default | purpose |
|---|---|---|
| `ADSB_ROOT` | parent of `code/` | project root |
| `ADSB_DATA_RAW` | `$ADSB_ROOT/data_raw` | downloaded BTS and aircraft-database CSVs |
| `ADSB_DATA_OUT` | `$ADSB_ROOT/data_out` | derived turnaround files |
| `ADSB_EU_SRC` | `$ADSB_ROOT/data_out` | derived OpenSky turnaround files |
| `ADSB_OUT` | `$ADSB_ROOT/out` | results |
| `ADSB_FIGS` | `$ADSB_ROOT/figures` | figures |

```bash
ADSB_DATA_RAW=/mnt/bigdisk/bts ADSB_DATA_OUT=/mnt/bigdisk/derived \
  python code/exp_ablation.py --outcome y
```

## 5. Methodological specification

Everything below is fixed in `code/common.py` and is stated here because the referee asked
for it explicitly.

### Turnaround construction
- **Pairing.** Flights are ordered per aircraft registration by departure time; an arrival
  into airport A is paired with that aircraft's next departure, retained only if that
  departure's origin is A. Cancelled and diverted flights are removed first.
- **Event times.** Reconstructed as *scheduled time + reported delay* rather than from
  clock arithmetic. This reproduces the reported clock times in **100.000%** of rows and is
  immune to the midnight wrap.
- **Duration.** Computed modulo 24 h, which is additionally immune to timezone differences.
  Agrees with a naive date-shifted construction in **100.00%** of retained pairs.
- **Retained range.** 20 to 720 minutes. `exp_robustness.py --part bounds` reports AUC
  0.879–0.896 across five alternative ranges including 1–1440.
- **Scheduled turnaround.** Wrapped into a **signed** window [−720, +720) minutes. An
  unsigned [0, 1440) wrap maps genuinely negative scheduled turnarounds — rotations already
  infeasible as scheduled, 1.95% of pairs — onto roughly +1400 minutes of imaginary buffer,
  in exactly the most stressed cases. The signed form raises the correlation between
  scheduled and actual ground time from 0.712 to 0.851.

### Features and information sources
| Source | Features |
|---|---|
| inbound status | `inbound_arr_delay` |
| schedule | `sched_turnaround_min` |
| time of day | `sched_hour`, `dow` |
| airport | `airport_code` |
| aircraft type | `type_code` |

`buffer_pressure` = `inbound_arr_delay` − `sched_turnaround_min` is a **derived** feature
belonging to both of its parents, and is removed whenever either is removed. Time-of-day
features come from the **scheduled** departure; using the actual departure hour leaks the
outcome, because that time is displaced by the delay being predicted.

### Split, model, seeds
- **Split.** Train 1–23 June 2024 (n = 435,694); test 24–30 June 2024 (n = 128,450). The
  boundary is a calendar day, so no day straddles it.
- **No validation split**, because no hyperparameter search was performed. Settings were
  fixed a priori: `HistGradientBoostingClassifier(max_iter=250, learning_rate=0.1,
  max_depth=6, random_state=0)`. `exp_robustness.py --part protocol` reports AUC
  0.883–0.887 across six alternative settings and identical results across five seeds.
- **Categorical encoding.** Ordinal, fitted on the **training split only**; levels unseen
  in training map to −1.
- **Bootstrap.** Percentile bootstrap, i.i.d. resampling of test rows: 2000 replications
  for headline intervals, 300–400 for per-stratum intervals.
- **Expected calibration error.** 10 equal-width bins on [0, 1].

### Metrics
AUC, AUPRC, recall, precision, alert rate, Brier score, ECE, and *unflagged propagated
delay* — the share of delay-minutes among true positives that the decision did not flag.
That last quantity is an **upper bound on addressable delay, not preventable delay**: no
intervention model is claimed, and detecting a delay does not imply it can be removed.

Every information condition is evaluated twice: at a fixed 0.5 threshold, and at the
threshold that issues the **same number of alerts** as the clean model. The second is
necessary because a degraded model's probabilities shrink toward the base rate, so part of
any apparent recall collapse is an operating-point shift rather than a loss of information.

## 6. Repository layout

```
code/
  common.py                  configuration, loading, feature groups, metrics, bootstrap
  build_turnarounds_us.py    BTS flights -> turnarounds
  validate_outcome.py        outcome validation against the BTS causal fields
  exp_information_quality.py noise / coarsening / missing, fixed and matched budgets
  exp_ablation.py            information-source ablation with retraining
  exp_decision_timing.py     decision-point ladder and decision-curve analysis
  exp_transfer.py            cross-airport, out-of-time, and remediation comparison
  exp_robustness.py          buffers, bounds, aircraft types, protocol, conditions
  exp_europe.py              European descriptive arm and re-audit
  make_figures.py            all figures
  legacy/                    the originally submitted scripts, for comparison
data_raw/                    source data (not redistributed; see section 3)
data_out/                    derived turnaround files
out/                         results, one CSV and one log per experiment (committed, so verify_manuscript.py runs standalone)
figures/                     600 dpi PNG and vector PDF
docs/CHANGES_FROM_SUBMITTED.md   every difference from the submitted version
```

## 7. Known limitations

- Analysis is at **block level**: turnaround durations, delays and their propagation. It
  says nothing about internal sub-process timings, which public data does not expose.
- The BTS causal fields decompose **arrival** delay and exist only for flights arriving 15
  or more minutes late, so 17.0% of `y = 1` cases carry no attribution. Those are, by
  construction, departures that were late but recovered en route.
- The causal fields are **carrier-reported attributions**, not independently audited.
- European turnarounds are **taxi-inclusive** ADS-B proxies, so systematically longer than
  the US gate-to-gate figures, and OpenSky carries no schedules, so the European outcome is
  excess ground time rather than delay. The European arm is descriptive only; see
  `docs/CHANGES_FROM_SUBMITTED.md` for the claims withdrawn after re-audit.
- Information degradation is a controlled perturbation, not a measurement of a specific
  real reporting pipeline. It bounds sensitivity rather than reproducing a particular fault.

## 8. Licence and citation

Code is released under the MIT licence (see `LICENSE`). The source datasets remain subject
to their own terms: BTS data are US Government public-domain works; OpenSky data are
subject to the OpenSky Network terms of use.

If you use this code, please cite the paper and this repository (see `CITATION.cff`).

## 9. Acknowledgements

This work makes use of data from the OpenSky Network and from the United States Bureau of
Transportation Statistics.
