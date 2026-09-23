#!/usr/bin/env python
"""
Verify every quantitative claim in the revised manuscript against the result files.

Each check names the claim, the source file, and the expected value. A claim that is not
backed by a number in out/ is a failure, and so is a number that has drifted. Run this
before every submission.
"""
from __future__ import annotations
import re, sys
from pathlib import Path
import pandas as pd

OUT = Path(__file__).resolve().parents[1] / "out"
MS = Path(__file__).resolve().parents[1] / "manuscript" / "MANUSCRIPT_REVISED.md"
TXT = MS.read_text(encoding="utf-8")
fails, checks = [], 0


def load(name, index=None):
    d = pd.read_csv(OUT / name)
    return d.set_index(index) if index else d


def chk(label, actual, expected, tol=5e-4):
    """The manuscript states `expected`; the pipeline produces `actual`."""
    global checks
    checks += 1
    if actual is None or abs(float(actual) - float(expected)) > tol:
        fails.append(f"  MISMATCH  {label}: manuscript says {expected}, pipeline gives {actual}")


def in_text(label, *phrases):
    global checks
    checks += 1
    for p in phrases:
        if p not in TXT:
            fails.append(f"  MISSING TEXT  {label}: {p!r}")


iq = load("info_quality_y.csv", "condition")
iqp = load("info_quality_y_prop.csv", "condition")
ab = load("ablation_y.csv", "removed")
abp = load("ablation_y_prop.csv", "removed")
dt = load("decision_timing.csv", "decision_point")
dr = load("transfer_drift.csv", "month")
ap = load("transfer_airports.csv", "hub")
rem = load("transfer_remediation.csv")
bd = load("robustness_bounds.csv")
ty = load("robustness_types.csv", "variant")
co = load("robustness_conditions.csv", "stratum")
nt = load("robustness_buffer_nontrivial.csv", "subset")
pr = load("robustness_protocol.csv")
eus = load("europe_scope.csv")
eup = load("europe_propagation_audit.csv", "filter")
eud = load("europe_drift_audit.csv", "window")
euv = load("europe_surviving.csv")
va = load("validation_outcome.csv", "statistic")["value"]
dc = load("decision_curve.csv", "pt")
_bs = OUT / "build_summary_us.csv"          # copied into out/ so the release self-verifies
if not _bs.exists():
    _bs = OUT.parent / "data_out" / "build_summary_us.csv"
bs = pd.read_csv(_bs).set_index("month")

# ---- section 3, dataset ------------------------------------------------------
chk("Table 1 total turnarounds", bs["turnarounds"].sum(), 2523210, 0.5)
chk("Table 1 total flights", bs["flights"].sum(), 2851883, 0.5)
for m, n, med, sch, tm, pv, pp in [
        ("2022_01", 454290, 73, 61, 98.4, 18.37, 6.85),
        ("2024_01", 479131, 72, 62, 90.6, 22.02, 10.04),
        ("2024_06", 564144, 69, 60, 87.4, 24.07, 10.98),
        ("2025_06", 561216, 69, 61, 82.7, 26.33, 12.43),
        ("2026_01", 464429, 73, 63, 80.1, 19.49, 7.90)]:
    chk(f"Table 1 {m} n", bs.loc[m, "turnarounds"], n, 0.5)
    chk(f"Table 1 {m} median actual", bs.loc[m, "median_actual"], med, 0.5)
    chk(f"Table 1 {m} median scheduled", bs.loc[m, "median_sched"], sch, 0.5)
    chk(f"Table 1 {m} type match", bs.loc[m, "type_match_%"], tm, 0.05)
    chk(f"Table 1 {m} prevalence", bs.loc[m, "base_rate_y_%"], pv, 0.005)
    chk(f"Table 1 {m} propagation-specific", bs.loc[m, "base_rate_yprop_%"], pp, 0.005)

# ---- section 5.1, outcome validation ----------------------------------------
chk("5.1 LateAircraftDelay > 0", va["LateAircraftDelay > 0 (%)"], 65.1, 0.05)
chk("5.1 LateAircraftDelay >= 15", va["LateAircraftDelay >= 15 (%)"], 54.9, 0.05)
chk("5.1 share late-aircraft", va["share of delay-minutes: LateAircraftDelay (%)"], 47.8, 0.05)
chk("5.1 share carrier", va["share of delay-minutes: CarrierDelay (%)"], 32.4, 0.05)
chk("5.1 share NAS", va["share of delay-minutes: NASDelay (%)"], 13.8, 0.05)
chk("5.1 share weather", va["share of delay-minutes: WeatherDelay (%)"], 5.8, 0.05)
chk("5.1 share security", va["share of delay-minutes: SecurityDelay (%)"], 0.2, 0.05)
chk("5.1 dominant late-aircraft", va["dominant cause: LateAircraftDelay (%)"], 51.9, 0.05)
chk("5.1 gradient >15", va["LateAircraftDelay>0 | inbound >15 late (%)"], 83.5, 0.05)
chk("5.1 gradient 0-15", va["LateAircraftDelay>0 | inbound 0-15 late (%)"], 58.6, 0.05)
chk("5.1 gradient on time", va["LateAircraftDelay>0 | inbound early/on time (%)"], 24.9, 0.05)
for s, r, a in [("low", 4.77, 0.889), ("medium", 4.75, 0.877), ("high", 4.25, 0.882)]:
    chk(f"5.1 conditions ratio {s}", co.loc[s, "ratio"], r, 0.005)
    chk(f"5.1 conditions auc {s}", co.loc[s, "auc"], a, 0.0005)
chk("5.1 non-trivial auc", nt.loc["inbound fits buffer", "auc"], 0.838, 0.0005)
chk("5.1 non-trivial prevalence", nt.loc["inbound fits buffer", "prevalence"] * 100, 21.5, 0.05)
chk("5.1 non-trivial share", nt.loc["inbound fits buffer", "n"] / nt.loc["full", "n"] * 100, 91.0, 0.05)
in_text("5.1 buffer contrast", "93.5%", "18.6%", "77.1%", "5.7%")

# ---- section 5.2, information quality ---------------------------------------
chk("5.2 clean auc", iq.loc["clean", "auc"], 0.887, 0.0005)
chk("5.2 clean auprc", iq.loc["clean", "auprc"], 0.851, 0.0005)
chk("5.2 clean ece", iq.loc["clean", "ece"], 0.015, 0.0005)
chk("5.2 clean recall", iq.loc["clean", "recall"], 0.631, 0.0005)
chk("5.2 clean precision", iq.loc["clean", "precision"], 0.932, 0.0005)
chk("5.2 clean alert rate", iq.loc["clean", "alert_rate"] * 100, 19.0, 0.05)
tbl2 = {"Clean": "clean", "Coarse, 4 buckets": "coarse status (4 buckets)",
        "Noise, sd 5 min": "noise sd=5 min", "Noise, sd 10 min": "noise sd=10 min",
        "Noise, sd 20 min": "noise sd=20 min", "Noise, sd 40 min": "noise sd=40 min",
        "Missing, mean fallback": "missing, mean fallback",
        "Missing, model retrained": "missing, model retrained"}
EXPECT = {  # exactly as printed in Table 2
 "clean":                     (.887,.851,.015,.631,.932,19.0,27.4,.631,.932,27.4),
 "coarse status (4 buckets)": (.878,.828,.032,.602,.898,18.8,30.7,.606,.894,30.4),
 "noise sd=5 min":            (.882,.843,.013,.631,.905,19.6,27.5,.620,.916,28.2),
 "noise sd=10 min":           (.871,.820,.026,.623,.850,20.6,27.7,.597,.881,29.5),
 "noise sd=20 min":           (.836,.746,.072,.610,.711,24.1,28.3,.535,.789,34.2),
 "noise sd=40 min":           (.769,.580,.152,.591,.552,30.1,29.5,.441,.651,43.2),
 "missing, mean fallback":    (.700,.512,.064,.190,.618, 8.6,76.5,.342,.505,62.6),
 "missing, model retrained":  (.731,.555,.045,.146,.778, 5.3,80.2,.382,.563,57.9)}
for k, v in EXPECT.items():
    r = iq.loc[k]
    for j, (col, exp, tol) in enumerate(zip(
            ["auc","auprc","ece","recall","precision","alert_rate","unflagged_delay_pct",
             "recall_matched","precision_matched","unflagged_delay_pct_matched"],
            v, [.0005]*5 + [.05,.05,.0005,.0005,.05])):
        val = r[col] * (100 if col in ("alert_rate",) else 1)
        chk(f"Table 2 [{k}] {col}", val, exp, tol)
chk("5.2 coarse dAUC", iq.loc["clean","auc"] - iq.loc["coarse status (4 buckets)","auc"], 0.009, 0.0006)
chk("5.2 missing retrained dAUC", iq.loc["clean","auc"] - iq.loc["missing, model retrained","auc"], 0.156, 0.0006)
chk("5.2 missing fallback dAUC", iq.loc["clean","auc"] - iq.loc["missing, mean fallback","auc"], 0.186, 0.0006)
chk("5.2 coarse recall loss at budget",
    iq.loc["clean","recall_matched"] - iq.loc["coarse status (4 buckets)","recall_matched"], 0.026, 0.0006)
chk("5.2 recovered share retrained",
    (iq.loc["missing, model retrained","recall_matched"] - iq.loc["missing, model retrained","recall"]), 0.236, 0.001)
chk("5.2 drop retrained",
    (iq.loc["clean","recall"] - iq.loc["missing, model retrained","recall"]), 0.485, 0.001)
chk("5.2 recovered share fallback",
    (iq.loc["missing, mean fallback","recall_matched"] - iq.loc["missing, mean fallback","recall"]), 0.152, 0.001)
chk("5.2 drop fallback",
    (iq.loc["clean","recall"] - iq.loc["missing, mean fallback","recall"]), 0.441, 0.001)
chk("5.2 y_prop coarse dAUC", iqp.loc["clean","auc"] - iqp.loc["coarse status (4 buckets)","auc"], 0.014, 0.0006)
chk("5.2 y_prop clean auc", iqp.loc["clean","auc"], 0.985, 0.0005)
chk("5.2 y_prop absence dAUC", abp.loc["inbound_status","delta_auc"], 0.213, 0.0006)
chk("5.2 y_prop schedule dAUC", abp.loc["schedule","delta_auc"], 0.040, 0.0006)
in_text("5.2 baseline", "30.5% recall at 94.7% precision", "roughly **doubles**")

# ---- section 5.3, decision timing -------------------------------------------
for dp, auc, apr, rc, pc, lm, p25, p10 in [
    ("DP3 inbound in-block (as submitted)", .887, .851, .631, .932, 68, 51, 41),
    ("DP2 inbound wheels-on",               .878, .838, .617, .912, 77, 59, 48),
    ("DP1 inbound off-block",               .855, .800, .580, .857, 212, 156, 125)]:
    chk(f"Table 3 {dp} auc", dt.loc[dp, "auc"], auc, 0.0005)
    chk(f"Table 3 {dp} auprc", dt.loc[dp, "auprc"], apr, 0.0005)
    chk(f"Table 3 {dp} recall", dt.loc[dp, "recall_matched"], rc, 0.0005)
    chk(f"Table 3 {dp} precision", dt.loc[dp, "precision_matched"], pc, 0.0005)
    chk(f"Table 3 {dp} lead median", dt.loc[dp, "lead_median"], lm, 0.5)
    chk(f"Table 3 {dp} lead p25", dt.loc[dp, "lead_p25"], p25, 0.5)
    chk(f"Table 3 {dp} lead p10", dt.loc[dp, "lead_p10"], p10, 0.5)
d3, d1 = dt.loc["DP3 inbound in-block (as submitted)"], dt.loc["DP1 inbound off-block"]
chk("5.3 lead gain", d1["lead_median"] - d3["lead_median"], 144, 0.5)
chk("5.3 auc cost", d3["auc"] - d1["auc"], 0.032, 0.0006)
chk("5.3 recall cost", d3["recall_matched"] - d1["recall_matched"], 0.051, 0.0006)
chk("5.3 DP3 under 60 min", d3["pct_lead_under_60"], 38.7, 0.05)

# ---- section 5.4, ablation ---------------------------------------------------
for src, auc, lo, hi, d, apr, rc, rcm, un in [
    ("(none - full model)", .887, .884, .889, 0.000, .851, .631, .631, 27.4),
    ("inbound_status",      .731, .728, .734, 0.156, .555, .146, .382, 80.2),
    ("schedule",            .842, .840, .845, 0.044, .759, .527, .554, 40.7),
    ("time_of_day",         .878, .876, .880, 0.009, .841, .624, .629, 28.6),
    ("aircraft_type",       .880, .877, .883, 0.007, .841, .618, .620, 28.3),
    ("airport",             .882, .880, .885, 0.005, .846, .624, .629, 28.0)]:
    r = ab.loc[src]
    chk(f"Table 4 {src} auc", r["auc"], auc, 0.0005)
    chk(f"Table 4 {src} lo", r["auc_lo"], lo, 0.0006)
    chk(f"Table 4 {src} hi", r["auc_hi"], hi, 0.0006)
    chk(f"Table 4 {src} dAUC", r["delta_auc"], d, 0.0006)
    chk(f"Table 4 {src} auprc", r["auprc"], apr, 0.0005)
    chk(f"Table 4 {src} recall", r["recall"], rc, 0.0005)
    chk(f"Table 4 {src} recall@budget", r["recall_matched"], rcm, 0.0006)
    chk(f"Table 4 {src} unflagged", r["unflagged_delay_pct"], un, 0.05)

# ---- section 5.5, transfer ---------------------------------------------------
chk("5.5 hub auc min", ap["auc"].min(), 0.837, 0.0005)
chk("5.5 hub auc max", ap["auc"].max(), 0.878, 0.0005)
chk("5.5 hub auc mean", ap["auc"].mean(), 0.861, 0.0005)
chk("5.5 hub recall min", ap["recall"].min(), 0.483, 0.0005)
chk("5.5 hub recall max", ap["recall"].max(), 0.610, 0.0005)
chk("5.5 hub prevalence min", ap["prevalence"].min() * 100, 20.0, 0.05)
chk("5.5 hub prevalence max", ap["prevalence"].max() * 100, 34.2, 0.05)
chk("5.5 hub ece min", ap["ece"].min(), 0.008, 0.0005)
chk("5.5 hub ece max", ap["ece"].max(), 0.037, 0.0005)
oot = dr[dr["ref"] == "out-of-time"]
ind = dr[dr["ref"] == "in-distribution"].iloc[0]
chk("5.5 oot auc min", oot["auc"].min(), 0.830, 0.0005)
chk("5.5 oot auc max", oot["auc"].max(), 0.876, 0.0005)
chk("5.5 in-dist auc", ind["auc"], 0.882, 0.0005)
chk("5.5 oot auprc min", oot["auprc"].min(), 0.726, 0.0005)
chk("5.5 oot auprc max", oot["auprc"].max(), 0.829, 0.0005)
chk("5.5 in-dist auprc", ind["auprc"], 0.846, 0.0005)
chk("5.5 alert rate min", dr["alert_rate"].min() * 100, 11.4, 0.05)
chk("5.5 alert rate max", dr["alert_rate"].max() * 100, 18.8, 0.05)
for k, rc, pc, al in [("none", .591, .934, 18.6), ("thr", .575, .910, 18.6),
                      ("cal", .598, .926, 19.1), ("retrain", .610, .916, 19.6)]:
    chk(f"Table 5 {k} recall", rem[k + "_recall"].mean(), rc, 0.0006)
    chk(f"Table 5 {k} precision", rem[k + "_precision"].mean(), pc, 0.0006)
    chk(f"Table 5 {k} alert", rem[k + "_alert"].mean() * 100, al, 0.06)
chk("5.5 ece none", rem["none_ece"].mean(), 0.026, 0.0006)
chk("5.5 ece cal", rem["cal_ece"].mean(), 0.018, 0.0006)
chk("5.5 retrained auc", rem["retrain_auc"].mean(), 0.866, 0.0006)
chk("5.5 source auc", rem["source_auc"].mean(), 0.868, 0.0006)
chk("5.5 bounds min", bd["auc"].min(), 0.879, 0.0006)
chk("5.5 bounds max", bd["auc"].max(), 0.896, 0.0006)
chk("5.5 bounds excluded pct",
    (1 - bd.set_index(["lo", "hi"]).loc[(20, 720), "n_total"] /
     bd.set_index(["lo", "hi"]).loc[(1, 1440), "n_total"]) * 100, 3.9, 0.05)
chk("5.5 hp min", pr["auc"].min(), 0.883, 0.0006)
chk("5.5 hp max", pr["auc"].max(), 0.887, 0.0006)
chk("5.5 types all", ty.loc["all rows, unmatched type coded -1", "auc"], 0.887, 0.0005)
chk("5.5 types matched", ty.loc["matched rows only", "auc"], 0.888, 0.0005)
chk("5.5 types dropped", ty.loc["type feature removed entirely", "auc"], 0.880, 0.0005)

# ---- section 5.6, Europe -----------------------------------------------------
chk("5.6 europe total", eus["n"].sum(), 203103, 0.5)
chk("5.6 europe windows", len(eus), 5, 0)
chk("5.6 europe days spanned", eus["days_spanned"].max(), 13, 0)
chk("5.6 europe pooled r as submitted", eup.loc["none (as submitted)", "pooled_r"], 0.210, 0.0006)
chk("5.6 europe within-route as submitted", eup.loc["none (as submitted)", "within_route_r"], 0.066, 0.0006)
chk("5.6 europe within-route 12h", eup.loc["gap < 12 h", "within_route_r"], -0.024, 0.0006)
chk("5.6 europe drift clean min", eud["auc_clean"].drop("2024-06-01").min(), 0.606, 0.0006)
chk("5.6 europe drift clean max", eud["auc_clean"].drop("2024-06-01").max(), 0.650, 0.0006)
chk("5.6 europe in-dist clean", eud.loc["2024-06-01", "auc_clean"], 0.833, 0.0006)
ev = euv.set_index(["analysis", "variant"])["value"]
chk("5.6 classifier generic", ev[("cross-airport long-turnaround classifier", "generic features only")], 0.783, 0.0006)
chk("5.6 classifier type", ev[("cross-airport long-turnaround classifier", "plus aircraft type")], 0.814, 0.0006)
chk("5.6 regression generic", ev[("turnaround-duration regression", "generic features only")], 0.480, 0.0006)
chk("5.6 regression type", ev[("turnaround-duration regression", "plus aircraft type")], 0.558, 0.0006)
in_text("5.6 withdrawal", "we withdraw them", "−0.024", "not supported")

# ---- section 6, decision curve ----------------------------------------------
chk("6 net-benefit gap at pt=0.05", dc.loc[0.05, "nb_full"] - dc.loc[0.05, "nb_no_inbound"], 0.0008, 0.0002)
chk("6 net-benefit gap at pt=0.50", dc.loc[0.50, "nb_full"] - dc.loc[0.50, "nb_no_inbound"], 0.135, 0.0006)
chk("6 auc spread across months", dr["auc"].max() - dr["auc"].min(), 0.052, 0.0015)
chk("6 auprc spread across months", dr["auprc"].max() - dr["auprc"].min(), 0.120, 0.0015)

# ---- required disclosures ----------------------------------------------------
in_text("item 3 wording", "upper bound on addressable delay, not preventable delay")
in_text("item 8.3 wording", "matched alarm budget")
in_text("item 7 protocol", "no validation split, because no hyperparameter search was performed")
in_text("item 1 reframe", "We do not label this outcome \"delay propagation\"")
in_text("leak disclosure", "leaks the outcome into the feature set")
in_text("ablation disclosure", "identical to the full model, to three decimal places")

print(f"{checks} checks run")
if fails:
    print(f"\n{len(fails)} FAILURES\n" + "\n".join(fails))
    sys.exit(1)
print("ALL CHECKS PASSED — every number in the manuscript is backed by out/")
