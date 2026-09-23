#!/usr/bin/env python
"""
Reviewer item 6: transfer across airports and time, and whether recalibration suffices.

The submitted paper reported stable AUC across airports and months and concluded that a
transferred model "calls for local recalibration rather than retraining". Stable AUC only
shows that the ranking survives; it says nothing about prevalence, about precision-recall
performance at a usable operating point, or about whether the probabilities themselves are
still meaningful. This script reports all of those, with sample sizes and bootstrap
intervals, and then tests the recalibration claim directly by comparing three remedies on
held-out target data:

  none          -- transfer the source model and keep the source threshold
  threshold     -- transfer the model, re-set only the decision threshold on target data
  recalibrate   -- transfer the model, fit an isotonic calibrator on target data, keep 0.5
  retrain       -- fit a fresh model on target data

Stages (run separately to stay inside a short wall-clock budget):
  --stage drift        out/transfer_drift.csv
  --stage airports     out/transfer_airports.csv
  --stage remediation  out/transfer_remediation.csv
"""
from __future__ import annotations
import argparse
import numpy as np, pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import recall_score, precision_score
from common import (OUT, MONTHS, TRAIN_MONTH, THR, SEED, load_month, split_train_test,
                    encode, fit, features_without, metrics, boot_ci,
                    threshold_for_alert_rate)

# generic feature set for transfer tests: airport identity removed, so the question is
# whether the mechanism transfers rather than whether the model memorised airports
GENERIC = [f for f in features_without() if f != "airport_code"]
HUBS = 8


def say(lines, s=""):
    print(s)
    lines.append(str(s))


def stage_drift():
    L = []
    say(L, "TEMPORAL TRANSFER: train on June 2024 (1-23), test out of time")
    ta = load_month(TRAIN_MONTH)
    tr, te_in = split_train_test(ta)
    clf = fit(tr, GENERIC)
    rows = []
    for m in MONTHS:
        if m == TRAIN_MONTH:
            te = te_in
        else:
            t = load_month(m)
            _, te = encode(tr, t)
        p = clf.predict_proba(te[GENERIC])[:, 1]
        mm = metrics(te["y"], p)
        lo, hi = boot_ci(te["y"].values, p, n_boot=300)
        rows.append({"month": m, "ref": "in-distribution" if m == TRAIN_MONTH else "out-of-time",
                     **mm, "auc_lo": lo, "auc_hi": hi})
        say(L, f"  {m}: n={mm['n']:>7,}  prevalence {mm['prevalence']*100:5.2f}%  "
               f"AUC {mm['auc']:.3f} ({lo:.3f}-{hi:.3f})  AUPRC {mm['auprc']:.3f}  "
               f"recall {mm['recall']:.3f}  precision {mm['precision']:.3f}  "
               f"alert {mm['alert_rate']*100:5.2f}%  ECE {mm['ece']:.3f}")
        if m != TRAIN_MONTH:
            del te
    res = pd.DataFrame(rows)
    res.to_csv(OUT / "transfer_drift.csv", index=False)
    o = res[res["ref"] == "out-of-time"]
    i = res[res["ref"] == "in-distribution"].iloc[0]
    say(L, f"\n  AUC: in-distribution {i['auc']:.3f}; out-of-time {o['auc'].min():.3f}-{o['auc'].max():.3f}")
    say(L, f"  AUPRC: in-distribution {i['auprc']:.3f}; out-of-time {o['auprc'].min():.3f}-{o['auprc'].max():.3f}")
    say(L, f"  prevalence: in-distribution {i['prevalence']*100:.2f}%; out-of-time "
           f"{o['prevalence'].min()*100:.2f}-{o['prevalence'].max()*100:.2f}%")
    say(L, f"  ECE: in-distribution {i['ece']:.3f}; out-of-time {o['ece'].min():.3f}-{o['ece'].max():.3f}")
    say(L, "  Discrimination is stable; calibration and the realised alert rate are not, which is")
    say(L, "  why the remediation comparison below is needed rather than assumed.")
    (OUT / "transfer_drift.txt").write_text("\n".join(L), encoding="utf-8")


def stage_airports():
    L = []
    say(L, "CROSS-AIRPORT TRANSFER (leave-one-hub-out, airport identity excluded)")
    ta = load_month(TRAIN_MONTH)
    vol = ta["airport"].value_counts()
    hubs = vol.head(HUBS).index.tolist()
    say(L, f"  busiest {HUBS} hubs: {hubs}")
    rows = []
    for h in hubs:
        tr_raw, te_raw = ta[ta["airport"] != h], ta[ta["airport"] == h]
        tr, te = encode(tr_raw, te_raw)
        c = fit(tr, GENERIC)
        p = c.predict_proba(te[GENERIC])[:, 1]
        mm = metrics(te["y"], p)
        lo, hi = boot_ci(te["y"].values, p, n_boot=300)
        rows.append({"hub": h, **mm, "auc_lo": lo, "auc_hi": hi})
        say(L, f"  {h}: n={mm['n']:>6,}  prevalence {mm['prevalence']*100:5.2f}%  "
               f"AUC {mm['auc']:.3f} ({lo:.3f}-{hi:.3f})  AUPRC {mm['auprc']:.3f}  "
               f"recall {mm['recall']:.3f}  precision {mm['precision']:.3f}  ECE {mm['ece']:.3f}")
    res = pd.DataFrame(rows)
    res.to_csv(OUT / "transfer_airports.csv", index=False)
    say(L, f"\n  AUC {res['auc'].min():.3f}-{res['auc'].max():.3f} (mean {res['auc'].mean():.3f})")
    say(L, f"  recall at a fixed 0.5 threshold {res['recall'].min():.3f}-{res['recall'].max():.3f}"
           f"  <- the operating point, not the ranking, is what moves")
    say(L, f"  prevalence {res['prevalence'].min()*100:.2f}-{res['prevalence'].max()*100:.2f}%")
    say(L, f"  ECE {res['ece'].min():.3f}-{res['ece'].max():.3f}")
    (OUT / "transfer_airports.txt").write_text("\n".join(L), encoding="utf-8")


def stage_remediation():
    L = []
    say(L, "DOES RECALIBRATION SUFFICE? three remedies on held-out target data")
    say(L, "  Source model: all other airports. Target airport data is split in time; the first")
    say(L, "  70% is the adaptation set (threshold / calibrator / retraining), the last 30% is")
    say(L, "  held out for evaluation. Every remedy is therefore judged on data none of them saw.")
    ta = load_month(TRAIN_MONTH)
    vol = ta["airport"].value_counts()
    hubs = vol.head(HUBS).index.tolist()
    rows = []
    for h in hubs:
        src_raw, tgt_raw = ta[ta["airport"] != h], ta[ta["airport"] == h].sort_values("date")
        cut = int(len(tgt_raw) * 0.70)
        adapt_raw, ev_raw = tgt_raw.iloc[:cut], tgt_raw.iloc[cut:]
        src, adapt = encode(src_raw, adapt_raw)
        _, ev = encode(src_raw, ev_raw)
        c = fit(src, GENERIC)
        p_ad = c.predict_proba(adapt[GENERIC])[:, 1]
        p_ev = c.predict_proba(ev[GENERIC])[:, 1]
        y_ev = ev["y"].values
        src_budget = float((c.predict_proba(src[GENERIC])[:, 1] > THR).mean())

        out = {"hub": h, "n_eval": len(ev), "prevalence_eval": float(y_ev.mean())}
        # (1) no adaptation
        d = (p_ev > THR).astype(int)
        out.update({"none_recall": recall_score(y_ev, d, zero_division=0),
                    "none_precision": precision_score(y_ev, d, zero_division=0),
                    "none_alert": d.mean(), "none_ece": metrics(y_ev, p_ev)["ece"]})
        # (2) threshold only, matched to the source alert rate on the adaptation set
        t_new = threshold_for_alert_rate(p_ad, src_budget)
        d = (p_ev > t_new).astype(int)
        out.update({"thr_recall": recall_score(y_ev, d, zero_division=0),
                    "thr_precision": precision_score(y_ev, d, zero_division=0),
                    "thr_alert": d.mean(), "thr_value": t_new})
        # (3) isotonic recalibration on the adaptation set, threshold kept at 0.5
        iso = IsotonicRegression(out_of_bounds="clip").fit(p_ad, adapt["y"].values)
        p_cal = iso.predict(p_ev)
        d = (p_cal > THR).astype(int)
        out.update({"cal_recall": recall_score(y_ev, d, zero_division=0),
                    "cal_precision": precision_score(y_ev, d, zero_division=0),
                    "cal_alert": d.mean(), "cal_ece": metrics(y_ev, p_cal)["ece"],
                    "cal_auc": metrics(y_ev, p_cal)["auc"]})
        # (4) retrain on target adaptation data
        c2 = fit(adapt, GENERIC)
        p_re = c2.predict_proba(ev[GENERIC])[:, 1]
        d = (p_re > THR).astype(int)
        m_re = metrics(y_ev, p_re)
        out.update({"retrain_recall": recall_score(y_ev, d, zero_division=0),
                    "retrain_precision": precision_score(y_ev, d, zero_division=0),
                    "retrain_alert": d.mean(), "retrain_ece": m_re["ece"],
                    "retrain_auc": m_re["auc"], "source_auc": metrics(y_ev, p_ev)["auc"]})
        rows.append(out)
        say(L, f"  {h} (n_eval={len(ev):,}, prevalence {y_ev.mean()*100:.1f}%)")
        say(L, f"    none        recall {out['none_recall']:.3f} precision {out['none_precision']:.3f} "
               f"alert {out['none_alert']*100:5.1f}%  ECE {out['none_ece']:.3f}")
        say(L, f"    threshold   recall {out['thr_recall']:.3f} precision {out['thr_precision']:.3f} "
               f"alert {out['thr_alert']*100:5.1f}%")
        say(L, f"    recalibrate recall {out['cal_recall']:.3f} precision {out['cal_precision']:.3f} "
               f"alert {out['cal_alert']*100:5.1f}%  ECE {out['cal_ece']:.3f}")
        say(L, f"    retrain     recall {out['retrain_recall']:.3f} precision {out['retrain_precision']:.3f} "
               f"alert {out['retrain_alert']*100:5.1f}%  ECE {out['retrain_ece']:.3f}  "
               f"AUC {out['retrain_auc']:.3f} vs source {out['source_auc']:.3f}")
    res = pd.DataFrame(rows)
    res.to_csv(OUT / "transfer_remediation.csv", index=False)
    say(L, "\n  MEANS ACROSS HUBS")
    for k in ["none", "thr", "cal", "retrain"]:
        say(L, f"    {k:<9} recall {res[k+'_recall'].mean():.3f}  "
               f"precision {res[k+'_precision'].mean():.3f}  alert {res[k+'_alert'].mean()*100:5.1f}%")
    say(L, f"    AUC: source-transferred {res['source_auc'].mean():.3f} vs "
           f"locally retrained {res['retrain_auc'].mean():.3f} "
           f"(difference {res['retrain_auc'].mean()-res['source_auc'].mean():+.3f})")
    say(L, f"    ECE: uncorrected {res['none_ece'].mean():.3f}, recalibrated "
           f"{res['cal_ece'].mean():.3f}, retrained {res['retrain_ece'].mean():.3f}")
    (OUT / "transfer_remediation.txt").write_text("\n".join(L), encoding="utf-8")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=["drift", "airports", "remediation"])
    a = ap.parse_args()
    {"drift": stage_drift, "airports": stage_airports, "remediation": stage_remediation}[a.stage]()
