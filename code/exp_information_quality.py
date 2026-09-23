#!/usr/bin/env python
"""
Core experiment: how the quality of the inbound-status signal governs the decision.

Changes from the originally submitted version, each demanded by a reviewer item:

  item 8 -- the coarse-status categories are now mutually exclusive and exhaustive, and
            include the 15-60 minute band that the submitted definition left undefined.
  item 4 -- buffer_pressure is a deterministic function of the inbound delay, so it is
            degraded consistently with it, and the "missing" condition is additionally
            run as a model RETRAINED without the inbound source, which is the correct
            counterfactual for a signal that is simply not available.
  items 2, 6 -- every condition is evaluated twice: at the fixed 0.5 threshold, and at the
            threshold that issues the SAME NUMBER OF ALERTS as the clean model. The second
            separates a genuine loss of discrimination from a shift of the operating point.
            Without it, most of the published recall collapse is a calibration artefact.
  item 1 -- everything is repeated for the propagation-specific outcome y_prop.

Outputs: out/info_quality_<outcome>.csv, out/info_quality.txt
"""
from __future__ import annotations
import numpy as np, pandas as pd
from sklearn.metrics import recall_score, precision_score
from common import (OUT, TRAIN_MONTH, SEED, THR, load_month, split_train_test, fit,
                    features_without, metrics, boot_ci, threshold_for_alert_rate,
                    leaked_delay_pct, ece)

# mutually exclusive, exhaustive coarse status categories (reviewer item 8)
COARSE = [("early or on time", -np.inf, 0.0),
          ("minor (0-15 min)", 0.0, 15.0),
          ("moderate (15-60 min)", 15.0, 60.0),
          ("major (over 60 min)", 60.0, np.inf)]
NOISE = [5, 10, 20, 40]
L = []


def say(s=""):
    print(s)
    L.append(str(s))


def coarsen(x: pd.Series, medians: dict) -> pd.Series:
    """Replace an exact inbound delay by the training-set median of its coarse bucket."""
    out = pd.Series(np.nan, index=x.index, dtype=float)
    for name, lo, hi in COARSE:
        m = (x > lo) & (x <= hi) if np.isfinite(lo) else (x <= hi)
        out[m] = medians[name]
    return out


def run(outcome: str):
    say("\n" + "=" * 78)
    say(f"OUTCOME: {outcome}")
    say("=" * 78)
    ta = load_month(TRAIN_MONTH)
    tr, te = split_train_test(ta)
    feats = features_without()
    say(f"  train {len(tr):,} ({tr['date'].min().date()} to {tr['date'].max().date()})  |  "
        f"test {len(te):,} ({te['date'].min().date()} to {te['date'].max().date()})")
    say(f"  test prevalence {te[outcome].mean()*100:.2f}%   features: {feats}")

    clf = fit(tr, feats, target=outcome)
    p0 = clf.predict_proba(te[feats])[:, 1]
    d0 = (p0 > THR).astype(int)
    m0 = metrics(te[outcome], p0)
    lo, hi = boot_ci(te[outcome].values, p0)
    say(f"\n  clean model: AUC {m0['auc']:.3f} (95% CI {lo:.3f}-{hi:.3f})  AUPRC {m0['auprc']:.3f}  "
        f"ECE {m0['ece']:.3f}  Brier {m0['brier']:.3f}")
    say(f"               at thr 0.5: recall {m0['recall']:.3f}  precision {m0['precision']:.3f}  "
        f"alert rate {m0['alert_rate']*100:.1f}%")
    budget = m0["alert_rate"]

    # transparent rule baseline: flag when the inbound delay has eaten the scheduled ground time
    rule = (te["inbound_arr_delay"] > te["sched_turnaround_min"]).astype(int).values
    say(f"  rule baseline (inbound delay > scheduled turnaround): fires {rule.mean()*100:.1f}%  "
        f"recall {recall_score(te[outcome], rule):.3f}  "
        f"precision {precision_score(te[outcome], rule, zero_division=0):.3f}")

    # model retrained without the inbound source -- the correct 'signal unavailable' case
    f_no_in = features_without(["inbound_status"])
    clf_no = fit(tr, f_no_in, target=outcome)
    p_no = clf_no.predict_proba(te[f_no_in])[:, 1]

    med = {name: (tr.loc[(tr["inbound_arr_delay"] > lo_) & (tr["inbound_arr_delay"] <= hi_)
                         if np.isfinite(lo_) else (tr["inbound_arr_delay"] <= hi_),
                         "inbound_arr_delay"].median())
           for name, lo_, hi_ in COARSE}
    say("  coarse-bucket representative values (training-set medians): "
        + ", ".join(f"{k} -> {v:.0f}" for k, v in med.items()))

    rng = np.random.default_rng(SEED)
    conds = {}
    conds["clean"] = p0
    for sd in NOISE:
        X = te[feats].copy()
        z = te["inbound_arr_delay"].values + rng.normal(0, sd, len(te))
        X["inbound_arr_delay"] = z
        X["buffer_pressure"] = z - te["sched_turnaround_min"].values
        conds[f"noise sd={sd} min"] = clf.predict_proba(X)[:, 1]
    X = te[feats].copy()
    z = coarsen(te["inbound_arr_delay"], med).values
    X["inbound_arr_delay"] = z
    X["buffer_pressure"] = z - te["sched_turnaround_min"].values
    conds["coarse status (4 buckets)"] = clf.predict_proba(X)[:, 1]
    X = te[feats].copy()
    mu = tr["inbound_arr_delay"].mean()
    X["inbound_arr_delay"] = mu
    X["buffer_pressure"] = mu - te["sched_turnaround_min"].values
    conds["missing, mean fallback"] = clf.predict_proba(X)[:, 1]
    conds["missing, model retrained"] = p_no

    rows = []
    for name, p in conds.items():
        m = metrics(te[outcome], p)
        d = (p > THR).astype(int)
        thr_b = threshold_for_alert_rate(p, budget)
        db = (p > thr_b).astype(int)
        rows.append({
            "condition": name, "auc": m["auc"], "auprc": m["auprc"], "ece": m["ece"],
            "recall": m["recall"], "precision": m["precision"], "alert_rate": m["alert_rate"],
            "flip_vs_clean": float((d != d0).mean()),
            "unflagged_delay_pct": leaked_delay_pct(te, d, outcome),
            "thr_matched": thr_b,
            "recall_matched": recall_score(te[outcome], db, zero_division=0),
            "precision_matched": precision_score(te[outcome], db, zero_division=0),
            "unflagged_delay_pct_matched": leaked_delay_pct(te, db, outcome),
        })
    res = pd.DataFrame(rows)
    res.to_csv(OUT / f"info_quality_{outcome}.csv", index=False)

    say("\n  AT THE FIXED 0.5 THRESHOLD (as published)")
    say("  " + res[["condition", "auc", "auprc", "recall", "precision", "alert_rate",
                    "flip_vs_clean", "unflagged_delay_pct"]].round(3).to_string(index=False)
        .replace("\n", "\n  "))
    say(f"\n  AT A MATCHED ALARM BUDGET (every condition issues {budget*100:.1f}% alerts)")
    say("  " + res[["condition", "thr_matched", "recall_matched", "precision_matched",
                    "unflagged_delay_pct_matched"]].round(3).to_string(index=False)
        .replace("\n", "\n  "))

    r = res.set_index("condition")
    say("\n  READING:")
    say(f"    noise sd=40 : AUC {r.loc['noise sd=40 min','auc']:.3f} vs clean {m0['auc']:.3f}; "
        f"recall holds ({r.loc['noise sd=40 min','recall']:.3f}) but precision falls to "
        f"{r.loc['noise sd=40 min','precision']:.3f}")
    say(f"    coarse      : AUC {r.loc['coarse status (4 buckets)','auc']:.3f}, recall "
        f"{r.loc['coarse status (4 buckets)','recall']:.3f}, "
        f"{r.loc['coarse status (4 buckets)','flip_vs_clean']*100:.1f}% of decisions change")
    mm = r.loc["missing, mean fallback"]
    mr = r.loc["missing, model retrained"]
    say(f"    missing     : AUC {mm['auc']:.3f} (fallback) / {mr['auc']:.3f} (retrained) "
        f"vs clean {m0['auc']:.3f}")
    say(f"                  recall at 0.5:      {mm['recall']:.3f} / {mr['recall']:.3f}  "
        f"vs clean {m0['recall']:.3f}")
    say(f"                  recall at matched:  {mm['recall_matched']:.3f} / "
        f"{mr['recall_matched']:.3f}  vs clean {m0['recall']:.3f}")
    say(f"    => of the recall drop at a fixed threshold, "
        f"{(m0['recall']-mm['recall']):.3f} total, "
        f"{(mm['recall_matched']-mm['recall']):.3f} is recovered simply by re-setting the")
    say("       threshold. The remainder is a genuine loss of discrimination.")
    return res


def main():
    for outcome in ["y", "y_prop"]:
        run(outcome)
    (OUT / "info_quality.txt").write_text("\n".join(L), encoding="utf-8")
    say(f"\nwrote out/info_quality_y.csv, out/info_quality_y_prop.csv, out/info_quality.txt")


if __name__ == "__main__":
    main()
