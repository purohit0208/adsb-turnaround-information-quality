#!/usr/bin/env python
"""
Reviewer item 4: value of each information SOURCE, ablated correctly.

The originally submitted Table 2 removed individual features. That is not a valid
information-source ablation, because buffer_pressure is exactly
inbound_arr_delay - sched_turnaround_min: dropping either parent while keeping the
derivative leaves the removed signal reconstructible from the two survivors. Here each
information source is removed together with every feature derived from it, and the model
is REFITTED for each ablation, which is the correct counterfactual for an operator who
does not have that source at all.

The submitted paper also reported two different "full model" rows (AUC 0.896 / recall
0.625 in Table 1 and AUC 0.900 / recall 0.636 in Table 2) without reconciling them. Here
Table 1 and Table 2 are produced by the same script, from the same split and the same
fitted model, so a single full-model row serves both.

Outputs: out/ablation_<outcome>.csv, out/ablation.txt
"""
from __future__ import annotations
import numpy as np, pandas as pd
from sklearn.metrics import recall_score, precision_score
from common import (OUT, TRAIN_MONTH, THR, SOURCES, load_month, split_train_test, fit,
                    features_without, metrics, boot_ci, threshold_for_alert_rate,
                    leaked_delay_pct)

L = []


def say(s=""):
    print(s)
    L.append(str(s))


def run(outcome: str):
    say("\n" + "=" * 78)
    say(f"INFORMATION-SOURCE ABLATION -- outcome {outcome}")
    say("=" * 78)
    ta = load_month(TRAIN_MONTH)
    tr, te = split_train_test(ta)
    full = features_without()
    clf = fit(tr, full, target=outcome)
    p0 = clf.predict_proba(te[full])[:, 1]
    m0 = metrics(te[outcome], p0)
    budget = m0["alert_rate"]
    lo0, hi0 = boot_ci(te[outcome].values, p0)
    say(f"  full model: features {full}")
    say(f"  AUC {m0['auc']:.3f} ({lo0:.3f}-{hi0:.3f})  AUPRC {m0['auprc']:.3f}  "
        f"recall {m0['recall']:.3f}  precision {m0['precision']:.3f}  "
        f"alert rate {budget*100:.1f}%  unflagged delay "
        f"{leaked_delay_pct(te, (p0 > THR).astype(int), outcome):.1f}%")

    rows = [{"removed": "(none - full model)", "features_left": len(full), **m0,
             "delta_auc": 0.0, "auc_lo": lo0, "auc_hi": hi0,
             "unflagged_delay_pct": leaked_delay_pct(te, (p0 > THR).astype(int), outcome),
             "recall_matched": m0["recall"], "precision_matched": m0["precision"]}]
    for src in SOURCES:
        f = features_without([src])
        if not f:
            continue
        c = fit(tr, f, target=outcome)
        p = c.predict_proba(te[f])[:, 1]
        m = metrics(te[outcome], p)
        lo, hi = boot_ci(te[outcome].values, p, n_boot=400)
        thr_b = threshold_for_alert_rate(p, budget)
        db = (p > thr_b).astype(int)
        dropped = [x for x in full if x not in f]
        rows.append({"removed": src, "features_left": len(f), **m,
                     "delta_auc": m0["auc"] - m["auc"], "auc_lo": lo, "auc_hi": hi,
                     "unflagged_delay_pct": leaked_delay_pct(te, (p > THR).astype(int), outcome),
                     "recall_matched": recall_score(te[outcome], db, zero_division=0),
                     "precision_matched": precision_score(te[outcome], db, zero_division=0)})
        say(f"    removed {src:<16} (also drops {dropped})")
    res = pd.DataFrame(rows).sort_values("delta_auc", ascending=False)
    res.to_csv(OUT / f"ablation_{outcome}.csv", index=False)
    say("\n  " + res[["removed", "auc", "delta_auc", "auprc", "recall", "precision",
                      "recall_matched", "unflagged_delay_pct"]].round(3)
        .to_string(index=False).replace("\n", "\n  "))

    say("\n  For comparison, the INVALID single-feature ablation used in the submitted paper")
    say("  (drop inbound_arr_delay but keep buffer_pressure, which reconstructs it):")
    f_bad = [x for x in full if x != "inbound_arr_delay"]
    c = fit(tr, f_bad, target=outcome)
    a_bad = metrics(te[outcome], c.predict_proba(te[f_bad])[:, 1])
    a_good = res.set_index("removed").loc["inbound_status"]
    say(f"    single-feature drop : AUC {a_bad['auc']:.3f}   recall {a_bad['recall']:.3f}")
    say(f"    correct source drop : AUC {a_good['auc']:.3f}   recall {a_good['recall']:.3f}")
    say(f"    the invalid version understates the loss by "
        f"{a_bad['auc'] - a_good['auc']:.3f} AUC.")
    return res


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--outcome", default="y", choices=["y", "y_prop"])
    a = ap.parse_args()
    run(a.outcome)
    f = OUT / f"ablation_{a.outcome}.txt"
    f.write_text("\n".join(L), encoding="utf-8")
    print(f"\nwrote out/ablation_{a.outcome}.csv and .txt")


if __name__ == "__main__":
    main()
