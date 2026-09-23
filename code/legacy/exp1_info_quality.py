#!/usr/bin/env python
"""
Experiment 1 (BTS US arm): information quality -> decision value for turnaround delay propagation.

Decision modelled: flag a next-leg departure as a propagation risk (predicted >15 min late) so an
operator can act (hold/buffer/swap). We train ONE fixed predictor on clean features, then DEGRADE the
most operationally-uncertain input -- the inbound arrival delay -- and measure how the *decision*
degrades (not just accuracy). The headline is the information-quality threshold beyond which the
decision materially fails. See DESIGN_ADSB_InfoQuality_v0.1.md (sections 4 and 10).

No leakage: features are all known before the next departure decision; the actual turnaround duration
(an outcome) is excluded.
"""
import argparse
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score, recall_score, precision_score


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="data_out/turnarounds_bts_2024_06.csv")
    ap.add_argument("--thr", type=float, default=0.5, help="decision threshold on P(delay>15)")
    args = ap.parse_args()

    ta = pd.read_csv(args.csv)
    ta["dep_dt"] = pd.to_datetime(ta["dep_dt"], errors="coerce")
    for c in ["inbound_arr_delay", "outbound_dep_delay", "sched_turnaround_min"]:
        ta[c] = pd.to_numeric(ta[c], errors="coerce")
    ta = ta.dropna(subset=["inbound_arr_delay", "outbound_dep_delay", "sched_turnaround_min", "dep_dt"])
    ta["hour"] = ta["dep_dt"].dt.hour
    ta["dow"] = ta["dep_dt"].dt.dayofweek
    ta["buffer_pressure"] = ta["inbound_arr_delay"] - ta["sched_turnaround_min"]
    ta["airport_code"] = ta["airport"].astype("category").cat.codes
    ta["y"] = (ta["outbound_dep_delay"] > 15).astype(int)
    ta = ta.sort_values("dep_dt").reset_index(drop=True)

    feats = ["inbound_arr_delay", "sched_turnaround_min", "buffer_pressure", "hour", "dow", "airport_code"]
    cut = int(len(ta) * 0.75)               # temporal split: train early, test late (honest generalisation)
    tr, te = ta.iloc[:cut], ta.iloc[cut:]
    print(f"train {len(tr):,} | test {len(te):,} | base rate next-leg >15 late (test) {te['y'].mean()*100:.1f}%")

    clf = HistGradientBoostingClassifier(max_iter=250, learning_rate=0.1, max_depth=6, random_state=0)
    clf.fit(tr[feats], tr["y"])

    p_clean = clf.predict_proba(te[feats])[:, 1]
    d_clean = (p_clean > args.thr).astype(int)
    auc_clean = roc_auc_score(te["y"], p_clean)
    print(f"\nclean-info predictor : AUC {auc_clean:.3f} | recall {recall_score(te['y'], d_clean):.3f} | "
          f"precision {precision_score(te['y'], d_clean):.3f}")
    d_rule = (te["inbound_arr_delay"] > te["sched_turnaround_min"]).astype(int)
    print(f"naive rule baseline  : recall {recall_score(te['y'], d_rule):.3f} | "
          f"precision {precision_score(te['y'], d_rule, zero_division=0):.3f}")

    true_prop = te["y"] == 1
    total_prop_delay = te.loc[true_prop, "outbound_dep_delay"].sum()
    rng = np.random.default_rng(0)
    tr_mean = tr["inbound_arr_delay"].mean()

    rows = []
    for sigma in [0, 5, 10, 20, 40]:
        Xd = te[feats].copy()
        noisy = te["inbound_arr_delay"].values + rng.normal(0, sigma, len(te))
        Xd["inbound_arr_delay"] = noisy
        Xd["buffer_pressure"] = noisy - te["sched_turnaround_min"].values
        p = clf.predict_proba(Xd)[:, 1]; d = (p > args.thr).astype(int)
        leaked = te.loc[true_prop & (d == 0), "outbound_dep_delay"].sum()
        rows.append({"condition": f"noise sd={sigma}m", "auc": roc_auc_score(te["y"], p),
                     "recall": recall_score(te["y"], d), "precision": precision_score(te["y"], d, zero_division=0),
                     "flip_vs_clean": float((d != d_clean).mean()), "leaked_delay_pct": leaked / total_prop_delay * 100})
    # missing inbound-delay info -> fall back to the training mean
    Xm = te[feats].copy(); Xm["inbound_arr_delay"] = tr_mean; Xm["buffer_pressure"] = tr_mean - te["sched_turnaround_min"].values
    p = clf.predict_proba(Xm)[:, 1]; d = (p > args.thr).astype(int)
    leaked = te.loc[true_prop & (d == 0), "outbound_dep_delay"].sum()
    rows.append({"condition": "missing (mean)", "auc": roc_auc_score(te["y"], p),
                 "recall": recall_score(te["y"], d), "precision": precision_score(te["y"], d, zero_division=0),
                 "flip_vs_clean": float((d != d_clean).mean()), "leaked_delay_pct": leaked / total_prop_delay * 100})

    res = pd.DataFrame(rows)
    Path("data_out").mkdir(exist_ok=True)
    res.to_csv("data_out/exp1_info_quality.csv", index=False)
    print("\n=== information quality -> decision value ===")
    print(res.round(3).to_string(index=False))

    # figure: how the decision degrades as inbound-delay information degrades
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    noise = res[res["condition"].str.startswith("noise")]
    x = [0, 5, 10, 20, 40]
    fig, ax1 = plt.subplots(figsize=(8, 5))
    ax1.plot(x, noise["recall"], "o-", color="#2b5d8a", label="recall (propagations caught)")
    ax1.set_xlabel("Noise on inbound-delay information (min, sd)")
    ax1.set_ylabel("Recall", color="#2b5d8a"); ax1.set_ylim(0, 1)
    ax2 = ax1.twinx()
    ax2.plot(x, noise["leaked_delay_pct"], "s--", color="#a14b46", label="undetected delay (%)")
    ax2.set_ylabel("Undetected propagated delay (%)", color="#a14b46")
    ax1.set_title("Experiment 1: turnaround propagation decision vs information quality")
    fig.tight_layout(); fig.savefig("data_out/exp1_info_quality.png", dpi=150, facecolor="white")
    print("\nwrote data_out/exp1_info_quality.csv and exp1_info_quality.png")


if __name__ == "__main__":
    main()
