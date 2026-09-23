#!/usr/bin/env python
"""
Experiment 2b (BTS US arm): temporal / seasonal drift.
Train the propagation model on June 2024, then test it on months spanning 2022-2026
(out-of-time), to see whether the mechanism holds across seasons and years.
Generic features only (transferable). See DESIGN section 4 (Experiment 2).
"""
import pandas as pd, numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score, recall_score, precision_score

feats = ["inbound_arr_delay", "sched_turnaround_min", "buffer_pressure", "hour", "dow"]


def load(tag):
    ta = pd.read_csv(f"data_out/turnarounds_bts_{tag}.csv")
    ta["dep_dt"] = pd.to_datetime(ta["dep_dt"], errors="coerce")
    for c in ["inbound_arr_delay", "outbound_dep_delay", "sched_turnaround_min"]:
        ta[c] = pd.to_numeric(ta[c], errors="coerce")
    ta = ta.dropna(subset=["inbound_arr_delay", "outbound_dep_delay", "sched_turnaround_min", "dep_dt"])
    ta["hour"] = ta["dep_dt"].dt.hour; ta["dow"] = ta["dep_dt"].dt.dayofweek
    ta["buffer_pressure"] = ta["inbound_arr_delay"] - ta["sched_turnaround_min"]
    ta["y"] = (ta["outbound_dep_delay"] > 15).astype(int)
    return ta.sort_values("dep_dt").reset_index(drop=True)


months = ["2022_01", "2024_01", "2024_06", "2025_06", "2026_01"]
data = {m: load(m) for m in months}

base = data["2024_06"]; cut = int(len(base) * 0.75)
clf = HistGradientBoostingClassifier(max_iter=200, learning_rate=0.1, max_depth=6, random_state=0)
clf.fit(base.iloc[:cut][feats], base.iloc[:cut]["y"])

rows = []
for m in months:
    te = data[m] if m != "2024_06" else base.iloc[cut:]   # held-out slice for the train month
    p = clf.predict_proba(te[feats])[:, 1]; d = (p > 0.5).astype(int)
    rows.append({"test_month": m, "n": len(te), "base_rate_%": te["y"].mean() * 100,
                 "auc": roc_auc_score(te["y"], p), "recall": recall_score(te["y"], d),
                 "precision": precision_score(te["y"], d, zero_division=0),
                 "ref": "in-dist (held-out)" if m == "2024_06" else "out-of-time"})
res = pd.DataFrame(rows)
res.to_csv("data_out/exp2b_temporal_drift.csv", index=False)
print("trained on June 2024 (first 75%); tested across months:\n")
print(res.round(3).to_string(index=False))
indist = res.loc[res["test_month"] == "2024_06", "auc"].iloc[0]
oot = res[res["test_month"] != "2024_06"]["auc"]
print(f"\nin-distribution AUC {indist:.3f} ; out-of-time AUC range {oot.min():.3f}-{oot.max():.3f} "
      f"(max drop {indist-oot.min():+.3f})")

import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
order = ["2022_01", "2024_01", "2024_06", "2025_06", "2026_01"]
r = res.set_index("test_month").loc[order]
fig, ax = plt.subplots(figsize=(9, 5))
colors = ["#2b5d8a" if t != "2024_06" else "#3a7d44" for t in order]
ax.bar(order, r["auc"], color=colors, alpha=.85)
ax.axhline(indist, color="#3a7d44", ls="--", label=f"train-month held-out AUC {indist:.3f}")
ax.set_ylim(0.5, 1.0); ax.set_ylabel("AUC"); ax.set_xlabel("test month (model trained on June 2024)")
ax.set_title("Experiment 2b: temporal/seasonal drift (train June 2024 -> test 2022-2026)")
ax.legend()
for i, t in enumerate(order):
    ax.text(i, r["auc"][t] + 0.01, f"{r['auc'][t]:.3f}", ha="center", fontsize=9)
plt.tight_layout(); plt.savefig("data_out/exp2b_temporal_drift.png", dpi=150, facecolor="white")
print("wrote data_out/exp2b_temporal_drift.png + .csv")
