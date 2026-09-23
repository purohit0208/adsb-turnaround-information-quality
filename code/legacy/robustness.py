#!/usr/bin/env python
"""
Robustness/hardening of the Experiment-1 findings (BTS June 2024):
  - bootstrap 95% CI on the clean-info AUC
  - decision-threshold sweep (precision/recall trade-off), incl. the missing-info case at each threshold
  - calibration of the predictor (reliability curve + ECE)
Makes the headline claims reviewer-proof rather than point estimates.
"""
import numpy as np, pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score, recall_score, precision_score

ta = pd.read_csv("data_out/turnarounds_bts_2024_06.csv")
ta["dep_dt"] = pd.to_datetime(ta["dep_dt"], errors="coerce")
for c in ["inbound_arr_delay", "outbound_dep_delay", "sched_turnaround_min"]:
    ta[c] = pd.to_numeric(ta[c], errors="coerce")
ta = ta.dropna(subset=["inbound_arr_delay", "outbound_dep_delay", "sched_turnaround_min", "dep_dt"])
ta["hour"] = ta["dep_dt"].dt.hour; ta["dow"] = ta["dep_dt"].dt.dayofweek
ta["buffer_pressure"] = ta["inbound_arr_delay"] - ta["sched_turnaround_min"]
ta["airport_code"] = ta["airport"].astype("category").cat.codes
ta["y"] = (ta["outbound_dep_delay"] > 15).astype(int)
ta = ta.sort_values("dep_dt").reset_index(drop=True)
feats = ["inbound_arr_delay", "sched_turnaround_min", "buffer_pressure", "hour", "dow", "airport_code"]
cut = int(len(ta) * 0.75); tr, te = ta.iloc[:cut], ta.iloc[cut:]

clf = HistGradientBoostingClassifier(max_iter=250, learning_rate=0.1, max_depth=6, random_state=0)
clf.fit(tr[feats], tr["y"])
p = clf.predict_proba(te[feats])[:, 1]; y = te["y"].values
auc = roc_auc_score(y, p)

# bootstrap CI on AUC
rng = np.random.default_rng(0); n = len(y); idx = np.arange(n)
aucs = [roc_auc_score(y[s], p[s]) for s in (rng.choice(idx, n, replace=True) for _ in range(300))]
lo, hi = np.percentile(aucs, [2.5, 97.5])
print(f"clean-info AUC {auc:.3f}  (95% bootstrap CI {lo:.3f}-{hi:.3f}, n_test={n:,})")

# missing-info predictions (inbound delay unknown -> training mean)
trm = tr["inbound_arr_delay"].mean()
Xm = te[feats].copy(); Xm["inbound_arr_delay"] = trm; Xm["buffer_pressure"] = trm - te["sched_turnaround_min"].values
pm = clf.predict_proba(Xm)[:, 1]
true_prop = y == 1; tot = te.loc[true_prop, "outbound_dep_delay"].sum()

print("\nthreshold sweep (clean vs missing inbound-delay info):")
print(f"{'thr':>4} {'recall':>7} {'prec':>6} | {'miss_recall':>11} {'miss_leak%':>10}")
rows = []
for thr in [0.2, 0.3, 0.4, 0.5, 0.6, 0.7]:
    d = (p > thr).astype(int); dm = (pm > thr).astype(int)
    leak = te.loc[true_prop & (dm == 0), "outbound_dep_delay"].sum() / tot * 100
    print(f"{thr:>4} {recall_score(y,d):>7.3f} {precision_score(y,d,zero_division=0):>6.3f} | "
          f"{recall_score(y,dm):>11.3f} {leak:>10.1f}")
    rows.append({"thr": thr, "recall": recall_score(y, d), "precision": precision_score(y, d, zero_division=0),
                 "missing_recall": recall_score(y, dm), "missing_leak_pct": leak})
pd.DataFrame(rows).to_csv("data_out/robustness_threshold_sweep.csv", index=False)

# calibration / ECE
bins = np.linspace(0, 1, 11); bid = np.clip(np.digitize(p, bins) - 1, 0, 9)
ece = 0.0; xs = []; ys = []
for b in range(10):
    m = bid == b
    if m.sum() > 0:
        conf, acc = p[m].mean(), y[m].mean()
        ece += m.sum() / len(p) * abs(acc - conf); xs.append(conf); ys.append(acc)
print(f"\ncalibration ECE = {ece:.3f} (lower is better; <0.05 is well-calibrated)")

import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
fig, ax = plt.subplots(1, 2, figsize=(13, 5))
ax[0].plot([0, 1], [0, 1], "k:", label="perfect")
ax[0].plot(xs, ys, "o-", color="#2b5d8a", label=f"model (ECE {ece:.3f})")
ax[0].set_xlabel("predicted P(next-leg >15 late)"); ax[0].set_ylabel("observed frequency")
ax[0].set_title("Calibration (reliability curve)"); ax[0].legend()
sw = pd.DataFrame(rows)
ax[1].plot(sw["thr"], sw["recall"], "o-", color="#2b5d8a", label="recall (clean)")
ax[1].plot(sw["thr"], sw["precision"], "o-", color="#a14b46", label="precision (clean)")
ax[1].plot(sw["thr"], sw["missing_recall"], "s--", color="#888", label="recall (inbound MISSING)")
ax[1].set_xlabel("decision threshold"); ax[1].set_ylabel("rate"); ax[1].set_ylim(0, 1)
ax[1].set_title("Threshold sweep: missing-info recall stays low everywhere"); ax[1].legend()
plt.tight_layout(); plt.savefig("data_out/robustness.png", dpi=150, facecolor="white")
print("wrote data_out/robustness.png + robustness_threshold_sweep.csv")
