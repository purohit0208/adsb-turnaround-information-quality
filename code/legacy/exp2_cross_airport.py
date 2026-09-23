#!/usr/bin/env python
"""
Experiment 2 (BTS US arm): cross-airport generalisation of the turnaround propagation model.
Tests whether a next-leg-delay propagation model transfers to airports it never saw in training.
Uses GENERIC features only (no airport identity), so the question is whether the underlying mechanism
generalises across airports. Includes a leave-one-hub-out sweep. See DESIGN section 4 (Experiment 2).
"""
import numpy as np, pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score, recall_score, precision_score, brier_score_loss

ta = pd.read_csv("data_out/turnarounds_bts_2024_06.csv")
ta["dep_dt"] = pd.to_datetime(ta["dep_dt"], errors="coerce")
for c in ["inbound_arr_delay", "outbound_dep_delay", "sched_turnaround_min"]:
    ta[c] = pd.to_numeric(ta[c], errors="coerce")
ta = ta.dropna(subset=["inbound_arr_delay", "outbound_dep_delay", "sched_turnaround_min", "dep_dt", "airport"])
ta["hour"] = ta["dep_dt"].dt.hour
ta["dow"] = ta["dep_dt"].dt.dayofweek
ta["buffer_pressure"] = ta["inbound_arr_delay"] - ta["sched_turnaround_min"]
ta["y"] = (ta["outbound_dep_delay"] > 15).astype(int)
feats = ["inbound_arr_delay", "sched_turnaround_min", "buffer_pressure", "hour", "dow"]  # generic, no airport id

vol = ta["airport"].value_counts()
top = vol[vol >= 2000].index.tolist()
d = ta[ta["airport"].isin(top)].copy()
print(f"{len(top)} airports with >=2000 turnarounds; {len(d):,} rows")


def fit_eval(tr, te, cal=False):
    clf = HistGradientBoostingClassifier(max_iter=150, learning_rate=0.12, max_depth=5, random_state=0)
    clf.fit(tr[feats], tr["y"])
    p = clf.predict_proba(te[feats])[:, 1]
    dd = (p > 0.5).astype(int)
    out = [roc_auc_score(te["y"], p), recall_score(te["y"], dd), precision_score(te["y"], dd, zero_division=0)]
    if cal:
        out.append(brier_score_loss(te["y"], p))
    return out


rng = np.random.default_rng(0)
aps = list(top); rng.shuffle(aps)
half = len(aps) // 2
train_aps, test_aps = set(aps[:half]), set(aps[half:])

trA = d[d["airport"].isin(train_aps)]
teB = d[d["airport"].isin(test_aps)]
ood = fit_eval(trA, teB, cal=True)
trs = trA.sort_values("dep_dt"); cut = int(len(trs) * 0.75)
iid = fit_eval(trs.iloc[:cut], trs.iloc[cut:], cal=True)
print(f"\nIID (same airports, temporal split): AUC {iid[0]:.3f} recall {iid[1]:.3f} prec {iid[2]:.3f} brier {iid[3]:.3f}")
print(f"OOD (held-out airports):             AUC {ood[0]:.3f} recall {ood[1]:.3f} prec {ood[2]:.3f} brier {ood[3]:.3f}")
print(f"transfer gap (AUC): {iid[0]-ood[0]:+.3f}")

hubs = vol.head(8).index.tolist()
rows = []
for h in hubs:
    a, r, p = fit_eval(d[d["airport"] != h], d[d["airport"] == h])
    rows.append({"hub": h, "n_test": int((d["airport"] == h).sum()), "auc": a, "recall": r, "precision": p})
res = pd.DataFrame(rows)
res.to_csv("data_out/exp2_cross_airport.csv", index=False)
print("\nleave-one-hub-out (train on all other airports, test on the hub):")
print(res.round(3).to_string(index=False))
print(f"\nmean leave-one-hub-out AUC: {res['auc'].mean():.3f} (range {res['auc'].min():.3f}-{res['auc'].max():.3f})")

import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
fig, ax = plt.subplots(figsize=(8, 5))
ax.bar(res["hub"], res["auc"], color="#2b5d8a", alpha=.85)
ax.axhline(iid[0], color="#3a7d44", ls="--", label=f"in-distribution AUC {iid[0]:.3f}")
ax.set_ylim(0.5, 1.0); ax.set_ylabel("AUC on held-out hub"); ax.set_xlabel("hub left out of training")
ax.set_title("Experiment 2: cross-airport generalisation (leave-one-hub-out)")
ax.legend()
plt.tight_layout(); plt.savefig("data_out/exp2_cross_airport.png", dpi=150, facecolor="white")
print("wrote data_out/exp2_cross_airport.csv and .png")
