#!/usr/bin/env python
"""
European (OpenSky ADS-B) arm — schedule-free analysis. OpenSky has no schedules, so we work in
data-relative terms (design section 3a): excess turnaround = turnaround - per-airport reference
(20th percentile = a min-feasible-turnaround proxy). Two results:
  (1) ground-time-state propagation: does an aircraft's excess turnaround persist to its next leg?
  (2) cross-airport generalisation of a "long-turnaround" classifier across EU hubs.
This is the independent, cross-region, schedule-free replication that complements the US/BTS arm.
"""
import numpy as np, pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score

ta = pd.read_csv("data_out/turnarounds_multi_2024-06-01_14d.csv")
ta["out_time"] = pd.to_datetime(ta["out_time"], errors="coerce", utc=True)
ta["in_time"] = pd.to_datetime(ta["in_time"], errors="coerce", utc=True)
ta = ta.dropna(subset=["out_time", "turnaround_min", "airport", "icao24"])
ta["hour"] = ta["out_time"].dt.hour; ta["dow"] = ta["out_time"].dt.dayofweek

# per-airport reference (20th pct) -> excess turnaround
ref = ta.groupby("airport")["turnaround_min"].quantile(0.20).rename("ref")
ta = ta.join(ref, on="airport")
ta["excess"] = ta["turnaround_min"] - ta["ref"]
print(f"EU turnarounds: {len(ta):,} across {ta['airport'].nunique()} hubs")
print("per-hub median turnaround / excess:")
print(ta.groupby("airport").agg(n=("turnaround_min", "size"), median_min=("turnaround_min", "median"),
                                median_excess=("excess", "median")).round(1).to_string())

# (1) ground-time-state propagation across an aircraft's consecutive legs (within sampled hubs)
ta = ta.sort_values(["icao24", "out_time"])
g = ta.groupby("icao24", sort=False)
nxt_airport = g["airport"].shift(-1); nxt_excess = g["excess"].shift(-1)
valid = (ta["next_dest"] == nxt_airport)            # only true consecutive legs between sampled hubs
prop = pd.DataFrame({"excess": ta["excess"], "next_excess": nxt_excess})[valid].dropna()
r = prop["excess"].corr(prop["next_excess"])
print(f"\n(1) ground-time-state propagation: corr(excess_N, excess_N+1) = {r:.3f} on {len(prop):,} consecutive-leg pairs")
hi = prop[prop["excess"] > 30]["next_excess"].mean(); lo = prop[prop["excess"] <= 0]["next_excess"].mean()
print(f"    next-leg excess when this leg ran >30 min long: {hi:.1f} min  vs  on/under reference: {lo:.1f} min")

# (2) cross-airport generalisation: predict a 'long turnaround' (> that hub's 75th pct) from generic features
ta["y"] = (ta["turnaround_min"] > ta.groupby("airport")["turnaround_min"].transform(lambda s: s.quantile(0.75))).astype(int)
for col in ["prev_origin", "next_dest"]:
    freq = ta[col].value_counts(normalize=True)
    ta[col + "_freq"] = ta[col].map(freq).fillna(0)
feats = ["hour", "dow", "prev_origin_freq", "next_dest_freq"]
aps = sorted(ta["airport"].unique()); half = len(aps) // 2
trA, teB = set(aps[:half]), set(aps[half:])
clf = HistGradientBoostingClassifier(max_iter=200, learning_rate=0.1, max_depth=5, random_state=0)
clf.fit(ta[ta["airport"].isin(trA)][feats], ta[ta["airport"].isin(trA)]["y"])
teX = ta[ta["airport"].isin(teB)]
auc = roc_auc_score(teX["y"], clf.predict_proba(teX[feats])[:, 1])
print(f"\n(2) cross-airport 'long-turnaround' classifier (generic features, no aircraft type/schedule):")
print(f"    train {sorted(trA)} -> test {sorted(teB)} : AUC {auc:.3f}")
print("    (modest by design: OpenSky lacks aircraft type + schedules; the aircraft-DB join would lift this)")

ta.drop(columns=["ref"]).to_csv("data_out/turnarounds_eu_2024-06_excess.csv", index=False)
print("\nwrote data_out/turnarounds_eu_2024-06_excess.csv")
