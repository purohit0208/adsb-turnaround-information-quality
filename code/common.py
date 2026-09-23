#!/usr/bin/env python
"""
Shared configuration, data loading, feature definitions and metrics for the US arm.

Everything a reviewer needs to reproduce a number is fixed here in one place:
the feature groups, the train/test split, the model hyperparameters, the encoding
rule, the seeds, the bootstrap procedure and the calibration binning.
"""
from __future__ import annotations
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import (roc_auc_score, average_precision_score, recall_score,
                             precision_score, brier_score_loss)

import os

# Project root: the parent of code/. Override any path with an environment variable, which
# lets the released repository run against a data directory held outside the checkout.
ROOT = Path(os.environ.get("ADSB_ROOT", Path(__file__).resolve().parents[1]))
DATA = Path(os.environ.get("ADSB_DATA_OUT", ROOT / "data_out"))
RAW = Path(os.environ.get("ADSB_DATA_RAW", ROOT / "data_raw"))
OUT = Path(os.environ.get("ADSB_OUT", ROOT / "out"))
FIGS = Path(os.environ.get("ADSB_FIGS", ROOT / "figures"))
for p in (OUT, FIGS):
    p.mkdir(parents=True, exist_ok=True)

SEED = 0
N_BOOT = 2000                 # bootstrap resamples for confidence intervals
ECE_BINS = 10                 # equal-width bins on [0,1] for expected calibration error
THR = 0.5                     # default operating threshold
MONTHS = ["2022_01", "2024_01", "2024_06", "2025_06", "2026_01"]
TRAIN_MONTH = "2024_06"
# explicit calendar split, on a day boundary so no day straddles train and test
SPLIT_DATE = pd.Timestamp("2024-06-24")     # train 1-23 June, test 24-30 June

HP = dict(max_iter=250, learning_rate=0.1, max_depth=6, random_state=SEED)

# ---------------------------------------------------------------- feature groups
# An "information source" is what an operator either has or does not have. Features
# derived deterministically from a source belong to that source and must be removed
# with it, otherwise an ablation leaks the removed signal back in. buffer_pressure is
# inbound_arr_delay - sched_turnaround_min and therefore belongs to BOTH parents.
SOURCES = {
    "inbound_status":     ["inbound_arr_delay"],
    "schedule":           ["sched_turnaround_min"],
    "time_of_day":        ["sched_hour", "dow"],
    "airport":            ["airport_code"],
    "aircraft_type":      ["type_code"],
}
DERIVED = {"buffer_pressure": ["inbound_status", "schedule"]}
ALL_FEATS = [f for v in SOURCES.values() for f in v] + list(DERIVED)


def features_without(dropped: list[str] | None = None) -> list[str]:
    """Feature list with whole information sources removed, including their derivatives."""
    dropped = set(dropped or [])
    feats = [f for s, fs in SOURCES.items() if s not in dropped for f in fs]
    feats += [d for d, parents in DERIVED.items() if not (set(parents) & dropped)]
    return feats


# ---------------------------------------------------------------- data
def load_month(tag: str) -> pd.DataFrame:
    ta = pd.read_csv(DATA / f"turnarounds_us_{tag}.csv", low_memory=False)
    ta["date"] = pd.to_datetime(ta["date"], errors="coerce")
    num = ["inbound_arr_delay", "inbound_dep_delay", "outbound_dep_delay",
           "sched_turnaround_min", "buffer_pressure", "turnaround_min",
           "lead_inblock", "lead_wheelson", "ob_LateAircraftDelay"]
    for c in num:
        ta[c] = pd.to_numeric(ta[c], errors="coerce")
    ta = ta.dropna(subset=["inbound_arr_delay", "outbound_dep_delay",
                           "sched_turnaround_min", "date", "sched_hour"])
    ta["sched_hour"] = ta["sched_hour"].astype(int)
    return ta.sort_values(["date", "next_cdep_min"]).reset_index(drop=True)


def encode(tr: pd.DataFrame, te: pd.DataFrame, cols=("airport", "typecode")):
    """Ordinal encoding fitted on the TRAINING split only; unseen levels -> -1.

    Fitting the encoder on the full dataset (as in the original submission) lets test-set
    category frequencies influence the code assignment. Codes are arbitrary integers for a
    tree model, so the effect is small, but the protocol is now clean.
    """
    tr, te = tr.copy(), te.copy()
    for c in cols:
        levels = pd.Index(sorted(tr[c].dropna().astype(str).unique()))
        m = pd.Series(np.arange(len(levels)), index=levels)
        name = {"airport": "airport_code", "typecode": "type_code"}[c]
        tr[name] = tr[c].astype(str).map(m).fillna(-1).astype(int)
        te[name] = te[c].astype(str).map(m).fillna(-1).astype(int)
    return tr, te


def split_train_test(ta: pd.DataFrame, split_date=SPLIT_DATE):
    tr = ta[ta["date"] < split_date]
    te = ta[ta["date"] >= split_date]
    return encode(tr, te)


# ---------------------------------------------------------------- model + metrics
def fit(tr: pd.DataFrame, feats: list[str], target: str = "y", **kw):
    clf = HistGradientBoostingClassifier(**{**HP, **kw})
    clf.fit(tr[feats], tr[target])
    return clf


def metrics(y, p, thr=THR, n=None) -> dict:
    d = (p > thr).astype(int)
    return {
        "n": int(len(y)) if n is None else n,
        "prevalence": float(np.mean(y)),
        "auc": float(roc_auc_score(y, p)),
        "auprc": float(average_precision_score(y, p)),
        "recall": float(recall_score(y, d, zero_division=0)),
        "precision": float(precision_score(y, d, zero_division=0)),
        "alert_rate": float(np.mean(d)),
        "brier": float(brier_score_loss(y, p)),
        "ece": ece(y, p),
    }


def ece(y, p, bins=ECE_BINS) -> float:
    """Expected calibration error, equal-width bins on [0,1]."""
    y, p = np.asarray(y), np.asarray(p)
    edges = np.linspace(0, 1, bins + 1)
    idx = np.clip(np.digitize(p, edges) - 1, 0, bins - 1)
    e = 0.0
    for b in range(bins):
        m = idx == b
        if m.any():
            e += m.mean() * abs(y[m].mean() - p[m].mean())
    return float(e)


def boot_ci(y, p, stat=roc_auc_score, n_boot=N_BOOT, seed=SEED, alpha=5.0):
    """Percentile bootstrap over test-set rows (i.i.d. resampling of turnarounds)."""
    rng = np.random.default_rng(seed)
    y, p = np.asarray(y), np.asarray(p)
    n = len(y)
    vals = np.empty(n_boot)
    for i in range(n_boot):
        s = rng.integers(0, n, n)
        if y[s].min() == y[s].max():
            vals[i] = np.nan
            continue
        vals[i] = stat(y[s], p[s])
    return float(np.nanpercentile(vals, alpha / 2)), float(np.nanpercentile(vals, 100 - alpha / 2))


def threshold_for_alert_rate(p, rate) -> float:
    """Threshold that makes exactly `rate` of cases alert -- for equal-alarm-budget tests."""
    return float(np.quantile(p, 1.0 - rate))


def leaked_delay_pct(te: pd.DataFrame, decision: np.ndarray, y_col="y") -> float:
    """Share of delay-minutes, among true positives, that the decision did not flag.

    NOTE (reviewer item 3): this is unflagged delay, i.e. an upper bound on what could be
    addressed if every flagged case were perfectly mitigated. It is not preventable delay,
    and no intervention model is claimed.
    """
    t = te[y_col] == 1
    tot = te.loc[t, "outbound_dep_delay"].sum()
    return float(te.loc[t & (decision == 0), "outbound_dep_delay"].sum() / tot * 100)
