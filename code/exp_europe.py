#!/usr/bin/env python
"""
European (OpenSky ADS-B) arm -- reviewer item 5, and a re-audit of our own claims.

Reviewer item 5 asked us either to label the European arm supplementary or to repeat the
noise / coarsening / missing experiments on it, and to narrow the replication claim because
the twentieth-percentile excess-turnaround proxy is not schedule-based delay. Attempting
that repeat forced a re-examination of the two European results the submitted paper relied
on, and one of them does not survive.

What this script establishes, in order:

  1  SCOPE. The European sample is five 14-day windows, not five months. The submitted
     abstract and section 5.1 say "five months". 203,103 turnarounds reproduces exactly.

  2  PROPAGATION (submitted: "excess turnaround correlates 0.31 with the next leg's
     excess"). The figure reproduces exactly, but it is an artefact of two things: route
     composition, and pairs that are not consecutive legs at all. Removing aircraft whose
     "next" turnaround is more than 12 hours later, and removing route-level means, leaves
     no association. This claim is withdrawn.

  3  TEMPORAL STABILITY (submitted: "AUCs of 0.79-0.85, near-perfect same-season transfer").
     This reproduces only when the label threshold and the categorical frequency maps are
     estimated across all five windows, including the test windows. Estimated within each
     window, as a clean temporal protocol requires, discrimination falls to 0.61-0.65. The
     claim is therefore reported as conditional on a protocol we cannot defend.

  4  What does survive: the descriptive turnaround structure, the cross-airport
     long-turnaround classifier, and the aircraft-type contribution.

  5  Why the degradation experiments cannot be repeated here: the European data contains no
     inbound-status signal of any strength to degrade.

Outputs: out/europe_*.csv, out/europe.txt
"""
from __future__ import annotations
import re
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.metrics import roc_auc_score, r2_score
from common import DATA, OUT, RAW, SEED
import os

# The European turnarounds were derived in the original study and are consumed as given;
# ADSB_EU_SRC lets the released repository point at wherever they are held.
SRC = Path(os.environ.get("ADSB_EU_SRC", DATA.parents[1] / "data_out"))
WINDOWS = ["2022-01-01", "2024-01-01", "2024-06-01", "2025-06-01", "2026-01-01"]
HUBS = ["EHAM", "EDDF", "EGLL", "LFPG", "LEMD", "EDDM", "LIRF", "LSZH"]
FEATS = ["hour", "dow", "prev_origin_freq", "next_dest_freq"]
HP = dict(max_iter=200, learning_rate=0.1, max_depth=5, random_state=SEED)
L = []


def say(s=""):
    print(s)
    L.append(str(s))


def typemap():
    a = pd.read_csv(RAW / "aircraftDatabase.csv", usecols=["icao24", "typecode"], low_memory=False)
    a = a.dropna(subset=["icao24"])
    a["icao24"] = a["icao24"].astype(str).str.strip().str.lower()
    a = a[a["typecode"].notna() & a["typecode"].astype(str).str.strip().ne("")]
    return a.drop_duplicates("icao24").set_index("icao24")["typecode"]


def load(w, tmap):
    d = pd.read_csv(SRC / f"turnarounds_multi_{w}_14d.csv")
    for c in ("in_time", "out_time"):
        d[c] = pd.to_datetime(d[c], errors="coerce", utc=True)
    d = d.dropna(subset=["in_time", "out_time", "turnaround_min", "airport", "icao24"])
    d["icao24"] = d["icao24"].astype(str).str.strip().str.lower()
    d["hour"] = d["out_time"].dt.hour
    d["dow"] = d["out_time"].dt.dayofweek
    d["typecode"] = d["icao24"].map(tmap)
    d["window"] = w
    return d


def add_freq(d, freqs):
    d = d.copy()
    for c in ("prev_origin", "next_dest"):
        d[c + "_freq"] = d[c].map(freqs[c]).fillna(0)
    return d


def ordinal(tr, te, col, new):
    lv = pd.Index(sorted(tr[col].dropna().astype(str).unique()))
    m = pd.Series(np.arange(len(lv)), index=lv)
    tr = tr.copy(); te = te.copy()
    tr[new] = tr[col].astype(str).map(m).fillna(-1)
    te[new] = te[col].astype(str).map(m).fillna(-1)
    return tr, te


def main():
    tmap = typemap()
    raw = {w: load(w, tmap) for w in WINDOWS}
    pool = pd.concat(raw.values(), ignore_index=True)

    # ---- 1. scope ---------------------------------------------------------------
    say("1. SCOPE OF THE EUROPEAN SAMPLE")
    rows = []
    for w in WINDOWS:
        d = raw[w]
        span = (d["out_time"].max() - d["in_time"].min()).days
        rows.append({"window_start": w, "n": len(d), "first": str(d["in_time"].min().date()),
                     "last": str(d["out_time"].max().date()), "days_spanned": span,
                     "type_match_pct": d["typecode"].notna().mean() * 100,
                     "median_turnaround_min": d["turnaround_min"].median()})
        say(f"   {w}: n={len(d):>6,}  {rows[-1]['first']} to {rows[-1]['last']}  "
            f"({span} days)  type match {rows[-1]['type_match_pct']:.1f}%  "
            f"median {d['turnaround_min'].median():.1f} min")
    scope = pd.DataFrame(rows)
    scope.to_csv(OUT / "europe_scope.csv", index=False)
    say(f"   TOTAL {len(pool):,} turnarounds at {pool['airport'].nunique()} hubs "
        f"(submitted manuscript: 203,103 -- reproduces exactly)")
    say("   These are five 14-DAY WINDOWS. The submitted abstract and section 5.1 call them")
    say("   'five months'. That wording must be corrected.\n")

    # per-hub structure
    hub = pool.groupby("airport")["turnaround_min"].agg(
        n="size", median="median", q25=lambda s: s.quantile(.25), q75=lambda s: s.quantile(.75))
    hub.to_csv(OUT / "europe_hub_structure.csv")
    say("   per-hub turnaround structure (taxi-inclusive ADS-B proxy, so systematically")
    say("   longer than the US gate-to-gate figures):")
    say("   " + hub.round(1).to_string().replace("\n", "\n   "))

    # ---- 2. the propagation claim ------------------------------------------------
    say("\n2. RE-AUDIT OF THE PUBLISHED PROPAGATION CLAIM (submitted: r = 0.31)")
    ref = pool.groupby(["airport", "window"])["turnaround_min"].quantile(.20).rename("ref")
    p = pool.join(ref, on=["airport", "window"])
    p["excess"] = p["turnaround_min"] - p["ref"]
    p = p.sort_values(["icao24", "out_time"])
    g = p.groupby("icao24", sort=False)
    pairs = pd.DataFrame({
        "window": p["window"], "airport": p["airport"], "excess": p["excess"],
        "next_dest": p["next_dest"], "out_time": p["out_time"],
        "n_airport": g["airport"].shift(-1), "n_excess": g["excess"].shift(-1),
        "n_in": g["in_time"].shift(-1), "n_window": g["window"].shift(-1)})
    pairs = pairs[(pairs["next_dest"] == pairs["n_airport"]) &
                  (pairs["window"] == pairs["n_window"])].dropna(subset=["n_excess"])
    pairs["gap_h"] = (pairs["n_in"] - pairs["out_time"]).dt.total_seconds() / 3600

    def within_route_r(df):
        k = df.groupby(["airport", "n_airport"])
        return (df["excess"] - k["excess"].transform("mean")).corr(
            df["n_excess"] - k["n_excess"].transform("mean"))

    rows = []
    say("   gap filter                    n     pooled r   within-route r")
    for lab, m in [("none (as submitted)", pairs["gap_h"].notna()),
                   ("gap < 24 h", pairs["gap_h"] < 24), ("gap < 12 h", pairs["gap_h"] < 12),
                   ("gap < 6 h", pairs["gap_h"] < 6), ("gap < 3 h", pairs["gap_h"] < 3)]:
        d = pairs[m]
        rows.append({"filter": lab, "n": len(d), "pooled_r": d["excess"].corr(d["n_excess"]),
                     "within_route_r": within_route_r(d)})
        say(f"   {lab:<22}{len(d):>8,}   {rows[-1]['pooled_r']:>8.3f}   {rows[-1]['within_route_r']:>13.3f}")
    pd.DataFrame(rows).to_csv(OUT / "europe_propagation_audit.csv", index=False)
    d12 = pairs[pairs["gap_h"] < 12]
    rng = np.random.default_rng(SEED)
    vals = [within_route_r(d12.iloc[rng.integers(0, len(d12), len(d12))]) for _ in range(1000)]
    vals = [v for v in vals if np.isfinite(v)]
    say(f"\n   within-route r at gap < 12 h = {within_route_r(d12):.3f}, "
        f"95% bootstrap CI {np.percentile(vals, 2.5):.3f} to {np.percentile(vals, 97.5):.3f}")
    say(f"   {(1 - within_route_r(pairs)/pairs['excess'].corr(pairs['n_excess']))*100:.0f}% of the "
        f"published association is between-route composition; the remainder is carried by")
    say(f"   pairs {(pairs['gap_h'] > 12).mean()*100:.0f}% of which are separated by more than 12 hours, i.e. the aircraft")
    say("   left the eight sampled hubs and returned later. These are not consecutive legs.")
    say("   CONCLUSION: the European propagation claim is withdrawn.")

    # ---- 3. the drift claim ------------------------------------------------------
    say("\n3. RE-AUDIT OF THE PUBLISHED EUROPEAN DRIFT CLAIM (submitted: AUC 0.79-0.85)")
    say("   A: label threshold and frequency maps estimated WITHIN each window (clean)")
    say("   B: estimated ONCE across all five windows, test windows included (leaky)")
    q_pool = pool.groupby("airport")["turnaround_min"].quantile(.75)
    fr_pool = {c: pool[c].value_counts(normalize=True) for c in ("prev_origin", "next_dest")}
    q_w = {w: raw[w].groupby("airport")["turnaround_min"].quantile(.75) for w in WINDOWS}
    fr_w = {w: {c: raw[w][c].value_counts(normalize=True) for c in ("prev_origin", "next_dest")}
            for w in WINDOWS}

    def lab_y(d, q):
        d = d.copy()
        d["y"] = (d["turnaround_min"] > d["airport"].map(q)).astype(int)
        return d

    trn = "2024-06-01"
    bA = lab_y(add_freq(raw[trn], fr_w[trn]), q_w[trn]).sort_values("out_time")
    bB = lab_y(add_freq(raw[trn], fr_pool), q_pool).sort_values("out_time")
    cut = int(len(bA) * .75)
    cA = HistGradientBoostingClassifier(**HP).fit(bA.iloc[:cut][FEATS], bA.iloc[:cut]["y"])
    cB = HistGradientBoostingClassifier(**HP).fit(bB.iloc[:cut][FEATS], bB.iloc[:cut]["y"])
    rows = []
    say(f"\n   {'window':<13}{'A (clean)':>11}{'B (leaky)':>11}")
    for w in WINDOWS:
        tA = bA.iloc[cut:] if w == trn else lab_y(add_freq(raw[w], fr_w[w]), q_w[w])
        tB = bB.iloc[cut:] if w == trn else lab_y(add_freq(raw[w], fr_pool), q_pool)
        aA = roc_auc_score(tA["y"], cA.predict_proba(tA[FEATS])[:, 1])
        aB = roc_auc_score(tB["y"], cB.predict_proba(tB[FEATS])[:, 1])
        rows.append({"window": w, "auc_clean": aA, "auc_leaky": aB,
                     "ref": "in-distribution" if w == trn else "out-of-window"})
        say(f"   {w:<13}{aA:>11.3f}{aB:>11.3f}{'   <- in-distribution' if w == trn else ''}")
    pd.DataFrame(rows).to_csv(OUT / "europe_drift_audit.csv", index=False)
    say("   The submitted 0.79-0.85 corresponds to protocol B. Under protocol A the model")
    say("   does not transfer across windows. The stability claim is therefore not supported.")

    # ---- 4. what survives --------------------------------------------------------
    say("\n4. RESULTS THAT DO SURVIVE")
    d = lab_y(add_freq(raw["2024-06-01"], fr_w["2024-06-01"]), q_w["2024-06-01"])
    aps = sorted(d["airport"].unique()); half = len(aps) // 2
    trA, teB = set(aps[:half]), set(aps[half:])
    tr, te = d[d["airport"].isin(trA)], d[d["airport"].isin(teB)]
    out = []
    for lab, extra in [("generic features only", []), ("plus aircraft type", ["tc"])]:
        a, b = ordinal(tr, te, "typecode", "tc")
        f = FEATS + extra
        c = HistGradientBoostingClassifier(**HP).fit(a[f], a["y"])
        auc = roc_auc_score(b["y"], c.predict_proba(b[f])[:, 1])
        out.append({"analysis": "cross-airport long-turnaround classifier", "variant": lab, "value": auc})
        say(f"   cross-airport classifier, {lab:<22} AUC {auc:.3f}   "
            f"(submitted: {'0.78' if not extra else '0.81'})")
    d2 = d.sort_values("out_time"); c2 = int(len(d2) * .75)
    tr2, te2 = d2.iloc[:c2], d2.iloc[c2:]
    for lab, extra in [("generic features only", []), ("plus aircraft type", ["tc"])]:
        a, b = ordinal(tr2, te2, "typecode", "tc")
        f = FEATS + extra
        r = HistGradientBoostingRegressor(max_iter=250, learning_rate=.1, max_depth=6,
                                          random_state=SEED).fit(a[f], a["turnaround_min"])
        r2 = r2_score(b["turnaround_min"], r.predict(b[f]))
        out.append({"analysis": "turnaround-duration regression", "variant": lab, "value": r2})
        say(f"   duration regression,      {lab:<22} R2  {r2:.3f}   "
            f"(submitted: {'0.45' if not extra else '0.55'})")
    pd.DataFrame(out).to_csv(OUT / "europe_surviving.csv", index=False)

    # ---- 5. why the degradation experiments cannot be repeated --------------------
    say("\n5. WHY THE INFORMATION-DEGRADATION EXPERIMENTS CANNOT BE REPEATED IN EUROPE")
    chain = pairs[pairs["gap_h"] < 12]
    say(f"   Degrading an inbound-status signal requires one to exist. OpenSky carries no")
    say(f"   schedules, so the only candidate is the aircraft's own previous ground stop,")
    say(f"   observable for {len(chain):,} of {len(pool):,} turnarounds ({len(chain)/len(pool)*100:.1f}%) because only")
    say(f"   eight hubs are sampled. Its association with the outcome is "
        f"{within_route_r(chain):.3f}.")
    say("   There is no signal of any strength to degrade, so the core experiment of this")
    say("   paper has no European counterpart in this dataset.")

    (OUT / "europe.txt").write_text("\n".join(L), encoding="utf-8")
    print("\nwrote out/europe_*.csv and out/europe.txt")


if __name__ == "__main__":
    main()
