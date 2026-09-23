#!/usr/bin/env python
"""
Build aircraft turnarounds from BTS Reporting Carrier On-Time Performance data (US).
FREE, public, NO approval needed. Gives real scheduled + actual gate times and delays, so it
supports the schedule-delay-propagation analysis OpenSky cannot. See ../DESIGN_ADSB_InfoQuality_v0.1.md (section 10).

HOW TO GET ONE MONTH OF DATA (about 3 minutes, no login):
  1) Open https://www.transtats.bts.gov/DL_SelectFields.aspx?gnoyr_VQ=FGJ
     (TranStats > Aviation > "Reporting Carrier On-Time Performance (1987-present)").
  2) Choose a Year and Month, then tick at least these fields:
       FlightDate, Tail_Number, Origin, Dest,
       CRSDepTime, DepTime, DepDelay, CRSArrTime, ArrTime, ArrDelay, Cancelled, Diverted
  3) Click "Download" -> you get a .zip containing a .csv. Unzip it.
  (The prezipped monthly files in the TranStats PREZIP area are the same data with UPPER_SNAKE
   column names; this script accepts both naming styles.)

RUN:
    python bts_turnaround.py --csv "C:\\path\\to\\BTS_month.csv"
Output (./bts_out/): turnarounds_bts_<file>.csv + a console summary incl. a quick propagation signal.
"""
from __future__ import annotations
import argparse, sys, re
from pathlib import Path
import pandas as pd

# Map common BTS column-name variants -> canonical names used below
COLMAP = {
    "fl_date": "date", "flightdate": "date",
    "tail_num": "tail", "tail_number": "tail",
    "origin": "origin", "dest": "dest",
    "crs_dep_time": "crs_dep", "crsdeptime": "crs_dep",
    "dep_time": "dep", "deptime": "dep",
    "dep_delay": "dep_delay", "depdelay": "dep_delay",
    "crs_arr_time": "crs_arr", "crsarrtime": "crs_arr",
    "arr_time": "arr", "arrtime": "arr",
    "arr_delay": "arr_delay", "arrdelay": "arr_delay",
    "cancelled": "cancelled", "diverted": "diverted",
}


def hhmm_to_min(v):
    """BTS local clock time as hhmm (e.g. 1435 -> 875 min). 2400 -> 0. None on bad input."""
    if pd.isna(v):
        return None
    try:
        x = int(float(v))
    except Exception:
        return None
    if x == 2400:
        x = 0
    return (x // 100) * 60 + (x % 100)


def build(df: pd.DataFrame, lo: float, hi: float) -> pd.DataFrame:
    df = df.copy()
    df.columns = [c.strip().lower() for c in df.columns]
    df = df.rename(columns={c: COLMAP[c] for c in df.columns if c in COLMAP})
    need = {"date", "tail", "origin", "dest", "dep", "arr"}
    miss = need - set(df.columns)
    if miss:
        raise SystemExit(f"CSV missing required columns {miss}. Got: {list(df.columns)[:25]}")

    for c in ("cancelled", "diverted"):
        if c in df.columns:
            df = df[df[c].fillna(0).astype(float) == 0]
    df = df.dropna(subset=["tail", "dep", "arr", "date"])
    df = df[df["tail"].astype(str).str.strip().ne("")]
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"])
    df["dep_min"] = df["dep"].map(hhmm_to_min)
    df["arr_min"] = df["arr"].map(hhmm_to_min)
    df = df.dropna(subset=["dep_min", "arr_min"])

    df["dep_dt"] = df["date"] + pd.to_timedelta(df["dep_min"], unit="m")
    df["arr_dt"] = df["date"] + pd.to_timedelta(df["arr_min"], unit="m")
    # arrival clock earlier than departure clock => landed after midnight
    df.loc[df["arr_min"] < df["dep_min"], "arr_dt"] += pd.Timedelta(days=1)

    # Vectorised pairing: each flight -> its tail's next flight (by dep time).
    # A turnaround exists where this flight's dest == the next flight's origin.
    df = df.sort_values(["tail", "dep_dt"])
    g = df.groupby("tail", sort=False)
    nxt_origin = g["origin"].shift(-1)
    nxt_dep_dt = g["dep_dt"].shift(-1)
    out = pd.DataFrame({
        "tail": df["tail"],
        "airport": df["dest"],
        "arr_dt": df["arr_dt"],
        "dep_dt": nxt_dep_dt,
        "turnaround_min": (nxt_dep_dt - df["arr_dt"]).dt.total_seconds() / 60.0,
        "inbound_arr_delay": df["arr_delay"] if "arr_delay" in df.columns else pd.NA,
        "outbound_dep_delay": g["dep_delay"].shift(-1) if "dep_delay" in df.columns else pd.NA,
        "prev_origin": df["origin"],
        "next_dest": g["dest"].shift(-1),
        "_next_origin": nxt_origin,
    })
    if "crs_arr" in df.columns and "crs_dep" in df.columns:
        ca = df["crs_arr"].map(hhmm_to_min)
        cd = g["crs_dep"].shift(-1).map(hhmm_to_min)
        out["sched_turnaround_min"] = (cd - ca) % (24 * 60)
    else:
        out["sched_turnaround_min"] = pd.NA
    out = out[out["airport"] == out["_next_origin"]].drop(columns="_next_origin")
    out = out[out["turnaround_min"].between(lo, hi)].reset_index(drop=True)
    out["turnaround_min"] = out["turnaround_min"].round(1)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="path to a BTS On-Time CSV")
    ap.add_argument("--min-min", type=float, default=20.0)
    ap.add_argument("--max-min", type=float, default=720.0)
    args = ap.parse_args()

    df = pd.read_csv(args.csv, low_memory=False)
    print(f"read {len(df)} flight rows from {args.csv}")
    ta = build(df, args.min_min, args.max_min)
    print(f"plausible turnarounds [{args.min_min:.0f},{args.max_min:.0f}] min: {len(ta)}")
    if len(ta):
        print("\nturnaround_min summary:")
        print(ta["turnaround_min"].describe(percentiles=[.1, .25, .5, .75, .9]).round(1).to_string())
        sub = ta[["inbound_arr_delay", "turnaround_min", "outbound_dep_delay"]].apply(pd.to_numeric, errors="coerce").dropna()
        if len(sub) > 30:
            corr = sub.corr()["outbound_dep_delay"].round(3)
            print("\nquick propagation signal (corr vs outbound departure delay):")
            print(corr.to_string())

    out = Path("bts_out"); out.mkdir(exist_ok=True)
    name = re.sub(r"[^A-Za-z0-9_]+", "_", Path(args.csv).stem)[:40]
    ta.to_csv(out / f"turnarounds_bts_{name}.csv", index=False)
    print(f"\nWrote bts_out/turnarounds_bts_{name}.csv ({len(ta)} rows)")
    print("Sanity: US hub turnarounds typically cluster ~30-90 min; the propagation corr should be")
    print("positive for inbound_arr_delay and negative-ish for turnaround_min (tight turns propagate delay).")


if __name__ == "__main__":
    main()
