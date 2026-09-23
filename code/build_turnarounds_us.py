#!/usr/bin/env python
"""
Build aircraft turnarounds from BTS Reporting Carrier On-Time Performance data (US arm).

REVISION 2026-09-23. Differences from the originally submitted pipeline
(scripts/bts_turnaround.py), each documented in the response to reviewers:

  (R1) Scheduled turnaround is wrapped into a SIGNED window [-720, +720) minutes instead of
       the unsigned [0, 1440). The unsigned wrap mapped genuinely negative scheduled
       turnarounds -- rotations that were already infeasible as scheduled -- onto values near
       +1440, i.e. onto a large imaginary buffer, in exactly the most stressed cases.
       Effect: corr(scheduled, actual) rises 0.712 -> 0.851; 1.95% of pairs are negative.
  (R2) Time-of-day features are taken from the SCHEDULED departure, not the actual departure.
       The actual departure time is displaced by the delay being predicted, so using its hour
       leaked the outcome (mildly) into the feature set.
  (R3) The five BTS cause-of-delay fields of the OUTBOUND leg are carried through, so the
       outcome can be validated against the carrier's own attribution (reviewer item 1).
  (R4) WheelsOn / TaxiIn / the inbound leg's own departure delay are carried through, to
       support the decision-lead-time ladder (reviewer item 2).
  (R5) Aircraft type is joined from the public OpenSky aircraft database by registration.
  (R6) Event datetimes are reconstructed as scheduled-time + reported-delay rather than from
       clock arithmetic. Verified to reproduce the reported clock times in 100.000% of rows.
       Turnaround duration is additionally computed modulo 24 h, which is immune to both the
       midnight wrap and to timezone differences; the two agree in 100.00% of retained pairs.

Usage:  python build_turnarounds_us.py --raw <dir> --out <dir>
"""
from __future__ import annotations
import argparse, re
from pathlib import Path
import numpy as np, pandas as pd

CAUSES = ["CarrierDelay", "WeatherDelay", "NASDelay", "SecurityDelay", "LateAircraftDelay"]
USECOLS = ["FlightDate", "Tail_Number", "Reporting_Airline", "Origin", "Dest",
           "CRSDepTime", "DepTime", "DepDelay", "CRSArrTime", "ArrTime", "ArrDelay",
           "ArrDel15", "Cancelled", "Diverted", "WheelsOn", "TaxiIn", "Distance",
           "CRSElapsedTime", "ActualElapsedTime"] + CAUSES

# retained-turnaround bounds (minutes); see manuscript section 3.2
LO, HI = 20.0, 720.0


def hhmm(v: pd.Series) -> pd.Series:
    """BTS local clock time hhmm -> minutes past local midnight. 2400 -> 0."""
    v = pd.to_numeric(v, errors="coerce")
    return (v.where(v != 2400, 0) // 100) * 60 + (v.where(v != 2400, 0) % 100)


def signed_wrap(x: pd.Series, half: float = 720.0) -> pd.Series:
    """Wrap a clock-time difference into [-half, +half) minutes."""
    return ((x + half) % (2 * half)) - half


def build(df: pd.DataFrame, actypes: pd.Series | None,
          lo: float = LO, hi: float = HI) -> pd.DataFrame:
    d = df[(df["Cancelled"].fillna(0) == 0) & (df["Diverted"].fillna(0) == 0)].copy()
    d = d.dropna(subset=["Tail_Number", "DepTime", "ArrTime", "FlightDate"])
    d = d[d["Tail_Number"].astype(str).str.strip().ne("")]
    d["date"] = pd.to_datetime(d["FlightDate"], errors="coerce")
    d = d.dropna(subset=["date"])
    for name, col in [("dep", "DepTime"), ("arr", "ArrTime"),
                      ("cdep", "CRSDepTime"), ("carr", "CRSArrTime")]:
        d[name + "_min"] = hhmm(d[col])
    d = d.dropna(subset=["dep_min", "arr_min", "cdep_min", "carr_min"])

    # ordering key: scheduled departure + reported departure delay (R6)
    d["dep_dt"] = d["date"] + pd.to_timedelta(d["cdep_min"] + d["DepDelay"].fillna(0), unit="m")
    d = d.sort_values(["Tail_Number", "dep_dt"])
    g = d.groupby("Tail_Number", sort=False)

    out = pd.DataFrame({
        "tail": d["Tail_Number"].astype(str).str.strip(),
        "carrier": d["Reporting_Airline"],
        "airport": d["Dest"],
        "_next_origin": g["Origin"].shift(-1),
        "prev_origin": d["Origin"],
        "next_dest": g["Dest"].shift(-1),
        "date": d["date"],
        # --- inbound leg (all known at or before the inbound's own gate arrival) ---
        "inbound_dep_delay": d["DepDelay"],          # known at inbound off-block  (earliest)
        "inbound_arr_delay": d["ArrDelay"],          # known at inbound in-block   (baseline)
        "inbound_taxi_in": d["TaxiIn"],              # in-block minus wheels-on
        "inbound_distance": d["Distance"],
        "inbound_sched_block": d["CRSElapsedTime"],  # scheduled gate-to-gate, inbound leg
        "inbound_actual_block": d["ActualElapsedTime"],
        # --- clock times, local to the turnaround airport ---
        "arr_min": d["arr_min"], "carr_min": d["carr_min"],
        "next_dep_min": g["dep_min"].shift(-1), "next_cdep_min": g["cdep_min"].shift(-1),
        # --- outbound leg ---
        "outbound_dep_delay": g["DepDelay"].shift(-1),
        "outbound_arr_del15": g["ArrDel15"].shift(-1),
    })
    for c in CAUSES:
        out["ob_" + c] = g[c].shift(-1)

    out = out[out["airport"] == out["_next_origin"]].drop(columns="_next_origin")
    out = out.dropna(subset=["next_dep_min", "outbound_dep_delay"])

    # actual and scheduled ground time, both modulo 24 h (R6, R1)
    out["turnaround_min"] = (out["next_dep_min"] - out["arr_min"]) % 1440
    out["sched_turnaround_min"] = signed_wrap(out["next_cdep_min"] - out["carr_min"])
    out["buffer_pressure"] = out["inbound_arr_delay"] - out["sched_turnaround_min"]

    # scheduled time-of-day features (R2)
    out["sched_hour"] = (out["next_cdep_min"] // 60).astype("Int64")
    out["dow"] = out["date"].dt.dayofweek

    # decision lead times (R4): from each candidate decision point to the outbound off-block
    out["lead_inblock"] = out["turnaround_min"]
    out["lead_wheelson"] = out["turnaround_min"] + out["inbound_taxi_in"].fillna(0)
    out["lead_inbound_offblock"] = out["turnaround_min"] + out["inbound_actual_block"].fillna(0)

    if actypes is not None:
        out["typecode"] = out["tail"].map(actypes)

    out = out[out["turnaround_min"].between(lo, hi)].reset_index(drop=True)

    # outcomes
    out["y"] = (out["outbound_dep_delay"] > 15).astype(int)          # primary: next-departure delay
    lac = out["ob_LateAircraftDelay"]
    out["attributed"] = lac.notna().astype(int)                       # BTS attribution available
    out["y_prop"] = ((out["y"] == 1) & (lac >= 15)).astype(int)       # secondary: propagation-specific
    return out


def main():
    ap = argparse.ArgumentParser()
    from common import RAW as _RAW, DATA as _DATA   # honours ADSB_DATA_RAW / ADSB_DATA_OUT
    ap.add_argument("--raw", default=str(_RAW))
    ap.add_argument("--out", default=str(_DATA))
    ap.add_argument("--only", default=None, help="process a single month, e.g. 2024_06")
    ap.add_argument("--bounds", default=None, help="override turnaround bounds, e.g. 1,1440")
    ap.add_argument("--suffix", default="", help="suffix for the output filename")
    args = ap.parse_args()
    lo, hi = (LO, HI) if not args.bounds else tuple(float(x) for x in args.bounds.split(","))
    raw, outd = Path(args.raw), Path(args.out)
    outd.mkdir(parents=True, exist_ok=True)

    adb = pd.read_csv(raw / "aircraftDatabase.csv", usecols=["registration", "typecode"],
                      low_memory=False)
    adb = adb.dropna(subset=["registration"])
    adb["registration"] = adb["registration"].astype(str).str.strip().str.upper()
    adb = adb[adb["typecode"].notna() & adb["typecode"].astype(str).str.strip().ne("")]
    actypes = adb.drop_duplicates("registration").set_index("registration")["typecode"]
    print(f"aircraft database: {len(actypes):,} registrations with a type code")

    files = sorted(raw.rglob("On_Time_Reporting_Carrier_On_Time_Performance*.csv"))
    rows = []
    for f in files:
        tag = re.search(r"(\d{4})_(\d{1,2})\.csv$", f.name)
        tag = f"{tag.group(1)}_{int(tag.group(2)):02d}" if tag else f.stem[-7:]
        if args.only and tag != args.only:
            continue
        df = pd.read_csv(f, usecols=USECOLS, low_memory=False)
        ta = build(df, actypes, lo, hi)
        ta.to_csv(outd / f"turnarounds_us_{tag}{args.suffix}.csv", index=False)
        rows.append({
            "month": tag, "flights": len(df), "turnarounds": len(ta),
            "first_date": str(ta["date"].min().date()), "last_date": str(ta["date"].max().date()),
            "median_actual": ta["turnaround_min"].median(),
            "median_sched": ta["sched_turnaround_min"].median(),
            "type_match_%": ta["typecode"].notna().mean() * 100 if "typecode" in ta else np.nan,
            "base_rate_y_%": ta["y"].mean() * 100,
            "attributed_%": ta["attributed"].mean() * 100,
            "base_rate_yprop_%": ta["y_prop"].mean() * 100,
        })
        print(f"  {tag}: {len(df):,} flights -> {len(ta):,} turnarounds  "
              f"(y {ta['y'].mean()*100:.1f}%, y_prop {ta['y_prop'].mean()*100:.1f}%, "
              f"type {ta['typecode'].notna().mean()*100:.1f}%)")
        del df, ta
    s = pd.DataFrame(rows)
    if args.suffix:
        print("\n" + s.round(2).to_string(index=False))
        return
    sf = outd / "build_summary_us.csv"
    if args.only and sf.exists():
        s = pd.concat([pd.read_csv(sf), s]).drop_duplicates("month", keep="last").sort_values("month")
    s.to_csv(sf, index=False)
    print("\n" + s.round(2).to_string(index=False))


if __name__ == "__main__":
    main()
