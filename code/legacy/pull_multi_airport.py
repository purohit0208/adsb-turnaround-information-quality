#!/usr/bin/env python
"""
Multi-airport turnaround pull from OpenSky (Phase 2 of the ADS-B study).
Loops the validated single-airport pairing (pilot_turnaround.pair_turnarounds) over a list
of airports and a date window, tags each turnaround with its airport, and writes one combined
dataset + a per-airport summary. See ../DESIGN_ADSB_InfoQuality_v0.1.md.

RUN ONLY AFTER:
  - OpenSky data access is granted (the pilot returns real data, not PERMISSION_DENIED), and
  - the one-airport pilot looks sane (median turnaround believable).

Respects OpenSky usage rules: pyopensky's flightlist already filters flights_data4 by the `day`
partition, and we query ONE airport at a time (well under the 2-concurrent-query limit). For long
windows, prefer several short runs over one huge run.

RUN:
    python pull_multi_airport.py --start 2024-06-03 --days 7
    python pull_multi_airport.py --airports EHAM,EDDF,EGLL --start 2024-06-03 --days 7
Output (./data_out/): turnarounds_multi_<start>_<days>d.csv  + summary_<start>_<days>d.csv
"""
from __future__ import annotations
import argparse, sys, time
from pathlib import Path
import pandas as pd

# Reuse the unit-tested pairing logic from the pilot
sys.path.insert(0, str(Path(__file__).resolve().parent))
from pilot_turnaround import pair_turnarounds, _norm_cols  # noqa: E402

# Major European hubs with good ADS-B coverage (override with --airports)
DEFAULT_AIRPORTS = ["EHAM", "EDDF", "EGLL", "LFPG", "LEMD", "EDDM", "LIRF", "LSZH"]


def pull_airport(Trino, A, start, stop, lo, hi):
    fl = Trino().flightlist(start, stop, airport=A)
    if fl is None or len(fl) == 0:
        return pd.DataFrame()
    fl = _norm_cols(fl)
    for c in ("firstseen", "lastseen"):
        fl[c] = pd.to_datetime(fl[c], utc=True)
    arr = fl[fl["estarrivalairport"] == A].copy()
    dep = fl[fl["estdepartureairport"] == A].copy()
    ta = pair_turnarounds(arr, dep, lo, hi)
    if len(ta):
        ta.insert(0, "airport", A)
    return ta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--airports", default=",".join(DEFAULT_AIRPORTS),
                    help="comma-separated ICAO codes")
    ap.add_argument("--start", default="2024-06-03")
    ap.add_argument("--days", type=int, default=7)
    ap.add_argument("--min-min", type=float, default=20.0)
    ap.add_argument("--max-min", type=float, default=360.0)
    args = ap.parse_args()

    try:
        from pyopensky.trino import Trino
    except ImportError:
        sys.exit("pyopensky not installed -> pip install pyopensky pandas matplotlib")

    airports = [a.strip().upper() for a in args.airports.split(",") if a.strip()]
    start = pd.Timestamp(args.start, tz="UTC")
    stop = start + pd.Timedelta(days=args.days)
    print(f"Pulling {len(airports)} airports  {start:%Y-%m-%d} .. {stop:%Y-%m-%d}")

    frames, summary = [], []
    for i, A in enumerate(airports, 1):
        print(f"[{i}/{len(airports)}] {A} ...", end=" ", flush=True)
        try:
            ta = pull_airport(Trino, A, start, stop, args.min_min, args.max_min)
        except Exception as e:
            print(f"ERROR ({e})"); summary.append({"airport": A, "n": 0, "median_min": None, "note": str(e)[:80]})
            continue
        n = len(ta)
        med = round(ta["turnaround_min"].median(), 1) if n else None
        print(f"{n} turnarounds, median {med} min")
        summary.append({"airport": A, "n": n, "median_min": med, "note": ""})
        if n:
            frames.append(ta)
        time.sleep(1.0)  # be polite between airports

    out = Path("data_out"); out.mkdir(exist_ok=True)
    tag = f"{args.start}_{args.days}d"
    allta = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    allta.to_csv(out / f"turnarounds_multi_{tag}.csv", index=False)
    pd.DataFrame(summary).to_csv(out / f"summary_{tag}.csv", index=False)

    print(f"\nTotal turnarounds: {len(allta)} across {len([s for s in summary if s['n']])} airports")
    print("Per-airport medians (sanity-check against realistic hub turnarounds, ~40-90 min):")
    print(pd.DataFrame(summary).to_string(index=False))
    print(f"\nWrote data_out/turnarounds_multi_{tag}.csv and summary_{tag}.csv")
    print("Next: aircraft-type join (OpenSky aircraft DB) + connecting-leg linkage for the predictor/Exp 1.")


if __name__ == "__main__":
    main()
