#!/usr/bin/env python
"""
PILOT (go/no-go): derive aircraft turnaround (ground) times at ONE airport from
OpenSky ADS-B via the pyopensky Trino interface, by pairing each aircraft's arrival
with its next departure. Validates the pairing approach before we scale to multiple
airports + the information-quality experiment (see ../DESIGN_ADSB_InfoQuality_v0.1.md).

ONE-TIME SETUP (your Windows machine):
  1) pip install pyopensky pandas matplotlib
  2) OpenSky account (done) + request data activation once at:
        https://opensky-network.org/my-opensky/request-data
  3) Find the config file and add your OpenSky login:
        python -c "from pyopensky.config import opensky_config_dir; print(opensky_config_dir)"
     In settings.conf set, under [default]:  username = <you>   password = <you>

RUN (start small):
    python pilot_turnaround.py --airport EHAM --start 2024-06-03 --days 3
Then scale to --days 7 (or another airport: EDDF Frankfurt, EGLL Heathrow, LEMD Madrid).

OUTPUT (./pilot_out/): turnarounds_<tag>.csv, hist_<tag>.png, and a console summary.

HONEST NOTE: firstseen/lastseen are airborne-segment proxies (first/last ADS-B sample),
so this turnaround proxy includes taxi-in + taxi-out. Phase 2 refines to true on/off-block
times via state_vectors on_ground. The pilot only needs to show the pairing is sane.
"""
from __future__ import annotations
import argparse, sys
from pathlib import Path
import pandas as pd


def _norm_cols(fl):
    """Map pyopensky/traffic flightlist airport columns to canonical names."""
    ren = {}
    for c in fl.columns:
        k = c.lower().replace("_", "")
        if k in ("estarrivalairport", "arrival", "arrivalairport"):
            ren[c] = "estarrivalairport"
        elif k in ("estdepartureairport", "departure", "departureairport"):
            ren[c] = "estdepartureairport"
    return fl.rename(columns=ren)


def pair_turnarounds(arr: pd.DataFrame, dep: pd.DataFrame, lo: float, hi: float) -> pd.DataFrame:
    """Per aircraft, pair each arrival (in) with the immediately following departure (out)."""
    ev = []
    for _, a in arr.iterrows():
        ev.append((a["icao24"], a["lastseen"], "in", a))
    for _, d in dep.iterrows():
        ev.append((d["icao24"], d["firstseen"], "out", d))
    E = pd.DataFrame(ev, columns=["icao24", "time", "kind", "row"])
    rows = []
    for icao, g in E.groupby("icao24"):
        g = g.sort_values("time").reset_index(drop=True)
        for i in range(len(g) - 1):
            if g.loc[i, "kind"] == "in" and g.loc[i + 1, "kind"] == "out":
                a, nd = g.loc[i, "row"], g.loc[i + 1, "row"]
                tt = (g.loc[i + 1, "time"] - g.loc[i, "time"]).total_seconds() / 60.0
                rows.append({
                    "icao24": icao,
                    "arr_callsign": a.get("callsign"), "dep_callsign": nd.get("callsign"),
                    "in_time": a["lastseen"], "out_time": nd["firstseen"],
                    "turnaround_min": round(tt, 1),
                    "prev_origin": a.get("estdepartureairport"),
                    "next_dest": nd.get("estarrivalairport"),
                })
    ta = pd.DataFrame(rows)
    if len(ta):
        ta = ta[(ta["turnaround_min"] >= lo) & (ta["turnaround_min"] <= hi)].reset_index(drop=True)
    return ta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--airport", default="EHAM", help="ICAO code (EHAM, EDDF, EGLL, LEMD, ...)")
    ap.add_argument("--start", default="2024-06-03", help="UTC date YYYY-MM-DD")
    ap.add_argument("--days", type=int, default=3)
    ap.add_argument("--min-min", type=float, default=20.0)
    ap.add_argument("--max-min", type=float, default=360.0)
    ap.add_argument("--limit", type=int, default=None, help="row cap for a quick smoke test")
    args = ap.parse_args()

    try:
        from pyopensky.trino import Trino
    except ImportError:
        sys.exit("pyopensky not installed -> pip install pyopensky pandas matplotlib")

    start = pd.Timestamp(args.start, tz="UTC")
    stop = start + pd.Timedelta(days=args.days)
    A = args.airport.upper()
    print(f"Querying flightlist: {A}  {start:%Y-%m-%d} .. {stop:%Y-%m-%d}")

    fl = Trino().flightlist(start, stop, airport=A, limit=args.limit)
    if fl is None or len(fl) == 0:
        sys.exit("No flights returned. Check activation/credentials, airport code, or window.")
    fl = _norm_cols(fl)
    print(f"  flightlist rows: {len(fl)} | columns: {list(fl.columns)}")
    miss = {"estarrivalairport", "estdepartureairport", "firstseen", "lastseen", "icao24"} - set(fl.columns)
    if miss:
        sys.exit(f"flightlist missing expected columns {miss}. Got {list(fl.columns)} -- paste these to me.")
    for c in ("firstseen", "lastseen"):
        fl[c] = pd.to_datetime(fl[c], utc=True)

    arr = fl[fl["estarrivalairport"] == A].copy()
    dep = fl[fl["estdepartureairport"] == A].copy()
    print(f"  arrivals into {A}: {len(arr)} | departures from {A}: {len(dep)}")

    ta = pair_turnarounds(arr, dep, args.min_min, args.max_min)
    print(f"  plausible turnarounds [{args.min_min:.0f},{args.max_min:.0f}] min: {len(ta)}")
    if len(ta):
        print("\n  turnaround_min summary:")
        print(ta["turnaround_min"].describe(percentiles=[.1, .25, .5, .75, .9]).round(1).to_string())

    out = Path("pilot_out"); out.mkdir(exist_ok=True)
    tag = f"{A}_{args.start}_{args.days}d"
    ta.to_csv(out / f"turnarounds_{tag}.csv", index=False)
    try:
        import matplotlib; matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        if len(ta):
            ax = ta["turnaround_min"].plot(kind="hist", bins=40, color="#2b5d8a")
            ax.set_xlabel("Turnaround proxy (min)"); ax.set_ylabel("flights")
            ax.set_title(f"{A} turnarounds, {args.start} +{args.days}d (n={len(ta)})")
            plt.savefig(out / f"hist_{tag}.png", dpi=150, bbox_inches="tight")
    except Exception as e:
        print("  (plot skipped:", e, ")")

    print(f"\nWrote pilot_out/turnarounds_{tag}.csv")
    print("Sanity check: median should land roughly in a realistic 40-90 min band for a hub;")
    print("if it is wildly off or n is tiny, paste the console output back and we adjust before scaling.")


if __name__ == "__main__":
    main()
