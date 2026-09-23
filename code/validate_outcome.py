#!/usr/bin/env python
"""
Reviewer item 1: is the modelled outcome really delay PROPAGATION?

The originally submitted paper labelled `next departure more than 15 minutes late` as
delay propagation without external validation. BTS publishes, for every flight that
arrives 15 or more minutes late, a five-way causal decomposition of that arrival delay,
one component of which -- LateAircraftDelay -- is the carrier's own attribution of the
delay to a late inbound aircraft. That is the closest thing to ground truth for
propagation available in public data, and we use it here to do three things:

  (a) establish exactly what the causal fields decompose and when they are populated,
      so their coverage limits are stated rather than assumed;
  (b) measure how much of the primary outcome is genuinely attributable to the inbound
      aircraft, unconditionally and conditional on the inbound having been late;
  (c) define a propagation-specific secondary outcome y_prop, on which the paper's
      information-quality results are then reproduced.

Outputs: out/validation_outcome.csv, out/validation_outcome.txt
"""
from __future__ import annotations
import numpy as np, pandas as pd
from common import DATA, OUT, MONTHS, TRAIN_MONTH, load_month

CAUSES = ["CarrierDelay", "WeatherDelay", "NASDelay", "SecurityDelay", "LateAircraftDelay"]
OB = ["ob_" + c for c in CAUSES]
L = []


def say(s=""):
    print(s)
    L.append(str(s))


def main():
    ta = load_month(TRAIN_MONTH)
    for c in OB:
        ta[c] = pd.to_numeric(ta[c], errors="coerce")
    say(f"BTS {TRAIN_MONTH}: {len(ta):,} turnarounds, "
        f"{ta['date'].min().date()} to {ta['date'].max().date()}")

    # ---- (a) what the causal fields decompose, and when they exist ----------------
    say("\n(a) COVERAGE AND MEANING OF THE BTS CAUSAL FIELDS")
    pop = ta[OB].notna().all(axis=1)
    say(f"  outbound legs with a causal decomposition : {pop.sum():,} of {len(ta):,} "
        f"({pop.mean()*100:.1f}%)")
    d15 = pd.to_numeric(ta["outbound_arr_del15"], errors="coerce")
    say(f"  populated when the outbound ARRIVED >=15 late : {pop[d15 == 1].mean()*100:.2f}% "
        f"(n={(d15 == 1).sum():,})")
    say(f"  populated when it arrived on time            : {pop[d15 == 0].mean()*100:.2f}% "
        f"(n={(d15 == 0).sum():,})")
    say("  => the decomposition explains ARRIVAL delay and exists only for late ARRIVALS.")

    # ---- (b) how much of the primary outcome is really propagation ----------------
    say("\n(b) HOW MUCH OF THE PRIMARY OUTCOME IS ATTRIBUTABLE TO THE INBOUND AIRCRAFT")
    y1 = ta["y"] == 1
    say(f"  primary outcome y=1 (next departure >15 late): {y1.sum():,} ({y1.mean()*100:.1f}%)")
    cov = pop[y1].mean()
    say(f"  of which a causal decomposition exists for   : {cov*100:.1f}%")
    say(f"  the remaining {(1-cov)*100:.1f}% are, by construction, departures that were late but")
    say("  recovered en route and arrived on time -- the cases with no downstream consequence.")
    z = ta[y1 & pop]
    lac = z["ob_LateAircraftDelay"]
    rows = [{"statistic": "y=1 with attribution", "value": len(z)},
            {"statistic": "LateAircraftDelay > 0 (%)", "value": (lac > 0).mean() * 100},
            {"statistic": "LateAircraftDelay >= 15 (%)", "value": (lac >= 15).mean() * 100}]
    say(f"  LateAircraftDelay > 0   : {(lac > 0).mean()*100:.1f}%")
    say(f"  LateAircraftDelay >= 15 : {(lac >= 15).mean()*100:.1f}%")
    tot = z[OB].sum()
    say("  share of attributed delay-minutes, by cause:")
    for c in CAUSES:
        v = tot["ob_" + c] / tot.sum() * 100
        say(f"    {c:<20}{v:6.1f}%")
        rows.append({"statistic": f"share of delay-minutes: {c} (%)", "value": v})
    dom = z[OB].idxmax(axis=1).str.replace("ob_", "", regex=False)
    say("  dominant (largest) component, share of y=1 flights:")
    for k, v in dom.value_counts(normalize=True).mul(100).items():
        say(f"    {k:<20}{v:6.1f}%")
        rows.append({"statistic": f"dominant cause: {k} (%)", "value": v})

    say("\n  conditional on the inbound aircraft's own lateness:")
    for lab, m in [("inbound >15 late", z["inbound_arr_delay"] > 15),
                   ("inbound 0-15 late", z["inbound_arr_delay"].between(0, 15, inclusive="left")),
                   ("inbound early/on time", z["inbound_arr_delay"] < 0)]:
        if m.sum():
            v = (z.loc[m, "ob_LateAircraftDelay"] > 0).mean() * 100
            say(f"    {lab:<22} n={m.sum():>7,}   LateAircraftDelay>0 in {v:5.1f}%")
            rows.append({"statistic": f"LateAircraftDelay>0 | {lab} (%)", "value": v})

    # ---- (c) the propagation-specific secondary outcome ---------------------------
    say("\n(c) PROPAGATION-SPECIFIC SECONDARY OUTCOME")
    say("  y_prop = 1 iff the next departure was >15 min late AND the carrier attributed")
    say("  >=15 min of the resulting arrival delay to a late inbound aircraft.")
    for tag in MONTHS:
        t = load_month(tag)
        say(f"    {tag}: n={len(t):,}  prevalence y={t['y'].mean()*100:5.2f}%  "
            f"y_prop={t['y_prop'].mean()*100:5.2f}%  "
            f"(y_prop / y = {t['y_prop'].sum()/max(t['y'].sum(),1)*100:4.1f}%)")
        rows.append({"statistic": f"prevalence y_prop {tag} (%)", "value": t["y_prop"].mean() * 100})
        del t
    say("\n  y_prop is a strict subset of y, so it is conservative: every y_prop case is one")
    say("  the carrier itself recorded as caused by the inbound aircraft.")

    pd.DataFrame(rows).to_csv(OUT / "validation_outcome.csv", index=False)
    (OUT / "validation_outcome.txt").write_text("\n".join(L), encoding="utf-8")
    say(f"\nwrote {OUT/'validation_outcome.csv'} and .txt")


if __name__ == "__main__":
    main()
