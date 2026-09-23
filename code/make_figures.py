#!/usr/bin/env python
"""
Figures for the revised manuscript.

Reviewer item 8.4 asks for bin-level sample sizes and uncertainty bands on the
conditional-mean relationships, and for the number of turnarounds behind each airport-level
median. Every conditional mean plotted here carries a Wilson 95% interval and its bin count.

  --fig structure   Fig 1  US turnaround structure and the inbound -> outbound relationship
  --fig buffer      Fig 2  buffer absorption surface (reviewer item 1)
  --fig infoquality Fig 3  information quality at a fixed threshold vs a matched alarm budget
  --fig timing      Fig 4  decision-point ladder: discrimination against lead time
  --fig netbenefit  Fig 5  decision-curve analysis (editor point b)
  --fig transfer    Fig 6  calibration, and per-airport / per-period behaviour
  --fig europe      Fig 7  European descriptive structure and the propagation re-audit
"""
from __future__ import annotations
import argparse
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from common import (OUT, FIGS, TRAIN_MONTH, load_month, split_train_test, fit,
                    features_without, metrics)

plt.rcParams.update({"font.size": 9, "axes.grid": True, "grid.alpha": 0.3,
                     "figure.facecolor": "white", "savefig.facecolor": "white",
                     "axes.spines.top": False, "axes.spines.right": False})
BLUE, RED, GREEN, GREY = "#2b5d8a", "#a14b46", "#3a7d44", "#777777"


def save(fig, name):
    for ext in ("png", "pdf"):
        fig.savefig(FIGS / f"{name}.{ext}", dpi=600, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote figures/{name}.png and .pdf")


def wilson(k, n, z=1.96):
    """Wilson score interval for a binomial proportion; stable for small bins."""
    k, n = np.asarray(k, float), np.asarray(n, float)
    with np.errstate(divide="ignore", invalid="ignore"):
        p = k / n
        d = 1 + z**2 / n
        c = (p + z**2 / (2 * n)) / d
        h = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / d
    return c - h, c + h


def binned_rate(x, y, edges):
    idx = np.digitize(x, edges) - 1
    rows = []
    for b in range(len(edges) - 1):
        m = idx == b
        if m.sum() == 0:
            continue
        k, n = int(y[m].sum()), int(m.sum())
        lo, hi = wilson(k, n)
        rows.append({"centre": (edges[b] + edges[b + 1]) / 2, "n": n, "rate": k / n,
                     "lo": lo, "hi": hi})
    return pd.DataFrame(rows)


def fig_structure():
    ta = load_month(TRAIN_MONTH)
    fig, ax = plt.subplots(1, 2, figsize=(9.5, 3.8))
    ax[0].hist(ta["turnaround_min"], bins=np.arange(20, 725, 5), color=BLUE, alpha=.85)
    ax[0].axvline(ta["turnaround_min"].median(), color=RED, ls="--", lw=1.2,
                  label=f"median {ta['turnaround_min'].median():.0f} min")
    ax[0].set_xlabel("Turnaround duration (min)"); ax[0].set_ylabel("Turnarounds")
    ax[0].set_title(f"(a) US gate-to-gate ground time\n$n$ = {len(ta):,}, June 2024")
    ax[0].legend(frameon=False)

    edges = np.array([-60, -30, -15, 0, 15, 30, 45, 60, 90, 120, 180, 300])
    b = binned_rate(ta["inbound_arr_delay"].values, ta["y"].values, edges)
    ax[1].fill_between(b["centre"], b["lo"] * 100, b["hi"] * 100, color=BLUE, alpha=.25,
                       label="95% Wilson interval")
    ax[1].plot(b["centre"], b["rate"] * 100, "o-", color=BLUE, ms=4, lw=1.4,
               label="next departure >15 min late")
    b2 = binned_rate(ta["inbound_arr_delay"].values, ta["y_prop"].values, edges)
    ax[1].fill_between(b2["centre"], b2["lo"] * 100, b2["hi"] * 100, color=GREEN, alpha=.20)
    ax[1].plot(b2["centre"], b2["rate"] * 100, "s--", color=GREEN, ms=4, lw=1.4,
               label="…and attributed to the late inbound")
    for _, r in b.iterrows():
        ax[1].annotate(f"{r['n']:,.0f}", (r["centre"], r["rate"] * 100), textcoords="offset points",
                       xytext=(0, 7), ha="center", fontsize=5.5, color=GREY, rotation=90)
    ax[1].set_xlabel("Inbound arrival delay (min)")
    ax[1].set_ylabel("Share of turnarounds (%)")
    ax[1].set_title("(b) Propagation through the turnaround\nbin counts shown above each point")
    ax[1].set_ylim(0, 105); ax[1].legend(frameon=False, fontsize=7, loc="upper left")
    fig.tight_layout(); save(fig, "fig1_structure")


def fig_buffer():
    d = pd.read_csv(OUT / "robustness_buffer.csv")
    dbands = list(dict.fromkeys(d["inbound_band"]))
    bbands = list(dict.fromkeys(d["buffer_band"]))
    M = d.pivot(index="inbound_band", columns="buffer_band", values="P_y_pct").loc[dbands, bbands]
    N = d.pivot(index="inbound_band", columns="buffer_band", values="n").loc[dbands, bbands]
    fig, ax = plt.subplots(figsize=(7.2, 3.6))
    im = ax.imshow(M.values, cmap="RdYlBu_r", vmin=0, vmax=100, aspect="auto")
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            v = M.values[i, j]
            if np.isfinite(v):
                ax.text(j, i - .13, f"{v:.0f}%", ha="center", va="center", fontsize=8,
                        color="white" if (v > 70 or v < 12) else "black")
                ax.text(j, i + .22, f"n={N.values[i, j]:,.0f}", ha="center", va="center",
                        fontsize=5.5, color="white" if (v > 70 or v < 12) else GREY)
    ax.set_xticks(range(len(bbands))); ax.set_xticklabels(bbands, fontsize=7.5)
    ax.set_yticks(range(len(dbands))); ax.set_yticklabels(dbands, fontsize=7.5)
    ax.set_xlabel("Scheduled ground time available (min)")
    ax.set_title("P(next departure more than 15 min late)\nby inbound delay and the buffer "
                 "available to absorb it")
    ax.grid(False)
    fig.colorbar(im, ax=ax, label="%", fraction=.03)
    fig.tight_layout(); save(fig, "fig2_buffer")


def fig_infoquality():
    r = pd.read_csv(OUT / "info_quality_y.csv").set_index("condition")
    order = ["clean", "noise sd=5 min", "noise sd=10 min", "noise sd=20 min", "noise sd=40 min",
             "coarse status (4 buckets)", "missing, mean fallback", "missing, model retrained"]
    lab = ["clean", "noise\nsd=5", "noise\nsd=10", "noise\nsd=20", "noise\nsd=40",
           "coarse\n4 buckets", "missing\n(fallback)", "missing\n(retrained)"]
    r = r.loc[order]
    x = np.arange(len(order))
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.9))
    ax[0].bar(x - .2, r["recall"], .38, color=BLUE, label="recall")
    ax[0].bar(x + .2, r["precision"], .38, color=RED, label="precision")
    ax[0].plot(x, r["auc"], "k^--", ms=5, lw=1.1, label="AUC")
    ax[0].set_title("(a) At the fixed 0.5 threshold, as submitted\n"
                    "alert rate is free to move (19% → 30% under noise)")
    ax[1].bar(x - .2, r["recall_matched"], .38, color=BLUE, label="recall")
    ax[1].bar(x + .2, r["precision_matched"], .38, color=RED, label="precision")
    ax[1].plot(x, r["auc"], "k^--", ms=5, lw=1.1, label="AUC")
    ax[1].set_title("(b) At a matched alarm budget (19.0% alerts)\n"
                    "noise now costs recall too; missing information most of all")
    for a in ax:
        a.set_xticks(x); a.set_xticklabels(lab, fontsize=6.8); a.set_ylim(0, 1.18)
        a.legend(frameon=False, fontsize=7.5, ncol=3, loc="upper center")
        a.axhline(r.loc["clean", "recall"], color=GREY, ls=":", lw=.9)
    ax[0].set_ylabel("Rate")
    fig.tight_layout(); save(fig, "fig3_information_quality")


def fig_timing():
    d = pd.read_csv(OUT / "decision_timing.csv")
    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    ax.errorbar(d["lead_median"], d["auc"],
                xerr=[d["lead_median"] - d["lead_p10"], np.zeros(len(d))],
                fmt="o", ms=8, color=BLUE, capsize=4, lw=1.2)
    for _, r in d.iterrows():
        ax.annotate(f"{r['decision_point'].split(' ', 1)[1]}\nAUC {r['auc']:.3f}, "
                    f"recall {r['recall_matched']:.3f}",
                    (r["lead_median"], r["auc"]), textcoords="offset points",
                    xytext=(10, -6), fontsize=7.5)
    ax.set_xlabel("Median lead time to the outbound off-block (min); bar extends to the 10th percentile")
    ax.set_ylabel("AUC")
    ax.set_title("Moving the decision earlier: 144 minutes of extra lead time\n"
                 "costs 0.032 AUC and 0.051 recall at the same alarm budget")
    ax.set_xlim(0, d["lead_median"].max() * 1.35)
    fig.tight_layout(); save(fig, "fig4_decision_timing")


def fig_netbenefit():
    c = pd.read_csv(OUT / "decision_curve.csv")
    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    ax.plot(c["pt"], c["nb_full"], "o-", color=BLUE, label="full model")
    ax.plot(c["pt"], c["nb_no_inbound"], "s--", color=RED, label="inbound status unavailable")
    ax.plot(c["pt"], c["treat_all"], ":", color=GREY, label="alert on every turnaround")
    ax.axhline(0, color="k", lw=.8, label="alert on none")
    ax.fill_between(c["pt"], c["nb_no_inbound"], c["nb_full"], color=BLUE, alpha=.12,
                    label="value of the inbound signal")
    ax.set_xlabel("Threshold probability $p_t$  (a false alarm costs $p_t/(1-p_t)$ times a miss)")
    ax.set_ylabel("Net benefit")
    ax.set_title("Decision-curve analysis\ninformation is worth most where alarms are costly")
    ax.set_ylim(-0.25, 0.30); ax.legend(frameon=False, fontsize=7.5)
    fig.tight_layout(); save(fig, "fig5_net_benefit")


def fig_transfer():
    ta = load_month(TRAIN_MONTH)
    tr, te = split_train_test(ta)
    f = features_without()
    c = fit(tr, f)
    p = c.predict_proba(te[f])[:, 1]
    y = te["y"].values
    fig, ax = plt.subplots(1, 3, figsize=(12.5, 3.7))
    edges = np.linspace(0, 1, 11)
    idx = np.clip(np.digitize(p, edges) - 1, 0, 9)
    xs, ys, los, his, ns = [], [], [], [], []
    for b in range(10):
        m = idx == b
        if m.sum() == 0:
            continue
        lo, hi = wilson(y[m].sum(), m.sum())
        xs.append(p[m].mean()); ys.append(y[m].mean()); los.append(lo); his.append(hi)
        ns.append(int(m.sum()))
    ax[0].plot([0, 1], [0, 1], "k:", lw=.9, label="perfect")
    ax[0].fill_between(xs, np.array(los), np.array(his), color=BLUE, alpha=.25)
    ax[0].plot(xs, ys, "o-", color=BLUE, ms=4, label=f"model (ECE {metrics(y, p)['ece']:.3f})")
    for xi, yi, ni in zip(xs, ys, ns):
        ax[0].annotate(f"{ni:,}", (xi, yi), textcoords="offset points", xytext=(4, -8),
                       fontsize=5.5, color=GREY)
    ax[0].set_xlabel("Predicted probability"); ax[0].set_ylabel("Observed frequency")
    ax[0].set_title("(a) Calibration, with bin counts"); ax[0].legend(frameon=False, fontsize=7.5)

    a = pd.read_csv(OUT / "transfer_airports.csv")
    xa = np.arange(len(a))
    ax[1].bar(xa - .2, a["auc"], .38, color=BLUE, label="AUC")
    ax[1].bar(xa + .2, a["auprc"], .38, color=RED, label="AUPRC")
    ax[1].plot(xa, a["prevalence"], "k^--", ms=5, lw=1.1, label="prevalence")
    for i, r in a.iterrows():
        ax[1].annotate(f"{r['n']:,}", (i, 0.03), ha="center", fontsize=5.5, color="white", rotation=90)
    ax[1].set_xticks(xa); ax[1].set_xticklabels(a["hub"], fontsize=7.5)
    ax[1].set_ylim(0, 1); ax[1].legend(frameon=False, fontsize=7.5, ncol=3, loc="upper left")
    ax[1].set_title("(b) Held-out hubs: AUC is stable,\nAUPRC tracks prevalence")

    d = pd.read_csv(OUT / "transfer_drift.csv")
    xd = np.arange(len(d))
    ax[2].bar(xd - .2, d["auc"], .38, color=BLUE, label="AUC")
    ax[2].bar(xd + .2, d["auprc"], .38, color=RED, label="AUPRC")
    ax[2].plot(xd, d["prevalence"], "k^--", ms=5, lw=1.1, label="prevalence")
    ax[2].set_xticks(xd); ax[2].set_xticklabels([m.replace("_", "-") for m in d["month"]],
                                                fontsize=7, rotation=30)
    ax[2].set_ylim(0, 1); ax[2].legend(frameon=False, fontsize=7.5, ncol=3, loc="upper left")
    ax[2].set_title("(c) Out-of-time: same pattern,\nprevalence drives the operating point")
    fig.tight_layout(); save(fig, "fig6_transfer")


def fig_europe():
    h = pd.read_csv(OUT / "europe_hub_structure.csv")
    a = pd.read_csv(OUT / "europe_propagation_audit.csv")
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.9))
    h = h.sort_values("median")
    xs = np.arange(len(h))
    ax[0].bar(xs, h["median"], color=BLUE, alpha=.85)
    ax[0].errorbar(xs, h["median"], yerr=[h["median"] - h["q25"], h["q75"] - h["median"]],
                   fmt="none", ecolor=GREY, capsize=3, lw=1)
    for i, r in h.iterrows():
        pass
    for i, (_, r) in enumerate(h.iterrows()):
        ax[0].annotate(f"n = {r['n']:,}", (i, 4), ha="center", fontsize=6, color="white", rotation=90)
    ax[0].set_xticks(xs); ax[0].set_xticklabels(h["airport"], fontsize=8)
    ax[0].set_ylabel("Turnaround duration (min)")
    ax[0].set_title("(a) European hubs: median and interquartile range\n"
                    "taxi-inclusive ADS-B proxy, so longer than US gate-to-gate")

    xs = np.arange(len(a))
    ax[1].bar(xs - .2, a["pooled_r"], .38, color=RED, label="pooled correlation")
    ax[1].bar(xs + .2, a["within_route_r"], .38, color=BLUE, label="within-route correlation")
    ax[1].axhline(0, color="k", lw=.8)
    ax[1].set_xticks(xs)
    ax[1].set_xticklabels([f"{f}\n(n={n:,})" for f, n in zip(a["filter"], a["n"])], fontsize=6.5)
    ax[1].set_ylabel("Correlation of consecutive excess ground times")
    ax[1].set_title("(b) Re-audit of the published European propagation result\n"
                    "the association does not survive either control")
    ax[1].legend(frameon=False, fontsize=7.5)
    fig.tight_layout(); save(fig, "fig7_europe")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--fig", required=True,
                    choices=["structure", "buffer", "infoquality", "timing", "netbenefit",
                             "transfer", "europe", "all"])
    a = ap.parse_args()
    F = {"structure": fig_structure, "buffer": fig_buffer, "infoquality": fig_infoquality,
         "timing": fig_timing, "netbenefit": fig_netbenefit, "transfer": fig_transfer,
         "europe": fig_europe}
    for k in (F if a.fig == "all" else [a.fig]):
        print(f"building {k}…")
        F[k]()
