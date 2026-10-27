"""What-if sweep: every New England storm with its peak gusts changed from -5 to +10 m/s, under several rules.

    python whatif_sweep.py                      # gust sweep, 3 rules, 39 storms, 2 seeds, 6 processes
    python whatif_sweep.py --knob precip_factor --values 0.5 1 1.5 2

The damage of each real storm is changed through the damage model (stormcrews/damage.py) and the restoration is
simulated; the value 0 (or 1 for factors) is the real storm. Writes results/whatif_<knob>.csv and .png.
"""
import argparse
import time
from pathlib import Path

import numpy as np

from stormcrews import Scenario, run_experiment
from stormcrews.rules import CurrentPractice, EarlyAid, ForecastWeighted

RULES = {"current practice": CurrentPractice(), "forecast weighted": ForecastWeighted(),
         "early aid x1.5": EarlyAid(1.5, 24.0)}
LABELS = {"gust_change": "change in peak gust in every county (m/s)", "precip_factor": "precipitation factor",
          "soil_moisture_change": "change in soil moisture (m3/m3)", "duration_factor": "duration factor",
          "customer_growth": "customer growth (share)", "footprint_scale": "footprint scale"}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--knob", default="gust_change")
    ap.add_argument("--values", type=float, nargs="+", default=[-5, -2.5, 0, 2.5, 5, 7.5, 10])
    ap.add_argument("--scenario", default="new_england")
    ap.add_argument("--seeds", type=int, default=2)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--out", default="results")
    a = ap.parse_args()
    base = Scenario.load(a.scenario)
    scs = {f"{v:g}": base.with_(what_if={a.knob: v}) for v in a.values}
    t0 = time.time()
    exp = run_experiment(RULES, scs, seeds=range(a.seeds), workers=min(a.workers, 6))
    s = exp.summary(); s["value"] = s["scenario"].astype(float)
    out = Path(a.out); out.mkdir(exist_ok=True)
    s.to_csv(out / f"whatif_{a.knob}.csv", index=False)
    print(f"{len(exp.table)} runs in {time.time() - t0:.0f} s")
    print(s[["value", "rule", "customers_interrupted", "saidi_min", "over_72h", "worst_county_h"]].to_string(index=False))
    figure(s, a.knob, out / f"whatif_{a.knob}.png", int(s["storms"].iloc[0]), a.seeds)


def replot(knob="gust_change", out="results", seeds=2):
    import pandas as pd
    s = pd.read_csv(Path(out) / f"whatif_{knob}.csv")
    figure(s, knob, Path(out) / f"whatif_{knob}.png", int(s["storms"].iloc[0]), seeds)


def figure(s, knob, path, storms, seeds, size=(12, 3.6)):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from stormcrews.visuals.plots import color_for, style
    style()
    fig, axes = plt.subplots(1, 3, figsize=size)
    panels = (("saidi_min", 1, "SAIDI contribution, sum over storms (min)"), ("over_72h", 1e6, "customers out > 72 h (M)"),
              ("worst_county_h", 1, "worst county, mean outage (h, mean over storms)"))
    real = 1.0 if "factor" in knob or "scale" in knob else 0.0
    for ax, (m, sc, lab) in zip(axes, panels):
        for rule, g in s.groupby("rule", sort=False):
            g = g.sort_values("value")
            ax.plot(g["value"], g[m] / sc, color=color_for(rule), lw=2, marker="o", ms=3.5, label=f"simulated: {rule}",
                    ls=(0, (4, 2)) if rule == "current practice" else "-")
        ax.axvline(real, color="#9AA1AC", lw=1, zorder=0)
        ax.set_title(lab); ax.set_xlabel(LABELS.get(knob, knob))
        if m == "saidi_min":
            ax.set_yscale("log"); ax.minorticks_off()
            ticks = [t for t in (1000, 3000, 10000, 30000, 100000) if g[m].min() / sc / 1.5 <= t <= s[m].max() / sc * 1.5]
            ax.set_yticks(ticks, [f"{t:,g}" for t in ticks])
    axes[0].text(real, axes[0].get_ylim()[1], " real storms", color="#6B7280", fontsize=8, va="top")
    axes[0].legend(loc="lower right", fontsize=8)
    fig.suptitle(f"What-if: New England, {storms} storms, {seeds} seeds, all values simulated", x=0.01, ha="left",
                 fontsize=11, fontweight="bold")
    fig.tight_layout(); fig.savefig(path); plt.close(fig)


if __name__ == "__main__":
    main()
