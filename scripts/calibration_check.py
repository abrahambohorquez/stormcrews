"""How well current practice in StormCrews reproduces every observed storm: customer-hours, peak and time to restore half
and nine tenths of the peak, for both regions and the three calibrations.

    python scripts/calibration_check.py            # writes results/calibration_check.csv and results/calibration_check.png
"""
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from stormcrews import Simulator  # noqa: E402
from stormcrews.rules import CurrentPractice  # noqa: E402


def shape(c, H):
    """Hours from the peak until the curve falls to one half and to one tenth of the peak, inside the storm window."""
    c = np.asarray(c[:H], float); p = int(np.argmax(c)); top = c[p]
    out = []
    for lvl in (0.5, 0.1):
        below = np.flatnonzero(c[p:] <= lvl * top)
        out.append(float(below[0]) if len(below) else float(H - p))
    return out


def one(args):
    region, storm, cal, seed = args
    res = Simulator(region=region, calibration=cal).run(CurrentPractice(), storm, seed)
    st = res.storm; H = st.hours
    sim = res.curve()
    t50o, t90o = shape(st.observed, H); t50s, t90s = shape(sim, H)
    return {"region": region, "storm": storm, "name": st.name, "calibration": cal, "seed": seed,
            "observed_customer_hours": st.observed.sum(), "simulated_customer_hours": res.metrics["customer_hours"],
            "observed_peak": st.observed.max(), "simulated_peak": sim.max(), "t50_observed": t50o, "t50_simulated": t50s,
            "t90_observed": t90o, "t90_simulated": t90s, "fleet": res.outcome["crews_local"]}


def main():
    jobs = []
    for region in ("ne", "pawv"):
        ids = Simulator(region=region).storm_ids
        jobs += [(region, s, cal, 0) for s in ids for cal in ("shape", "equal", "anchored")]
    workers = int(sys.argv[1]) if len(sys.argv) > 1 else 6
    with ProcessPoolExecutor(max_workers=min(workers, 6)) as ex:
        df = pd.DataFrame(list(ex.map(one, jobs, chunksize=4)))
    (ROOT / "results").mkdir(exist_ok=True)
    df.to_csv(ROOT / "results" / "calibration_check.csv", index=False)
    df["chi_ratio"] = df["simulated_customer_hours"] / df["observed_customer_hours"]
    df["peak_ratio"] = df["simulated_peak"] / df["observed_peak"]
    g = df.groupby(["region", "calibration"]).agg(
        storms=("storm", "size"), chi_ratio_median=("chi_ratio", "median"),
        chi_within_10pct=("chi_ratio", lambda x: float((abs(x - 1) <= 0.1).mean())),
        peak_ratio_median=("peak_ratio", "median"),
        t50_abs_error_h=("t50_simulated", lambda x: float(np.median(abs(x - df.loc[x.index, "t50_observed"])))),
        t90_abs_error_h=("t90_simulated", lambda x: float(np.median(abs(x - df.loc[x.index, "t90_observed"])))))
    print(g.round(3).to_string())
    g.round(3).to_csv(ROOT / "results" / "calibration_summary.csv")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from stormcrews.visuals.plots import GRAY, NAVY, style
    style()
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))
    d = df[df["calibration"] == "shape"]
    for region, c, lab in (("ne", NAVY, "New England"), ("pawv", GRAY, "Pennsylvania and West Virginia")):
        x = d[d["region"] == region]
        axes[0].scatter(x["observed_customer_hours"] / 1e6, x["simulated_customer_hours"] / 1e6, s=28, color=c,
                        edgecolor="white", linewidth=0.6, label=lab, zorder=3)
        axes[1].scatter(x["t90_observed"], x["t90_simulated"], s=28, color=c, edgecolor="white", linewidth=0.6,
                        label=lab, zorder=3)
    lim = [0.5, 120]
    axes[0].plot(lim, lim, color="#9AA1AC", lw=1); axes[0].set_xscale("log"); axes[0].set_yscale("log")
    axes[0].set_xlim(lim); axes[0].set_ylim(lim)
    for f in (axes[0].set_xticks, axes[0].set_yticks):
        f([1, 3, 10, 30, 100], ["1", "3", "10", "30", "100"])
    axes[0].minorticks_off()
    axes[0].set_xlabel("observed customer-hours (M)"); axes[0].set_ylabel("simulated, current practice (M)")
    axes[0].set_title("Customer-hours of every storm")
    m = float(max(d["t90_observed"].max(), d["t90_simulated"].max())) + 5
    axes[1].plot([0, m], [0, m], color="#9AA1AC", lw=1); axes[1].set_xlim(0, m); axes[1].set_ylim(0, m)
    axes[1].set_xlabel("observed hours from peak to 10 % of it"); axes[1].set_ylabel("simulated")
    axes[1].set_title("Length of the restoration tail")
    axes[0].legend(loc="upper left")
    fig.suptitle("Calibration check: current practice against the observed storms (shape calibration, seed 0)",
                 x=0.01, ha="left", fontsize=12, fontweight="bold")
    fig.tight_layout(); fig.savefig(ROOT / "results" / "calibration_check.png"); plt.close(fig)


if __name__ == "__main__":
    main()
