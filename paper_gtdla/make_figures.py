"""Figures of the GT&D Latin America 2027 paper, as PDF in paper_gtdla/figures. Run from the repository root after
paper_gtdla/experiments.py.

    python paper_gtdla/make_figures.py              # every figure of the paper
    python paper_gtdla/make_figures.py problem      # Fig. 1, Tropical Storm Isaias as EAGLE-I recorded it
    python paper_gtdla/make_figures.py validation   # Fig. 3, observed, fitted and held-out restoration of six storms
    python paper_gtdla/make_figures.py isaias_maps  # Fig. 4, Isaias by county, observed and simulated
    python paper_gtdla/make_figures.py demos        # Fig. 5, gust what-if, crews per 1,000 customers out, aid lag

Fig. 2 is drawn in TikZ inside main.tex. `explore` and `study_area` draw two README images (scripts/make_figures.py).

Palette: navy for the simulator, red tile for observed data, gray for alternatives and baselines.
"""
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))

from stormcrews import Simulator  # noqa: E402
from stormcrews.rules import CurrentPractice  # noqa: E402
from experiments import ANCHORS, CAL, _with_cal, base  # noqa: E402

FIG = HERE / "figures"; FIG.mkdir(exist_ok=True)
EXT = "pdf"                      # scripts/make_figures.py sets FIG and EXT to render the README images
RES = HERE / "results"
NAVY, RED, GRAY, INK = "#1E2761", "#B03A2E", "#6B7280", "#1F2933"
LIGHT_RED = "#F3DEDA"
VALIDATION = (("ne", 10, "NE, Isaias, Aug. 2020"), ("ne", 0, "NE, Riley, Mar. 2018"), ("ne", 23, "NE, Dec. 2022"),
              ("pawv", 45, "PA-WV, Apr. 2025"), ("pawv", 0, "PA-WV, Riley, Mar. 2018"), ("pawv", 11, "PA-WV, Jun. 2020"))


def style():
    plt.rcParams.update({
        "font.family": "serif", "font.serif": ["Times New Roman", "STIXGeneral"], "mathtext.fontset": "stix",
        "font.size": 7, "axes.titlesize": 7, "axes.titleweight": "normal", "axes.titlelocation": "left",
        "axes.labelsize": 7, "axes.labelcolor": INK, "axes.edgecolor": "#9AA1AC", "axes.linewidth": 0.5,
        "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": "#E6E8EB",
        "grid.linewidth": 0.4, "xtick.color": INK, "ytick.color": INK, "xtick.labelsize": 6.5,
        "ytick.labelsize": 6.5, "xtick.major.width": 0.5, "ytick.major.width": 0.5, "xtick.major.size": 2,
        "ytick.major.size": 2, "legend.frameon": False, "text.color": INK, "savefig.dpi": 300, "pdf.fonttype": 42,
        "savefig.bbox": "tight", "savefig.pad_inches": 0.02, "figure.facecolor": "white"})


def validation():
    """Observed EAGLE-I customers out (red), current practice with the storm's own anchored parameters (navy) and with
    the repair time and dispersion predicted from the other storms (gray, dashed), seed 0."""
    style()
    par = pd.read_csv(RES / "holdout_parameters.csv").set_index(["region", "storm"])
    fig, axes = plt.subplots(1, 6, figsize=(7.16, 1.38))
    for i, (ax, (reg, sid, title)) in enumerate(zip(axes, VALIDATION)):
        sim = Simulator(base(reg)); st = sim.region.storm(sid)
        fit = sim.run(CurrentPractice(), sid, seed=0)
        p = par.loc[(reg, sid)]
        held = sim.run(CurrentPractice(), _with_cal(st, repair_mean_h=p["repair_pred_h"], size_sigma=p["sigma_pred"]),
                       seed=0)
        obs = st.observed / 1e3; x = np.arange(len(obs)) + 1
        H = int(len(obs) * 1.25)
        ax.plot(x, obs, color=RED, lw=0.75, drawstyle="steps-pre")
        c = fit.curve()[:H] / 1e3; ax.plot(np.arange(len(c)) + 1, c, color=NAVY, lw=0.75)
        h = held.curve()[:H] / 1e3; ax.plot(np.arange(len(h)) + 1, h, color=GRAY, lw=0.7, ls=(0, (2.5, 1.2)))
        ax.set_xlim(0, H); ax.set_ylim(0, None)
        ax.set_title(title, fontsize=6.5)
        ax.set_xlabel("hours", fontsize=6.3, labelpad=1)
        if i == 0:
            k = int(np.argmax(obs))
            ax.annotate("observed", (k + 2, obs[k]), xytext=(k + 30, obs[k] * 0.98), color=RED, fontsize=6,
                        va="center", arrowprops=dict(arrowstyle="-", color=RED, lw=0.4))
            j = int(len(c) * 0.42)
            ax.annotate("fitted", (j, c[j]), xytext=(j + 35, c[j] + obs.max() * 0.2), color=NAVY, fontsize=6,
                        arrowprops=dict(arrowstyle="-", color=NAVY, lw=0.4))
            j2 = int(len(h) * 0.55)
            ax.annotate("held out", (j2, h[j2]), xytext=(j2 + 25, h[j2] + obs.max() * 0.32), color=GRAY,
                        fontsize=6, arrowprops=dict(arrowstyle="-", color=GRAY, lw=0.4))
    axes[0].set_ylabel("customers out (thousands)", fontsize=6.5)
    fig.tight_layout(w_pad=0.35)
    fig.savefig(FIG / f"validation.{EXT}"); plt.close(fig)


def _totals(df, by):
    """Sum over the storms of each seed, then the mean over seeds."""
    cols = ["saidi_min", "customers_interrupted", "over_72h", "local_crew_hours", "aid_crew_hours"]
    return df.groupby(by + ["seed"])[cols].sum().groupby(level=list(range(len(by)))).mean()


def demos():
    style()
    fig, axes = plt.subplots(1, 3, figsize=(7.16, 2.0))
    # (a) gust what-if: SAIDI contribution and interruptions relative to the real storms
    g = _totals(pd.read_csv(RES / "gust.csv"), ["gust_change"])
    x = g.index.to_numpy(float); s = g["saidi_min"] / g.loc[0.0, "saidi_min"]
    n = g["customers_interrupted"] / g.loc[0.0, "customers_interrupted"]
    ax = axes[0]
    ax.plot(x, s, color=NAVY, lw=0.8, marker="o", ms=1.8)
    ax.plot(x, n, color=GRAY, lw=0.7, marker="o", ms=1.6)
    ax.set_yscale("log"); ax.minorticks_off()
    ax.set_yticks([0.5, 1, 2, 5, 10], ["0.5", "1", "2", "5", "10"])
    ax.axvline(0, color="#C4C9D1", lw=0.5, zorder=0)
    ax.text(x[-1] + 0.6, s.iloc[-1], "SAIDI\ncontribution", color=NAVY, fontsize=6, ha="left", va="center")
    ax.text(x[-1] + 0.6, n.iloc[-1], "customers\ninterrupted", color=GRAY, fontsize=6, ha="left", va="center")
    ax.set_xlim(-6, 16.5); ax.set_xticks([-5, 0, 5, 10])
    ax.set_xlabel("change in peak gust, every county (m/s)")
    ax.set_ylabel("relative to the real storms")
    ax.set_title("(a) Weather what-if")
    # (b) crews per 1,000 customers out at the peak
    c = pd.read_csv(RES / "crews.csv")
    f = _totals(c[c["sweep"] == "fleet"], ["crews_per_1000"])
    ax = axes[1]
    ax.plot(f.index, f["saidi_min"], color=NAVY, lw=0.8, marker="o", ms=1.8)
    lo, hi = min(ANCHORS.values()), max(ANCHORS.values())
    for v in ANCHORS.values():
        ax.plot([v, v], [0, 0.04], transform=ax.get_xaxis_transform(), color=GRAY, lw=0.8, clip_on=False)
    ax.text(lo + 0.05, 0.07, "public filings", transform=ax.get_xaxis_transform(), color=GRAY, fontsize=6)
    ax.set_xlim(lo - 0.2, hi + 0.2)
    sq = f.loc[4.0, "saidi_min"]
    ax.annotate("anchored fleet", (4.0, sq), xytext=(4.25, sq + 900), fontsize=6, color=NAVY,
                arrowprops=dict(arrowstyle="-", color=NAVY, lw=0.4))
    ax.set_xlabel("crews per 1,000 customers out at the peak")
    ax.set_ylabel("SAIDI contribution (min)")
    ax.set_ylim(0, None)
    ax.set_title("(b) Crew resources")
    # (c) mutual-aid lag
    lag = pd.concat([c[c["sweep"] == "lag"], c[(c["sweep"] == "fleet") & (c["fleet_factor"] == 1.0)]])
    L = _totals(lag, ["aid_lead_h"])
    ax = axes[2]
    ax.plot(L.index, L["saidi_min"], color=NAVY, lw=0.8, marker="o", ms=1.8)
    ax.annotate("24 h, default", (24, L.loc[24.0, "saidi_min"]), xytext=(30, L.loc[24.0, "saidi_min"] - 170),
                fontsize=6, color=NAVY, arrowprops=dict(arrowstyle="-", color=NAVY, lw=0.4))
    ax.set_xticks([12, 24, 36, 48, 72])
    ax.set_xlabel("hours from the aid request to arrival")
    ax.set_ylabel("SAIDI contribution (min)")
    ax.set_ylim(L["saidi_min"].min() * 0.9, L["saidi_min"].max() * 1.05)
    ax.set_title("(c) Mutual-aid lag")
    fig.tight_layout(w_pad=1.0)
    fig.savefig(FIG / f"demos.{EXT}"); plt.close(fig)


def explore():
    """The interactive explorer's figure (stormcrews.explore, as notebooks/explore.ipynb draws it) for Isaias under the
    anchored calibration: current practice pinned as the baseline, then three quarters of the local fleet."""
    from stormcrews import explore as E
    style()
    E._rc = lambda: plt.rc_context({"font.size": 9, "axes.titlesize": 9.5, "xtick.labelsize": 8.5,
                                    "ytick.labelsize": 8.5, "axes.labelsize": 9, "axes.grid": True})
    base_ = E.run_case(E.default_settings(calibration="anchored"))
    run_ = E.run_case(E.default_settings(calibration="anchored", fleet_factor=0.75))
    fig = E.storm_figure(run_, base_, label="75 % local fleet", baseline_label="pinned: current practice")
    fig.set_size_inches(9.6, 2.6)
    fig.tight_layout(w_pad=1.0)
    fig.savefig(FIG / f"explore.{EXT}"); plt.close(fig)


def study_area():
    """The two regions, shaded by observed customer-hours per customer over all storms (the existing map style)."""
    from stormcrews.core.data import region
    from stormcrews.visuals import plots as P
    style()
    regs = [region("ne"), region("pawv")]
    vals = [sum(R.storm(i).observed_county for i in R.storm_ids) / R.customers for R in regs]
    vmax = float(np.ceil(max(np.nanmax(v) for v in vals) / 10) * 10)
    fig, axes = plt.subplots(1, 2, figsize=(3.5, 1.75), gridspec_kw={"width_ratios": [1, 1.45]})
    for ax, R, v in zip(axes, regs, vals):
        sm = P.draw_counties(ax, R, v, P.OUTAGE, vmax=vmax, state_labels=True, label_size=5.5)
        for t in ax.texts:
            t.set_fontweight("normal")
    fig.tight_layout(rect=(0, 0.17, 1, 1), w_pad=0.1)
    cax = fig.add_axes([0.2, 0.12, 0.6, 0.035])
    cb = fig.colorbar(sm, cax=cax, orientation="horizontal"); cb.outline.set_visible(False); cb.ax.grid(False)
    cb.ax.tick_params(labelsize=5.5, length=2)
    cb.set_label("observed customer-hours per customer, all storms", fontsize=6)
    fig.savefig(FIG / f"study_area.{EXT}"); plt.close(fig)


def isaias_facts():
    """Observed numbers of Tropical Storm Isaias used by the problem figure and the macros."""
    from stormcrews.core.data import region
    R = region("ne"); st = R.storm(10)
    obs = st.observed; pk = int(np.argmax(obs))
    cum = np.cumsum(st.rises.sum(1))
    res = Simulator(base("ne")).run(CurrentPractice(), 10, seed=1)
    return {"obs": obs, "cum": cum, "peak_h": pk, "peak": float(obs[pk]),
            "after_peak_share": float(1 - cum[pk] / cum[-1]),
            "damage90_h": int(np.searchsorted(cum, 0.9 * cum[-1])),
            "out": {d: float(obs[pk + d]) for d in (24, 72, 96)},
            "saidi_obs": float(60 * obs.sum() / R.customers.sum()), "customers": float(R.customers.sum()),
            "aid_request_h": float(res.outcome["aid_requested_at"]), "aid_arrival_h": float(res.outcome["aid_requested_at"]) + 24.0}


def problem():
    """Tropical Storm Isaias as the control room lived it: customers out (observed), the customers who had lost power
    so far (observed damage arrivals), the long tail after the peak and when the usual mutual aid arrives."""
    style()
    f = isaias_facts(); obs = f["obs"] / 1e3; cum = f["cum"] / 1e3; pk = f["peak_h"]
    t = np.arange(len(obs)) + 1 - pk                    # hours from the peak of customers out
    fig, ax = plt.subplots(figsize=(3.5, 1.7))
    ax.plot(t, cum, color=GRAY, lw=0.8, drawstyle="steps-pre")
    ax.plot(t, obs, color=RED, lw=0.9, drawstyle="steps-pre")
    ax.text(t[-1] - 2, cum[-1] + 45, "customers who lost power so far", color=GRAY, fontsize=6, ha="right", va="bottom")
    ax.text(46, obs[pk + 46] + 90, "customers out", color=RED, fontsize=6, ha="left")
    for d in (24, 72, 96):
        y = f["out"][d] / 1e3
        ax.plot([d, d], [0, y], color=RED, lw=0.4, ls=(0, (1.5, 1.5)))
        ax.plot(d, y, "o", color=RED, ms=2)
        ax.text(d + 2, y + 25, f"{y:,.0f} k\nat {d} h", color=RED, fontsize=5.6, ha="left", va="bottom")
    a = f["aid_arrival_h"] - pk
    ax.axvline(a, color=NAVY, lw=0.6, ls=(0, (3, 1.5)))
    ax.text(a + 2, 1460, "usual mutual aid\narrives (model)", color=NAVY, fontsize=5.6, ha="left", va="top")
    ax.set_xlim(t[0] - 1, t[-1] + 3); ax.set_ylim(0, 1950)
    ax.set_xlabel("hours from the peak of customers out")
    ax.set_ylabel("customers (thousands)")
    ax.set_xticks([-48, -24, 0, 24, 48, 72, 96, 120])
    fig.tight_layout()
    fig.savefig(FIG / f"problem.{EXT}"); plt.close(fig)


def isaias_maps():
    """Tropical Storm Isaias by county: observed customer-hours per customer, the same simulated under current practice
    (anchored calibration, seed 1) and the simulated change with peak gusts 5 m/s stronger (existing map style)."""
    from stormcrews.visuals import plots as P
    style()
    sim = Simulator(base("ne"))
    real = sim.run(CurrentPractice(), 10, seed=1)
    gust = Simulator(base("ne", what_if={"gust_change": 5.0})).run(CurrentPractice(), 10, seed=1)
    R = sim.region; st = real.storm
    obs = st.observed_county / R.customers
    s0 = P.customer_hours_per_customer(real); d = P.customer_hours_per_customer(gust) - s0
    vmax = float(np.ceil(max(np.nanmax(obs), np.nanmax(s0)) / 10) * 10)
    fig, axes = plt.subplots(1, 3, figsize=(7.16, 1.95))
    for ax, v, t in ((axes[0], obs, "Observed (EAGLE-I)"), (axes[1], s0, "Simulated, current practice")):
        sm = P.draw_counties(ax, R, v, P.DURATION, vmax=vmax, state_labels=True, label_size=5.5)
        ax.set_title(t, fontsize=7, loc="left")
    sm2 = P.draw_counties(axes[2], R, d, P.OUTAGE, vmax=float(np.ceil(np.nanmax(d) / 10) * 10), state_labels=True,
                          label_size=5.5)
    axes[2].set_title("Simulated change with +5 m/s gust", fontsize=7, loc="left")
    for ax in axes:
        for t in ax.texts:
            t.set_fontweight("normal")
    fig.tight_layout(rect=(0, 0.14, 1, 1), w_pad=0.2)
    for sm_, (x0, w), lab in ((sm, (0.12, 0.45), "customer-hours per customer"),
                              (sm2, (0.71, 0.24), "change in customer-hours per customer")):
        cax = fig.add_axes([x0, 0.1, w, 0.03])
        cb = fig.colorbar(sm_, cax=cax, orientation="horizontal"); cb.outline.set_visible(False); cb.ax.grid(False)
        cb.ax.tick_params(labelsize=5.5, length=2); cb.set_label(lab, fontsize=6)
    fig.savefig(FIG / f"isaias_maps.{EXT}"); plt.close(fig)


if __name__ == "__main__":
    for name in sys.argv[1:] or ["problem", "validation", "isaias_maps", "demos"]:
        globals()[name]()
    print("figures in", FIG)
