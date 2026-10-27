"""Figures for StormCrews: a frame of a storm in progress, restoration curves, county maps and rule comparisons.

House style: navy #1E2761 for the simulator, red tile #B03A2E for observed data, gray #6B7280 for alternatives and
baselines; thin lines and no chart junk. Maps keep their navy and red ramps.
"""
from functools import lru_cache

import matplotlib

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

from ..core.data import DATA

NAVY, RED, GRAY, LIGHT = "#1E2761", "#B03A2E", "#6B7280", "#EEF0F3"
GOLD, PURPLE, INK = "#B8862B", "#7A5195", "#1F2933"
OUTAGE = LinearSegmentedColormap.from_list("outage", ["#F7F7F8", "#E9C4BE", "#C9705F", RED, "#6E1F17"])
DURATION = LinearSegmentedColormap.from_list("duration", ["#F4F6F9", "#B9C6DA", "#5D7AA6", "#16305B", "#0A1830"])
RULE_COLORS = {"current practice": GRAY, "forecast weighted": GOLD, "early aid, peak error": "#6F8FC0",
               "early aid": NAVY, "observed": INK}


def color_for(name):
    for k, c in RULE_COLORS.items():
        if name.lower().startswith(k.lower()):
            return c
    return PURPLE


def style():
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 10, "axes.titlesize": 11, "axes.titleweight": "bold",
        "axes.titlelocation": "left", "axes.labelsize": 10, "axes.labelcolor": INK, "axes.edgecolor": "#C4C9D1",
        "axes.linewidth": 0.8, "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
        "grid.color": "#E3E6EA", "grid.linewidth": 0.6, "xtick.color": GRAY, "ytick.color": GRAY,
        "xtick.labelsize": 9, "ytick.labelsize": 9, "legend.frameon": False, "legend.fontsize": 9,
        "figure.facecolor": "white", "axes.facecolor": "white", "savefig.facecolor": "white",
        "savefig.bbox": "tight", "savefig.dpi": 160, "text.color": INK})


@lru_cache(maxsize=4)
def geometry(code):
    """County and state polygons ordered like the region's counties, or None if geopandas is not installed."""
    try:
        import geopandas as gpd
        import pandas as pd
    except ImportError:
        return None
    c = gpd.read_file(DATA / f"{code}_counties.gpkg")
    s = gpd.read_file(DATA / f"{code}_states.gpkg")
    order = pd.read_csv(DATA / f"{code}_counties.csv")["fips"].astype(int)
    c["fips"] = c["fips"].astype(int)
    c = c.set_index("fips").loc[order].reset_index()
    return c, s


def _map_axes(ax):
    ax.set_axis_off(); ax.set_aspect("equal")


def draw_counties(ax, region, values, cmap, vmin=0.0, vmax=None, edge="white", norm=None, state_labels=False,
                  label_size=7):
    """Choropleth of one value per county on Census 2020 boundaries: light gray where there is no value, white
    county edges, gray state borders and, optionally, state labels with a white halo. Falls back to centroid bubbles
    without geopandas."""
    g = geometry(region.code)
    vmax = vmax if vmax is not None else max(float(np.nanmax(values)), 1e-9)
    norm = norm or matplotlib.colors.Normalize(vmin, vmax)
    if g is not None:
        c, s = g
        c = c.assign(v=values)
        c.plot(ax=ax, color=[cmap(norm(v)) if np.isfinite(v) else LIGHT for v in values], edgecolor=edge, linewidth=0.4)
        s.boundary.plot(ax=ax, color="#8A919C", linewidth=0.8)
        if state_labels:
            import matplotlib.patheffects as pe
            for _, r in s.iterrows():
                p = r.geometry.representative_point()
                ax.text(p.x, p.y, r["STUSPS"], ha="center", va="center", fontsize=label_size, fontweight="bold",
                        color=INK, path_effects=[pe.withStroke(linewidth=2.2, foreground="white")], zorder=6)
    else:
        lon, lat = region.counties["lon"], region.counties["lat"]
        ax.scatter(lon, lat, s=80, c=[cmap(norm(v)) for v in values], edgecolor="white", linewidth=0.5)
    _map_axes(ax)
    return matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap)


def _colorbar(fig, sm, ax, label, fmt=None):
    cb = fig.colorbar(sm, ax=ax, orientation="horizontal", fraction=0.04, pad=0.02, aspect=35, format=fmt)
    cb.outline.set_visible(False); cb.ax.tick_params(labelsize=8, color=GRAY, length=2); cb.ax.grid(False)
    cb.set_label(label, fontsize=9, color=INK)
    return cb


def _thousands(x, _=None):
    return f"{x / 1e3:,.0f}k" if abs(x) >= 1e3 else f"{x:,.0f}"


# ---------------------------------------------------------------------------------------------- a storm in progress
def render_state(env, return_array=True, figsize=(11, 6.2)):
    """One frame of a Session: map of customers out and crews, the regional curve so far, and the status."""
    style()
    p, R, storm = env.state, env.region, env.storm
    if env._fig is None or not return_array:
        if env._fig is not None:
            plt.close(env._fig)
        env._fig = plt.figure(figsize=figsize)
    fig = env._fig; fig.clf()
    gs = fig.add_gridspec(2, 2, width_ratios=[1.55, 1], height_ratios=[1, 1.05], wspace=0.08, hspace=0.35)
    ax_map = fig.add_subplot(gs[:, 0]); ax_curve = fig.add_subplot(gs[0, 1]); ax_txt = fig.add_subplot(gs[1, 1])

    share = p.out / R.customers
    # the color scale only grows during a storm (largest county share seen so far), so frames stay comparable
    key = (storm.id, env.engine_seed)
    if getattr(env, "_scale_key", None) != key:
        env._scale_key, env._vmax = key, 0.05
    env._vmax = max(env._vmax, float(np.ceil(share.max() * 20) / 20))
    sm = draw_counties(ax_map, R, share, OUTAGE, vmax=env._vmax)
    _colorbar(fig, sm, ax_map, "share of customers out (crew jobs)",
              fmt=matplotlib.ticker.PercentFormatter(1.0, decimals=0))
    lon, lat = R.counties["lon"].to_numpy(), R.counties["lat"].to_numpy()
    w = p.working
    if w.sum() > 0:
        size = 400 * np.sqrt(w / max(w.max(), 1))
        ax_map.scatter(lon[w > 0], lat[w > 0], s=size[w > 0], color=NAVY, alpha=0.85, edgecolor="white",
                       linewidth=0.8, zorder=5)
    ax_map.set_title(f"{storm.name}   |   hour {p.t:5.1f}", loc="left", fontsize=12, color=INK)
    ax_map.text(0.0, 0.985, "shading: share of customers out    dots: crews working, area in proportion",
                transform=ax_map.transAxes, fontsize=8, color=GRAY, va="top")

    # regional curve: observed vs the simulation so far
    obs = storm.observed if storm.observed is not None else np.zeros(storm.hours)
    if storm.observed is not None:
        ax_curve.plot(np.arange(len(obs)) + 1, obs, color=RED, lw=1, drawstyle="steps-pre", label="observed (EAGLE-I)")
    h = env.history
    if h:
        t = [x["t"] for x in h]; out = [x["out"] for x in h]
        ax_curve.plot(t, out, color=NAVY, lw=1.4, label="simulated (crew jobs)")
        ax_curve.scatter([t[-1]], [out[-1]], color=NAVY, s=22, zorder=5)
    if p.aid_requested_at is not None:
        ax_curve.axvline(p.aid_requested_at, color=GRAY, lw=0.8, ls=(0, (3, 2)))
        ax_curve.text(p.aid_requested_at - 1, ax_curve.get_ylim()[1] * 0.97, "aid requested ", color=GRAY, fontsize=8,
                      va="top", ha="right")
    ax_curve.set_xlim(0, max(len(obs), p.t + 6)); ax_curve.yaxis.set_major_formatter(_thousands)
    ax_curve.set_xlabel("hours from the start of the storm"); ax_curve.set_title("Customers out, region")
    ax_curve.legend(loc="upper right", fontsize=8)

    ax_txt.set_axis_off()
    aid = ("not requested" if p.aid_requested_at is None else
           f"{p.aid_crews:,} crews, asked at hour {p.aid_requested_at:.0f}"
           + (", on site" if p.crews > env.local_crews else ", on the way"))
    rows = [("Customers out (crew jobs)", f"{p.out.sum():,.0f}"),
            ("Jobs waiting", f"{p.waiting_jobs.sum():,}"),
            ("Crews on duty", f"{p.crews:,}  (local {env.local_crews:,})"),
            ("Crews working", f"{p.working.sum():,}"),
            ("Mutual aid", aid),
            ("Customer-hours so far, restored jobs", f"{p.customer_hours_so_far / 1e6:,.2f} M")]
    for i, (k, v) in enumerate(rows):
        y = 1.0 - i * 0.175
        ax_txt.text(0.0, y, k, fontsize=8.5, color=GRAY, transform=ax_txt.transAxes, va="top")
        ax_txt.text(0.0, y - 0.07, v, fontsize=11, color=INK, fontweight="bold", transform=ax_txt.transAxes, va="top")

    if return_array:
        fig.canvas.draw()
        return np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()
    plt.pause(0.001)
    return None


# ---------------------------------------------------------------------------------------------- analysis figures
def plot_restoration_curves(storm, curves, ax=None, title=None, hours=None):
    """Observed regional curve against the curves of several rules (dict name -> customers out per hour)."""
    style()
    ax = ax or plt.subplots(figsize=(8, 4.2))[1]
    obs = storm.observed if storm.observed is not None else np.zeros(0)
    if storm.observed is not None:
        x = np.arange(len(obs)) + 1
        ax.fill_between(x, obs, color=LIGHT, step="pre", lw=0, zorder=1)
        ax.plot(x, obs, color=INK, lw=1.2, drawstyle="steps-pre", zorder=3, label="observed (EAGLE-I)")
    H = hours or max([len(obs)] + [len(c) for c in curves.values()])
    for name, c in curves.items():
        ref = name == "current practice"          # the reference, dashed on top so it shows where others overlap
        ax.plot(np.arange(len(c)) + 1, c, color=color_for(name), lw=2, label=f"simulated: {name}", zorder=5 if ref else 4,
                ls=(0, (4, 2)) if ref else "-")
    ax.set_xlim(0, H); ax.set_ylim(0, None)
    ax.yaxis.set_major_formatter(_thousands)
    ax.set_xlabel("hours from the start of the storm"); ax.set_ylabel("customers out")
    ax.set_title(title or storm.name)
    ax.legend(loc="upper right")
    return ax


def plot_county_map(region, values, ax=None, title="", label="", cmap=DURATION, vmax=None, fmt=None):
    style()
    if ax is None:
        fig, ax = plt.subplots(figsize=(5, 5.5))
    sm = draw_counties(ax, region, values, cmap, vmax=vmax)
    _colorbar(ax.figure, sm, ax, label, fmt=fmt)
    ax.set_title(title)
    return ax


def plot_paired_intervals(comparison, metric="customer_hours", scenarios=None, titles=None, xlabel=None, path=None):
    """Relative change of each rule against the baseline, with its bootstrap interval, one panel per scenario."""
    from matplotlib.ticker import PercentFormatter
    style()
    c = comparison[comparison["metric"] == metric]
    scenarios = scenarios or list(dict.fromkeys(c["scenario"]))
    rules = list(dict.fromkeys(c["rule"]))
    fig, axes = plt.subplots(1, len(scenarios), figsize=(3.3 * len(scenarios) + 1.6, 0.42 * len(rules) + 1.4),
                             sharey=True, squeeze=False)
    axes = axes[0]
    lo = min(0.0, float(c["relative_low"].min())); hi = max(0.0, float(c["relative_high"].max()))
    pad = 0.08 * (hi - lo or 1)
    for ax, sc in zip(axes, scenarios):
        d = c[c["scenario"] == sc].set_index("rule").reindex(rules)
        y = np.arange(len(rules))[::-1]
        ax.axvline(0, color="#9AA1AC", lw=1, zorder=1)
        for yi, (rule, r) in zip(y, d.iterrows()):
            if not np.isfinite(r["relative"]):
                continue
            col = color_for(rule)
            ax.plot([r["relative_low"], r["relative_high"]], [yi, yi], color=col, lw=2.2, solid_capstyle="round", zorder=2)
            ax.scatter([r["relative"]], [yi], s=46, color=col, edgecolor="white", linewidth=1.2, zorder=3)
        ax.set_yticks(y)
        if ax is not axes[0]:
            ax.tick_params(labelleft=False, length=0)
        ax.set_xlim(lo - pad, hi + pad); ax.xaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
        ax.grid(axis="y", visible=False)
        ax.set_title((titles or {}).get(sc, sc), fontsize=10)
        ax.set_xlabel(xlabel or f"{metric.replace('_', ' ')} vs baseline", fontsize=9)
    axes[0].set_yticks(np.arange(len(rules))[::-1], rules)
    fig.tight_layout()
    if path:
        fig.savefig(path); plt.close(fig)
    return fig


def plot_validation(results, ncols=3, path=None, size=None, title=None):
    """Observed (EAGLE-I) against simulated current practice, one panel per storm. `results`: list of Result."""
    style()
    n = len(results); nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=size or (4.2 * ncols, 2.7 * nrows), squeeze=False)
    for ax, res in zip(axes.ravel(), results):
        st = res.storm; obs = st.observed; x = np.arange(len(obs)) + 1
        ax.fill_between(x, obs, color=LIGHT, step="pre", lw=0, zorder=1)
        ax.plot(x, obs, color=INK, lw=1.0, drawstyle="steps-pre", zorder=3, label="observed (EAGLE-I)")
        c = res.curve()
        ax.plot(np.arange(len(c)) + 1, c, color=RED, lw=1.6, zorder=4, label="simulated: current practice")
        ax.set_xlim(0, max(len(obs), 1) * 1.25); ax.set_ylim(0, None); ax.yaxis.set_major_formatter(_thousands)
        reg = "NE" if st.region == "ne" else "PA-WV"
        ax.set_title(f"{reg}: {st.name}", fontsize=9)
        ch = res.metrics["customer_hours"] / max(float(obs.sum()), 1.0)
        ax.text(0.98, 0.55, f"customer-hours,\nsimulated / observed: {ch:.2f}", transform=ax.transAxes, ha="right",
                va="top", fontsize=7.5, color=GRAY)
    for ax in axes.ravel()[n:]:
        ax.set_visible(False)
    axes[0, 0].legend(loc="upper right", fontsize=7.5)
    for ax in axes[-1]:
        ax.set_xlabel("hours from the start of the storm")
    if title:
        fig.suptitle(title, x=0.01, ha="left", fontsize=11, fontweight="bold")
    fig.tight_layout()
    if path:
        fig.savefig(path); plt.close(fig)
    return fig


# ---------------------------------------------------------------------------------------------- data provenance
# The provenance diagram is drawn in TikZ (docs/figures/provenance.tex), not here; docs/figures/provenance.png is
# rendered from that PDF so the paper and the README show the same figure.


# ---------------------------------------------------------------------------------------------- maps
def customer_hours_per_customer(result):
    """Simulated customer-hours per customer, by county, from a Result."""
    inter = result.storm.rises.sum(0)
    ch = np.nan_to_num(result.metrics["county_mean_outage_h"]) * inter
    N = result.storm.customers if result.storm.customers is not None else None
    if N is None:
        from ..core.data import region as _r
        N = _r(result.storm.region).customers
    return ch / N


def _cbar(fig, ax, sm, label, fs=8):
    cb = fig.colorbar(sm, ax=ax, orientation="horizontal", fraction=0.05, pad=0.02, aspect=28, shrink=0.9)
    cb.outline.set_visible(False); cb.ax.grid(False); cb.ax.tick_params(labelsize=fs - 1, length=2)
    cb.set_label(label, fontsize=fs)
    return cb


def study_area(path=None, size=(12, 4.6), fs=8, title=True):
    """The two regions: observed storm customer-hours per customer over the whole catalog, by county."""
    from ..core.data import region as _r
    style()
    regs = [_r("ne"), _r("pawv")]
    vals = [sum(R.storm(i).observed_county for i in R.storm_ids) / R.customers for R in regs]
    vmax = float(np.ceil(max(np.nanmax(v) for v in vals) / 10) * 10)
    fig, axes = plt.subplots(1, 2, figsize=size, gridspec_kw={"width_ratios": [1, 1.35]})
    for ax, R, v in zip(axes, regs, vals):
        sm = draw_counties(ax, R, v, OUTAGE, vmax=vmax, state_labels=True, label_size=fs)
    _cbar(fig, list(axes), sm, "observed storm customer-hours per customer, all storms 2018 to 2025 (EAGLE-I)", fs)
    fig.canvas.draw()
    top = max(ax.get_position().y1 for ax in axes)
    for ax, R in zip(axes, regs):
        fig.text(ax.get_position().x0, top + 0.02, f"{R.title}: {len(R.counties)} counties, {len(R.storm_ids)} storms",
                 fontsize=fs + 1, fontweight="bold", color=INK, ha="left", va="bottom")
    if path:
        fig.savefig(path); plt.close(fig)
    return fig


def footprint_maps(real, simulated, whatif=None, whatif_label="", path=None, size=(13, 4.8), fs=8):
    """One storm by county: observed and simulated customer-hours per customer, and a what-if change if given."""
    st = real.storm
    obs = st.observed_county / (st.customers if st.customers is not None else real_customers(st))
    sim = customer_hours_per_customer(simulated)
    panels = [("Observed (EAGLE-I)", obs), ("Simulated: " + simulated.rule, sim)]
    style()
    n = 3 if whatif is not None else 2
    fig, axes = plt.subplots(1, n, figsize=size)
    vmax = float(np.ceil(max(np.nanmax(obs), np.nanmax(sim)) / 5) * 5)
    for ax, (t, v) in zip(axes, panels):
        sm = draw_counties(ax, simulated_region(st), v, DURATION, vmax=vmax, state_labels=True, label_size=fs - 1)
        ax.set_title(t, fontsize=fs + 1)
    _cbar(fig, list(axes[:2]), sm, "customer-hours per customer", fs)
    if whatif is not None:
        d = customer_hours_per_customer(whatif) - sim
        sm2 = draw_counties(axes[2], simulated_region(st), d, OUTAGE, vmax=float(np.ceil(np.nanmax(d) / 5) * 5),
                            state_labels=True, label_size=fs - 1)
        axes[2].set_title(f"Simulated what-if: {whatif_label}", fontsize=fs + 1)
        _cbar(fig, axes[2], sm2, "change in customer-hours per customer", fs)
    if path:
        fig.savefig(path); plt.close(fig)
    return fig


def real_customers(storm):
    from ..core.data import region as _r
    return _r(storm.region).customers


def simulated_region(storm):
    from ..core.data import region as _r
    return _r(storm.region)
