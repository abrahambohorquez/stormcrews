"""Point-and-click exploration of StormCrews, for notebooks (notebooks/explore.ipynb).

    from stormcrews.explore import Explorer
    Explorer()                    # three tabs: a real storm, a synthetic storm, two rules side by side

The functions below the widgets work without ipywidgets and are what the app calls:

    settings = default_settings()                 # Isaias, shape calibration, current practice, seed 1
    result = run_case(settings)                   # one simulated storm (a stormcrews Result)
    card(result)                                  # the metrics the app shows, as rows of (label, value, unit)
    fig = storm_figure(result, baseline)          # restoration curve and county map, as in the app

Every number shown comes from one run of the simulator with the settings on screen.
"""
from copy import deepcopy

import numpy as np

from .core.data import NAMES, TITLES, region as load_region
from .rules import REFERENCE
from .scenario import CALIBRATIONS, Scenario
from .simulator import Simulator

NAVY, RED, GRAY, INK, LIGHT = "#1E2761", "#B03A2E", "#6B7280", "#1F2933", "#EEF0F3"
REGIONS = ("ne", "pawv")
AID_MODES = ("normal", "none")
JOB_ORDERS = ("largest", "oldest", "guard", "sweep")

DEFAULTS = {
    "region": "ne", "storm": 10, "calibration": "shape", "rule": "current practice", "seed": 1,
    # weather and network what-if (neutral values replay the real storm)
    "gust_change": 0.0, "precip_factor": 1.0, "soil_moisture_change": 0.0,
    "shift_east_km": 0.0, "shift_north_km": 0.0, "footprint_scale": 1.0, "duration_factor": 1.0,
    "auto_share": None,
    # operations
    "fleet_factor": 1.0, "aid": "normal", "aid_lead_h": 24.0, "aid_share": 0.5,
    "job_order": "largest", "decision_h": 6.0,
}


def default_settings(**changes):
    s = deepcopy(DEFAULTS); s.update(changes)
    return s


# ---------------------------------------------------------------------------------------------- storms
def storm_label(meta, code):
    n = (meta.get("name") or "").rsplit("_", 1)[0]
    name = NAMES.get(n, n.title()) if n else meta.get("type", "Storm")
    region = "NE" if code == "ne" else "PA-WV"
    return f"{region} | {meta['start'][:10]} | {name}"


def storm_options(code=None):
    """(label, (region, storm id)) for every shipped storm, New England first."""
    out = []
    for c in (REGIONS if code is None else (code,)):
        for m in load_region(c).meta:
            out.append((storm_label(m, c), (c, m["id"])))
    return out


# ---------------------------------------------------------------------------------------------- running
def what_if(s):
    w = {"gust_change": float(s["gust_change"]), "precip_factor": float(s["precip_factor"]),
         "soil_moisture_change": float(s["soil_moisture_change"]),
         "footprint_shift_km": (float(s["shift_east_km"]), float(s["shift_north_km"])),
         "footprint_scale": float(s["footprint_scale"]), "duration_factor": float(s["duration_factor"]),
         "auto_share": None if s["auto_share"] is None else float(s["auto_share"])}
    return w


def scenario(s):
    return Scenario(name="explore", region=s["region"], calibration=s["calibration"], decision_h=float(s["decision_h"]),
                    fleet_factor=float(s["fleet_factor"]), aid=s["aid"], aid_lead_h=float(s["aid_lead_h"]),
                    aid_share=float(s["aid_share"]), job_order=s["job_order"], what_if=what_if(s))


def run_case(s, storm=None):
    """One run with the settings `s` (see DEFAULTS). `storm` replaces the shipped storm (a synthetic one)."""
    sim = Simulator(scenario(s))
    return sim.run(REFERENCE[s["rule"]], s["storm"] if storm is None else storm, seed=int(s["seed"]))


def crew_hours(m):
    return m["local_crew_hours"] + m["aid_crew_hours"]


CARD = (  # key, label, scale, decimals, unit
    ("customer_hours", "Customer-hours out", 1e6, 1, "million"),
    ("saidi_min", "SAIDI contribution", 1, 0, "min per customer served"),
    ("customers_interrupted", "Customers interrupted", 1e3, 0, "thousand"),
    ("caidi_h", "Mean outage per customer", 1, 1, "h"),
    ("over_24h", "Customers out beyond 24 h", 1e3, 1, "thousand"),
    ("over_72h", "Customers out beyond 72 h", 1e3, 1, "thousand"),
    ("over_96h", "Customers out beyond 96 h", 1e3, 1, "thousand"),
    ("restoration_end_h", "Last customer restored", 1, 0, "h after the window opens"),
    ("crew_hours", "Crew-hours paid", 1e3, 0, "thousand"),
    ("crews_max", "Peak crews on duty", 1, 0, "crews"),
    ("worst_county_h", "Worst county, mean outage", 1, 1, "h per customer out"),
)


SHORT_UNIT = {"million": "M", "min per customer served": "min", "thousand": "k", "h": "h",
              "h after the window opens": "h", "crews": "", "h per customer out": "h"}


def _value(m, key):
    return crew_hours(m) if key == "crew_hours" else float(m[key])


def card(result, baseline=None):
    """Rows (label, value text, unit, change against the baseline text or '')."""
    rows = []
    for key, label, sc, d, unit in CARD:
        v = _value(result.metrics, key)
        ch = ""
        if baseline is not None:
            b = _value(baseline.metrics, key)
            diff = (v - b) / sc
            ch = f"{diff:+,.{d}f}" + (f" ({100 * (v / b - 1):+.0f} %)" if b else "")
        rows.append((label, f"{v / sc:,.{d}f}", unit, ch))
    return rows


def card_html(result, baseline=None, title=None):
    rows = card(result, baseline)
    head = ("<tr><th style='text-align:left'>Metric</th><th style='text-align:right'>Value</th><th></th>"
            + ("<th style='text-align:right'>Change vs pinned</th>" if baseline is not None else "") + "</tr>")
    body = "".join(
        f"<tr><td>{a}</td><td style='text-align:right'><b>{b}</b></td><td style='color:{GRAY}'>{c}</td>"
        + (f"<td style='text-align:right;color:{NAVY}'>{d}</td>" if baseline is not None else "") + "</tr>"
        for a, b, c, d in rows)
    t = f"<div style='color:{INK};font-weight:600;margin:2px 0 4px 0'>{title}</div>" if title else ""
    return (f"<div style='font-family:sans-serif;font-size:12px'>{t}<table style='border-collapse:collapse'>"
            f"{head}{body}</table></div>")


# ---------------------------------------------------------------------------------------------- figures
def _rc():
    import matplotlib.pyplot as plt
    return plt.rc_context({"font.family": "DejaVu Sans", "font.size": 8.5, "axes.titlesize": 9,
                           "axes.titleweight": "normal", "axes.titlelocation": "left", "axes.edgecolor": "#C4C9D1",
                           "axes.linewidth": 0.6, "axes.spines.top": False, "axes.spines.right": False,
                           "axes.grid": True, "grid.color": "#E9ECEF", "grid.linewidth": 0.5, "xtick.color": GRAY,
                           "ytick.color": GRAY, "axes.labelcolor": INK, "text.color": INK,
                           "figure.facecolor": "white", "savefig.facecolor": "white"})


def _label(ax, x, y, text, color):
    import matplotlib.patheffects as pe
    ax.annotate(text, (x, y), xytext=(7, 3), textcoords="offset points", color=color, fontsize=8, va="bottom",
                path_effects=[pe.withStroke(linewidth=2.5, foreground="white")])


def plot_curves(ax, curves, observed=None, hours=None):
    """`curves`: list of (label, customers out per hour, color, linestyle). Observed EAGLE-I in red tile."""
    top = 0.0
    if observed is not None:
        x = np.arange(len(observed)) + 1
        ax.plot(x, observed / 1e3, color=RED, lw=0.9, drawstyle="steps-pre")
        i = int(np.argmax(observed))
        ax.annotate("observed (EAGLE-I)", (x[i], observed[i] / 1e3), xytext=(6, 2), textcoords="offset points",
                    color=RED, fontsize=8)
        top = max(top, observed.max())
    for k, (lab, c, color, ls) in enumerate(curves):
        x = np.arange(len(c)) + 1
        ax.plot(x, c / 1e3, color=color, lw=1.1, ls=ls)
        p = int(np.argmax(c)); level = (0.45 - 0.25 * k) * c[p]       # label where the curve falls to this level
        below = np.flatnonzero(c[p:] <= level)
        j = p + int(below[0]) if len(below) else len(c) - 1
        _label(ax, x[j], c[j] / 1e3, lab, color)
        top = max(top, c.max())
    ax.set_xlim(0, hours or ax.get_xlim()[1]); ax.set_ylim(0, top / 1e3 * 1.12 if top > 0 else 1)
    ax.set_xlabel("hours since the storm window opened"); ax.set_ylabel("customers out (thousands)")


def plot_map(ax, result, vmax=None, title="Simulated customer-hours per customer"):
    from .visuals.plots import DURATION, customer_hours_per_customer, draw_counties
    R = load_region(result.storm.region)
    v = customer_hours_per_customer(result)
    vmax = vmax or max(float(np.ceil(np.nanmax(v) / 5) * 5), 1.0)
    sm = draw_counties(ax, R, v, DURATION, vmax=vmax, state_labels=True, label_size=7)
    ax.set_title(title, fontsize=9)
    cb = ax.figure.colorbar(sm, ax=ax, orientation="horizontal", fraction=0.045, pad=0.02, aspect=30, shrink=0.85)
    cb.outline.set_visible(False); cb.ax.grid(False); cb.ax.tick_params(labelsize=7, length=2)
    cb.set_label("hours per customer served", fontsize=8)
    return v


def storm_figure(result, baseline=None, label="simulated", baseline_label="pinned baseline", with_card=False):
    """Restoration curve and county map of one run; the pinned baseline in gray. With `with_card` a third panel
    lists the metrics (used for the README image)."""
    import matplotlib.pyplot as plt
    with _rc():
        ncol = 3 if with_card else 2
        widths = [1.5, 1.0, 1.25] if with_card else [1.5, 1.0]
        fig, axes = plt.subplots(1, ncol, figsize=(11.5 if with_card else 9.6, 3.6),
                                 gridspec_kw={"width_ratios": widths})
        st = result.storm
        curves = [(label, result.curve(), NAVY, "-")]
        if baseline is not None:
            curves.append((baseline_label, baseline.curve(), GRAY, (0, (3, 2))))
        H = max(len(c) for _, c, _, _ in curves)
        plot_curves(axes[0], curves, st.observed if st.real and st.observed is not None else None, hours=H)
        axes[0].set_title(f"{TITLES[st.region]}: {st.name}")
        plot_map(axes[1], result)
        if with_card:
            ax = axes[2]; ax.set_axis_off()
            rows = card(result, baseline)
            ax.set_title("Metrics of this run" + (" (change vs pinned)" if baseline is not None else ""), fontsize=9)
            for i, (a, b, c, d) in enumerate(rows):
                y = 0.95 - i * 0.087
                u = SHORT_UNIT.get(c, c)
                ax.text(0.0, y, a + (f" ({u})" if u else ""), fontsize=7.6, color=INK, transform=ax.transAxes,
                        va="top")
                ax.text(0.77, y, b, fontsize=7.6, color=INK, transform=ax.transAxes, va="top", ha="right",
                        fontweight="bold")
                if d:
                    ax.text(1.0, y, d, fontsize=7.2, color=NAVY, transform=ax.transAxes, va="top", ha="right")
        fig.tight_layout(w_pad=1.0)
    return fig


def synthetic_figure(region, gust, result):
    import matplotlib.pyplot as plt
    from .visuals.plots import OUTAGE, draw_counties
    with _rc():
        fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.4), gridspec_kw={"width_ratios": [1.0, 1.0, 1.4]})
        sm = draw_counties(axes[0], region, gust, OUTAGE, vmin=float(np.floor(gust.min())),
                           vmax=float(np.ceil(gust.max())), state_labels=True, label_size=7)
        axes[0].set_title("Peak gust of the synthetic storm", fontsize=9)
        cb = fig.colorbar(sm, ax=axes[0], orientation="horizontal", fraction=0.045, pad=0.02, aspect=30, shrink=0.85)
        cb.outline.set_visible(False); cb.ax.grid(False); cb.ax.tick_params(labelsize=7, length=2)
        cb.set_label("m/s", fontsize=8)
        plot_map(axes[1], result)
        plot_curves(axes[2], [("simulated", result.curve(), NAVY, "-")])
        axes[2].set_title("Simulated restoration")
        fig.tight_layout(w_pad=1.0)
    return fig


def compare_figure(a, b, label_a, label_b):
    import matplotlib.pyplot as plt
    with _rc():
        fig, ax = plt.subplots(figsize=(7.5, 3.2))
        st = a.storm
        curves = [(label_a, a.curve(), NAVY, "-"), (label_b, b.curve(), GRAY, (0, (3, 2)))]
        plot_curves(ax, curves, st.observed if st.real and st.observed is not None else None,
                    hours=max(len(c) for _, c, _, _ in curves))
        ax.set_title(f"{TITLES[st.region]}: {st.name}")
        fig.tight_layout()
    return fig


def compare_html(a, b, label_a, label_b):
    ra, rb = card(a), card(b)
    body = ""
    for (lab, va, unit, _), (_, vb, _, _), (key, _, sc, d, _) in zip(ra, rb, CARD):
        x, y = _value(a.metrics, key), _value(b.metrics, key)
        diff = f"{(y - x) / sc:+,.{d}f}" + (f" ({100 * (y / x - 1):+.0f} %)" if x else "")
        body += (f"<tr><td>{lab}</td><td style='text-align:right'><b>{va}</b></td>"
                 f"<td style='text-align:right'><b>{vb}</b></td><td style='color:{GRAY}'>{unit}</td>"
                 f"<td style='text-align:right;color:{NAVY}'>{diff}</td></tr>")
    return ("<div style='font-family:sans-serif;font-size:12px'><table style='border-collapse:collapse'>"
            f"<tr><th style='text-align:left'>Metric</th><th>{label_a}</th><th>{label_b}</th><th></th>"
            f"<th>second minus first</th></tr>{body}</table></div>")


# ---------------------------------------------------------------------------------------------- widgets
def _show(out, fig=None, html=None):
    import matplotlib.pyplot as plt
    from IPython.display import HTML, display
    with out:
        out.clear_output(wait=True)
        if fig is not None:
            display(fig); plt.close(fig)
        if html is not None:
            display(HTML(html))


class StormTab:
    """A real storm with every knob, the metrics card and a pinned baseline."""

    def __init__(self, run_now=True):
        import ipywidgets as W
        self.W = W
        D = DEFAULTS
        st = {"description_width": "150px"}; lay = W.Layout(width="360px")

        def slider(v, lo, hi, step, desc, fmt=".1f"):
            return W.FloatSlider(value=v, min=lo, max=hi, step=step, description=desc, readout_format=fmt,
                                 continuous_update=False, style=st, layout=lay)

        self.storm = W.Dropdown(options=storm_options(), value=(D["region"], D["storm"]), description="Storm",
                                style=st, layout=W.Layout(width="460px"))
        self.calibration = W.Dropdown(options=list(CALIBRATIONS), value=D["calibration"], description="Calibration",
                                      style=st, layout=lay)
        self.rule = W.Dropdown(options=list(REFERENCE), value=D["rule"], description="Rule", style=st, layout=lay)
        self.seed = W.BoundedIntText(value=D["seed"], min=0, max=10_000, description="Seed", style=st, layout=lay)
        self.gust = slider(0, -10, 10, 0.5, "Gust change (m/s)")
        self.rain = slider(1, 0, 3, 0.1, "Rain factor (x)")
        self.soil = slider(0, -0.1, 0.1, 0.01, "Soil moisture change", ".2f")
        self.east = slider(0, -200, 200, 10, "Footprint east (km)", ".0f")
        self.north = slider(0, -200, 200, 10, "Footprint north (km)", ".0f")
        self.scale = slider(1, 0.5, 2, 0.05, "Footprint scale (x)", ".2f")
        self.duration = slider(1, 0.5, 2, 0.05, "Duration factor (x)", ".2f")
        self.keep_auto = W.Checkbox(value=True, description="Keep the measured automatic share", style=st, layout=lay)
        self.auto = slider(0.1, 0, 0.6, 0.01, "Automatic share", ".2f")
        self.fleet = slider(1, 0.25, 2, 0.05, "Fleet factor (x)", ".2f")
        self.aid = W.Dropdown(options=list(AID_MODES), value=D["aid"], description="Mutual aid", style=st, layout=lay)
        self.aid_lead = slider(24, 0, 96, 2, "Aid lead time (h)", ".0f")
        self.aid_share = slider(0.5, 0, 2, 0.05, "Aid size (per local crew)", ".2f")
        self.job_order = W.Dropdown(options=list(JOB_ORDERS), value=D["job_order"], description="Job order",
                                    style=st, layout=lay)
        self.decision = slider(6, 1, 24, 1, "Decision interval (h)", ".0f")
        self.auto_run = W.Checkbox(value=True, description="Run on every change", style=st, layout=lay)
        self.run_btn = W.Button(description="Run", button_style="primary", layout=W.Layout(width="110px"))
        self.pin_btn = W.Button(description="Pin as baseline", layout=W.Layout(width="140px"))
        self.clear_btn = W.Button(description="Clear baseline", layout=W.Layout(width="130px"))
        self.status = W.HTML()
        self.plot_out = W.Output(); self.card_out = W.Output()
        self.result = None; self.baseline = None; self.baseline_label = None

        knobs = [self.storm, self.calibration, self.rule, self.seed, self.gust, self.rain, self.soil,
                 self.east, self.north, self.scale, self.duration, self.keep_auto, self.auto, self.fleet, self.aid,
                 self.aid_lead, self.aid_share, self.job_order, self.decision]
        for w in knobs:
            w.observe(self._changed, names="value")
        self.run_btn.on_click(lambda _: self.update())
        self.pin_btn.on_click(lambda _: self.pin())
        self.clear_btn.on_click(lambda _: self.clear())

        acc = W.Accordion(children=[
            W.VBox([self.storm, self.calibration, self.rule, self.seed]),
            W.VBox([self.gust, self.rain, self.soil, self.east, self.north, self.scale, self.duration,
                    self.keep_auto, self.auto]),
            W.VBox([self.fleet, self.aid, self.aid_lead, self.aid_share, self.job_order, self.decision])])
        for i, t in enumerate(("Storm, calibration and rule", "Weather and network what-if", "Crews and operations")):
            acc.set_title(i, t)
        acc.selected_index = 0
        self.widget = W.VBox([acc, W.HBox([self.run_btn, self.pin_btn, self.clear_btn, self.auto_run]), self.status,
                              self.plot_out, self.card_out])
        if run_now:
            self.update()

    def settings(self):
        region, sid = self.storm.value
        return default_settings(
            region=region, storm=sid, calibration=self.calibration.value, rule=self.rule.value, seed=self.seed.value,
            gust_change=self.gust.value, precip_factor=self.rain.value, soil_moisture_change=self.soil.value,
            shift_east_km=self.east.value, shift_north_km=self.north.value,
            footprint_scale=self.scale.value, duration_factor=self.duration.value,
            auto_share=None if self.keep_auto.value else self.auto.value, fleet_factor=self.fleet.value,
            aid=self.aid.value, aid_lead_h=self.aid_lead.value, aid_share=self.aid_share.value,
            job_order=self.job_order.value, decision_h=self.decision.value)

    def _changed(self, _):
        if self.auto_run.value:
            self.update()

    def describe(self, s):
        bits = [s["rule"], f"{s['calibration']} calibration", f"seed {s['seed']}"]
        if s["gust_change"]:
            bits.append(f"gust {s['gust_change']:+g} m/s")
        if s["fleet_factor"] != 1:
            bits.append(f"fleet x{s['fleet_factor']:g}")
        return ", ".join(bits)

    def update(self):
        s = self.settings()
        self.status.value = f"<span style='color:{GRAY}'>running...</span>"
        try:
            self.result = run_case(s)
        except Exception as e:                         # an impossible combination: say so and keep the last run
            self.status.value = f"<span style='color:{RED}'>Could not run: {e}</span>"
            return None
        self.label = self.describe(s)
        fig = storm_figure(self.result, self.baseline, baseline_label=self.baseline_label or "pinned baseline")
        _show(self.plot_out, fig=fig)
        _show(self.card_out, html=card_html(self.result, self.baseline, title=f"This run: {self.label}"
                                            + (f"; pinned: {self.baseline_label}" if self.baseline else "")))
        self.status.value = (f"<span style='color:{GRAY}'>{self.result.storm.name}: "
                             f"{len(self.result.outcome['t_arr']):,} repair jobs simulated.</span>")
        return self.result

    def pin(self):
        if self.result is not None:
            self.baseline, self.baseline_label = self.result, "pinned: " + self.label
            self.update()

    def clear(self):
        self.baseline = self.baseline_label = None
        self.update()


class SyntheticTab:
    """A storm that never happened: a Gaussian gust field with the timing and crews of a real template storm."""

    def __init__(self, run_now=True):
        import ipywidgets as W
        st = {"description_width": "150px"}; lay = W.Layout(width="360px")
        self.region = W.Dropdown(options=[("New England", "ne"), ("Pennsylvania and West Virginia", "pawv")],
                                 value="ne", description="Region", style=st, layout=lay)
        self.template = W.Dropdown(description="Timing and crews of", style=st, layout=W.Layout(width="460px"))
        self.lat = W.FloatSlider(description="Center latitude", step=0.05, continuous_update=False, style=st, layout=lay)
        self.lon = W.FloatSlider(description="Center longitude", step=0.05, continuous_update=False, style=st, layout=lay)
        self.peak = W.FloatSlider(value=38, min=20, max=50, step=1, description="Peak gust (m/s)",
                                  continuous_update=False, style=st, layout=lay)
        self.radius = W.FloatSlider(value=120, min=30, max=400, step=10, description="Radius (km)",
                                    continuous_update=False, style=st, layout=lay)
        self.rule = W.Dropdown(options=list(REFERENCE), value="current practice", description="Rule", style=st,
                               layout=lay)
        self.seed = W.BoundedIntText(value=1, min=0, max=10_000, description="Seed", style=st, layout=lay)
        self.run_btn = W.Button(description="Run", button_style="primary")
        self.status = W.HTML(); self.out = W.Output()
        self.region.observe(lambda _: self._region(), names="value")
        self._region(first=True)
        self.run_btn.on_click(lambda _: self.update())
        self.widget = W.VBox([W.VBox([self.region, self.template, self.lat, self.lon, self.peak, self.radius,
                                      self.rule, self.seed]), self.run_btn, self.status, self.out])
        if run_now:
            self.update()

    def _region(self, first=False):
        R = load_region(self.region.value)
        self.template.options = storm_options(self.region.value)
        self.template.value = (self.region.value, 10 if self.region.value == "ne" else R.storm_ids[0])
        la, lo = R.counties["lat"], R.counties["lon"]
        for w, x, c in ((self.lat, la, (41.8 if self.region.value == "ne" else la.median())),
                        (self.lon, lo, (-72.7 if self.region.value == "ne" else lo.median()))):
            w.min, w.max = -1e9, 1e9
            w.min, w.max = float(np.floor(x.min())), float(np.ceil(x.max())); w.value = float(round(c, 2))

    def update(self):
        from .synthetic import gust_field, synthetic_storm
        self.status.value = f"<span style='color:{GRAY}'>running...</span>"
        code, tid = self.template.value
        R = load_region(code)
        gust = gust_field(R, (self.lat.value, self.lon.value), self.peak.value, self.radius.value)
        storm = synthetic_storm(R, gust, tid)
        s = default_settings(region=code, storm=tid, rule=self.rule.value, seed=self.seed.value)
        self.result = run_case(s, storm=storm)
        _show(self.out, fig=synthetic_figure(R, gust, self.result),
              html=card_html(self.result, title="Synthetic storm, shape calibration of the template storm"))
        self.status.value = ""
        return self.result


class CompareTab:
    """Two rules on the same storm, scenario and seed: the same damage, jobs and repair times."""

    def __init__(self, run_now=True):
        import ipywidgets as W
        st = {"description_width": "150px"}; lay = W.Layout(width="360px")
        self.storm = W.Dropdown(options=storm_options(), value=("ne", 10), description="Storm", style=st,
                                layout=W.Layout(width="460px"))
        self.calibration = W.Dropdown(options=list(CALIBRATIONS), value="shape", description="Calibration", style=st,
                                      layout=lay)
        self.rule_a = W.Dropdown(options=list(REFERENCE), value="current practice", description="First rule",
                                 style=st, layout=lay)
        self.rule_b = W.Dropdown(options=list(REFERENCE), value="early aid x1.5", description="Second rule",
                                 style=st, layout=lay)
        self.seed = W.BoundedIntText(value=1, min=0, max=10_000, description="Seed", style=st, layout=lay)
        self.run_btn = W.Button(description="Run both", button_style="primary")
        self.status = W.HTML(); self.out = W.Output()
        self.run_btn.on_click(lambda _: self.update())
        self.widget = W.VBox([self.storm, self.calibration, self.rule_a, self.rule_b, self.seed, self.run_btn,
                              self.status, self.out])
        if run_now:
            self.update()

    def update(self):
        self.status.value = f"<span style='color:{GRAY}'>running...</span>"
        region, sid = self.storm.value
        base = default_settings(region=region, storm=sid, calibration=self.calibration.value, seed=self.seed.value)
        try:
            a = run_case(dict(base, rule=self.rule_a.value)); b = run_case(dict(base, rule=self.rule_b.value))
        except Exception as e:
            self.status.value = f"<span style='color:{RED}'>Could not run: {e}</span>"
            return None
        _show(self.out, fig=compare_figure(a, b, self.rule_a.value, self.rule_b.value),
              html=compare_html(a, b, self.rule_a.value, self.rule_b.value))
        self.status.value = ""
        self.results = (a, b)
        return a, b


class Explorer:
    """The three tabs in one widget. Display it as the last expression of a notebook cell."""

    def __init__(self, run_now=True):
        import ipywidgets as W
        self.storm = StormTab(run_now); self.synthetic = SyntheticTab(run_now); self.compare = CompareTab(run_now)
        self.widget = W.Tab(children=[self.storm.widget, self.synthetic.widget, self.compare.widget])
        for i, t in enumerate(("Real storm", "Synthetic storm", "Two rules")):
            self.widget.set_title(i, t)

    def _ipython_display_(self):
        from IPython.display import display
        display(self.widget)
