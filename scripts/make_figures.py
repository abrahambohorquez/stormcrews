"""Builds every image of the README in docs/figures, in the palette of the paper: navy for the simulator, red tile for
observed data, gray for alternatives and baselines.

    python scripts/make_figures.py            # every image (runs paper_gtdla/experiments.py results; needs pdflatex
                                              # and pdftoppm only for the two diagrams)
    python scripts/make_figures.py hero rules # some of them

Images: hero.png, problem.png, engine.png, provenance.png, validation.png, isaias_maps.png, demos.png, rules.png,
explore.png and isaias.gif.
"""
import shutil
import subprocess
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "paper_gtdla"))
sys.path.insert(0, str(ROOT / "scripts"))

import importlib.util  # noqa: E402

_spec = importlib.util.spec_from_file_location("paper_figures", ROOT / "paper_gtdla" / "make_figures.py")
PF = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(PF)      # the paper's figure functions
from stormcrews import Simulator  # noqa: E402
from stormcrews.core.data import region  # noqa: E402
from stormcrews.rules import CurrentPractice  # noqa: E402
from stormcrews.visuals import plots as P  # noqa: E402

FIG = ROOT / "docs" / "figures"; FIG.mkdir(parents=True, exist_ok=True)
TIKZ = FIG / "_tikz"
NAVY, RED, GRAY, INK = PF.NAVY, PF.RED, PF.GRAY, PF.INK


_paper_style = PF.style


def style():
    _paper_style()
    plt.rcParams.update({"savefig.dpi": 300})


def paper_figures():
    """The paper's figures as PNG: the problem, the validation, the Isaias maps and the illustrations."""
    PF.FIG, PF.EXT = FIG, "png"
    PF.style = style
    for name in ("problem", "validation", "isaias_maps", "demos"):
        getattr(PF, name)()


def hero():
    """Tropical Storm Isaias observed and simulated, next to the two regions of the shipped data."""
    style()
    fig = plt.figure(figsize=(8.6, 2.6))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.35, 0.85, 1.2], wspace=0.08)
    ax = fig.add_subplot(gs[0])
    res = Simulator("new_england", calibration="anchored").run(CurrentPractice(), 10, seed=1)
    st = res.storm; obs = st.observed / 1e3; c = res.curve() / 1e3
    ax.plot(np.arange(len(obs)) + 1, obs, color=RED, lw=0.9, drawstyle="steps-pre")
    ax.plot(np.arange(len(c)) + 1, c, color=NAVY, lw=0.9)
    ax.text(108, 700, "observed (EAGLE-I)", color=RED, fontsize=7)
    ax.text(178, 90, "simulated", color=NAVY, fontsize=7)
    ax.set_xlim(0, len(c) + 2); ax.set_ylim(0, None)
    ax.set_xlabel("hours since the storm window opened"); ax.set_ylabel("customers out (thousands)")
    ax.set_title("Tropical Storm Isaias, New England, August 2020", fontsize=7.5)
    regs = [region("ne"), region("pawv")]
    vals = [sum(R.storm(i).observed_county for i in R.storm_ids) / R.customers for R in regs]
    vmax = float(np.ceil(max(np.nanmax(v) for v in vals) / 10) * 10)
    for k, (R, v) in enumerate(zip(regs, vals)):
        a = fig.add_subplot(gs[k + 1])
        sm = P.draw_counties(a, R, v, P.OUTAGE, vmax=vmax, state_labels=True, label_size=5.5)
        for t in a.texts:
            t.set_fontweight("normal")
        a.set_title(f"{R.title}: {len(R.storm_ids)} storms", fontsize=7)
    cax = fig.add_axes([0.53, 0.06, 0.36, 0.03])
    cb = fig.colorbar(sm, cax=cax, orientation="horizontal"); cb.outline.set_visible(False); cb.ax.grid(False)
    cb.ax.tick_params(labelsize=6, length=2); cb.set_label("observed customer-hours per customer, all storms", fontsize=6.5)
    fig.savefig(FIG / "hero.png"); plt.close(fig)


def rules():
    """Three restoration rules on the 39 New England storms: the SAIDI contribution and the worst county."""
    style()
    d = pd.read_csv(ROOT / "paper_gtdla" / "results" / "rules.csv")
    d = d[d["rule"] != "largest first, 48 h guard"]
    per_seed = d.groupby(["rule", "seed"]).agg(saidi=("saidi_min", "sum"), worst=("worst_county_h", "mean"),
                                               travel=("travel_crew_hours", "sum"))
    t = per_seed.groupby(level=0).mean()
    order = ["largest first", "proportional dispatch", "oldest first"]
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 1.55), sharey=True)
    for ax, (col, lab) in zip(axes, (("saidi", "SAIDI contribution, 39 storms (min)"),
                                     ("worst", "worst county, mean outage (h)"),
                                     ("travel", "crew-hours driving (thousands)"))):
        sc = 1e3 if col == "travel" else 1
        for i, r in enumerate(order):
            v = t.loc[r, col] / sc
            colr = GRAY if r == "largest first" else NAVY
            ax.plot([0, v], [i, i], color="#D5D9DF", lw=0.8, zorder=1)
            ax.plot(v, i, "o", color=colr, ms=4, zorder=2)
            ax.text(v, i + 0.28, f"{v:,.0f}" if col != "worst" else f"{v:.1f}", color=colr, fontsize=6.5,
                    ha="center", va="bottom")
        ax.set_xlabel(lab); ax.set_xlim(0, t[col].max() / sc * 1.18); ax.set_ylim(-0.6, 2.8)
        ax.grid(axis="y", visible=False)
    axes[0].set_yticks(range(3), ["largest job first\n(default)", "proportional\ndispatch", "oldest job first"])
    fig.tight_layout(w_pad=1.2)
    fig.savefig(FIG / "rules.png"); plt.close(fig)


def _tex_to_png(tex, png, dpi=250):
    TIKZ.mkdir(exist_ok=True)
    shutil.copyfile(tex, TIKZ / tex.name)
    subprocess.run(["pdflatex", "-interaction=nonstopmode", tex.name], cwd=TIKZ, check=True, capture_output=True)
    stem = tex.with_suffix("").name                  # relative names: pdftoppm mishandles accented absolute paths
    subprocess.run(["pdftoppm", "-r", str(dpi), "-png", "-singlefile", stem + ".pdf", stem], cwd=TIKZ, check=True,
                   capture_output=True)
    shutil.move(str(TIKZ / (stem + ".png")), str(png))


def diagrams():
    """The restoration loop and the data provenance, both drawn in TikZ."""
    _tex_to_png(FIG / "engine.tex", FIG / "engine.png")
    _tex_to_png(ROOT / "docs" / "figures" / "provenance.tex", FIG / "provenance.png", dpi=200)


def explore():
    """The interactive notebook's figure, with the metrics card (scripts/notebook_screenshot.py)."""
    import notebook_screenshot
    notebook_screenshot.main()


def frames():
    """An animation of Isaias under current practice, as Session.render() draws its frames."""
    fr = []
    Simulator("new_england", calibration="anchored").run(CurrentPractice(), 10, seed=1, frames=fr)
    try:
        from PIL import Image
    except ImportError:
        print("pillow is not installed: isaias.gif skipped")
        return
    imgs = [Image.fromarray(f).resize((f.shape[1] * 3 // 4, f.shape[0] * 3 // 4), Image.LANCZOS)
            .quantize(colors=96, method=Image.Quantize.MEDIANCUT) for f in fr[:34]]
    imgs[0].save(FIG / "isaias.gif", save_all=True, append_images=imgs[1:], duration=350, loop=0, optimize=True)


ALL = ("hero", "paper_figures", "rules", "diagrams", "explore", "frames")

if __name__ == "__main__":
    for name in sys.argv[1:] or ALL:
        globals()[name]()
    print("images in", FIG)
