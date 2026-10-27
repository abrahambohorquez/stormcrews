"""Renders the README image of the interactive notebook (notebooks/explore.ipynb): the figure the "Real storm" tab
draws for Tropical Storm Isaias under early aid, with current practice pinned as the baseline, and the metrics card.
The same functions draw the notebook, so the image shows what a user sees after pressing "Pin as baseline" and
switching the rule.

    python scripts/notebook_screenshot.py          # writes docs/figures/explore.png
"""
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from stormcrews import explore as E  # noqa: E402


def main():
    base = E.run_case(E.default_settings())                             # Isaias, current practice, seed 1
    run = E.run_case(E.default_settings(rule="early aid x1.5"))
    fig = E.storm_figure(run, base, label="early aid x1.5", baseline_label="pinned: current practice",
                         with_card=True)
    out = ROOT / "docs" / "figures" / "explore.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=200, bbox_inches="tight")
    print("wrote", out)


if __name__ == "__main__":
    main()
