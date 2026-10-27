"""Compares the reference rules under several scenarios, on every storm, with paired intervals.

    python evaluator.py                                          # the three shipped scenarios below, 3 seeds
    python evaluator.py --scenarios new_england stress_test --seeds 5

Every rule of a scenario sees the same storm draws for a given seed, so each rule is compared with current practice
storm by storm; intervals resample storms. Writes to results/:
    runs.csv                one row per scenario, rule, storm and seed, with every metric
    summary.csv, .md        per scenario and rule: totals over the storms (mean over seeds)
    comparison.csv, .md     each rule against current practice: relative change and 95 % bootstrap interval
    counties.csv            mean outage per county, scenario and rule
    paired_saidi.png        relative change in the SAIDI contribution with its interval, one panel per scenario
"""
import argparse
import time
from pathlib import Path

from stormcrews import run_experiment, shipped
from stormcrews.rules import REFERENCE

TITLES = {"new_england": "New England", "pennsylvania_west_virginia": "Pennsylvania and West Virginia",
          "stress_test": "Stress test (no aid, 75 % fleet)"}
ORDER = ["new_england", "pennsylvania_west_virginia", "stress_test"]
METRICS = ("saidi_min", "customer_hours", "over_24h", "over_72h", "over_96h", "worst_county_h", "restoration_end_h")


def markdown_summary(s):
    head = ["Scenario", "Rule", "SAIDI (min)", "Customer-hours (M)", "Out > 24 h (k)", "Out > 72 h (k)",
            "Out > 96 h (k)", "Worst county (h)", "Paid crew-hours (k)"]
    lines = ["| " + " | ".join(head) + " |", "|" + "|".join(["---", "---"] + ["---:"] * (len(head) - 2)) + "|"]
    for _, r in s.iterrows():
        lines.append("| " + " | ".join([TITLES.get(r["scenario"], r["scenario"]), r["rule"],
                                        f"{r['saidi_min']:,.0f}", f"{r['customer_hours'] / 1e6:,.1f}",
                                        f"{r['over_24h'] / 1e3:,.0f}", f"{r['over_72h'] / 1e3:,.1f}",
                                        f"{r['over_96h'] / 1e3:,.1f}", f"{r['worst_county_h']:,.1f}",
                                        f"{(r['local_crew_hours'] + r['aid_crew_hours']) / 1e3:,.0f}"]) + " |")
    return "\n".join(lines) + "\n"


def markdown_comparison(c):
    piv = c.pivot_table(index=["scenario", "rule"], columns="metric",
                        values=["relative", "relative_low", "relative_high", "share_storms_lower"], sort=False)
    head = ["Scenario", "Rule", "SAIDI", "Out > 72 h", "Out > 96 h", "Worst county", "Storms with lower SAIDI"]
    lines = ["| " + " | ".join(head) + " |", "|" + "|".join(["---", "---"] + ["---:"] * (len(head) - 2)) + "|"]
    f = lambda sc, ru, m: (f"{100 * piv.loc[(sc, ru), ('relative', m)]:+.1f} % "
                           f"[{100 * piv.loc[(sc, ru), ('relative_low', m)]:+.1f}, "
                           f"{100 * piv.loc[(sc, ru), ('relative_high', m)]:+.1f}]")
    for (sc, ru) in piv.index:
        lines.append("| " + " | ".join([TITLES.get(sc, sc), ru, f(sc, ru, "saidi_min"), f(sc, ru, "over_72h"),
                                        f(sc, ru, "over_96h"), f(sc, ru, "worst_county_h"),
                                        f"{100 * piv.loc[(sc, ru), ('share_storms_lower', 'saidi_min')]:.0f} %"])
                     + " |")
    return "\n".join(lines) + "\n"


def figures(c, out):
    import matplotlib
    matplotlib.use("Agg")

    from stormcrews.visuals.plots import plot_paired_intervals
    scs = [x for x in ORDER if x in set(c["scenario"])] + [x for x in dict.fromkeys(c["scenario"]) if x not in ORDER]
    plot_paired_intervals(c, "saidi_min", scenarios=scs, titles=TITLES,
                          xlabel="simulated SAIDI contribution vs current practice", path=out / "paired_saidi.png")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scenarios", nargs="+", default=[x for x in ORDER if x in shipped()])
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--boot", type=int, default=2000)
    ap.add_argument("--out", default="results")
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(exist_ok=True)
    t0 = time.time()
    exp = run_experiment(REFERENCE, a.scenarios, seeds=range(a.seeds), workers=min(a.workers, 6))
    print(f"{len(exp.table)} runs in {time.time() - t0:.0f} s")
    exp.table.to_csv(out / "runs.csv", index=False)
    exp.county.to_csv(out / "counties.csv", index=False)
    s = exp.summary(); s.to_csv(out / "summary.csv", index=False)
    c = exp.compare("current practice", metrics=METRICS, n_boot=a.boot); c.to_csv(out / "comparison.csv", index=False)
    (out / "summary.md").write_text(markdown_summary(s), encoding="utf-8")
    (out / "comparison.md").write_text(markdown_comparison(c), encoding="utf-8")
    figures(c, out)
    print(markdown_comparison(c))


if __name__ == "__main__":
    main()
