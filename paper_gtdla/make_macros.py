"""Writes paper_gtdla/macros.tex: every number of the GT&D Latin America 2027 paper, from paper_gtdla/results/,
results/calibration_summary.csv and the shipped data. Run from the repository root after paper_gtdla/experiments.py.

    python paper_gtdla/make_macros.py
"""
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))

import stormcrews as sc  # noqa: E402
from stormcrews.core.engine import EngineConfig  # noqa: E402
from stormcrews.scenario import Scenario  # noqa: E402
from experiments import ANCHOR_PER_1000, ANCHORS, SEEDS  # noqa: E402

RES = HERE / "results"
M = {}


def put(name, value):
    assert name.isalpha(), name
    assert name not in M, name
    M[name] = value


def num(x, d=0):
    return f"{x:,.{d}f}"


def pct(x, d=0):
    """A relative change as a signed percent, e.g. +12 or $-$7."""
    return f"{100 * x:+.{d}f}".replace("-", "$-$")


def apct(x, d=0):
    return f"{abs(100 * x):.{d}f}"


def totals(df, by):
    cols = ["saidi_min", "customer_hours", "customers_interrupted", "over_24h", "over_72h", "over_96h",
            "local_crew_hours", "aid_crew_hours", "busy_crew_hours", "travel_crew_hours"]
    t = df.groupby(by + ["seed"])[cols].sum()
    t["paid_crew_hours"] = t["local_crew_hours"] + t["aid_crew_hours"]
    mean = df.groupby(by + ["seed"])[["restoration_end_h", "worst_county_h", "caidi_h"]].agg(
        {"restoration_end_h": "median", "worst_county_h": "mean", "caidi_h": "median"})
    t = t.join(mean)
    return t.groupby(level=list(range(len(by)))).mean()


# ------------------------------------------------------------------ data
R = {"NE": sc.region("ne"), "PA": sc.region("pawv")}
for k, r in R.items():
    put(f"nStorms{k}", str(len(r.meta))); put(f"nCounties{k}", str(len(r.counties)))
    put(f"custM{k}", num(r.customers.sum() / 1e6, 1))
put("nStorms", str(sum(len(r.meta) for r in R.values())))
put("nCounties", str(sum(len(r.counties) for r in R.values())))
autos = np.concatenate([[x["auto_share"] for x in r.meta] for r in R.values()])
put("autoMin", f"{100 * autos.min():.0f}"); put("autoMax", f"{100 * autos.max():.0f}")
E = EngineConfig(); S = Scenario()
put("custPerJob", f"{E.customers_per_job:g}"); put("speedKmh", f"{E.speed_kmh:g}"); put("setupH", f"{E.setup_h:g}")
put("repairCV", f"{E.repair_cv:g}"); put("aidLeadH", f"{E.aid_lead_h:g}"); put("aidShare", f"{E.aid_share:g}")
put("triggerPct", f"{100 * E.reactive_trigger:g}"); put("guardH", f"{E.guard_h:g}")
put("autoLoMin", f"{60 * E.auto_h[0]:g}"); put("autoHiMin", f"{60 * E.auto_h[1]:g}")
put("decisionH", f"{S.decision_h:g}")
put("anchorRatio", f"{ANCHOR_PER_1000:.1f}")
put("anchorMin", f"{min(ANCHORS.values()):.1f}"); put("anchorMax", f"{max(ANCHORS.values()):.1f}")
put("nAnchors", {5: "five"}.get(len(ANCHORS), str(len(ANCHORS))))
fl = {k: np.median([x["calibrations"]["anchored"]["fleet"] for x in r.meta]) for k, r in R.items()}
put("fleetNE", num(fl["NE"])); put("fleetPA", num(fl["PA"]))
rep = np.concatenate([[x["calibrations"]["anchored"]["repair_mean_h"] for x in r.meta] for r in R.values()])
put("repairMed", f"{np.median(rep):.1f}"); put("repairMax", f"{rep.max():.0f}")
put("nSeeds", str(len(SEEDS)))

# ------------------------------------------------------------------ validation
cal = pd.read_csv(ROOT / "results" / "calibration_summary.csv")
put("peakAllMin", f"{cal['peak_ratio_median'].min():.2f}"); put("peakAllMax", f"{cal['peak_ratio_median'].max():.2f}")
h = pd.read_csv(RES / "holdout.csv")
h["chr"] = h["customer_hours"] / h["observed_customer_hours"]; h["pk"] = h["peak_sim"] / h["peak_obs"]
h["ech"] = np.abs(np.log(h["chr"]))
for key, reg in (("NE", "ne"), ("PA", "pawv")):
    for fk, fit in (("In", "in sample"), ("Out", "held out")):
        d = h[(h["region"] == reg) & (h["fit"] == fit)]
        p = f"{fk}{key}"
        put(f"chRatio{p}", f"{d['chr'].median():.2f}")
        put(f"chErr{p}", f"{100 * (np.exp(d['ech'].median()) - 1):.0f}")
        put(f"chWithin{p}", f"{100 * (d['ech'] <= np.log(1.25)).mean():.0f}")
        put(f"chPooled{p}", f"{d['customer_hours'].sum() / d['observed_customer_hours'].sum():.2f}")
        put(f"peakRatio{p}", f"{d['pk'].median():.2f}")
        put(f"tFifty{p}", f"{np.median(np.abs(d['t50_sim'] - d['t50_obs'])):.0f}")
        put(f"tNinety{p}", f"{np.median(np.abs(d['t90_sim'] - d['t90_obs'])):.0f}")
        put(f"nVal{p}", str(len(d)))

# ------------------------------------------------------------------ (a) gust what-if
g = totals(pd.read_csv(RES / "gust.csv"), ["gust_change"])
z, f5 = g.loc[0.0], g.loc[5.0]
put("saidiNE", num(z["saidi_min"]))
put("chNE", num(z["customer_hours"] / 1e6))
put("gustSaidiX", f"{f5['saidi_min'] / z['saidi_min']:.1f}")
put("gustInterX", f"{f5['customers_interrupted'] / z['customers_interrupted']:.1f}")
put("gustOverSeventyTwoX", f"{f5['over_72h'] / z['over_72h']:.1f}")
put("gustOverNinetySixX", f"{f5['over_96h'] / z['over_96h']:.1f}")
put("gustEndZero", num(z["restoration_end_h"])); put("gustEndFive", num(f5["restoration_end_h"]))
put("gustCaidiZero", f"{z['caidi_h']:.1f}"); put("gustCaidiFive", f"{f5['caidi_h']:.1f}")
put("gustPaidCH", pct(f5["paid_crew_hours"] / z["paid_crew_hours"] - 1))
put("gustSaidiMinusFiveX", f"{g.loc[-5.0, 'saidi_min'] / z['saidi_min']:.2f}")
put("overSeventyTwoNE", num(z["over_72h"] / 1e3)); put("overNinetySixNE", num(z["over_96h"] / 1e3))
put("overTwentyFourNE", num(z["over_24h"] / 1e6, 1)); put("interNE", num(z["customers_interrupted"] / 1e6, 1))

# ------------------------------------------------------------------ (b) crews and aid lag
c = pd.read_csv(RES / "crews.csv")
f = totals(c[c["sweep"] == "fleet"], ["crews_per_1000"])
base_ = f.loc[ANCHOR_PER_1000]
for lab, v in (("Two", 2.0), ("Three", 3.0), ("Five", 5.0), ("Max", max(ANCHORS.values())), ("Min", min(ANCHORS.values()))):
    r_ = f.loc[min(f.index, key=lambda x: abs(x - v))]
    put(f"crews{lab}Saidi", pct(r_["saidi_min"] / base_["saidi_min"] - 1))
    put(f"crews{lab}Paid", pct(r_["paid_crew_hours"] / base_["paid_crew_hours"] - 1))
    put(f"crews{lab}OverSeventyTwo", pct(r_["over_72h"] / base_["over_72h"] - 1))
    put(f"crews{lab}Travel", pct(r_["travel_crew_hours"] / base_["travel_crew_hours"] - 1))
lag = totals(pd.concat([c[c["sweep"] == "lag"], c[(c["sweep"] == "fleet") & (c["fleet_factor"] == 1.0)]]), ["aid_lead_h"])
for lab, v in (("Twelve", 12.0), ("FortyEight", 48.0), ("SeventyTwo", 72.0)):
    put(f"lag{lab}Saidi", pct(lag.loc[v, "saidi_min"] / lag.loc[24.0, "saidi_min"] - 1))
    put(f"lag{lab}OverNinetySix", pct(lag.loc[v, "over_96h"] / lag.loc[24.0, "over_96h"] - 1))

# ------------------------------------------------------------------ (c) rules
ru = totals(pd.read_csv(RES / "rules.csv"), ["rule"])
b = ru.loc["largest first"]
RK = {"largest first": "Lar", "oldest first": "Old", "largest first, 48 h guard": "Gua", "proportional dispatch": "Pro"}
for rule, k in RK.items():
    r_ = ru.loc[rule]
    put(f"ru{k}Saidi", num(r_["saidi_min"]))
    put(f"ru{k}OverTwentyFour", num(r_["over_24h"] / 1e6, 2))
    put(f"ru{k}OverSeventyTwo", num(r_["over_72h"] / 1e3))
    put(f"ru{k}OverNinetySix", num(r_["over_96h"] / 1e3))
    put(f"ru{k}Worst", f"{r_['worst_county_h']:.1f}")
    put(f"ru{k}Travel", num(r_["travel_crew_hours"] / 1e3))
    put(f"ru{k}SaidiRel", pct(r_["saidi_min"] / b["saidi_min"] - 1, 1))
    put(f"ru{k}WorstRel", pct(r_["worst_county_h"] / b["worst_county_h"] - 1))
    put(f"ru{k}TravelRel", pct(r_["travel_crew_hours"] / b["travel_crew_hours"] - 1, 1))

# ------------------------------------------------------------------ sensitivity: trigger and customers per job
tr = totals(pd.read_csv(RES / "trigger.csv"), ["reactive_trigger"])
put("trigLowSaidi", pct(tr.loc[0.005, "saidi_min"] / tr.loc[0.01, "saidi_min"] - 1, 1))
put("trigHighSaidi", pct(tr.loc[0.02, "saidi_min"] / tr.loc[0.01, "saidi_min"] - 1, 1))
pj = totals(pd.read_csv(RES / "perjob.csv"), ["customers_per_job", "case"])
fit = pd.read_csv(RES / "perjob_fit.csv")
for lab, v in (("Low", 20.0), ("Mid", 35.0), ("High", 55.0)):
    real = pj.loc[(v, "real")]
    put(f"pj{lab}", f"{v:g}")
    put(f"pj{lab}Repair", f"{fit[fit['customers_per_job'] == v]['repair_mean_h'].median():.1f}")
    put(f"pj{lab}Saidi", num(real["saidi_min"]))
    put(f"pj{lab}Gust", f"{pj.loc[(v, 'gust +5'), 'saidi_min'] / real['saidi_min']:.2f}")
    put(f"pj{lab}Fleet", pct(pj.loc[(v, 'fleet 75 %'), 'saidi_min'] / real['saidi_min'] - 1))
    put(f"pj{lab}Lag", pct(pj.loc[(v, 'aid lag 48 h'), 'saidi_min'] / real['saidi_min'] - 1))
    put(f"pj{lab}Busy", num(real["busy_crew_hours"] / 1e6, 2))
    put(f"pj{lab}OverNinetySix", num(real["over_96h"] / 1e3))
put("nRunsDemo", num(sum(len(pd.read_csv(RES / f)) for f in ("gust.csv", "crews.csv", "rules.csv", "trigger.csv",
                                                                "perjob.csv"))))

# ------------------------------------------------------------------ Tropical Storm Isaias (problem figure and maps)
from make_figures import isaias_facts  # noqa: E402
from stormcrews.visuals.plots import customer_hours_per_customer  # noqa: E402
fi = isaias_facts()
put("isaiasPeak", num(fi["peak"] / 1e3)); put("isaiasSaidiObs", num(fi["saidi_obs"]))
put("isaiasAfterPeak", f"{100 * fi['after_peak_share']:.0f}")
put("isaiasDamageNinety", str(fi["damage90_h"] - fi["peak_h"]))
for d, lab in ((24, "TwentyFour"), (72, "SeventyTwo"), (96, "NinetySix")):
    put(f"isaiasOut{lab}", num(fi["out"][d] / 1e3))
put("isaiasAidAfterPeak", f"{fi['aid_arrival_h'] - fi['peak_h']:.0f}")
RN = R["NE"]; ist = RN.storm(10); ct = (RN.counties["state"] == "CT").to_numpy()
ires = sc.Simulator(sc.Scenario(region="ne", calibration="anchored")).run(sc.rules.CurrentPractice(), 10, seed=1)
chs = customer_hours_per_customer(ires) * RN.customers
put("isaiasCHObsCT", f"{ist.observed_county[ct].sum() / RN.customers[ct].sum():.1f}")
put("isaiasCHObsRest", f"{ist.observed_county[~ct].sum() / RN.customers[~ct].sum():.1f}")
put("isaiasCHSimCT", f"{chs[ct].sum() / RN.customers[ct].sum():.1f}")
put("isaiasCHSimRest", f"{chs[~ct].sum() / RN.customers[~ct].sum():.1f}")

out_ = subprocess.run([sys.executable, "-m", "pytest", "--collect-only", "-q", "-p", "no:cacheprovider"], cwd=ROOT,
                      capture_output=True, text=True).stdout
put("nTests", out_.strip().splitlines()[-1].split()[0])
put("version", sc.__version__)

# unsigned twins of every signed change, for sentences that already say "rises" or "falls"
for k, v in list(M.items()):
    if v.startswith("+") or v.startswith("$-$"):
        M[k + "Abs"] = v.lstrip("+").replace("$-$", "")
(HERE / "macros.tex").write_text("% generated by paper_gtdla/make_macros.py; do not edit\n"
                                 + "\n".join(f"\\newcommand{{\\{k}}}{{{v}}}" for k, v in sorted(M.items())) + "\n",
                                 encoding="utf-8")
print(len(M), "macros written")
