"""Runs behind the GT&D Latin America 2027 paper. Every run uses the anchored calibration (the local fleet set from the
line crews per customer out that utilities reported in public filings), current practice unless stated, and fixed
seeds. Results go to paper_gtdla/results/.

    python paper_gtdla/experiments.py gust        # (a) peak gust changed from -5 to +10 m/s, every New England storm
    python paper_gtdla/experiments.py crews       # (b) local fleet and mutual-aid lag, every New England storm
    python paper_gtdla/experiments.py rules       # (c) job order and dispatch rules, every New England storm
    python paper_gtdla/experiments.py perjob      # customers per repair job 20, 35, 55, with the repair time refitted
    python paper_gtdla/experiments.py trigger     # the reactive aid trigger at 0.5, 1 and 2 % of customers out
    python paper_gtdla/experiments.py holdout     # leave-one-storm-out check of the anchored calibration, both regions
    python paper_gtdla/experiments.py all

At most four worker processes (--workers).
"""
import argparse
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from stormcrews import Scenario, Simulator  # noqa: E402
from stormcrews.core.data import region as load_region  # noqa: E402
from stormcrews.rules import CurrentPractice  # noqa: E402

OUT = Path(__file__).resolve().parent / "results"
SEEDS = (0, 1, 2)
CAL = "anchored"
# Line crews at work per 1,000 customers out at the utility's peak, from the five public filings that anchor the fleet
# (line crews internal and external, tree crews excluded, with the same utility's peak customers out)
ANCHORS = {"Central Maine Power, Oct. 2019 (press release)": 3.06,
           "Eversource Connecticut, Isaias 2020 (CT legislative forum)": 4.04,
           "Eversource New Hampshire, Dec. 2022 (NH PUC DE 24-041)": 5.11,
           "Duquesne Light, Apr. 2025 (PA PUC post-storm report)": 1.63,
           "FirstEnergy Pennsylvania, Apr. 2025 (PA PUC post-storm report)": 5.90}
ANCHOR_PER_1000 = 4.0          # the New England median of the filings, which sets the anchored fleet of every storm
METRICS = ("customer_hours", "saidi_min", "customers_interrupted", "over_24h", "over_72h", "over_96h",
           "restoration_end_h", "worst_county_h", "crews_max", "local_crew_hours", "aid_crew_hours",
           "busy_crew_hours", "travel_crew_hours", "caidi_h")
_SIMS = {}


def _sim(sc):
    key = repr(sc)
    if key not in _SIMS:
        _SIMS[key] = Simulator(sc)
    return _SIMS[key]


def _one(job):
    """job: (label dict, scenario, storm id or Storm, seed, execution dict)."""
    label, sc, storm, seed, execution = job
    res = _sim(sc).run(CurrentPractice(), storm, seed, execution=execution or None)
    row = dict(label)
    row.update({"storm": res.storm.id, "seed": seed})
    row.update({k: res.metrics[k] for k in METRICS})
    curve = res.curve()
    row["peak_sim"] = float(curve.max())
    if res.storm.observed is not None:
        st = res.storm; H = st.hours
        row["observed_customer_hours"] = float(st.observed.sum()); row["peak_obs"] = float(st.observed.max())
        row["t50_obs"], row["t90_obs"] = shape(st.observed, H)
        row["t50_sim"], row["t90_sim"] = shape(curve, H)
    return row


def shape(c, H):
    """Hours from the peak until the curve falls to one half and to one tenth of the peak, inside the storm window."""
    c = np.asarray(c[:H], float); p = int(np.argmax(c)); top = c[p]
    out = []
    for lvl in (0.5, 0.1):
        below = np.flatnonzero(c[p:] <= lvl * top)
        out.append(float(below[0]) if len(below) else float(H - p))
    return out


def run(jobs, workers):
    if workers > 1:
        with ProcessPoolExecutor(max_workers=min(workers, 4)) as ex:
            rows = list(ex.map(_one, jobs, chunksize=2))
    else:
        rows = [_one(j) for j in jobs]
    return pd.DataFrame(rows)


def base(region="ne", **kw):
    return Scenario(name=region, region=region, calibration=CAL, **kw)


def storms(sc):
    return Simulator(sc).storm_ids


def gust(workers):
    jobs = []
    for dg in (-5.0, -2.5, 0.0, 2.5, 5.0, 7.5, 10.0):
        sc = base(what_if={"gust_change": dg} if dg else None)
        jobs += [({"gust_change": dg}, sc, s, k, None) for s in storms(sc) for k in SEEDS]
    run(jobs, workers).to_csv(OUT / "gust.csv", index=False)


def crews(workers):
    jobs = []
    for f in (0.4, 0.5, 0.75, 1.0, 1.25, 1.475):
        sc = base(fleet_factor=f)
        jobs += [({"sweep": "fleet", "fleet_factor": f, "aid_lead_h": 24.0, "crews_per_1000": ANCHOR_PER_1000 * f},
                  sc, s, k, None) for s in storms(sc) for k in SEEDS]
    for lag in (12.0, 36.0, 48.0, 72.0):
        sc = base(aid_lead_h=lag)
        jobs += [({"sweep": "lag", "fleet_factor": 1.0, "aid_lead_h": lag, "crews_per_1000": ANCHOR_PER_1000},
                  sc, s, k, None) for s in storms(sc) for k in SEEDS]
    run(jobs, workers).to_csv(OUT / "crews.csv", index=False)


RULES = {"largest first": {}, "oldest first": {"job_order": "oldest"},
         "largest first, 48 h guard": {"job_order": "guard"},
         "proportional dispatch": {"dispatch": "relative"}}


def rules(workers):
    jobs = []
    sc = base()
    for name, ex in RULES.items():
        jobs += [({"rule": name}, sc, s, k, ex) for s in storms(sc) for k in SEEDS]
    run(jobs, workers).to_csv(OUT / "rules.csv", index=False)


def trigger(workers):
    jobs = []
    for tr in (0.005, 0.01, 0.02):
        sc = base(reactive_trigger=tr)
        jobs += [({"reactive_trigger": tr}, sc, s, k, None) for s in storms(sc) for k in SEEDS]
    run(jobs, workers).to_csv(OUT / "trigger.csv", index=False)


# ------------------------------------------------------------------ customers per job, with the repair time refitted
def _with_cal(storm, **changes):
    meta = dict(storm.meta); cals = dict(meta["calibrations"]); c = dict(cals[CAL]); c.update(changes)
    cals[CAL] = c; meta["calibrations"] = cals
    return replace(storm, meta=meta)


def _refit(job):
    """Repair time that makes current practice reproduce the storm's observed customer-hours with `cpj` customers per
    job, the anchored fleet and dispersion kept (bisection on the log of the repair time, seed 0)."""
    region, sid, cpj = job
    sc = base(region, customers_per_job=cpj); sim = _sim(sc); st = sim.region.storm(sid)
    target = st.observed.sum(); r0 = st.calibration(CAL)["repair_mean_h"]
    lo, hi = r0 / 8, r0 * 8
    for _ in range(18):
        mid = float(np.sqrt(lo * hi))
        ch = sim.run(CurrentPractice(), _with_cal(st, repair_mean_h=mid), 0).metrics["customer_hours"]
        if abs(ch / target - 1) < 0.01:
            break
        lo, hi = (mid, hi) if ch < target else (lo, mid)
    return {"region": region, "storm": sid, "customers_per_job": cpj, "repair_mean_h": mid, "repair_anchored_h": r0,
            "chi_ratio_fit": ch / target}


def perjob(workers):
    R = load_region("ne"); ids = storms(base())
    fits = [{"region": "ne", "storm": s, "customers_per_job": 35.0,
             "repair_mean_h": R.storm(s).calibration(CAL)["repair_mean_h"],
             "repair_anchored_h": R.storm(s).calibration(CAL)["repair_mean_h"], "chi_ratio_fit": np.nan} for s in ids]
    jobs = [("ne", s, c) for c in (20.0, 55.0) for s in ids]
    with ProcessPoolExecutor(max_workers=min(workers, 4)) as ex:
        fits += list(ex.map(_refit, jobs))
    fits = pd.DataFrame(fits); fits.to_csv(OUT / "perjob_fit.csv", index=False)
    # the demonstrations under each assumption: real storms, +5 m/s gust, 75 % fleet, aid lag 48 h
    cases = {"real": {}, "gust +5": {"what_if": {"gust_change": 5.0}}, "fleet 75 %": {"fleet_factor": 0.75},
             "aid lag 48 h": {"aid_lead_h": 48.0}}
    out = []
    for cpj, g in fits.groupby("customers_per_job"):
        rep = dict(zip(g["storm"], g["repair_mean_h"]))
        for case, kw in cases.items():
            sc = base(customers_per_job=cpj, **kw)
            jobs = [({"customers_per_job": cpj, "case": case}, sc, _with_cal(R.storm(s), repair_mean_h=rep[s]), k, None)
                    for s in ids for k in SEEDS]
            out.append(run(jobs, workers))
    pd.concat(out).to_csv(OUT / "perjob.csv", index=False)


# ------------------------------------------------------------------ leave one storm out
def holdout(workers):
    """For each storm, the repair time and job-size dispersion come from the other storms of its region: log repair
    time regressed on log peak customers out, and the median dispersion. The fleet keeps the anchor (public filings
    times the storm's peak). Current practice is then simulated on the held-out storm and compared with what was
    observed; the in-sample anchored calibration is run alongside with the same seed."""
    jobs, pred = [], []
    for reg in ("ne", "pawv"):
        R = load_region(reg); sc = base(reg); ids = storms(sc)
        m = {x["id"]: x for x in R.meta}
        lp = np.log([m[s]["peak_customers_out"] for s in ids])
        lr = np.log([m[s]["calibrations"][CAL]["repair_mean_h"] for s in ids])
        sg = np.array([m[s]["calibrations"][CAL]["size_sigma"] for s in ids])
        for i, s in enumerate(ids):
            keep = np.arange(len(ids)) != i
            b, a = np.polyfit(lp[keep], lr[keep], 1)
            r_hat = float(np.exp(a + b * lp[i])); s_hat = float(np.round(np.median(sg[keep]) * 2) / 2)
            pred.append({"region": reg, "storm": s, "repair_pred_h": r_hat, "repair_fit_h": float(np.exp(lr[i])),
                         "sigma_pred": s_hat, "sigma_fit": float(sg[i]), "slope": b})
            st = R.storm(s)
            jobs.append(({"region": reg, "fit": "held out"}, sc, _with_cal(st, repair_mean_h=r_hat, size_sigma=s_hat), 0, None))
            jobs.append(({"region": reg, "fit": "in sample"}, sc, s, 0, None))
    pd.DataFrame(pred).to_csv(OUT / "holdout_parameters.csv", index=False)
    run(jobs, workers).to_csv(OUT / "holdout.csv", index=False)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("what", choices=("gust", "crews", "rules", "perjob", "trigger", "holdout", "all"))
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()
    OUT.mkdir(exist_ok=True)
    todo = ("gust", "crews", "rules", "trigger", "holdout", "perjob") if a.what == "all" else (a.what,)
    for w in todo:
        t0 = time.time(); globals()[w](min(a.workers, 4)); print(f"{w}: {time.time() - t0:.0f} s", flush=True)
