"""Runs behind the papers. Every run uses the anchored calibration (the local fleet set from the
line crews per customer out that utilities reported in public filings), current practice unless stated, and fixed
seeds. Results go to experiments/results/.

    python experiments/experiments.py gust        # (a) peak gust changed from -5 to +10 m/s, every New England storm
    python experiments/experiments.py crews       # (b) local fleet and mutual-aid lag, every New England storm
    python experiments/experiments.py rules       # (c) job order and dispatch rules, every New England storm
    python experiments/experiments.py perjob      # customers per repair job 20, 35, 55, with the repair time refitted
    python experiments/experiments.py trigger     # the reactive aid trigger at 0.5, 1 and 2 % of customers out
    python experiments/experiments.py holdout     # leave-one-storm-out check of the anchored calibration, both regions
    python experiments/experiments.py all

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
SEEDS = (0, 1, 2, 3, 4)
CAL = "anchored_territory"     # the anchored fleet, each utility territory with its own crews (scripts/calibrate_territories.py)
TERRITORIES = True
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
    return Scenario(name=region, region=region, calibration=CAL, territories=TERRITORIES, **kw)


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
def arrival_hours(storm):
    """Hours over which the middle 80 % of the storm's damage arrives (from the 10th to the 90th percentile of the
    cumulative regional rises). An input of the simulator, not an outcome of the restoration."""
    by_h = storm.rises.sum(1); cs = np.cumsum(by_h) / by_h.sum()
    return float(np.searchsorted(cs, 0.9) - np.searchsorted(cs, 0.1))


def storm_features(storm):
    """Per-storm quantities read from the simulator inputs only (rises, weather, the peak of customers out)."""
    r = storm.rises; tot = r.sum(); by_c = r.sum(0)
    return {"log_peak": float(np.log(storm.meta["peak_customers_out"])), "log_tot": float(np.log(tot)),
            "arrival_h": arrival_hours(storm), "log_ncty": float(np.log((by_c > 0.01 * by_c.max()).sum())),
            "gust_w": float((storm.weather[:, 0] * by_c).sum() / tot)}


# Candidate predictors of the held-out repair time, all from the inputs: (features, whether sigma is regressed too)
CANDIDATES = {"peak": (["log_peak"], False),
              "peak+tot": (["log_peak", "log_tot"], False),
              "peak+arrival": (["log_peak", "arrival_h"], False),
              "tot+arrival+ncty": (["log_tot", "arrival_h", "log_ncty"], False),
              "peak+arrival+gust": (["log_peak", "arrival_h", "gust_w"], False),
              "peak+arrival, sigma regressed": (["log_peak", "arrival_h"], True),
              "peak+arrival+tot": (["log_peak", "arrival_h", "log_tot"], False)}
CHOSEN = "peak+arrival, sigma regressed"


def _holdout_jobs(reg, candidate, label):
    """Leave-one-storm-out jobs of one candidate predictor in one region, every seed; returns (jobs, parameter rows)."""
    feats, sig_reg = CANDIDATES[candidate]
    R = load_region(reg); sc = base(reg); ids = storms(sc)
    m = {x["id"]: x for x in R.meta}
    S = {s: R.storm(s) for s in ids}
    F = pd.DataFrame([storm_features(S[s]) for s in ids], index=ids)
    X = np.column_stack([np.ones(len(ids))] + [F[f].to_numpy(float) for f in feats])
    lr = np.log([m[s]["calibrations"][CAL]["repair_mean_h"] for s in ids])
    sg = np.array([m[s]["calibrations"][CAL]["size_sigma"] for s in ids])
    jobs, pred = [], []
    for i, s in enumerate(ids):
        keep = np.arange(len(ids)) != i
        b = np.linalg.lstsq(X[keep], lr[keep], rcond=None)[0]
        r_hat = float(np.exp(X[i] @ b))
        if sig_reg:
            bs = np.linalg.lstsq(X[keep], sg[keep], rcond=None)[0]
            s_hat = float(np.clip(np.round((X[i] @ bs) * 2) / 2, 0, 3.5))
        else:
            s_hat = float(np.round(np.median(sg[keep]) * 2) / 2)
        row = {"region": reg, "candidate": candidate, "storm": s, "repair_pred_h": r_hat,
               "repair_fit_h": float(np.exp(lr[i])), "sigma_pred": s_hat, "sigma_fit": float(sg[i]),
               "log_customer_hours_obs": float(np.log(S[s].observed.sum()))}
        row.update({f: float(F.loc[s, f]) for f in F.columns})
        row.update({f"coef_{f}": float(c) for f, c in zip(feats, b[1:])})
        pred.append(row)
        for k in SEEDS:
            jobs.append((dict(label, storm_id=s), sc, _with_cal(S[s], repair_mean_h=r_hat, size_sigma=s_hat), k, None))
    return jobs, pred, sc, ids


def holdout(workers):
    """For each storm, the repair time and job-size dispersion come from the other storms of its region: log repair
    time and the job-size dispersion both regressed on the log peak customers out and on the hours over which the
    damage arrives (arrival_hours). The fleet keeps the anchor (public filings times the storm's peak). Current practice is
    then simulated on the held-out storm with every seed in SEEDS and compared with what was observed; the in-sample
    anchored calibration is run alongside with the same seeds.

    The two predictors (CHOSEN) were picked by this same leave-one-out check among the CANDIDATES, all built from
    the simulator inputs; holdout_select() runs every candidate and lets the selection itself be held out."""
    jobs, pred = [], []
    for reg in ("ne", "pawv"):
        j, p, sc, ids = _holdout_jobs(reg, CHOSEN, {"region": reg, "fit": "held out"})
        jobs += j; pred += p
        jobs += [({"region": reg, "fit": "in sample"}, sc, s, k, None) for s in ids for k in SEEDS]
    pd.DataFrame(pred).to_csv(OUT / "holdout_parameters.csv", index=False)
    df = run(jobs, workers)
    df.drop(columns=["storm_id"], errors="ignore").to_csv(OUT / "holdout.csv", index=False)


def holdout_select(workers):
    """Every candidate predictor of the repair time under the same leave-one-storm-out check, so that the choice of
    predictors can itself be held out (nested selection, computed in make_macros) and compared with a regression of
    customer-hours on the same inputs that uses no simulator."""
    jobs, pred = [], []
    for reg in ("ne", "pawv"):
        for c in CANDIDATES:
            j, p, _, _ = _holdout_jobs(reg, c, {"region": reg, "candidate": c})
            jobs += j; pred += p
    pd.DataFrame(pred).to_csv(OUT / "holdout_candidates_parameters.csv", index=False)
    run(jobs, workers).drop(columns=["storm_id"], errors="ignore").to_csv(OUT / "holdout_candidates.csv", index=False)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("what", choices=("gust", "crews", "rules", "perjob", "trigger", "holdout", "holdout_select", "all"))
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()
    OUT.mkdir(exist_ok=True)
    todo = ("gust", "crews", "rules", "trigger", "holdout", "holdout_select", "perjob") if a.what == "all" else (a.what,)
    for w in todo:
        t0 = time.time(); globals()[w](min(a.workers, 4)); print(f"{w}: {time.time() - t0:.0f} s", flush=True)
