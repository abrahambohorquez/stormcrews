"""Experiments: rules x scenarios x storms x seeds, with tidy tables and paired comparisons.

    exp = run_experiment(REFERENCE, ["new_england", "stress_test"], seeds=range(3), workers=6)
    exp.table                      # one row per scenario, rule, storm and seed, every metric
    exp.summary()                  # per scenario and rule: totals over storms (mean over seeds) and means
    exp.compare("current practice")  # paired differences with bootstrap intervals over storms

Every rule of a scenario sees the same storm draws for a given seed (common random numbers: the same jobs, sizes,
repair times and automatic restorations), so differences between rules are paired.
"""
import warnings
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

from .metrics import ADDITIVE, AVERAGED
from .scenario import Scenario
from .simulator import Simulator, rule_name

_SIMS = {}


def _sim(scenario):
    key = repr(scenario)
    if key not in _SIMS:
        _SIMS[key] = Simulator(scenario)
    return _SIMS[key]


def _one(job):
    scenario, name, rule, storm, seed, keep_curves = job
    res = _sim(scenario).run(rule, storm, seed)
    row = res.row(); row["rule"] = name
    return row, res.metrics["county_mean_outage_h"], (res.curve() if keep_curves else None)


def _as_rules(rules):
    if isinstance(rules, dict):
        return dict(rules)
    if not isinstance(rules, (list, tuple)):
        rules = [rules]
    return {rule_name(r): r for r in rules}


def _as_scenarios(scenarios):
    if isinstance(scenarios, dict) and scenarios and all(isinstance(v, Scenario) for v in scenarios.values()):
        return [s.with_(name=n) for n, s in scenarios.items()]
    if not isinstance(scenarios, (list, tuple)):
        scenarios = [scenarios]
    return [Scenario.load(s) for s in scenarios]


class Experiment:
    def __init__(self, table, county, curves, scenarios):
        self.table, self.county, self.curves, self.scenarios = table, county, curves, scenarios

    def summary(self):
        """Per scenario and rule: additive metrics summed over storms, the others averaged; then mean over seeds."""
        t = self.table
        agg = {**{k: "sum" for k in ADDITIVE}, **{k: "mean" for k in AVERAGED}}
        per_seed = t.groupby(["scenario", "rule", "seed"], sort=False).agg(agg)
        out = per_seed.groupby(level=[0, 1], sort=False).mean().reset_index()
        out.insert(2, "storms", t.groupby(["scenario", "rule"], sort=False)["storm"].nunique().to_numpy())
        out.insert(3, "seeds", t.groupby(["scenario", "rule"], sort=False)["seed"].nunique().to_numpy())
        return out

    def compare(self, baseline="current practice", metrics=("customer_hours", "saidi_min", "over_24h", "over_72h",
                                                              "over_96h", "worst_county_h", "restoration_end_h"),
                n_boot=2000, level=0.95, seed=0):
        return paired_comparison(self.table, baseline, metrics, n_boot, level, seed)


def run_experiment(rules, scenarios="new_england", storms=None, seeds=(0, 1, 2), workers=1, keep_curves=False):
    """Crosses rules x scenarios x storms x seeds.

    rules       a rule, a list of rules, or a dict name -> rule (a rule has decide(state), or is a function of it;
                with workers > 1 rules must be picklable, so use classes rather than lambdas)
    scenarios   a Scenario, a shipped scenario name, a YAML path, or a list or dict of them
    storms      storm ids (default: the scenario's storms)
    """
    rules = _as_rules(rules); scs = _as_scenarios(scenarios)
    jobs = []
    for sc in scs:
        ids = list(storms) if storms is not None else _sim(sc).storm_ids
        jobs += [(sc, n, r, s, int(k), keep_curves) for n, r in rules.items() for s in ids for k in seeds]
    if workers > 1:
        with ProcessPoolExecutor(max_workers=min(int(workers), 6)) as ex:
            res = list(ex.map(_one, jobs, chunksize=4))
    else:
        res = [_one(j) for j in jobs]
    table = pd.DataFrame([r for r, _, _ in res])
    rows = []
    for sc in scs:
        sim = _sim(sc); fips = sim.counties["fips"].to_numpy()
        for n in rules:
            mats = [c for (r, c, _), j in zip(res, jobs) if j[0] is sc and j[1] == n]
            with np.errstate(all="ignore"):
                mean = np.nanmean(np.vstack(mats), axis=0) if mats else np.full(len(fips), np.nan)
            rows += [{"scenario": sc.name, "rule": n, "fips": f, "mean_outage_h": v} for f, v in zip(fips, mean)]
    county = pd.DataFrame(rows)
    curves = {(r["scenario"], r["rule"], r["storm"], r["seed"]): c for r, _, c in res} if keep_curves else {}
    return Experiment(table, county, curves, scs)


def paired_comparison(table, baseline="current practice", metrics=("customer_hours",), n_boot=2000, level=0.95, seed=0):
    """Each rule against the baseline, storm by storm (means over seeds), with bootstrap intervals over storms.

    Additive metrics (customer-hours, SAIDI minutes, customers over a threshold, crew-hours) are compared as totals over
    the storms: relative = sum(rule) / sum(baseline) - 1. The others (worst county, restoration time) as means over
    the storms. The interval resamples storms with replacement. `share_storms_lower` is the share of storms where
    the rule gives a lower value than the baseline.
    """
    rng = np.random.default_rng(seed)
    lo_q, hi_q = (1 - level) / 2, 1 - (1 - level) / 2
    rows = []
    for sc, g in table.groupby("scenario", sort=False):
        per = g.groupby(["rule", "storm"], sort=False)[list(metrics)].mean()
        if baseline not in per.index.get_level_values(0):
            raise ValueError(f"baseline {baseline!r} not in the table")
        base = per.loc[baseline]
        for rule in per.index.get_level_values(0).unique():
            if rule == baseline:
                continue
            r = per.loc[rule].reindex(base.index)
            n = len(base)
            idx = rng.integers(0, n, size=(n_boot, n))
            for m in metrics:
                b, x = base[m].to_numpy(float), r[m].to_numpy(float)
                ok = np.isfinite(b) & np.isfinite(x)
                if m in ADDITIVE:
                    stat = lambda xb, bb: np.nansum(xb, axis=-1) / np.nansum(bb, axis=-1) - 1
                    bv, rv = np.nansum(b), np.nansum(x)
                else:
                    b = np.where(ok, b, np.nan); x = np.where(ok, x, np.nan)
                    stat = lambda xb, bb: np.nanmean(xb, axis=-1) / np.nanmean(bb, axis=-1) - 1
                    bv, rv = np.nanmean(b), np.nanmean(x)
                with np.errstate(all="ignore"), warnings.catch_warnings():
                    warnings.simplefilter("ignore", RuntimeWarning)
                    boot = stat(x[idx], b[idx])
                    rel = rv / bv - 1 if bv else np.nan
                    q_lo, q_hi = (float(np.nanquantile(boot, q)) if np.isfinite(boot).any() else np.nan
                                  for q in (lo_q, hi_q))
                rows.append({"scenario": sc, "rule": rule, "metric": m, "baseline": baseline, "baseline_value": bv,
                             "rule_value": rv, "difference": rv - bv, "relative": rel,
                             "relative_low": q_lo, "relative_high": q_hi,
                             "share_storms_lower": float(np.mean(x[ok] < b[ok])) if ok.any() else np.nan,
                             "storms": int(ok.sum())})
    return pd.DataFrame(rows)
