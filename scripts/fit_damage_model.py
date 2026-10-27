"""Fits the storm damage model of StormCrews, writes stormcrews/data/damage_model.json and the damage forecast of every
storm in stormcrews/data/<region>.npz.

One observation per storm and county, in both regions (85 storms x their counties). The response is the storm's
customer interruptions in the county, y (the sum of the hourly rises above the pre-storm baseline); the features are the
peak gust G (m/s) with a change of slope at 20 m/s, accumulated precipitation P (mm), mean soil moisture S (m3/m3)
and customers per square mile D, with a fixed effect per storm. Two weightings:

  equal  (default) every county-storm cell counts the same. Least squares on the log interruption rate,
             log(1 + 1000 y / N) = alpha_e + b1 G + b2 (G - 20)_+ + b3 log(1 + P) + b4 S + b5 log D + e
         Fitted pooled over both regions (shipped default) and per region (equal_ne, equal_pawv).
  count  Poisson pseudo-likelihood with the customers as exposure, E[y] = N exp(alpha_e + b . x). It weights cells by
         their expected counts, so large, dense counties dominate.

Intervals are clustered by county. Fit quality: R2 (or deviance explained) with and without the covariates, in sample
and on counties held out in five folds (storm effects re-estimated on the rest).

The damage forecast a rule may read (`state.forecast`) is the expected interruptions per county of the "equal"
specification fitted on the other storms of the same region, with the median storm effect of those storms and the
storm's own weather, capped at the largest interruptions per customer in the data. It never sees the storm's outages.

    python scripts/fit_damage_model.py
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from stormcrews.core.data import region  # noqa: E402

KNOT = 20.0
FEATURES = ["gust", "gust_above_20", "log1p_precip", "soil_moisture", "log_density"]


def cells():
    rows = []
    for code in ("ne", "pawv"):
        R = region(code); c = R.counties
        for m in R.meta:
            st = R.storm(m["id"]); w = st.weather
            for k in range(len(c)):
                rows.append({"region": code, "storm": f"{code}_{m['id']}", "storm_id": m["id"], "fips": int(c["fips"].iloc[k]),
                             "y": float(st.rises[:, k].sum()), "customers": float(c["customers"].iloc[k]),
                             "gust": w[k, 0], "precip": w[k, 1], "soil_moisture": w[k, 2],
                             "density": c["density"].iloc[k]})
    d = pd.DataFrame(rows)
    d["gust_above_20"] = np.clip(d["gust"] - KNOT, 0, None)
    d["log1p_precip"] = np.log1p(d["precip"]); d["log_density"] = np.log(d["density"])
    d["log_rate"] = np.log1p(1000 * d["y"] / d["customers"])
    return d


def design(d, with_x=True):
    fe = pd.get_dummies(d["storm"], prefix="", prefix_sep="", dtype=float)
    return pd.concat([fe, d[FEATURES]], axis=1) if with_x else fe


def fit(d, kind, with_x=True):
    import statsmodels.api as sm
    X = design(d, with_x); groups = pd.factorize(d["fips"])[0]
    if kind == "count":
        m = sm.GLM(d["y"], X, family=sm.families.Poisson(), offset=np.log(d["customers"]))
        return m.fit(cov_type="cluster", cov_kwds={"groups": groups}, maxiter=300)
    return sm.OLS(d["log_rate"], X).fit(cov_type="cluster", cov_kwds={"groups": groups})


def poisson_deviance(y, mu):
    with np.errstate(divide="ignore", invalid="ignore"):
        t = np.where(y > 0, y * np.log(y / mu), 0.0)
    return float(2 * np.sum(t - (y - mu)))


def quality(d, kind):
    """In-sample and held-out-county fit, with and without the covariates."""
    rng = np.random.default_rng(2026)
    fips = d["fips"].unique(); fold = dict(zip(fips, rng.integers(0, 5, len(fips))))
    f_ = d["fips"].map(fold).to_numpy()
    pred = {True: np.zeros(len(d)), False: np.zeros(len(d))}
    for f in range(5):
        tr, te = d[f_ != f], d[f_ == f]
        for wx in (True, False):
            r = fit(tr, kind, wx)
            Xte = design(te, wx).reindex(columns=r.params.index, fill_value=0.0).to_numpy()
            eta = Xte @ r.params.to_numpy()
            pred[wx][f_ == f] = np.exp(eta) * te["customers"].to_numpy() if kind == "count" else eta
    full, fe = fit(d, kind), fit(d, kind, False)
    if kind == "count":
        y = d["y"].to_numpy(); null = poisson_deviance(y, d["customers"].to_numpy() * y.sum() / d["customers"].sum())
        score = lambda mu: 1 - poisson_deviance(y, mu) / null
        return {"measure": "deviance explained", "in_sample": score(full.mu), "in_sample_storm_effects_only": score(fe.mu),
                "held_out_counties": score(pred[True]), "held_out_counties_storm_effects_only": score(pred[False])}
    y = d["log_rate"].to_numpy(); tss = float(((y - y.mean()) ** 2).sum())
    score = lambda p: 1 - float(((y - p) ** 2).sum()) / tss
    return {"measure": "R2 of log(1 + interruptions per 1000 customers)", "in_sample": float(full.rsquared),
            "in_sample_storm_effects_only": float(fe.rsquared), "held_out_counties": score(pred[True]),
            "held_out_counties_storm_effects_only": score(pred[False])}


def spec(d, kind):
    r = fit(d, kind); ci = r.conf_int()
    out = {"kind": kind, "cells": int(len(d)), "storms": int(d["storm"].nunique()), "counties": int(d["fips"].nunique()),
           "coefficients": {f: {"estimate": float(r.params[f]), "se": float(r.bse[f]), "low": float(ci.loc[f, 0]),
                                "high": float(ci.loc[f, 1])} for f in FEATURES},
           "covariance": r.cov_params().loc[FEATURES, FEATURES].to_numpy().tolist(),
           "storm_effects": {s: float(r.params[s]) for s in d["storm"].unique()},
           "fit": quality(d, kind)}
    if kind == "equal":
        res = d["log_rate"].to_numpy() - r.fittedvalues.to_numpy()
        out["residual_sd"] = float(res.std()); out["smearing"] = float(np.mean(np.exp(res)))
    else:
        ok = d["y"].to_numpy() > 0
        out["residual_sd"] = float(np.std(np.log(d["y"].to_numpy()[ok] / r.mu[ok])))
    return out


def forecasts(d):
    """Leave-one-storm-out damage forecast of every storm, written into the region's npz (other arrays untouched)."""
    from stormcrews.core.data import DATA
    cap = float((d["y"] / d["customers"]).max())
    for code in ("ne", "pawv"):
        dr = d[d["region"] == code].reset_index(drop=True)
        path = DATA / f"{code}.npz"
        with np.load(path) as z:
            arrays = {k: z[k] for k in z.files}
        for sid in dr["storm_id"].unique():
            train = dr[dr["storm_id"] != sid].reset_index(drop=True)
            X = design(train).to_numpy(float); y = train["log_rate"].to_numpy(float)
            b, *_ = np.linalg.lstsq(X, y, rcond=None)
            cols = list(design(train).columns); nfe = len(cols) - len(FEATURES)
            smear = float(np.mean(np.exp(y - X @ b)))
            alpha = float(np.median(b[:nfe])); beta = b[nfe:]
            te = dr[dr["storm_id"] == sid]
            eta = alpha + te[FEATURES].to_numpy(float) @ beta
            n = te["customers"].to_numpy(float)
            yhat = n * np.minimum(np.clip(np.exp(eta) * smear - 1.0, 0.0, None) / 1000.0, cap)
            arrays[f"forecast_{sid}"] = yhat.astype(np.float32)
        np.savez_compressed(path, **arrays)
        print(code, "forecasts written for", dr["storm_id"].nunique(), "storms")


def main():
    d = cells()
    specs = {"equal": spec(d, "equal"), "count": spec(d, "count")}
    for code in ("ne", "pawv"):
        specs[f"equal_{code}"] = spec(d[d["region"] == code].reset_index(drop=True), "equal")
    shares = d["y"] / d["customers"]
    out = {"description": "Storm damage per county; see scripts/fit_damage_model.py", "default": "equal",
           "knot_gust_ms": KNOT, "features": FEATURES, "specs": specs,
           "max_interruptions_per_customer": float(shares.max()),
           "p99_interruptions_per_customer": float(shares.quantile(0.99))}
    (ROOT / "stormcrews" / "data" / "damage_model.json").write_text(json.dumps(out, indent=1), encoding="utf-8",
                                                                    newline="\n")
    print(f"{len(d)} cells, {d['storm'].nunique()} storms, {d['fips'].nunique()} counties")
    for name, s in specs.items():
        print(name, s["kind"], {k: round(v, 3) for k, v in s["fit"].items() if isinstance(v, float)})
        for f in FEATURES:
            c = s["coefficients"][f]; print(f"  {f:15s} {c['estimate']:+.4f}  [{c['low']:+.4f}, {c['high']:+.4f}]")
    forecasts(d)


if __name__ == "__main__":
    main()
