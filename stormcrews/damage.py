"""The storm damage model: customer interruptions per county as a function of weather and network features.

Fitted by scripts/fit_damage_model.py on every storm and county of both regions, with a fixed effect per storm and
intervals clustered by county, and shipped in data/damage_model.json. Features: peak gust G (m/s) with a change of
slope at 20 m/s, accumulated precipitation P (mm), mean soil moisture S (m3/m3) and customers per square mile D.

Specifications (`model(spec)`):
  "equal"       default. Every county-storm cell counts the same: least squares on the log interruption rate,
                    log(1 + 1000 y / N) = alpha_e + b1 G + b2 (G - 20)_+ + b3 log(1 + P) + b4 S + b5 log D
                pooled over both regions
  "equal_ne", "equal_pawv"   the same fitted on one region
  "count"       Poisson with the customers as exposure, E[y] = N exp(alpha_e + b . x); cells weigh by their expected
                counts, so large, dense counties dominate

What-if changes use the coefficients as a multiplier exp(b . dx) on a real storm's damage (the storm effect and
everything the model does not see stay as observed); synthetic storms use the level with a template's storm effect.
"""
import json
from dataclasses import dataclass, field, replace
from functools import lru_cache

import numpy as np

from .core.data import DATA


@dataclass
class DamageModel:
    spec: str
    kind: str                        # "equal" (log rate, cells weigh equally) or "count" (Poisson, exposure)
    coef: dict                       # feature -> estimate
    cov: np.ndarray                  # covariance of the feature coefficients (clustered by county)
    features: list
    knot: float
    storm_effects: dict = field(default_factory=dict)
    info: dict = field(default_factory=dict, repr=False)

    # ------------------------------------------------------------------ pieces
    def x(self, gust, precip, soil, density):
        """Feature matrix (counties x features) in the order of `features`."""
        gust = np.asarray(gust, float)
        cols = {"gust": gust, "gust_above_20": np.clip(gust - self.knot, 0, None),
                "log1p_precip": np.log1p(np.asarray(precip, float)), "soil_moisture": np.asarray(soil, float),
                "log_density": np.log(np.asarray(density, float))}
        return np.column_stack([np.broadcast_to(cols[f], np.shape(gust)) for f in self.features])

    def beta(self):
        return np.array([self.coef[f] for f in self.features])

    def rate(self, alpha, gust, precip, soil, density):
        """Expected interruptions per customer."""
        eta = alpha + self.x(gust, precip, soil, density) @ self.beta()
        if self.kind == "count":
            return np.exp(eta)
        return np.clip(np.exp(eta) * self.info.get("smearing", 1.0) - 1.0, 0.0, None) / 1000.0

    def storm_effect(self, region_code, storm_id):
        return self.storm_effects[f"{region_code}_{storm_id}"]

    def draw(self, seed):
        """A copy with coefficients drawn from their sampling distribution, to carry the model's uncertainty."""
        rng = np.random.default_rng(seed)
        b = rng.multivariate_normal(self.beta(), self.cov)
        return replace(self, coef=dict(zip(self.features, b)))

    def with_coefficients(self, **overrides):
        """A copy with some coefficients replaced, e.g. a gust effect from the literature."""
        c = dict(self.coef); c.update({k: float(v) for k, v in overrides.items()})
        return replace(self, coef=c)

    def effect(self, feature_delta, level=0.95):
        """Multiplier of damage for a change of the features, with its delta-method interval.

        feature_delta: dict feature -> change (e.g. {"gust": 5, "gust_above_20": 5} above the knot,
        {"log1p_precip": 0.5} for half a log point more precipitation).
        """
        d = np.array([feature_delta.get(f, 0.0) for f in self.features])
        z = {0.9: 1.645, 0.95: 1.96, 0.99: 2.576}[level]
        m = float(d @ self.beta()); s = float(np.sqrt(d @ self.cov @ d))
        return np.exp(m), np.exp(m - z * s), np.exp(m + z * s)


@lru_cache(maxsize=8)
def _load(spec):
    d = json.loads((DATA / "damage_model.json").read_text(encoding="utf-8"))
    spec = spec or d["default"]
    s = d["specs"][spec]; f = d["features"]
    info = {k: v for k, v in s.items() if k not in ("coefficients", "covariance", "storm_effects")}
    info.update({"max_interruptions_per_customer": d["max_interruptions_per_customer"],
                 "p99_interruptions_per_customer": d["p99_interruptions_per_customer"],
                 "coefficients": s["coefficients"], "default": d["default"], "specs": sorted(d["specs"])})
    return DamageModel(spec=spec, kind=s["kind"], coef={k: s["coefficients"][k]["estimate"] for k in f},
                       cov=np.array(s["covariance"]), features=f, knot=float(d["knot_gust_ms"]),
                       storm_effects=s["storm_effects"], info=info)


def model(spec=None):
    """The shipped damage model; `spec` is "equal" (default), "equal_ne", "equal_pawv" or "count"."""
    return _load(spec)
