"""Synthetic storms: storms that never happened, built from a gust field through the damage model.

    from stormcrews.synthetic import gust_field, synthetic_storm
    R = region("ne")
    gust = gust_field(R, center=(41.8, -72.7), peak=38, radius_km=120)          # a strong storm over Hartford
    storm = synthetic_storm(R, gust, template=10)                               # with the timing of Isaias
    Simulator("new_england").run(CurrentPractice(), storm, seed=0)

Damage per county is the damage model's expectation with the storm effect of the template (default specification:
    y_k = N_k (exp(alpha_template + b . x_k) s - 1) / 1000, with s the smearing factor of the log fit),
capped at the largest interruptions per customer seen in the data; precipitation and soil moisture come from the
template storm unless given. With `noise_seed`, y_k is multiplied by a lognormal draw with the model's residual spread
(renormalized to keep the expected total). Hours follow the template's regional profile: every county gets
y_k r(h), with r the template's hourly rises summed over counties and normalized. The template also lends its share of
automatic restoration, its calibrated crews and its peak hour. The damage forecast is the
model's expectation with the median storm effect of the region.
"""
from dataclasses import replace

import numpy as np

from .core.data import Storm
from .damage import model as default_model
from .whatif import _xy_km


def gust_field(region, center, peak, radius_km, background=None):
    """Peak gust per county (m/s): a Gaussian bump of height `peak` at center (lat, lon) over a background level
    (by default the 10th percentile of the observed county peak gusts of the region)."""
    c = region.counties
    if background is None:
        background = float(np.quantile(np.concatenate([region.storm(i).weather[:, 0] for i in region.storm_ids]), 0.10))
    p = _xy_km(c)
    lat0 = np.radians(c["lat"].mean())
    q = np.array([np.radians(center[1]) * 6371.0 * np.cos(lat0), np.radians(center[0]) * 6371.0])
    d2 = ((p - q) ** 2).sum(1)
    return background + (peak - background) * np.exp(-d2 / (2 * radius_km ** 2))


def synthetic_storm(region, gust, template, precip=None, soil_moisture=None, noise_seed=None, damage_model=None,
                    name=None, spec=None):
    """A Storm built from a peak gust per county, with the timing and response of a real template storm."""
    M = damage_model or default_model(spec)
    t = region.storm(region.find(template)) if not isinstance(template, Storm) else template
    c = region.counties; N = region.customers
    gust = np.asarray(gust, float)
    if gust.shape != (len(c),):
        raise ValueError(f"gust must have one value per county ({len(c)})")
    P = t.weather[:, 1] if precip is None else np.broadcast_to(np.asarray(precip, float), gust.shape)
    S = t.weather[:, 2] if soil_moisture is None else np.broadcast_to(np.asarray(soil_moisture, float), gust.shape)
    D = c["density"].to_numpy(float)
    alpha = M.storm_effect(region.code, t.id)
    cap = float(M.info.get("max_interruptions_per_customer", np.inf))
    y = N * np.minimum(M.rate(alpha, gust, P, S, D), cap)
    if noise_seed is not None:
        s = float(M.info.get("residual_sd", 0.0))
        z = np.random.default_rng(noise_seed).normal(-s * s / 2, s, len(y))
        y = np.minimum(y * np.exp(z), cap * N)
    prof = t.rises.sum(1); prof = prof / prof.sum()
    rises = np.outer(prof, y)
    a_med = float(np.median([v for k, v in M.storm_effects.items() if k.startswith(region.code + "_")]))
    forecast = N * np.minimum(M.rate(a_med, gust, P, S, D), cap)
    meta = dict(t.meta)
    meta.update({"name": name or "synthetic", "type": f"synthetic ({t.meta.get('type', '')} timing)", "synthetic": True,
                 "template": t.id, "peak_customers_out": None, "observed_customer_hours": None})
    return replace(t, id=100000 + t.id, rises=rises, forecast=forecast, meta=meta, observed=None, observed_county=None,
                   weather=np.column_stack([gust, P, S]), customers=None, real=False)
