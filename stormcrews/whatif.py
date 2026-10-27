"""What-if changes to a real storm: weather and network knobs applied through the damage model.

Every knob changes the hourly rises x[h, k] of customers newly out in county k. With every knob at its neutral value
the storm is returned untouched, so a what-if scenario at defaults replays the real storm bit for bit.

Order of application:
1. footprint   the damage pattern per customer moves (shift_km = east, north) or spreads (scale > 1) around the
               damage-weighted centroid: county k takes the hourly damage per customer, and the weather, of the county
               nearest to c + (p_k - c - shift) / scale; nothing if no county is within reach
2. weather and network, through the damage model (default specification "equal": every county-storm cell weighs the
   same; see damage.py), a multiplier per county
       m_k = exp( b1 dG + b2 [(G + dG - 20)_+ - (G - 20)_+] + b3 [log(1 + fP) - log(1 + P)] + b4 dS )
             x (1 + g_k)^(1 + b5)
   dG gust change (m/s), f precipitation factor, dS soil-moisture change (m3/m3), g customer growth (customers and
   damage grow, density enters through b5).
   Interruptions per customer are capped at the largest share observed in the data.
3. duration    the cumulative hourly profile of every county is stretched in time by `duration_factor`
4. automation  the share restored automatically within the hour is set to `auto_share`
The damage forecast is changed by the same footprint and multipliers.
"""
from dataclasses import dataclass, field, fields, replace

import numpy as np

from .damage import model as default_model

NEUTRAL = {"gust_change": 0.0, "precip_factor": 1.0, "soil_moisture_change": 0.0, "customer_growth": 0.0, "footprint_shift_km": (0.0, 0.0), "footprint_scale": 1.0, "duration_factor": 1.0,
           "auto_share": None, "model_draw": None, "coefficients": None}


@dataclass
class WhatIf:
    gust_change: object = 0.0               # m/s, a number or {fips: change}
    precip_factor: object = 1.0             # multiplies accumulated precipitation, number or {fips: factor}
    soil_moisture_change: object = 0.0      # m3/m3, number or {fips: change}
    customer_growth: object = 0.0           # share, number or {fips: share}
    footprint_shift_km: tuple = (0.0, 0.0)  # east, north
    footprint_scale: float = 1.0            # > 1 spreads the damage over a wider area
    duration_factor: float = 1.0            # > 1 makes the damage arrive more slowly
    auto_share: float = None                # share restored automatically (grid automation); None keeps the storm's
    model_draw: int = None                  # seed of a draw of the model coefficients (uncertainty); None = estimates
    coefficients: dict = None               # override coefficients, e.g. {"gust": 0.1}
    damage_spec: str = None                 # damage model: None or "equal" (default, counties weigh equally),
                                            # "equal_region" (fitted on the storm's region), "count" (exposure-weighted)

    def __post_init__(self):
        self.footprint_shift_km = tuple(float(v) for v in (self.footprint_shift_km or (0.0, 0.0)))
        self.footprint_scale = float(self.footprint_scale); self.duration_factor = float(self.duration_factor)
        if self.footprint_scale <= 0 or self.duration_factor <= 0:
            raise ValueError("footprint_scale and duration_factor must be positive")
        if self.auto_share is not None and not 0 <= float(self.auto_share) < 1:
            raise ValueError("auto_share must be in [0, 1)")

    @classmethod
    def from_dict(cls, d):
        if d is None:
            return cls()
        if isinstance(d, WhatIf):
            return d
        known = {f.name for f in fields(cls)}
        unknown = set(d) - known
        if unknown:
            raise ValueError(f"unknown what_if knobs: {sorted(unknown)}")
        return cls(**d)

    def is_neutral(self):
        return all(_neutral(getattr(self, k), v) for k, v in NEUTRAL.items())

    def weather_neutral(self):
        return all(_neutral(getattr(self, k), NEUTRAL[k]) for k in
                   ("gust_change", "precip_factor", "soil_moisture_change", "customer_growth"))


def _neutral(v, n):
    if isinstance(v, dict):
        return all(_neutral(x, n) for x in v.values()) if n is not None else not v
    if isinstance(n, tuple):
        return tuple(v) == n
    return v == n


def per_county(value, counties, neutral):
    """A number or {fips: value} as one value per county."""
    if isinstance(value, dict):
        f = counties["fips"].astype(int).map({int(k): float(v) for k, v in value.items()})
        return f.fillna(neutral).to_numpy(float)
    return np.full(len(counties), float(value))


def _xy_km(counties):
    lat, lon = counties["lat"].to_numpy(float), counties["lon"].to_numpy(float)
    lat0 = np.radians(lat.mean())
    return np.column_stack([np.radians(lon) * 6371.0 * np.cos(lat0), np.radians(lat) * 6371.0])


def footprint_map(counties, damage, shift_km=(0.0, 0.0), scale=1.0):
    """Source county of every county after moving the damage pattern; -1 where nothing reaches."""
    p = _xy_km(counties)
    w = damage / damage.sum() if damage.sum() > 0 else np.full(len(p), 1 / len(p))
    c = (p * w[:, None]).sum(0)
    q = c + (p - c - np.asarray(shift_km, float)) / scale
    d = np.sqrt(((q[:, None, :] - p[None, :, :]) ** 2).sum(-1))
    nn = np.sqrt(((p[:, None, :] - p[None, :, :]) ** 2).sum(-1) + np.eye(len(p)) * 1e12).min(1)
    reach = np.max(nn)                                    # a source farther than any county's nearest neighbor is out
    src = d.argmin(1)
    src[d.min(1) > reach] = -1
    return src


def stretch(rises, factor):
    """Stretches the cumulative hourly profile of every county in time; totals are kept."""
    H, K = rises.shape
    H2 = max(1, int(np.ceil(H * factor)))
    cum = np.vstack([np.zeros((1, K)), np.cumsum(rises, axis=0)])
    t_old = np.arange(H + 1) * factor
    t_new = np.arange(H2 + 1).astype(float)
    new = np.column_stack([np.interp(t_new, t_old, cum[:, k]) for k in range(K)])
    out = np.clip(np.diff(new, axis=0), 0, None)
    tot, tot2 = rises.sum(0), out.sum(0)
    return out * np.divide(tot, tot2, out=np.zeros(K), where=tot2 > 0)


def apply(storm, region, whatif, damage_model=None):
    """A modified copy of `storm` (or the storm itself when every knob is neutral)."""
    wi = WhatIf.from_dict(whatif)
    if wi.is_neutral():
        return storm
    spec = wi.damage_spec
    if spec == "equal_region":
        spec = f"equal_{region.code}"
    M = damage_model or default_model(spec)
    if wi.coefficients:
        M = M.with_coefficients(**wi.coefficients)
    if wi.model_draw is not None:
        M = M.draw(wi.model_draw)
    c = region.counties
    N = (storm.customers if storm.customers is not None else region.customers).astype(float)
    rises, forecast, weather = storm.rises.copy(), storm.forecast.copy(), storm.weather.copy()
    meta = dict(storm.meta); meta["what_if"] = True

    # 1. footprint
    if wi.footprint_shift_km != (0.0, 0.0) or wi.footprint_scale != 1.0:
        src = footprint_map(c, rises.sum(0), wi.footprint_shift_km, wi.footprint_scale)
        ok = src >= 0; s = np.where(ok, src, 0)
        rate = rises / N; frate = forecast / N
        rises = np.where(ok, rate[:, s], 0.0) * N
        forecast = np.where(ok, frate[s], 0.0) * N
        weather = weather[s]
        weather[~ok, 0] = np.minimum(weather[~ok, 0], M.knot)          # outside the footprint: no damaging gust

    # 2. weather and network multiplier
    N2 = N
    if not wi.weather_neutral():
        G, P, S = weather[:, 0], weather[:, 1], weather[:, 2]
        dG = per_county(wi.gust_change, c, 0.0); fP = per_county(wi.precip_factor, c, 1.0)
        dS = per_county(wi.soil_moisture_change, c, 0.0)
        g = per_county(wi.customer_growth, c, 0.0)
        b = M.coef
        logm = (b["gust"] * dG + b["gust_above_20"] * (np.clip(G + dG - M.knot, 0, None) - np.clip(G - M.knot, 0, None))
                + b["log1p_precip"] * (np.log1p(fP * P) - np.log1p(P)) + b["soil_moisture"] * dS)
        m = np.exp(logm) * (1 + g) ** (1 + b["log_density"])
        N2 = N * (1 + g)
        cap = float(M.info.get("max_interruptions_per_customer", np.inf)) * N2
        tot = rises.sum(0) * m
        m = np.where(tot > cap, m * cap / np.maximum(tot, 1e-12), m)
        rises = rises * m; forecast = forecast * m
        weather = np.column_stack([G + dG, fP * P, S + dS])

    # 3. duration
    if wi.duration_factor != 1.0:
        rises = stretch(rises, wi.duration_factor)
        meta["peak_hour"] = int(round(meta["peak_hour"] * wi.duration_factor))
        meta["hours"] = rises.shape[0]

    # 4. automation
    if wi.auto_share is not None:
        meta["auto_share"] = float(wi.auto_share)

    return replace(storm, rises=rises, forecast=forecast, meta=meta, weather=weather,
                   customers=N2 if N2 is not N else storm.customers, observed=None, observed_county=None, real=False)
