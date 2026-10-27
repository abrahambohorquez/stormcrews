"""Loads the storms and counties that ship with StormCrews."""
import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

DATA = Path(__file__).resolve().parents[1] / "data"
REGIONS = {"new_england": "ne", "ne": "ne", "pennsylvania_west_virginia": "pawv", "pawv": "pawv"}
TITLES = {"ne": "New England", "pawv": "Pennsylvania and West Virginia"}
NAMES = {"riley": "Nor'easter Riley", "isaias": "Tropical Storm Isaias", "noreaster": "October nor'easter",
         "winter": "December winter storm"}


@dataclass
class Storm:
    id: int
    region: str
    rises: np.ndarray            # hours x counties, customers losing power above the pre-storm baseline
    forecast: np.ndarray         # expected interruptions per county, fitted on the other storms of the region
    meta: dict
    observed: np.ndarray = None          # observed regional customers out above baseline, at the end of every hour
    observed_county: np.ndarray = None   # observed customer-hours per county
    weather: np.ndarray = None           # counties x (peak gust m/s, precipitation mm, mean soil moisture m3/m3), ERA5
    customers: np.ndarray = None         # customers per county when they differ from the region's (what-if, synthetic)
    real: bool = True                    # False after a what-if change or for a synthetic storm

    @property
    def name(self):
        n = (self.meta.get("name") or "").rsplit("_", 1)[0]
        label = NAMES.get(n, n.title()) if n else self.meta.get("type", "Storm")
        if self.meta.get("synthetic"):
            return f"Synthetic storm, timing of {self.meta['start'][:10]}"
        return f"{label}, {self.meta['start'][:10]}" + ("" if self.real else " (what-if)")

    @property
    def hours(self):
        return self.rises.shape[0]

    def calibration(self, kind):
        return self.meta["calibrations"][kind]


class Region:
    def __init__(self, name="ne"):
        self.code = REGIONS[name]; self.title = TITLES[self.code]
        self.counties = pd.read_csv(DATA / f"{self.code}_counties.csv", float_precision="round_trip")
        self.meta = json.loads((DATA / f"{self.code}_storms.json").read_text(encoding="utf-8"))
        self._arr = np.load(DATA / f"{self.code}.npz")
        la, lo = np.radians(self.counties["lat"].to_numpy()), np.radians(self.counties["lon"].to_numpy())
        d = (np.sin((la[:, None] - la[None, :]) / 2) ** 2
             + np.cos(la[:, None]) * np.cos(la[None, :]) * np.sin((lo[:, None] - lo[None, :]) / 2) ** 2)
        self.dist_km = 6371.0 * 2 * np.arcsin(np.sqrt(d))

    @property
    def customers(self):
        return self.counties["customers"].to_numpy(float)

    @property
    def storm_ids(self):
        return [m["id"] for m in self.meta]

    def storm(self, i):
        m = next(x for x in self.meta if x["id"] == i)
        a = self._arr
        return Storm(i, self.code, a[f"rises_{i}"].astype(float), a[f"forecast_{i}"].astype(float), m,
                     a[f"observed_{i}"].astype(float), a[f"observed_county_{i}"].astype(float),
                     a[f"weather_{i}"].astype(float))

    def find(self, name):
        """Storm id from an id, a name fragment ("isaias") or a start date ("2020-08-02")."""
        if isinstance(name, (int, np.integer)) or str(name).isdigit():
            return int(name)
        key = str(name).lower().replace(" ", "_")
        hits = [m["id"] for m in self.meta if key in (m.get("name") or "").lower() or str(m["start"]).startswith(str(name))]
        if not hits:
            raise KeyError(f"no storm matches {name!r} in {self.title}")
        return hits[0]

    def summary(self):
        """One row per storm: dates, type, duration, peak customers out, observed customer-hours and parameters."""
        return pd.DataFrame([{k: m[k] for k in ("id", "start", "name", "type", "hours", "peak_customers_out",
                                                "observed_customer_hours", "auto_share")} for m in self.meta])


@lru_cache(maxsize=4)
def region(name="ne"):
    return Region(name)
