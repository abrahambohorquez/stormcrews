"""Scenarios: the world in which restoration rules are compared.

A scenario fixes the region and its storms, the calibration of crews and repairs, and the stresses applied to them:
the size of the local fleet, whether mutual aid comes or not, a scaling of the damage of the region or of single
counties, the order in which jobs are served inside a county, operational parameters (repair time, customers per
job, travel speed, aid lead time and size) and weather and network what-if knobs (`what_if`, see whatif.py). At
defaults a scenario replays the real storms exactly.

    Scenario()                                   # New England, every storm, shape calibration
    Scenario.load("stress_test.yaml")            # a shipped scenario, by name
    Scenario.load("my_scenario.yaml")            # or any YAML file
    Scenario(region="pawv", fleet_factor=0.8, aid="none", damage_scale={42003: 0.7})
"""
from dataclasses import asdict, dataclass, fields, replace
from pathlib import Path

import numpy as np
import yaml

CONFIG_DIR = Path(__file__).resolve().parent / "config"
CALIBRATIONS = ("shape", "equal", "anchored", "anchored_low", "anchored_high")
AID_MODES = ("normal", "none")


@dataclass
class Scenario:
    name: str = "new_england"
    region: str = "ne"                      # ne (New England) | pawv (Pennsylvania and West Virginia)
    storms: object = "all"                  # all | list of storm ids
    calibration: str = "shape"              # shape | equal | anchored | anchored_low | anchored_high
    decision_h: float = 6.0                 # hours between decision points (decisions also come when aid arrives)
    fleet_factor: float = 1.0               # scales the calibrated local fleet
    aid: str = "normal"                     # normal (the requested crews come) | none (nobody comes)
    reactive_aid: bool = True               # request aid when `reactive_trigger` of the region is out, if the rule
                                            # has not asked
    reactive_trigger: float = 0.01          # share of the region's customers out that triggers the reactive request
    job_order: str = "largest"              # largest | oldest | guard (largest, but a job waiting 48 h jumps the queue)
    damage_scale: object = 1.0              # scalar, or {fips: factor}
    # operations (defaults are the calibrated research values)
    repair_factor: float = 1.0              # multiplies the calibrated mean repair time
    customers_per_job: float = 35.0         # customers behind one repair job
    speed_kmh: float = 60.0                 # crew travel speed between counties
    aid_lead_h: float = 24.0                # hours from the aid request to arrival
    aid_share: float = 0.5                  # aid crews per local crew at multiplier 1
    # weather and network knobs applied to the real storm through the damage model (see whatif.py)
    what_if: object = None

    def __post_init__(self):
        self.decision_h = float(self.decision_h); self.fleet_factor = float(self.fleet_factor)
        from .whatif import WhatIf
        self.what_if = WhatIf.from_dict(self.what_if)
        for k in ("repair_factor", "customers_per_job", "speed_kmh", "aid_lead_h", "aid_share", "reactive_trigger"):
            setattr(self, k, float(getattr(self, k)))
        if not 0 < self.reactive_trigger < 1:
            raise ValueError("reactive_trigger must be a share of customers between 0 and 1")
        if self.customers_per_job <= 0 or self.speed_kmh <= 0 or self.repair_factor <= 0:
            raise ValueError("customers_per_job, speed_kmh and repair_factor must be positive")
        if self.aid_lead_h < 0 or self.aid_share < 0 or self.fleet_factor <= 0 or self.decision_h <= 0:
            raise ValueError("aid_lead_h and aid_share must be non-negative, fleet_factor and decision_h positive")
        if self.aid not in AID_MODES:
            raise ValueError(f"aid must be one of {AID_MODES}, not {self.aid!r}")
        if self.calibration not in CALIBRATIONS:
            raise ValueError(f"calibration must be one of {CALIBRATIONS}, not {self.calibration!r}")

    @classmethod
    def load(cls, source="new_england.yaml", **overrides):
        """A scenario from a YAML file, the name of a shipped one, or a dict; keyword arguments override it."""
        if isinstance(source, Scenario):
            return replace(source, **overrides)
        if isinstance(source, (str, Path)):
            p = Path(source)
            if not p.exists():
                p = CONFIG_DIR / p.name
                if p.suffix == "":
                    p = p.with_suffix(".yaml")
            d = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
            d.setdefault("name", p.stem)
        else:
            d = dict(source)
        known = {f.name for f in fields(cls)}
        unknown = set(d) - known
        if unknown:
            raise ValueError(f"unknown scenario keys: {sorted(unknown)}")
        d.update(overrides)
        return cls(**d)

    def with_(self, **changes):
        """A copy with some fields changed."""
        return replace(self, **changes)

    def storm_ids(self, region):
        """Storms of the scenario that have its calibration (the anchored_low variant could not reproduce six storms
        of Pennsylvania and West Virginia, which are left out of it)."""
        ids = list(region.storm_ids) if self.storms == "all" else [int(i) for i in self.storms]
        if self.storms == "all":
            has = {m["id"] for m in region.meta if self.calibration in m["calibrations"]}
            ids = [i for i in ids if i in has]
        return ids

    def damage_vector(self, counties):
        """Damage multiplier per county, in the order of the region's counties."""
        if isinstance(self.damage_scale, dict):
            f = counties["fips"].astype(int).map({int(k): float(v) for k, v in self.damage_scale.items()})
            return f.fillna(1.0).to_numpy(float)
        return np.broadcast_to(np.asarray(self.damage_scale, float), (len(counties),)).copy()

    def to_dict(self):
        return asdict(self)


def shipped():
    """Names of the scenarios that ship with StormCrews."""
    return sorted(p.stem for p in CONFIG_DIR.glob("*.yaml"))
