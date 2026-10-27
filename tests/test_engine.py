"""Fidelity of the engine to the research simulator it was extracted from.

With the current-practice rule the steppable engine must give, bit for bit, the result of `simulate` in the research
simulator (`<research>/ambiente/simulador_v2.py`). The research code is not part of this repository; when no such folder
is next to StormCrews (or STORMCREWS_RESEARCH does not point to one), the tests are skipped. Nothing in the research
folder is written: bytecode caching is turned off before importing it.
"""
import importlib
import os
import sys
from pathlib import Path

import numpy as np
import pytest

from stormcrews.core.data import Region
from stormcrews.core.engine import Decision, EngineConfig, run

def _research_root():
    env = os.environ.get("STORMCREWS_RESEARCH")
    if env:
        return Path(env)
    hit = next(Path(__file__).resolve().parents[2].glob("*/ambiente/simulador_v2.py"), None)
    return hit.parents[1] if hit else Path("__missing__")


RESEARCH = _research_root()
AMBIENTE = RESEARCH / "ambiente"
DATA = {"ne": RESEARCH / "datos", "pawv": RESEARCH / "replica_pawv" / "datos"}

pytestmark = pytest.mark.skipif(not (AMBIENTE / "simulador_v2.py").exists(), reason="research simulator not available")

CASES = {"ne": [1, 4, 13, 23], "pawv": [1, 8, 21, 43]}
CALIBRATIONS = ("shape", "anchored", "equal")
KEYS = (("chi", "customer_hours"), ("t_end", "t_end"))


def research(region):
    """Imports the research simulator for one region (its data folder is read when the package is imported)."""
    sys.dont_write_bytecode = True
    os.environ["TORMENTAS_DATOS"] = str(DATA[region])
    for m in [m for m in sys.modules if m.startswith("tormentas_sim") or m == "simulador_v2"]:
        del sys.modules[m]
    if str(AMBIENTE) not in sys.path:
        sys.path.insert(0, str(AMBIENTE))
    D = importlib.import_module("tormentas_sim.datos")
    pol = importlib.import_module("tormentas_sim.politicas")
    v2 = importlib.import_module("simulador_v2")
    return D, pol, v2


def drive(gen, customers):
    """Runs the engine generator with the current-practice decision (crews follow what is out, reactive aid)."""
    p = next(gen)
    try:
        while True:
            p = gen.send(Decision(p.out + 1e-6 * customers, 0.0))
    except StopIteration as stop:
        return stop.value


@pytest.mark.parametrize("region", ["ne", "pawv"])
def test_engine_equals_research_simulator(region):
    D, pol, v2 = research(region)
    cust = D.counties()["customers"].to_numpy(float)
    dist = D.distances_km()
    R = Region(region)
    # the shipped data are the research inputs, exactly
    assert np.array_equal(cust, R.customers)
    assert np.array_equal(dist, R.dist_km)
    for e in CASES[region]:
        rises, net = D.arrivals(e)
        storm = R.storm(e)
        assert np.array_equal(rises, storm.rises)
        fall = np.clip(net[:-1] - net[1:], 0, None)
        phi = float(np.minimum(rises[:-1], fall).sum() / max(rises.sum(), 1.0))
        assert phi == storm.meta["auto_share"]
        for kind in CALIBRATIONS:
            cal = storm.calibration(kind)
            seed = 2000 + e
            a = v2.simulate(rises * (1 - phi), cust, dist, pol.StatusQuo(), cal["fleet"],
                            v2.Config(repair_mean_h=cal["repair_mean_h"]), seed=seed, size_sigma=cal["size_sigma"])
            b = drive(run(storm.rises * (1 - storm.meta["auto_share"]), R.customers, R.dist_km, cal["fleet"],
                          EngineConfig(repair_mean_h=cal["repair_mean_h"], size_sigma=cal["size_sigma"]), seed=seed),
                      R.customers)
            for ka, kb in KEYS:
                assert a[ka] == b[kb], (region, e, kind, ka, a[ka], b[kb])
            assert a["incidents"] == b["jobs"]
            assert a["aid_request_h"] == b["aid_requested_at"]
            if a["aid_request_h"] is not None:          # the research reports the would-be aid even when never asked
                assert a["crews_aid"] == b["aid_crews"]


@pytest.mark.parametrize("region", ["ne", "pawv"])
def test_simulator_at_default_knobs_equals_research_simulator(region):
    """The full Simulator path, with every what-if and operational knob written out at its neutral value."""
    from stormcrews import Simulator
    from stormcrews.rules import CurrentPractice
    D, pol, v2 = research(region)
    cust = D.counties()["customers"].to_numpy(float); dist = D.distances_km()
    neutral = {"gust_change": 0.0, "precip_factor": 1.0, "soil_moisture_change": 0.0, "customer_growth": 0.0, "footprint_shift_km": (0.0, 0.0), "footprint_scale": 1.0,
               "duration_factor": 1.0, "auto_share": None}
    sim = Simulator(region=region, what_if=neutral, fleet_factor=1.0, repair_factor=1.0, customers_per_job=35.0,
                    speed_kmh=60.0, aid_lead_h=24.0, aid_share=0.5)
    for e in CASES[region][:2]:
        ses = sim.start(e, seed=9)
        res = sim.run(CurrentPractice(), e, seed=9)
        rises, net = D.arrivals(e)
        phi = res.storm.meta["auto_share"]; cal = res.storm.calibration("shape")
        a = v2.simulate(rises * (1 - phi), cust, dist, pol.StatusQuo(), cal["fleet"],
                        v2.Config(repair_mean_h=cal["repair_mean_h"]), seed=ses.engine_seed, size_sigma=cal["size_sigma"])
        assert a["chi"] == res.outcome["customer_hours"] and a["t_end"] == res.outcome["t_end"]
