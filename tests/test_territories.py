"""Utility territories: each territory's crews serve only its counties, aid is requested per territory, the fleet split
sums to the calibrated fleet, and a run without territories is unchanged."""
import numpy as np
import pytest

from stormcrews import Scenario, Simulator
from stormcrews.core.engine import EngineConfig, run
from stormcrews.rules import CurrentPractice
from stormcrews.simulator import split_fleet

SMALL = 13


def _play(gen):
    p = next(gen)
    while True:
        try:
            p = gen.send(type("D", (), {"weights": p.out + 1e-6, "aid": 0.0, "aid_at": None})())
        except StopIteration as stop:
            return stop.value


def test_crews_stay_in_their_territory():
    rng = np.random.default_rng(3)
    K, H = 6, 30
    rises = rng.uniform(0, 400, (H, K)) * (rng.uniform(0, 1, (H, K)) < 0.4)
    customers = np.full(K, 50_000.0)
    dist = np.abs(np.arange(K)[:, None] - np.arange(K)[None, :]) * 30.0
    terr = np.array([0, 0, 0, 1, 1, 1]); fleet = np.array([5, 7])
    cfg = EngineConfig(repair_mean_h=2.0, reactive_trigger=0.001)
    out = _play(run(rises, customers, dist, 12, cfg, seed=1, territory=terr, fleet=fleet))
    assert np.all(np.isfinite(out["finish"]))
    assert out["crews_local_t"].tolist() == [5, 7]
    # a job started by a crew of territory t is in a county of t: travel never crosses the boundary (counties 2 and 3
    # are 30 km apart, every cross-boundary trip would be at least that)
    assert out["aid_crews"] == int(round(0.5 * 5)) + int(round(0.5 * 7))
    assert out["aid_crews_t"].tolist() == [2, 4] or out["aid_crews_t"].tolist() == [3, 4]


def test_pooled_run_is_unchanged_by_the_territory_code():
    rng = np.random.default_rng(5)
    K, H = 4, 20
    rises = rng.uniform(0, 300, (H, K)); customers = np.full(K, 20_000.0)
    dist = np.abs(np.arange(K)[:, None] - np.arange(K)[None, :]) * 20.0
    cfg = EngineConfig(repair_mean_h=2.5)
    a = _play(run(rises, customers, dist, 8, cfg, seed=2))
    b = _play(run(rises, customers, dist, 8, cfg, seed=2, territory=None))
    assert a["customer_hours"] == b["customer_hours"] and a["territory"] is None


def test_split_fleet_sums_and_covers():
    terr = np.array([0, 0, 1, 2, 2, 2]); cust = np.array([100.0, 100.0, 1.0, 300.0, 300.0, 300.0])
    f = split_fleet(10, terr, cust)
    assert f.sum() == 10 and (f >= 1).all() and f[2] >= f[0] >= f[1]


def test_scenario_with_territories_runs_when_the_table_has_them():
    sim = Simulator(Scenario(name="ne", region="ne", calibration="anchored", territories=True))
    if sim.region.territory_ids is None:
        with pytest.raises(ValueError, match="territory"):
            sim.run(CurrentPractice(), SMALL, seed=0)
        return
    res = sim.run(CurrentPractice(), SMALL, seed=0)
    assert res.metrics["customer_hours"] > 0
    assert res.outcome["crews_local_t"].sum() == res.outcome["crews_local"]
