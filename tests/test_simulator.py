"""Properties of the simulator, the rules, the scenarios and the experiment runner that hold for any rule."""
import numpy as np
import pandas as pd
import pytest

from stormcrews import Decision, Scenario, Simulator, paired_comparison, run_experiment, shipped
from stormcrews.core.data import region
from stormcrews.core.engine import EngineConfig, run
from stormcrews.metrics import restoration_curve
from stormcrews.rules import REFERENCE, CurrentPractice, EarlyAid, ForecastWeighted

SMALL = {"ne": 13, "pawv": 8}     # two small storms keep the tests fast


class Shuffled:
    """A rule that ignores outages: random weights and a random aid request, for property tests."""
    name = "shuffled"

    def __init__(self, seed=0):
        self.seed = seed

    def decide(self, state):
        rng = np.random.default_rng([self.seed, state.storm.id, state.seed, int(state.t * 100)])
        return rng.uniform(0, 1, len(state.customers)), float(rng.choice([0.0, 1.0, 2.0]))


@pytest.mark.parametrize("name", ["ne", "pawv"])
def test_customers_conserved_and_every_job_finished(name):
    sim = Simulator(region=name)
    res = sim.run(CurrentPractice(), SMALL[name], seed=3)
    o, s = res.outcome, res.storm
    assert np.all(np.isfinite(o["finish"])) and np.all(o["finish"] > o["t_arr"])
    total = o["customers"].sum() + res.auto["customers"].sum()
    assert np.isclose(total, s.rises.sum(), rtol=1e-9)
    assert np.isclose(res.metrics["customers_interrupted"], s.rises.sum(), rtol=1e-9)
    K = len(sim.counties)
    by_county = np.bincount(o["county"], o["customers"], K) + np.bincount(res.auto["county"], res.auto["customers"], K)
    assert np.allclose(by_county, s.rises.sum(0))


def test_zero_damage_means_no_outage():
    res = Simulator(damage_scale=0.0).run(CurrentPractice(), SMALL["ne"])
    m = res.metrics
    assert m["customer_hours"] == 0.0 and m["customers_interrupted"] == 0.0 and m["over_24h"] == 0.0
    assert len(res.history) == 1


def test_engine_zero_damage():
    R = region("ne")
    gen = run(np.zeros((10, len(R.customers))), R.customers, R.dist_km, 50, EngineConfig(), seed=0)
    p = next(gen)
    with pytest.raises(StopIteration) as stop:
        gen.send(Decision(p.out + 1.0))
    out = stop.value.value
    assert out["jobs"] == 0 and out["customer_hours"] == 0.0 and out["t_end"] == 0.0


@pytest.mark.parametrize("rule", [CurrentPractice(), EarlyAid(mult=2.0, hours_before_peak=36), Shuffled(4)])
def test_running_customer_hours_add_up_to_the_storm(rule):
    res = Simulator().run(rule, SMALL["ne"], seed=5)
    o, m = res.outcome, res.metrics
    seen = [h["customer_hours_so_far"] for h in res.history]
    assert all(b >= a - 1e-6 for a, b in zip(seen, seen[1:]))            # customer-hours only accrue
    assert seen[-1] <= o["customer_hours"] + 1e-6
    auto_ch = float((res.auto["customers"] * res.auto["hours"]).sum())
    assert np.isclose(m["customer_hours"], o["customer_hours"] + auto_ch, rtol=1e-9)


def test_determinism_and_common_random_numbers():
    sim = Simulator()
    a = sim.run(Shuffled(1), SMALL["ne"], seed=11); b = sim.run(Shuffled(1), SMALL["ne"], seed=11)
    assert a.metrics["customer_hours"] == b.metrics["customer_hours"] and a.history == b.history
    c = sim.run(CurrentPractice(), SMALL["ne"], seed=11); d = sim.run(CurrentPractice(), SMALL["ne"], seed=12)
    assert c.metrics["customer_hours"] != d.metrics["customer_hours"]
    # same seed, different rules: the same jobs and the same automatic restorations
    assert np.array_equal(a.outcome["t_arr"], c.outcome["t_arr"]) and np.array_equal(a.auto["hours"], c.auto["hours"])


def test_rules_in_any_form():
    sim = Simulator(); s = SMALL["ne"]
    base = sim.run(CurrentPractice(), s, seed=0).metrics["customer_hours"]
    eps = 1e-6
    forms = [lambda st: st.out + eps * st.customers,
             lambda st: (st.out + eps * st.customers, 0.0),
             lambda st: {"weights": st.out + eps * st.customers},
             lambda st: Decision(st.out + eps * st.customers, 0.0)]
    for f in forms:
        assert sim.run(f, s, seed=0).metrics["customer_hours"] == base
    with pytest.raises(ValueError):
        sim.run(lambda st: np.ones(3), s)
    with pytest.raises(ValueError):
        sim.run(lambda st: -np.ones(len(st.customers)), s)


def test_session_step_by_step_equals_run():
    sim = Simulator(); rule = ForecastWeighted()
    ses = sim.start(SMALL["ne"], seed=7)
    while not ses.done:
        st = ses.state
        assert st.t >= 0 and st.out.shape == (len(sim.counties),)
        ses.step(rule.decide(st))
    assert ses.result(rule).metrics["customer_hours"] == sim.run(rule, SMALL["ne"], seed=7).metrics["customer_hours"]
    with pytest.raises(RuntimeError):
        ses.step(rule.decide(ses.state))


def test_metrics_are_consistent():
    res = Simulator().run(CurrentPractice(), SMALL["ne"], seed=1)
    m = res.metrics
    assert m["over_96h"] <= m["over_72h"] <= m["customers_interrupted"]
    assert np.isclose(res.curve().sum(), m["customer_hours"], rtol=0.05)


def test_scenarios_load_and_stress():
    assert set(shipped()) == {"new_england", "pennsylvania_west_virginia", "stress_test", "what_if"}
    for name in shipped():
        assert Scenario.load(name).name == name
    with pytest.raises(ValueError):
        Scenario.load({"region": "ne", "unknown_key": 1})
    sim = Simulator("stress_test")
    res = sim.run(EarlyAid(), SMALL["ne"], seed=0)
    assert res.outcome["aid_crews"] == 0 and res.metrics["aid_crew_hours"] == 0.0
    assert res.outcome["crews_local"] == max(1, round(0.75 * res.storm.calibration("shape")["fleet"]))


def test_damage_scale_by_county():
    sim = Simulator(); fips = int(sim.counties["fips"].iloc[0])
    ses = Simulator(damage_scale={fips: 0.0}).start(SMALL["ne"], seed=0)
    assert ses.rises[:, 0].sum() == 0 and np.array_equal(ses.rises[:, 1:], ses.storm.rises[:, 1:])


def test_no_aid_brings_no_crews():
    st = SMALL["ne"]
    a = Simulator().run(EarlyAid(), st, seed=0); b = Simulator(aid="none").run(EarlyAid(), st, seed=0)
    assert a.outcome["aid_crews"] == int(round(0.5 * a.outcome["crews_local"] * 1.5))
    assert b.outcome["aid_crews"] == 0 and b.metrics["crews_max"] == b.outcome["crews_local"]
    assert b.metrics["customer_hours"] > a.metrics["customer_hours"]


def test_experiment_runner_and_paired_comparison():
    exp = run_experiment({"current practice": CurrentPractice(), "early aid": EarlyAid()},
                         ["new_england", Scenario(name="small fleet", fleet_factor=0.8)], storms=[13, 16], seeds=[0, 1])
    t = exp.table
    assert len(t) == 2 * 2 * 2 * 2 and set(t["scenario"]) == {"new_england", "small fleet"}
    for col in ("customer_hours", "saidi_min", "over_24h", "over_72h", "over_96h", "worst_county_h", "caidi_h",
                "crews_max", "local_crew_hours", "aid_crew_hours", "busy_crew_hours", "travel_crew_hours"):
        assert col in t
    s = exp.summary()
    assert len(s) == 4 and (s["storms"] == 2).all()
    cmp = exp.compare("current practice")
    row = cmp[(cmp["scenario"] == "new_england") & (cmp["metric"] == "customer_hours")].iloc[0]
    assert row["relative_low"] <= row["relative"] <= row["relative_high"]
    same = t[t["rule"] == "current practice"].copy(); same["rule"] = "copy"
    c0 = paired_comparison(pd.concat([t, same]), "current practice", ["customer_hours"])
    z = c0[c0["rule"] == "copy"]
    assert np.allclose(z["relative"], 0) and np.allclose(z["relative_low"], 0) and np.allclose(z["relative_high"], 0)
    assert len(exp.county) == 2 * 2 * len(Simulator().counties)


def test_reference_rules_run():
    sim = Simulator(decision_h=12)
    for name, rule in REFERENCE.items():
        res = sim.run(rule, SMALL["ne"], seed=0)
        assert res.metrics["customer_hours"] > 0, name


def test_render_frame():
    ses = Simulator().start(SMALL["ne"], seed=0)
    for _ in range(3):
        ses.step(CurrentPractice().decide(ses.state))
    frame = ses.render()
    assert frame.ndim == 3 and frame.shape[2] == 3 and frame.dtype == np.uint8
    ses.close()


def test_restoration_curve_matches_engine_helper():
    res = Simulator().run(CurrentPractice(), SMALL["ne"], seed=0)
    assert np.array_equal(res.curve(), restoration_curve(res.outcome, res.auto))
