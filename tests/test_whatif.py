"""What-if knobs, operational knobs, the damage model and synthetic storms."""
import numpy as np
import pytest

from stormcrews import Scenario, Simulator
from stormcrews.cli import main as cli
from stormcrews.core.data import region
from stormcrews.damage import model
from stormcrews.rules import CurrentPractice, EarlyAid
from stormcrews.synthetic import gust_field, synthetic_storm
from stormcrews.whatif import WhatIf, apply, stretch

SMALL = 13
NEUTRAL = {"gust_change": 0.0, "precip_factor": 1.0, "soil_moisture_change": 0.0, "customer_growth": 0.0, "footprint_shift_km": [0, 0], "footprint_scale": 1.0, "duration_factor": 1.0,
           "auto_share": None}
OPS_DEFAULT = {"fleet_factor": 1.0, "repair_factor": 1.0, "customers_per_job": 35.0, "speed_kmh": 60.0,
               "aid_lead_h": 24.0, "aid_share": 0.5}


@pytest.mark.parametrize("name", ["ne", "pawv"])
def test_neutral_knobs_replay_the_real_storm_bit_for_bit(name):
    R = region(name); st = R.storm(R.storm_ids[3])
    assert apply(st, R, WhatIf.from_dict(NEUTRAL)) is st
    a = Simulator(region=name).run(CurrentPractice(), st.id, seed=4)
    b = Simulator(region=name, what_if=NEUTRAL, **OPS_DEFAULT).run(CurrentPractice(), st.id, seed=4)
    for k in ("customer_hours", "saidi_min", "restoration_end_h", "over_72h", "local_crew_hours", "aid_crew_hours"):
        assert a.metrics[k] == b.metrics[k], k
    assert np.array_equal(a.outcome["finish"], b.outcome["finish"])


def test_more_gust_rain_and_wet_soil_mean_more_damage():
    R = region("ne"); st = R.storm(10)
    totals = [apply(st, R, {"gust_change": g}).rises.sum(0) for g in (-5, -2, 0.001, 2, 5, 10)]
    for lo, hi in zip(totals, totals[1:]):
        assert np.all(hi >= lo - 1e-9) and hi.sum() > lo.sum()
    rain = [apply(st, R, {"precip_factor": f}).rises.sum() for f in (0.5, 1.001, 2.0)]
    assert rain[0] < rain[1] < rain[2]
    soil = [apply(st, R, {"soil_moisture_change": s}).rises.sum() for s in (-0.05, 0.001, 0.05)]
    assert soil[0] < soil[1] < soil[2]
    cap = model().info["max_interruptions_per_customer"] * R.customers
    assert np.all(totals[-1] <= cap + 1e-6)
    sim = [Simulator(what_if={"gust_change": g}).run(CurrentPractice(), 10, seed=0).metrics["customer_hours"]
           for g in (-3, 0, 3)]
    assert sim[0] < sim[1] < sim[2]


def test_customers_conserved_under_what_if():
    wi = {"gust_change": 3, "precip_factor": 1.4, "customer_growth": 0.1, "footprint_shift_km": (40, -20),
          "footprint_scale": 1.2, "duration_factor": 1.3, "auto_share": 0.25}
    res = Simulator(what_if=wi).run(CurrentPractice(), SMALL, seed=1)
    st = res.storm
    assert not st.real and st.observed is None and st.meta["auto_share"] == 0.25
    total = res.outcome["customers"].sum() + res.auto["customers"].sum()
    assert np.isclose(total, st.rises.sum(), rtol=1e-9)
    assert np.all(np.isfinite(res.outcome["finish"]))


def test_duration_and_footprint_keep_what_they_should():
    R = region("ne"); st = R.storm(10)
    x = stretch(st.rises, 1.7)
    assert x.shape[0] == int(np.ceil(st.hours * 1.7)) and np.allclose(x.sum(0), st.rises.sum(0)) and (x >= 0).all()
    s = apply(st, R, {"footprint_scale": 1.0, "footprint_shift_km": (0.0, 0.0), "duration_factor": 0.6})
    assert np.allclose(s.rises.sum(0), st.rises.sum(0))
    moved = apply(st, R, {"footprint_shift_km": (150, 0)})
    assert moved.rises.shape == st.rises.shape and (moved.rises >= 0).all()
    assert not np.allclose(moved.rises.sum(0), st.rises.sum(0))


def test_customer_growth_follows_the_model():
    R = region("ne"); st = R.storm(10); b6 = model().coef["log_density"]
    s = apply(st, R, {"customer_growth": 0.2})
    assert np.allclose(s.customers, R.customers * 1.2)
    assert np.isclose(s.rises.sum(), st.rises.sum() * 1.2 ** (1 + b6), rtol=1e-9)


def test_model_uncertainty_draws_and_overrides():
    M = model()
    assert M.draw(1).coef != M.coef and M.draw(1).coef == M.draw(1).coef
    R = region("ne"); st = R.storm(10)
    wetter = apply(st, R, {"soil_moisture_change": 0.02, "coefficients": {"soil_moisture": 10.0}}).rises.sum()
    assert np.isclose(wetter, st.rises.sum() * np.exp(0.2), rtol=1e-9)
    m, lo, hi = M.effect({"gust": 5})
    assert lo < m < hi


def test_operational_knobs():
    base = Simulator().run(EarlyAid(), SMALL, seed=2)
    slow = Simulator(repair_factor=2.0).run(EarlyAid(), SMALL, seed=2)
    assert slow.metrics["customer_hours"] > base.metrics["customer_hours"]
    big = Simulator(aid_share=1.0).run(EarlyAid(), SMALL, seed=2)
    assert big.outcome["aid_crews"] == int(round(1.0 * base.outcome["crews_local"] * 1.5))
    late = Simulator(aid_lead_h=48).run(EarlyAid(), SMALL, seed=2)
    assert late.metrics["customer_hours"] >= base.metrics["customer_hours"] - 1e-6
    fewer_jobs = Simulator(customers_per_job=70).run(CurrentPractice(), SMALL, seed=2)
    assert fewer_jobs.outcome["jobs"] < base.outcome["jobs"]


def test_yaml_what_if_block(tmp_path):
    p = tmp_path / "s.yaml"
    p.write_text("region: ne\nfleet_factor: 0.9\nwhat_if:\n  gust_change: 2.5\n  precip_factor: 1.2\n", encoding="utf-8")
    sc = Scenario.load(str(p))
    assert sc.what_if.gust_change == 2.5 and sc.fleet_factor == 0.9
    with pytest.raises(ValueError):
        Scenario.load({"what_if": {"wind": 3}})


def test_synthetic_storms_are_valid():
    R = region("ne")
    g1 = gust_field(R, (41.8, -72.7), 30, 120); g2 = gust_field(R, (41.8, -72.7), 38, 120)
    a, b = synthetic_storm(R, g1, "isaias"), synthetic_storm(R, g2, "isaias")
    for s in (a, b):
        assert s.rises.shape == (R.storm(10).hours, len(R.counties))
        assert np.all(np.isfinite(s.rises)) and (s.rises >= 0).all() and not s.real
        assert np.all(s.rises.sum(0) <= model().info["max_interruptions_per_customer"] * R.customers + 1e-6)
    assert b.rises.sum() > a.rises.sum()
    n1, n2 = synthetic_storm(R, g1, 10, noise_seed=3), synthetic_storm(R, g1, 10, noise_seed=3)
    assert np.array_equal(n1.rises, n2.rises)
    res = Simulator().run(CurrentPractice(), a, seed=0)
    assert np.isclose(res.metrics["customers_interrupted"], a.rises.sum(), rtol=1e-9)


def test_command_line(capsys):
    assert cli(["whatif", "--storm", str(SMALL), "--gust", "2", "--rule", "current"]) == 0
    out = capsys.readouterr().out
    assert "what-if" in out and "real storm" in out
    assert cli(["synthetic", "--center", "44", "-70", "--peak", "30", "--template", str(SMALL)]) == 0


def test_damage_specifications():
    eq, cnt = model(), model("count")
    assert eq.spec == "equal" and eq.kind == "equal" and cnt.kind == "count"
    assert set(eq.info["specs"]) == {"equal", "equal_ne", "equal_pawv", "count"}
    assert eq.features == cnt.features == ["gust", "gust_above_20", "log1p_precip", "soil_moisture", "log_density"]
    m, lo, hi = eq.effect({"gust": 2.0})
    assert lo > 1.0                          # stronger gusts below the knot mean more damage
    R = region("ne"); st = R.storm(10)
    a = apply(st, R, {"soil_moisture_change": -0.02}).rises.sum()
    b = apply(st, R, {"soil_moisture_change": -0.02, "damage_spec": "count"}).rises.sum()
    c = apply(st, R, {"soil_moisture_change": -0.02, "damage_spec": "equal_region"}).rises.sum()
    assert a < st.rises.sum() and np.isclose(a, st.rises.sum() * np.exp(-0.02 * eq.coef["soil_moisture"]), rtol=1e-9)
    assert not np.isclose(a, b) and not np.isclose(a, c)
    assert apply(st, R, {"damage_spec": "count"}) is st          # choosing a model alone changes nothing
