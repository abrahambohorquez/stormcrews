"""Scenario configuration: every shipped YAML loads and runs, unknown keys are rejected, the documented keys are the
keys the code reads, and the reliability metrics and the reactive trigger behave as documented."""
import re
from dataclasses import fields
from pathlib import Path

import numpy as np
import pytest

from stormcrews import Scenario, Simulator, WhatIf, shipped
from stormcrews.rules import CurrentPractice

ROOT = Path(__file__).resolve().parents[1]
DOC = ROOT / "docs" / "configuration.md"
SMALL = 13          # a short New England storm


def _documented(section):
    """Keys in the first column of the table under `## section` of docs/configuration.md."""
    text = DOC.read_text(encoding="utf-8")
    part = text.split(f"## {section}\n", 1)[1].split("\n## ", 1)[0]
    return set(re.findall(r"^\| `([a-z_]+)` \|", part, flags=re.M))


@pytest.mark.parametrize("name", shipped())
def test_every_shipped_scenario_loads_and_runs(name):
    sim = Simulator(name)
    assert sim.scenario.name == name
    if sim.scenario.region == "ne":
        res = sim.run(CurrentPractice(), SMALL, seed=0)
        assert res.metrics["customer_hours"] > 0


def test_unknown_keys_are_rejected():
    with pytest.raises(ValueError, match="unknown scenario keys"):
        Scenario.load({"region": "ne", "fleet_factr": 0.8})
    with pytest.raises(ValueError, match="unknown what_if knobs"):
        Scenario(what_if={"gust": 3})


def test_bad_values_are_rejected():
    for bad in ({"aid": "lots"}, {"calibration": "magic"}, {"reactive_trigger": 1.5},
                {"customers_per_job": 0}, {"fleet_factor": 0}):
        with pytest.raises(ValueError):
            Scenario(**bad)


def test_documented_keys_match_the_code():
    assert _documented("Scenario keys") == {f.name for f in fields(Scenario)}
    assert _documented("`what_if`") == {f.name for f in fields(WhatIf)}


def test_saidi_and_crew_metrics():
    res = Simulator("new_england").run(CurrentPractice(), SMALL, seed=0)
    m = res.metrics
    served = res.storm.customers if res.storm.customers is not None else Simulator("new_england").region.customers
    assert np.isclose(m["saidi_min"], 60 * m["customer_hours"] / served.sum())
    assert np.isclose(m["caidi_h"], m["customer_hours"] / m["customers_interrupted"])
    assert m["crews_max"] == res.outcome["crews_local"] + m["aid_crews"]
    assert 0 < m["travel_crew_hours"] < m["busy_crew_hours"]


def test_reactive_trigger_moves_the_request():
    hours = {}
    for tr in (0.002, 0.01, 0.05):
        res = Simulator("new_england", reactive_trigger=tr).run(CurrentPractice(), 10, seed=0)
        hours[tr] = res.outcome["aid_requested_at"]
    assert hours[0.002] < hours[0.01] < hours[0.05]
    default = Simulator("new_england").run(CurrentPractice(), 10, seed=0)
    assert default.outcome["aid_requested_at"] == hours[0.01]
