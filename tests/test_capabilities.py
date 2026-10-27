"""Execution options, the extra calibrations and scheduled aid requests: properties that hold everywhere."""
import numpy as np
import pytest

from stormcrews import Decision, Scenario, Simulator
from stormcrews.core.data import region
from stormcrews.rules import CurrentPractice

SMALL = 13


def test_defaults_are_unchanged_by_the_options():
    a = Simulator().run(CurrentPractice(), SMALL, seed=3)
    b = Simulator().run(CurrentPractice(), SMALL, seed=3,
                        execution={"dispatch": "absolute", "replan_after_aid": False, "queue_view": False,
                                   "job_order": "largest"})
    assert np.array_equal(a.outcome["finish"], b.outcome["finish"])
    assert a.metrics["customer_hours"] == b.metrics["customer_hours"]


def test_new_calibrations_and_storm_lists():
    R = region("pawv")
    sc = Scenario(region="pawv", calibration="anchored_low")
    ids = sc.storm_ids(R)
    assert 0 < len(ids) < len(R.storm_ids)
    with pytest.raises(ValueError):
        Simulator(sc).run(CurrentPractice(), next(i for i in R.storm_ids if i not in ids))
    for cal in ("anchored_low", "anchored_high"):
        assert Simulator(calibration=cal).run(CurrentPractice(), SMALL, seed=0).metrics["customer_hours"] > 0


@pytest.mark.parametrize("execution", [{"job_order": "sweep", "sweep_share": 0.2}, {"dispatch": "relative"},
                                       {"replan_after_aid": True, "queue_view": True}])
def test_execution_options_conserve_customers(execution):
    res = Simulator().run(CurrentPractice(), SMALL, seed=2, execution=execution)
    o = res.outcome
    assert np.all(np.isfinite(o["finish"])) and np.all(o["t_start"] >= o["t_arr"] - 1e-12)
    assert np.isclose(o["customers"].sum() + res.auto["customers"].sum(), res.storm.rises.sum())
    assert np.isclose(o["crew_hours"].sum(), (o["finish"] - o["t_start"]).sum())


def test_queue_view_and_replan():
    ses = Simulator().start(SMALL, seed=0, execution={"queue_view": True, "replan_after_aid": True})
    seen_after = False
    while not ses.done:
        st = ses.state
        assert sum(len(c) for c, _ in st.queue) == int(st.waiting_jobs.sum())
        assert all(np.all(t <= st.t + 1e-12) for _, t in st.queue)
        assert st.crews_at.sum() == st.crews
        seen_after |= st.after_aid
        ses.step(st.out + 1e-6 * st.customers)
    assert seen_after


class OnTime:
    """Current allocation, with mutual aid committed at the first decision for 24 h before the regional peak."""
    name = "aid on time"

    def decide(self, state):
        w = state.out + 1e-6 * state.customers
        if state.t == 0:
            return Decision(w, 1.5, state.storm.meta["peak_hour"] - 24.0)
        return w, 0.0


def test_scheduled_request():
    res = Simulator().run(OnTime(), SMALL, seed=0)
    assert res.outcome["aid_requested_at"] == res.storm.meta["peak_hour"] - 24.0
    assert res.outcome["aid_crews"] == int(round(0.5 * res.outcome["crews_local"] * 1.5))


def test_proportional_dispatch_changes_where_crews_go():
    a = Simulator().run(CurrentPractice(), 10, seed=0)
    b = Simulator().run(CurrentPractice(), 10, seed=0, execution={"dispatch": "relative"})
    assert np.array_equal(a.outcome["t_arr"], b.outcome["t_arr"])
    assert not np.array_equal(a.outcome["finish"], b.outcome["finish"])
