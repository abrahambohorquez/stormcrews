"""The notebook explorer (stormcrews/explore.py): its runs are ordinary simulator runs, and its widgets update."""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402

from stormcrews import Simulator, explore as E  # noqa: E402
from stormcrews.rules import CurrentPractice  # noqa: E402

SMALL = ("ne", 1)       # a short New England storm, so the tests stay fast


def test_default_run_is_a_plain_simulator_run():
    s = E.default_settings(region=SMALL[0], storm=SMALL[1])
    r = E.run_case(s)
    ref = Simulator("new_england").run(CurrentPractice(), SMALL[1], seed=1)
    assert r.metrics["customer_hours"] == ref.metrics["customer_hours"]
    assert r.storm.real                                    # neutral knobs replay the real storm
    rows = E.card(r)
    assert [a for a, *_ in rows] == [c[1] for c in E.CARD]
    assert rows[1][1] == f"{r.metrics['saidi_min']:,.0f}"


def test_knobs_reach_the_scenario_and_baseline_change():
    base = E.run_case(E.default_settings(region=SMALL[0], storm=SMALL[1]))
    s = E.default_settings(region=SMALL[0], storm=SMALL[1], gust_change=5.0, fleet_factor=0.8, aid="none",
                           aid_lead_h=36.0, job_order="oldest", auto_share=0.2)
    sc = E.scenario(s)
    assert sc.fleet_factor == 0.8 and sc.aid == "none" and sc.job_order == "oldest" and sc.aid_lead_h == 36.0
    assert sc.what_if.gust_change == 5.0 and sc.what_if.auto_share == 0.2
    r = E.run_case(s)
    assert not r.storm.real
    assert r.metrics["customers_interrupted"] > base.metrics["customers_interrupted"]
    rows = E.card(r, base)
    assert all(d for *_, d in rows)                        # every metric has its change against the baseline
    fig = E.storm_figure(r, base, with_card=True)
    assert len(fig.axes) >= 3
    plt.close("all")


def test_storm_options_cover_both_regions():
    opts = E.storm_options()
    assert len(opts) == 85 and ("ne", 10) in [v for _, v in opts]
    assert any(v[0] == "pawv" for _, v in opts)


def test_widgets_update():
    pytest.importorskip("ipywidgets")
    tab = E.StormTab(run_now=False)
    tab.auto_run.value = False
    tab.storm.value = SMALL
    first = tab.update()
    assert first is not None and np.isfinite(first.metrics["saidi_min"])
    tab.pin()
    assert tab.baseline is first
    tab.gust.value = 5.0
    second = tab.update()
    assert second.metrics["customer_hours"] > first.metrics["customer_hours"]
    tab.calibration.value = "anchored_low"
    tab.storm.value = ("pawv", next(m["id"] for m in E.load_region("pawv").meta
                                    if "anchored_low" not in m["calibrations"]))
    assert tab.update() is None and "Could not run" in tab.status.value
    plt.close("all")
