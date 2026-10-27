"""Reference restoration rules. Each has `decide(state)` and returns (county weights, aid multiplier).

CurrentPractice        crews in proportion to the customers out in each county; mutual aid requested reactively when
                       1 % of the region is out (done by the simulator when the rule has not asked). What utilities
                       describe, and the status quo the calibration reproduces.
ForecastWeighted       crews follow what is out plus what the damage forecast still expects in each county
EarlyAid               current allocation, and mutual aid (times `mult`) requested `hours_before_peak` hours before the
                       regional peak. It reads the hour of the peak from the storm, so it plays a perfect timing
                       forecast; `peak_error_h` adds a normal error to it, drawn per storm and seed.

Writing your own takes a few lines:

    class OldestFirst:
        name = "oldest waiting first"
        def decide(self, state):
            return state.out * (1 + state.wait_h / 24), 0.0

or just a function: `lambda state: state.out + state.forecast`.
"""
import numpy as np

EPS = 1e-6


class CurrentPractice:
    name = "current practice"

    def weights(self, state):
        return state.out + EPS * state.customers

    def aid(self, state):
        return 0.0

    def decide(self, state):
        return self.weights(state), self.aid(state)


class ForecastWeighted(CurrentPractice):
    name = "forecast weighted"

    def weights(self, state):
        return state.out + np.clip(state.forecast - state.arrived, 0, None) + EPS * state.customers


class EarlyAid(CurrentPractice):
    def __init__(self, mult=1.5, hours_before_peak=24.0, peak_error_h=0.0, seed=0):
        self.mult, self.before, self.err, self.seed = mult, hours_before_peak, peak_error_h, seed
        self.name = (f"early aid x{mult:g}, {hours_before_peak:g} h before the peak"
                     + (f", peak error {peak_error_h:g} h" if peak_error_h else ""))

    def request_hour(self, state):
        """Hour of the request for this storm and seed (the same at every decision point)."""
        error = 0.0
        if self.err:
            error = np.random.default_rng([self.seed, state.storm.id, state.seed]).normal(0.0, self.err)
        return state.storm.meta["peak_hour"] + error - self.before

    def aid(self, state):
        if state.aid_requested:
            return 0.0
        return self.mult if state.t >= self.request_hour(state) - state.scenario.decision_h / 2 else 0.0


REFERENCE = {"current practice": CurrentPractice(), "forecast weighted": ForecastWeighted(),
             "early aid x1.5": EarlyAid(1.5, 24.0), "early aid, peak error 12 h": EarlyAid(1.5, 24.0, peak_error_h=12.0)}
