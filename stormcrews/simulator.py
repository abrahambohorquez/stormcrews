"""Running restoration rules on real storms.

    sim = Simulator("new_england.yaml")
    result = sim.run(CurrentPractice(), storm=10, seed=1)       # Tropical Storm Isaias
    result.metrics["saidi_min"], result.curve()

A rule is any object with a method `decide(state)`, or any function of the state. At every decision point (the start
of the storm, every `decision_h` hours, and when mutual aid arrives) it receives a `State` and returns where the crews
should be and, optionally, a mutual-aid request:

    weights                         an array with one non-negative weight per county
    (weights, aid)                  plus a multiplier of the usual mutual-aid request (0 for none)
    {"weights": ..., "aid": ...}    the same as a dict
    Decision(weights, aid, aid_at)  aid_at, at the first decision only: the hour the request is placed

Crews are spread in proportion to the weights. Only the first aid request of a storm counts; aid x1 brings half the
local fleet. For step-by-step control, `sim.start(storm, seed)` returns a `Session` that stops at every decision point.

A rule can also say how its plan is executed, with an `execution` dict (any subset; the scenario gives the rest):
    job_order          largest | oldest | guard | sweep          which job of a county goes first
    guard_h, sweep_share                                         their parameters
    dispatch           absolute | relative                       where a free crew goes when its county is full
    replan_after_aid   True: decide again once aid crews have joined
    queue_view         True: the state carries the jobs waiting in each county (`state.queue`)
"""
from dataclasses import dataclass, field
from itertools import count

import numpy as np

from .core.data import Storm, region as load_region
from .core.engine import Decision, EngineConfig, run
from .metrics import restoration_curve, storm_metrics
from .scenario import Scenario
from .whatif import apply as apply_what_if

_RUN_IDS = count(1)
EXECUTION = ("job_order", "guard_h", "sweep_share", "dispatch", "replan_after_aid", "queue_view")


@dataclass
class State:
    """What a rule sees at a decision point. Arrays have one value per county, in the order of `counties`."""
    t: float                         # hours since the start of the storm window
    out: np.ndarray                  # customers out waiting for a crew or under repair
    waiting_jobs: np.ndarray         # repair jobs waiting
    working: np.ndarray              # crews working or travelling to a job
    wait_h: np.ndarray               # hours the oldest waiting job has waited
    arrived: np.ndarray              # customers whose outage has appeared so far (crew jobs)
    restored: np.ndarray             # customers restored by crews so far
    crews: int                       # crews on duty, local plus mutual aid that has arrived
    local_crews: int
    aid_requested_at: float          # hour of the aid request, None if not requested
    aid_crews: int                   # crews the request brings
    customer_hours_so_far: float     # customer-hours of the jobs restored so far
    forecast: np.ndarray             # expected customer interruptions per county, from the damage forecast
    customers: np.ndarray            # customers per county
    counties: object                 # DataFrame: fips, name, state, customers, lat, lon, density
    storm: object                    # the Storm: id, name, hours, meta (peak_hour, type, dates, ...)
    seed: int
    scenario: Scenario
    crews_at: np.ndarray = None      # crews standing in each county (working, travelling or idle)
    queue: list = None               # with queue_view: per county, (customers, arrival hour) of the waiting jobs
    dist_km: np.ndarray = None       # distances between county centroids
    repair_mean_h: float = None      # mean hours on site per job (calibrated) and setup hours per job
    setup_h: float = None
    after_aid: bool = False          # the second decision of an aid arrival (replan_after_aid)
    run: int = 0                     # identifies the run, so a rule with its own random stream can restart it
    oracle: object = field(default=None, repr=False)   # oracle(): the storm's hourly crew damage, past and future;
                                                       # perfect information, only for bounds (a perfect forecast)

    @property
    def aid_requested(self):
        return self.aid_requested_at is not None


def as_decision(x, K):
    """Turns what a rule returns into a Decision, with checks."""
    aid_at = None
    if isinstance(x, Decision):
        w, aid, aid_at = x.weights, x.aid, x.aid_at
    elif isinstance(x, dict):
        w, aid, aid_at = x["weights"], x.get("aid", 0.0), x.get("aid_at")
    elif isinstance(x, tuple) and len(x) == 2:
        w, aid = x
    else:
        w, aid = x, 0.0
    w = np.asarray(w, float)
    if w.shape != (K,):
        raise ValueError(f"a rule must return {K} county weights, got shape {w.shape}")
    if not np.all(np.isfinite(w)):
        raise ValueError("county weights must be finite")
    if (w < 0).any():
        # customers out can be a rounding error below zero (e.g. -1e-10) when a county has just been cleared
        if w.min() < -1e-6 * max(1.0, float(np.abs(w).max())):
            raise ValueError("county weights must be non-negative")
        w = np.clip(w, 0.0, None)
    aid = float(aid or 0.0)
    if aid < 0 or not np.isfinite(aid):
        raise ValueError("the aid multiplier must be a non-negative number")
    return Decision(w, aid, None if aid_at is None else float(aid_at))


def rule_name(rule):
    return getattr(rule, "name", None) or getattr(rule, "__name__", None) or type(rule).__name__


@dataclass
class Result:
    """What one rule did in one storm with one seed."""
    scenario: str
    rule: str
    storm: object
    seed: int
    metrics: dict
    outcome: dict = field(repr=False)
    auto: dict = field(repr=False)
    history: list = field(repr=False)

    def curve(self):
        """Customers out at the end of every hour (crew jobs and automatic restorations), like `storm.observed`."""
        return restoration_curve(self.outcome, self.auto)

    def row(self):
        """The metrics as a flat dict for a table (the per-county array left out)."""
        r = {"scenario": self.scenario, "rule": self.rule, "storm": self.storm.id, "storm_name": self.storm.name,
             "seed": self.seed}
        r.update({k: v for k, v in self.metrics.items() if k != "county_mean_outage_h"})
        r["aid_requested_at"] = np.nan if self.outcome["aid_requested_at"] is None else self.outcome["aid_requested_at"]
        r["local_crews"] = self.outcome["crews_local"]
        r["peak_customers_out"] = self.storm.meta.get("peak_customers_out", np.nan)
        r["decisions"] = len(self.history)
        return r


class Session:
    """One storm, stopped at every decision point. `state` is what a rule would see; `step(decision)` moves on."""

    def __init__(self, sim, storm, seed=0, rule=None, execution=None, engine_seed=None, auto_seed=None):
        """`rule` (or `execution`) sets how the plan is executed (see the module docstring). `engine_seed` and
        `auto_seed`, when given, replace the two random streams drawn from `seed` (jobs and repairs; automatic
        restorations), which is how a run of the research simulator is reproduced exactly."""
        self.sim, self.scenario, self.region = sim, sim.scenario, sim.region
        base = storm if isinstance(storm, Storm) else self.region.storm(self.region.find(storm))
        self.real_storm = base
        self.run_id = next(_RUN_IDS)
        self.storm = apply_what_if(base, self.region, self.scenario.what_if)
        self.seed = int(seed)
        sc = self.scenario
        self.customers = self.storm.customers if self.storm.customers is not None else self.region.customers
        scale = sc.damage_vector(self.region.counties)
        self.rises = self.storm.rises * scale
        self.forecast = self.storm.forecast * scale
        if sc.calibration not in self.storm.meta["calibrations"]:
            raise ValueError(f"storm {self.storm.id} has no {sc.calibration!r} calibration")
        cal = self.storm.calibration(sc.calibration)
        self.local_crews = max(1, int(round(cal["fleet"] * sc.fleet_factor)))
        phi = self.storm.meta["auto_share"]
        rng = np.random.default_rng(self.seed)
        self.engine_seed = int(rng.integers(2 ** 31)) if engine_seed is None else int(engine_seed)
        K = len(self.customers)
        a_c = (phi * self.rises).ravel()
        a_rng = rng if auto_seed is None else np.random.default_rng(auto_seed)
        self.auto = {"customers": a_c, "hours": a_rng.uniform(0.25, 1.0, size=a_c.shape),
                     "county": np.tile(np.arange(K), self.storm.hours),
                     "hour": np.repeat(np.arange(self.storm.hours), K)}
        ex = {"job_order": sc.job_order}
        ex.update(getattr(rule, "execution", None) or {})
        ex.update(execution or {})
        unknown = set(ex) - set(EXECUTION)
        if unknown:
            raise ValueError(f"unknown execution options: {sorted(unknown)}")
        self.execution = ex
        self.cfg = EngineConfig(repair_mean_h=cal["repair_mean_h"] * sc.repair_factor, size_sigma=cal["size_sigma"],
                                rebalance_h=sc.decision_h, customers_per_job=sc.customers_per_job,
                                speed_kmh=sc.speed_kmh, aid_lead_h=sc.aid_lead_h, aid_share=sc.aid_share,
                                reactive_trigger=sc.reactive_trigger,
                                reactive_aid=sc.reactive_aid and sc.aid != "none",
                                aid_comes=sc.aid != "none", **ex)
        self._crew_rises = self.rises * (1 - phi)
        self._gen = run(self._crew_rises, self.customers, self.region.dist_km, self.local_crews, self.cfg,
                        seed=self.engine_seed)
        self._point = next(self._gen)
        self.outcome = None; self.metrics = None; self.history = []; self.last_decision = None
        self._fig = None
        self._snapshot()

    # ------------------------------------------------------------------ state
    @property
    def state(self):
        p, R = self._point, self.region
        return State(t=p.t, out=p.out.copy(), waiting_jobs=p.waiting_jobs.copy(), working=p.working.copy(),
                     wait_h=p.wait_h.copy(), arrived=p.arrived.copy(), restored=p.restored.copy(), crews=p.crews,
                     local_crews=self.local_crews, aid_requested_at=p.aid_requested_at, aid_crews=p.aid_crews,
                     customer_hours_so_far=p.customer_hours, forecast=self.forecast,
                     customers=self.customers, counties=R.counties, storm=self.storm, seed=self.seed,
                     scenario=self.scenario, crews_at=p.crews_at.copy(), queue=p.queue, dist_km=R.dist_km,
                     repair_mean_h=self.cfg.repair_mean_h, setup_h=self.cfg.setup_h, after_aid=p.after_aid,
                     run=self.run_id, oracle=lambda: self._crew_rises)

    @property
    def done(self):
        return self.outcome is not None

    def _snapshot(self):
        p = self._point
        self.history.append({"t": p.t, "out": float(p.out.sum()), "crews": int(p.crews), "working": int(p.working.sum()),
                             "aid_requested": p.aid_requested_at is not None,
                             "customer_hours_so_far": p.customer_hours})

    def step(self, decision):
        """Applies a decision and runs to the next decision point. Returns True when the last customer is back."""
        if self.done:
            raise RuntimeError("the storm is over")
        d = as_decision(decision, len(self.customers)); self.last_decision = d
        try:
            self._point = self._gen.send(d)
            self._snapshot()
        except StopIteration as stop:
            self.outcome = stop.value
            self.metrics = storm_metrics(self.outcome, self.auto, self.region.counties, customers=self.customers)
        return self.done

    def result(self, rule="manual"):
        if not self.done:
            raise RuntimeError("the storm is not over yet")
        return Result(self.scenario.name, rule if isinstance(rule, str) else rule_name(rule), self.storm, self.seed,
                      self.metrics, self.outcome, self.auto, self.history)

    def render(self, return_array=True):
        """A frame: county map of customers out and crews, the regional curve so far, and the status."""
        from .visuals.plots import render_state
        return render_state(self, return_array=return_array)

    def close(self):
        if self._fig is not None:
            import matplotlib.pyplot as plt
            plt.close(self._fig); self._fig = None


class Simulator:
    """Runs rules on the storms of a scenario."""

    def __init__(self, scenario=None, **overrides):
        self.scenario = Scenario.load(scenario if scenario is not None else Scenario(), **overrides)
        self.region = load_region(self.scenario.region)

    @property
    def storm_ids(self):
        return self.scenario.storm_ids(self.region)

    @property
    def counties(self):
        return self.region.counties

    def start(self, storm, seed=0, rule=None, **options):
        """A Session stopped at the first decision point. `rule` sets how the plan is executed; options: execution,
        engine_seed, auto_seed (see Session)."""
        return Session(self, storm, seed, rule=rule, **options)

    def run(self, rule, storm, seed=0, frames=None, **options):
        """Plays `rule` through `storm`. With a list as `frames`, a rendered frame is appended at every decision."""
        decide = rule.decide if hasattr(rule, "decide") else rule
        s = self.start(storm, seed, rule=rule, **options)
        while not s.done:
            s.step(decide(s.state))
            if frames is not None and not s.done:
                frames.append(s.render())
        s.close()
        return s.result(rule)
