"""Discrete-event restoration engine, steppable.

Damage is replayed from a real storm. Every hourly rise of customers out in a county becomes repair jobs that arrive at
random inside that hour; a measured share of the customers comes back by itself in 15 to 60 minutes (reclosers and
remote switching). Crews travel between county centroids, take a job in the county they serve (largest first by
default, as utilities do), repair it in a lognormal time and move where the target allocation is short. Mutual aid
joins a lead time after it is requested.

The engine is a generator. It stops at every decision point (the start, every `rebalance_h` hours and when mutual aid
arrives), yields a `DecisionPoint` with the full state, and waits for a `Decision`:
    weights   target share of crews per county (any non-negative vector, normalized inside)
    aid       mutual-aid multiplier to request now (0 for none); only the first request counts
    aid_at    optional, at the first decision only: the hour the request is placed (a request committed in advance)
When the last job is restored it returns an `Outcome`. With the current-practice baseline it reproduces, bit for bit,
the simulator of the research project it comes from (tests/test_engine.py).

Options, all off by default so that a default run is unchanged:
    job_order         largest | oldest | guard | sweep (largest first, but one crew in 1 / sweep_share always takes the
                      job of its county that has waited longest)
    dispatch          absolute: a free crew goes to the county short of most crews (utility practice); relative: to the
                      county furthest below its target in proportion (shortfall over target)
    replan_after_aid  a second decision point right after aid crews join, so a rule can replan with the larger fleet
    queue_view        decision points also carry the jobs waiting in each county (customers and arrival hour), what a
                      control room sees

Territories (`territory`, an integer id per county, with `fleet` crews per territory): each utility territory keeps
its own crews, which only serve its counties; targets are set within each territory, and mutual aid is requested and
joins territory by territory (the reactive trigger counts the territory's customers). With `territory=None` (the
default) one pooled fleet serves the region and every result is unchanged.
"""
import heapq
from dataclasses import dataclass

import numpy as np


@dataclass
class EngineConfig:
    customers_per_job: float = 35.0          # customers behind one repair job; Connecticut storm evidence
    repair_mean_h: float = 3.0
    repair_cv: float = 0.6
    setup_h: float = 0.5
    speed_kmh: float = 60.0
    rebalance_h: float = 6.0
    aid_lead_h: float = 24.0                 # request to arrival, Isaias average in Connecticut
    aid_share: float = 0.5                   # mutual-aid crews per local crew at multiplier 1
    reactive_trigger: float = 0.01           # share of customers out that triggers the usual reactive request
    reactive_aid: bool = True                # request aid reactively if the rule has not asked yet
    job_order: str = "largest"               # largest | oldest | guard | sweep
    guard_h: float = 48.0
    size_sigma: float = 0.0                  # dispersion of job sizes (lognormal), 0 for equal jobs
    aid_comes: bool = True                   # False: requests are accepted but no aid crew comes
    auto_h: tuple = (0.25, 1.0)              # duration of automatic restorations
    sweep_share: float = 0.1                 # job_order 'sweep': share of crews that always take the oldest job
    dispatch: str = "absolute"               # absolute | relative
    replan_after_aid: bool = False           # a second decision point once aid crews have joined
    queue_view: bool = False                 # decision points carry the waiting jobs of every county


@dataclass
class DecisionPoint:
    t: float
    out: np.ndarray                          # customers out per county (waiting plus under repair)
    waiting_jobs: np.ndarray
    working: np.ndarray                      # crews working or travelling per county
    wait_h: np.ndarray                       # hours the oldest waiting job of each county has waited
    arrived: np.ndarray                      # customers whose outage has appeared, per county
    restored: np.ndarray                     # customers restored by crews, per county
    crews: int
    aid_requested_at: float = None
    aid_crews: int = 0
    customer_hours: float = 0.0              # cumulative, crew jobs finished so far
    crews_at: np.ndarray = None              # crews standing in each county (working, travelling or idle)
    queue: list = None                       # with queue_view: per county, (customers, arrival hour) of waiting jobs
    after_aid: bool = False                  # True at the second decision point of an aid arrival


@dataclass
class Decision:
    weights: np.ndarray
    aid: float = 0.0
    aid_at: float = None                     # hour the request is placed, when committed in advance (first decision)


def make_jobs(rises, cfg, rng):
    h, k = np.nonzero(rises > 0)
    r = rises[h, k]
    n = np.maximum(1, np.rint(r / cfg.customers_per_job)).astype(int)
    t = np.repeat(h.astype(float), n) + rng.uniform(0, 1, n.sum())
    cty = np.repeat(k, n)
    cust = np.repeat(r / n, n)
    if cfg.size_sigma > 0:
        w = rng.lognormal(0.0, cfg.size_sigma, n.sum())
        starts = np.concatenate([[0], np.cumsum(n)[:-1]])
        cust = np.repeat(r / np.add.reduceat(w, starts), n) * w
    o = np.argsort(t)
    return t[o], cty[o], cust[o]


def run(rises, customers, dist_km, n_local, cfg, seed=0, territory=None, fleet=None, aid_fleet=None):
    """Generator. `rises` are the crew jobs' customers per hour and county (automatic restorations removed). The
    local crews start in proportion to customers. With `territory` (an integer id per county) and `fleet` (local
    crews per territory, summing to `n_local`), every territory keeps its own fleet; `aid_fleet` gives the aid crews
    each territory's usual request brings (by default `aid_share` times its local fleet)."""
    if cfg.job_order not in ("largest", "oldest", "guard", "sweep"):
        raise ValueError(f"job_order must be largest, oldest, guard or sweep, not {cfg.job_order!r}")
    if cfg.dispatch not in ("absolute", "relative"):
        raise ValueError(f"dispatch must be absolute or relative, not {cfg.dispatch!r}")
    rng = np.random.default_rng(seed)
    K = len(customers)
    t_arr, c_arr, k_arr = make_jobs(rises, cfg, rng)
    n = len(t_arr); finish = np.full(n, np.nan)
    s2 = np.log(1 + cfg.repair_cv ** 2); mu = np.log(cfg.repair_mean_h) - s2 / 2
    repair = rng.lognormal(mu, np.sqrt(s2), n)

    pooled = territory is None
    terr = np.zeros(K, int) if pooled else np.asarray(territory, int)
    T = 1 if pooled else int(terr.max()) + 1
    fleet_t = np.array([int(n_local)]) if pooled else np.asarray(fleet, int)
    if not pooled and (len(fleet_t) != T or fleet_t.sum() != n_local):
        raise ValueError("fleet must give the local crews of every territory and sum to n_local")
    aid_t = None if pooled else (np.rint(cfg.aid_share * fleet_t).astype(int) if aid_fleet is None else np.asarray(aid_fleet, int))
    cust_t = np.bincount(terr, weights=customers.astype(float), minlength=T)

    queues = [[] for _ in range(K)]; waitq = [[] for _ in range(K)]; started = np.zeros(n, bool)
    waiting = np.zeros(K); qn = np.zeros(K, int); inrepair = np.zeros(K); working = np.zeros(K, int)
    idle = [[] for _ in range(K)]; loc = []; crew_terr = []; arrived = np.zeros(K); restored = np.zeros(K)
    S = {"crews": 0, "target": np.zeros(K), "seq": 0, "ch": 0.0, "aid_at": None, "aid_n": 0,
         "crews_t": np.zeros(T, int), "aid_at_t": [None] * T, "aid_n_t": np.zeros(T, int)}
    done = []
    ev = []
    sweep_every = max(1, int(round(1 / cfg.sweep_share))) if cfg.job_order == "sweep" else 0
    relative = cfg.dispatch == "relative"
    start_w = customers.astype(float)
    # crew time per job and county
    t_start = np.full(n, np.nan); t_travel = np.zeros(n)
    crew_hours = np.zeros(K); travel_hours = np.zeros(K)

    def push(t, kind, data):
        heapq.heappush(ev, (t, S["seq"], kind, data)); S["seq"] += 1

    def wait_h(t):
        old = np.full(K, np.nan)
        for k in range(K):
            while waitq[k] and started[waitq[k][0][1]]:
                heapq.heappop(waitq[k])
            if waitq[k]:
                old[k] = waitq[k][0][0]
        return np.where(np.isnan(old), 0.0, t - np.nan_to_num(old, nan=t))

    def account():
        """Adds the jobs finished since the last decision point to the running customer-hours (vectorized)."""
        if not done:
            return
        i = np.asarray(done); done.clear()
        S["ch"] += float((k_arr[i] * (finish[i] - t_arr[i])).sum())

    def queue_view():
        """Customers and arrival hour of the jobs waiting in each county, in the order the county queue holds them."""
        out = []
        for k in range(K):
            ii = np.fromiter((e[2] for e in queues[k]), int, len(queues[k]))
            ii = ii[~started[ii]]
            out.append((k_arr[ii], t_arr[ii]))
        return out

    def point(t, after_aid=False):
        account()
        return DecisionPoint(t=t, out=waiting + inrepair, waiting_jobs=qn.copy(), working=working.copy(), wait_h=wait_h(t),
                             arrived=arrived.copy(), restored=restored.copy(), crews=S["crews"], aid_requested_at=S["aid_at"],
                             aid_crews=S["aid_n"], customer_hours=S["ch"],
                             crews_at=np.bincount(np.asarray(loc, int), minlength=K) if loc else np.zeros(K, int),
                             queue=queue_view() if cfg.queue_view else None, after_aid=after_aid)

    def set_target(w, t):
        w = np.asarray(w, float)
        if pooled:
            S["target"] = S["crews"] * w / w.sum() if w.sum() > 0 else np.zeros(K)
            return
        tot = np.bincount(terr, weights=w, minlength=T)
        with np.errstate(divide="ignore", invalid="ignore"):
            S["target"] = np.where(tot[terr] > 0, S["crews_t"][terr] * w / tot[terr], 0.0)

    def add_crews(m, weights, t_idx=0):
        w = np.asarray(weights, float)
        if not pooled:
            w = np.where(terr == t_idx, w, 0.0)
            if w.sum() <= 0:
                w = np.where(terr == t_idx, start_w, 0.0)
        w = w / w.sum()
        alloc = np.floor(m * w).astype(int); alloc[np.argsort(-(m * w - alloc))[: m - alloc.sum()]] += 1
        for k, a in enumerate(alloc):
            for _ in range(a):
                loc.append(k); crew_terr.append(t_idx); idle[k].append(len(loc) - 1)
        S["crews"] += m; S["crews_t"][t_idx] += m

    def request(t, mult, t_idx=None):
        """An aid request at hour t: for the region (pooled), or for one territory, or for every territory that has
        not asked yet (t_idx None with territories)."""
        if pooled:
            m = int(round(cfg.aid_share * n_local * mult)) if cfg.aid_comes else 0
            S["aid_at"], S["aid_n"] = t, m
            if m > 0:
                push(t + cfg.aid_lead_h, "aid", (m, 0))
            return
        for ti in (range(T) if t_idx is None else [t_idx]):
            if S["aid_at_t"][ti] is not None:
                continue
            m = int(round(aid_t[ti] * mult)) if cfg.aid_comes else 0
            S["aid_at_t"][ti] = t; S["aid_n_t"][ti] = m; S["aid_n"] += m
            if S["aid_at"] is None:
                S["aid_at"] = t
            if m > 0:
                push(t + cfg.aid_lead_h, "aid", (m, ti))

    def start(c, k, t):
        i = None
        if cfg.job_order == "sweep" and c % sweep_every == 0:       # a sweep crew takes the job waiting longest
            while waitq[k] and started[waitq[k][0][1]]:
                heapq.heappop(waitq[k])
            if waitq[k]:
                _, i = heapq.heappop(waitq[k])
        if cfg.job_order == "guard":
            while waitq[k] and started[waitq[k][0][1]]:
                heapq.heappop(waitq[k])
            if waitq[k] and t - waitq[k][0][0] > cfg.guard_h:
                _, i = heapq.heappop(waitq[k])
        while i is None or started[i]:
            _, _, i = heapq.heappop(queues[k])
        started[i] = True; qn[k] -= 1
        waiting[k] = max(0.0, waiting[k] - k_arr[i]); inrepair[k] += k_arr[i]; working[k] += 1
        travel = 0.0 if loc[c] == k else dist_km[loc[c], k] / cfg.speed_kmh
        crew_hours[k] += travel + cfg.setup_h + repair[i]; travel_hours[k] += travel
        t_start[i] = t; t_travel[i] = travel
        loc[c] = k
        push(t + travel + cfg.setup_h + repair[i], "free", (c, k, i))

    def choose(c):
        has = qn > 0 if pooled else (qn > 0) & (terr == crew_terr[c])
        if not has.any():
            return None
        deficit = S["target"] - working; here = loc[c]
        if has[here] and deficit[here] >= 0:
            return here
        cand = np.flatnonzero(has & (deficit > 0))
        if len(cand):
            if relative:                         # furthest below its target in proportion, not in crews
                return int(cand[np.argmax(deficit[cand] / np.maximum(S["target"][cand], 1e-9))])
            return int(cand[np.argmax(deficit[cand])])
        cand = np.flatnonzero(has)
        return int(cand[np.argmax(waiting[cand] / (1 + working[cand]))])

    def dispatch_idle(t, k=None):
        if k is not None and qn[k] > 0 and idle[k]:
            start(idle[k].pop(), k, t); return
        for j in range(K):
            while idle[j] and (qn > 0).any():
                c = idle[j][-1]; dest = choose(c)
                if dest is None:                     # nothing for this crew (pooled: no job anywhere; territories:
                    if pooled:                       # none in its territory), move on to the next county's crews
                        return
                    break
                idle[j].pop(); start(c, dest, t)

    if pooled:
        add_crews(n_local, start_w)
    else:
        for ti in range(T):
            if fleet_t[ti] > 0:
                add_crews(int(fleet_t[ti]), start_w, ti)
    d = yield point(0.0)
    set_target(d.weights, 0.0)
    for i in range(n):
        push(t_arr[i], "arr", i)
    push(cfg.rebalance_h, "reb", None)
    if d.aid > 0:
        request(0.0 if getattr(d, "aid_at", None) is None else float(d.aid_at), d.aid)
    remaining = n; t_end = 0.0; trigger = cfg.reactive_trigger * customers.sum()
    trigger_t = cfg.reactive_trigger * cust_t

    while ev and remaining > 0:
        t, _, kind, data = heapq.heappop(ev)
        if kind == "arr":
            i = data; k = c_arr[i]
            heapq.heappush(queues[k], (t, 0.0, i) if cfg.job_order == "oldest" else (-k_arr[i], t, i))
            heapq.heappush(waitq[k], (t, i)); waiting[k] += k_arr[i]; qn[k] += 1; arrived[k] += k_arr[i]
            if cfg.reactive_aid:
                if pooled:
                    if S["aid_at"] is None and (waiting + inrepair).sum() >= trigger:
                        request(t, 1.0)
                else:
                    ti = terr[k]
                    if S["aid_at_t"][ti] is None and (waiting + inrepair)[terr == ti].sum() >= trigger_t[ti]:
                        request(t, 1.0, ti)
            dispatch_idle(t, k)
        elif kind == "free":
            c, k, i = data
            finish[i] = t; inrepair[k] -= k_arr[i]; working[k] -= 1; restored[k] += k_arr[i]; remaining -= 1; t_end = t
            done.append(i)
            dest = choose(c)
            if dest is None:
                idle[loc[c]].append(c)
            else:
                start(c, dest, t)
        elif kind == "reb":
            d = yield point(t)
            set_target(d.weights, t)
            if d.aid > 0 and S["aid_at"] is None:
                request(t, d.aid)
            dispatch_idle(t); push(t + cfg.rebalance_h, "reb", None)
        elif kind == "aid":
            m, ti = data
            d = yield point(t)
            set_target(d.weights, t)
            tg = S["target"]
            add_crews(m, np.maximum(tg, 0) + 1e-9 if tg.sum() > 0 else start_w, ti)
            if cfg.replan_after_aid:
                d = yield point(t, after_aid=True)
            set_target(d.weights, t); dispatch_idle(t)

    dur = finish - t_arr
    if pooled:
        aid_hours = 0.0 if S["aid_at"] is None else max(0.0, t_end - S["aid_at"]) * S["aid_n"]
    else:
        aid_hours = float(sum(max(0.0, t_end - a) * m for a, m in zip(S["aid_at_t"], S["aid_n_t"]) if a is not None))
    return {"customer_hours": float((k_arr * dur).sum()), "t_end": t_end, "jobs": n,
            "aid_requested_at": S["aid_at"], "aid_crews": S["aid_n"], "crews_local": n_local, "aid_hours": aid_hours,
            "t_arr": t_arr, "finish": finish, "customers": k_arr, "county": c_arr,
            "t_start": t_start, "t_travel": t_travel, "crew_hours": crew_hours, "travel_hours": travel_hours,
            "territory": None if pooled else terr, "crews_local_t": None if pooled else fleet_t,
            "aid_crews_t": None if pooled else S["aid_n_t"], "aid_requested_at_t": None if pooled else S["aid_at_t"]}
