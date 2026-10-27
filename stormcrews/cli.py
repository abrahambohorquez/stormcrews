"""Command line for StormCrews.

    stormcrews storms --region ne
    stormcrews whatif --storm isaias --gust +5 --rain 1.5 --rule current --rule early_aid
    stormcrews whatif --region pawv --storm 45 --fleet 0.8 --aid none --seeds 3
    stormcrews synthetic --center 41.8 -72.7 --peak 38 --radius 120 --template isaias --rule current

`whatif` replays the real storm and the changed one under each rule with the same seeds and prints both.
"""
import argparse
import sys

import numpy as np

from . import rules as R
from .core.data import region as load_region
from .scenario import Scenario
from .simulator import Simulator
from .synthetic import gust_field, synthetic_storm

RULES = {"current": R.CurrentPractice, "forecast": R.ForecastWeighted, "early_aid": lambda: R.EarlyAid(1.5, 24.0),
         "early_aid_error": lambda: R.EarlyAid(1.5, 24.0, peak_error_h=12.0)}
COLUMNS = [("customers_interrupted", "interruptions k", 1e3, 1), ("customer_hours", "cust-hours M", 1e6, 1),
           ("saidi_min", "SAIDI min", 1, 0), ("over_24h", "out>24h k", 1e3, 1), ("over_72h", "out>72h k", 1e3, 1),
           ("over_96h", "out>96h k", 1e3, 1), ("worst_county_h", "worst cty h", 1, 1),
           ("restoration_end_h", "end h", 1, 0)]


def _table(rows):
    head = f"{'':34s}" + "".join(f"{c[1]:>16s}" for c in COLUMNS)
    print(head)
    for label, m in rows:
        print(f"{label[:34]:34s}" + "".join(f"{m[k] / s:>16,.{d}f}" for k, _, s, d in COLUMNS))


def _mean(results):
    return {k: float(np.nanmean([r.metrics[k] for r in results])) for k, *_ in COLUMNS}


def _scenario(a, what_if=None):
    sc = Scenario.load(a.scenario) if a.scenario else Scenario(region=a.region)
    ops = {k: v for k, v in (("fleet_factor", a.fleet), ("repair_factor", a.repair), ("customers_per_job", a.job_size),
                             ("speed_kmh", a.speed), ("aid_lead_h", a.aid_lead), ("aid_share", a.aid_share),
                             ("aid", a.aid), ("job_order", a.job_order)) if v is not None}
    return sc.with_(what_if=what_if, **ops)


def _ops(p):
    g = p.add_argument_group("operations")
    g.add_argument("--fleet", type=float, help="fleet factor (1 = calibrated)")
    g.add_argument("--repair", type=float, help="repair time factor")
    g.add_argument("--job-size", type=float, help="customers per repair job")
    g.add_argument("--speed", type=float, help="crew travel speed, km/h")
    g.add_argument("--aid-lead", type=float, help="hours from aid request to arrival")
    g.add_argument("--aid-share", type=float, help="aid crews per local crew at multiplier 1")
    g.add_argument("--aid", choices=("normal", "none"))
    g.add_argument("--job-order", choices=("largest", "oldest", "guard"))
    p.add_argument("--region", default="ne", choices=("ne", "pawv"))
    p.add_argument("--scenario", help="a shipped scenario name or YAML file (overrides --region)")
    p.add_argument("--rule", action="append", choices=sorted(RULES), help="repeat for several rules")
    p.add_argument("--seeds", type=int, default=1)


def main(argv=None):
    ap = argparse.ArgumentParser(prog="stormcrews", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("storms", help="list the storms of a region")
    s.add_argument("--region", default="ne", choices=("ne", "pawv"))

    w = sub.add_parser("whatif", help="a real storm, changed")
    w.add_argument("--storm", required=True, help="id, name fragment (isaias) or start date")
    g = w.add_argument_group("weather and network")
    g.add_argument("--gust", type=float, default=0.0, help="gust change, m/s")
    g.add_argument("--rain", type=float, default=1.0, help="precipitation factor")
    g.add_argument("--soil", type=float, default=0.0, help="soil-moisture change, m3/m3")
    g.add_argument("--growth", type=float, default=0.0, help="customer growth, share")
    g.add_argument("--shift-km", type=float, nargs=2, default=(0.0, 0.0), metavar=("EAST", "NORTH"))
    g.add_argument("--spread", type=float, default=1.0, help="footprint scale")
    g.add_argument("--duration", type=float, default=1.0, help="duration factor of the hourly profile")
    g.add_argument("--auto-share", type=float, help="share restored automatically within the hour")
    _ops(w)

    y = sub.add_parser("synthetic", help="a storm that never happened, from a gust field")
    y.add_argument("--center", type=float, nargs=2, required=True, metavar=("LAT", "LON"))
    y.add_argument("--peak", type=float, required=True, help="peak gust at the center, m/s")
    y.add_argument("--radius", type=float, default=120.0, help="km")
    y.add_argument("--template", default="isaias", help="real storm lending timing and response")
    y.add_argument("--noise-seed", type=int)
    _ops(y)

    a = ap.parse_args(argv)
    if a.cmd == "storms":
        print(load_region(a.region).summary().to_string(index=False)); return 0

    rules = {k: RULES[k]() for k in (a.rule or ["current"])}
    if a.cmd == "whatif":
        wi = {"gust_change": a.gust, "precip_factor": a.rain, "soil_moisture_change": a.soil,
              "customer_growth": a.growth, "footprint_shift_km": tuple(a.shift_km), "footprint_scale": a.spread,
              "duration_factor": a.duration, "auto_share": a.auto_share}
        real, changed = Simulator(_scenario(a)), Simulator(_scenario(a, wi))
        sid = real.region.find(a.storm)
        rows = []
        for name, rule in rules.items():
            rows.append((f"{name}, real storm", _mean([real.run(rule, sid, k) for k in range(a.seeds)])))
            rows.append((f"{name}, what-if", _mean([changed.run(rule, sid, k) for k in range(a.seeds)])))
        print(f"{real.region.storm(sid).name}, {a.seeds} seed(s); all values simulated")
        _table(rows)
    else:
        sim = Simulator(_scenario(a))
        tpl = sim.region.find(a.template)
        storm = synthetic_storm(sim.region, gust_field(sim.region, tuple(a.center), a.peak, a.radius), tpl,
                                noise_seed=a.noise_seed)
        print(f"{storm.name}: {storm.rises.sum() / 1e3:,.1f} k interruptions expected; all values simulated")
        _table([(name, _mean([sim.run(rule, storm, k) for k in range(a.seeds)])) for name, rule in rules.items()])
    return 0


if __name__ == "__main__":
    sys.exit(main())
