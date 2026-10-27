"""StormCrews quick start.

1. Replay Tropical Storm Isaias (New England, August 2020) under current practice and under an early mutual-aid
   request, and print what each does to the reliability indices.
2. Write a rule of your own in a few lines and run it.
3. Step through a storm decision point by decision point.
4. Change the weather of a real storm, and build a storm that never happened.

    python example.py
"""
import numpy as np

from stormcrews import Simulator
from stormcrews.rules import CurrentPractice, EarlyAid

ISAIAS = 10

sim = Simulator("new_england")
print(f"{sim.region.title}: {len(sim.counties)} counties, {len(sim.storm_ids)} storms\n")

for rule in (CurrentPractice(), EarlyAid(mult=1.5, hours_before_peak=24)):
    res = sim.run(rule, storm=ISAIAS, seed=1)
    m = res.metrics
    print(f"{res.rule}, {res.storm.name}")
    print(f"  {m['restoration_end_h']:.0f} hours to restore {m['customers_interrupted']:,.0f} customer interruptions")
    print(f"  SAIDI contribution {m['saidi_min']:,.0f} min, CAIDI {m['caidi_h']:,.1f} h, "
          f"{m['crews_max']:,.0f} crews on duty at most")
    print(f"  {m['customer_hours'] / 1e6:,.1f} M customer-hours, {m['over_72h'] / 1e3:,.1f} k customers out > 72 h, "
          f"{(m['local_crew_hours'] + m['aid_crew_hours']) / 1e3:,.0f} k paid crew-hours\n")


# 2. A rule of your own: favor counties whose oldest job has waited longest, and ask for double aid once a fifth of
#    the forecast damage has appeared.
class PatientFirst:
    name = "patient first"

    def decide(self, state):
        weights = state.out * (1 + state.wait_h / 24)
        seen = state.arrived.sum() / max(state.forecast.sum(), 1.0)
        aid = 2.0 if seen > 0.2 and not state.aid_requested else 0.0
        return weights, aid


res = sim.run(PatientFirst(), storm=ISAIAS, seed=1)
print(f"{res.rule}: SAIDI contribution {res.metrics['saidi_min']:,.0f} min, aid asked at hour "
      f"{res.outcome['aid_requested_at']:.0f}, {res.metrics['over_72h'] / 1e3:,.1f} k customers out > 72 h\n")

# 3. Step by step: look at the state at every decision point and decide.
session = sim.start(storm=23, seed=0)               # the December 2022 winter storm
while not session.done:
    state = session.state
    busiest = state.counties["name"].iloc[int(np.argmax(state.out))]
    if state.t % 24 == 0:
        print(f"  hour {state.t:4.0f}: {state.out.sum():8,.0f} customers out, most in {busiest}, "
              f"{state.crews} crews on duty")
    session.step(CurrentPractice().decide(state))
print(f"{session.storm.name}: restored in {session.metrics['restoration_end_h']:.0f} h, "
      f"SAIDI contribution {session.metrics['saidi_min']:,.0f} min\n")

# 4. What-if: Isaias with peak gusts 3 m/s stronger and 20 % more rain, through the damage model (all simulated).
from stormcrews import gust_field, synthetic_storm  # noqa: E402

for label, s in (("real Isaias", Simulator("new_england")),
                 ("Isaias, gust +3 m/s, rain x1.2", Simulator("new_england", what_if={"gust_change": 3, "precip_factor": 1.2}))):
    m = s.run(CurrentPractice(), storm=ISAIAS, seed=1).metrics
    print(f"{label:32s} {m['customers_interrupted'] / 1e6:5.2f} M interruptions, SAIDI {m['saidi_min']:6,.0f} min, "
          f"{m['over_72h'] / 1e3:6,.1f} k customers out > 72 h")

# 5. A storm that never happened: peak gust 38 m/s centered on Hartford, with the timing of Isaias.
storm = synthetic_storm(sim.region, gust_field(sim.region, center=(41.8, -72.7), peak=38, radius_km=120), template=ISAIAS)
m = sim.run(CurrentPractice(), storm, seed=1).metrics
print(f"{storm.name}: {m['customers_interrupted'] / 1e6:.2f} M interruptions, SAIDI contribution {m['saidi_min']:,.0f} min")
