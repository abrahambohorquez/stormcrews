# Scenario configuration

A scenario fixes the world a rule is tested in: the region and its storms, the calibration, the crews and mutual aid,
and weather what-ifs. It is a YAML file, a Python dict or a `stormcrews.Scenario`. At default values a scenario replays the real storms exactly.

```python
from stormcrews import Scenario, Simulator

Simulator("new_england")                                   # a shipped scenario, by name
Simulator("my_scenario.yaml")                              # any YAML file
Simulator("new_england", calibration="anchored", fleet_factor=0.8)   # a shipped scenario with keys overridden
Scenario.load({"region": "pawv", "aid": "none"})           # a dict
```

Unknown keys are rejected with a `ValueError` that names them, so a typo never passes silently. The shipped files are
in [`stormcrews/config/`](../stormcrews/config) and every key below is read by the code
(`tests/test_config.py` checks that this table and the `Scenario` class list the same keys).

## Scenario keys

| Key | Type | Unit | Default | Meaning |
|---|---|---|---|---|
| `name` | text | | file name | Label of the scenario in result tables |
| `region` | `ne` or `pawv` | | `ne` | New England (67 counties, 39 storms) or Pennsylvania and West Virginia (122 counties, 46 storms) |
| `storms` | `all` or list of ids | | `all` | Storms to run; `stormcrews storms --region ne` lists the ids |
| `calibration` | `shape`, `equal`, `anchored`, `anchored_low`, `anchored_high` | | `shape` | Fleet, repair time and job-size dispersion of every storm (see the README) |
| `decision_h` | number | h | 6 | Hours between decision points; decisions also come when mutual aid arrives |
| `fleet_factor` | number > 0 | | 1 | Multiplies the calibrated local fleet |
| `aid` | `normal`, `none` | | `normal` | Mutual aid comes in full, or nobody comes |
| `reactive_aid` | true or false | | true | Place the usual request when `reactive_trigger` of the customers are out and the rule has not asked yet |
| `reactive_trigger` | number in (0, 1) | share of customers | 0.01 | Share of the region's customers out that triggers the usual request |
| `job_order` | `largest`, `oldest`, `guard`, `sweep` | | `largest` | Which job of an area goes first: the one with most customers, the oldest, the largest unless a job has waited 48 h, or the largest with one crew in ten always taking the oldest |
| `damage_scale` | number or `{fips: factor}` | | 1 | Multiplies the replayed damage, everywhere or in single counties |
| `repair_factor` | number > 0 | | 1 | Multiplies the calibrated mean repair time |
| `customers_per_job` | number > 0 | customers | 35 | Customers behind one repair job on average (an assumption; see the paper's sensitivity) |
| `speed_kmh` | number > 0 | km/h | 60 | Crew travel speed between area centroids |
| `aid_lead_h` | number ≥ 0 | h | 24 | Hours from an aid request to the arrival of the crews |
| `aid_share` | number ≥ 0 | crews per local crew | 0.5 | Aid crews per local crew at multiplier 1 |
| `what_if` | null or mapping | | null | Weather and network changes applied to the real storm through the damage model (keys below) |

## `what_if`

Numbers apply to every county; a mapping `{fips: value}` applies to single counties. Neutral values replay the storm.

| Key | Type | Unit | Neutral | Meaning |
|---|---|---|---|---|
| `gust_change` | number or mapping | m/s | 0 | Added to each county's peak gust |
| `precip_factor` | number or mapping | | 1 | Multiplies accumulated precipitation |
| `soil_moisture_change` | number or mapping | m³/m³ | 0 | Added to mean soil moisture |
| `customer_growth` | number or mapping | share | 0 | More customers, and more damage through the density term |
| `footprint_shift_km` | [east, north] | km | [0, 0] | Moves the damage pattern |
| `footprint_scale` | number > 0 | | 1 | Above 1 spreads the damage over a wider area |
| `duration_factor` | number > 0 | | 1 | Above 1 makes the damage arrive more slowly |
| `auto_share` | null or number in [0, 1) | share | null | Share restored without a crew within the hour; null keeps each storm's measured share |
| `model_draw` | null or integer | | null | Seed of a draw of the damage-model coefficients, for uncertainty |
| `coefficients` | null or mapping | | null | Overrides damage-model coefficients, e.g. `{gust: 0.1}` |
| `damage_spec` | null, `equal`, `equal_region`, `count` | | null | Damage model: counties weigh equally (default), fitted on the storm's region, or exposure weighted |

## An annotated custom scenario

Save as `my_scenario.yaml` and run with `Simulator("my_scenario.yaml")` or
`stormcrews whatif --scenario my_scenario.yaml --storm isaias`.

```yaml
# Pennsylvania and West Virginia with the fleet set from public crew counts, a smaller local fleet,
# slower mutual aid and stronger gusts.
name: pawv_stress           # label in result tables
region: pawv                # Pennsylvania and West Virginia
storms: all                 # or a list of ids, e.g. [0, 11, 45]
calibration: anchored       # fleet from public filings, repair time fitted per storm
decision_h: 6.0             # the rule decides every 6 hours and when aid arrives

fleet_factor: 0.8           # 80 % of the calibrated local fleet
aid: normal                 # aid comes in full
reactive_aid: true          # usual request when 1 % of customers are out ...
reactive_trigger: 0.01      # ... this share
aid_lead_h: 36.0            # aid takes 36 hours to arrive instead of 24
aid_share: 0.5              # half a crew per local crew

job_order: largest          # largest job first inside each county
customers_per_job: 35.0     # the default assumption

what_if:
  gust_change: 3.0          # 3 m/s stronger peak gusts in every county
```
