# Changelog

## Unreleased

**Added**
- Utility territories: `territories: true` gives each utility territory its own crews, which serve only its counties,
  and its own mutual-aid request; the calibrated fleet is split in proportion to the customers each territory serves,
  and the usual aid joins the territories in proportion to the damage each receives.
  The territory of every county comes from the public HIFLD service-area polygons with overrides from the utilities'
  town lists (`stormcrews/data/TERRITORIES.md`, `scripts/add_territories.py`), and `scripts/calibrate_territories.py`
  writes the `anchored_territory` calibration (anchored fleet, repair time refitted under territories). A run without
  territories is unchanged bit for bit (`tests/test_territories.py`). The papers now use `anchored_territory`.

**Changed**
- Every experiment behind the papers now runs 5 seeds per storm (`SEEDS` in `experiments/experiments.py`), the
  held-out check included, and the macros average seeds per storm before taking medians.
- Leave-one-storm-out check (`experiments/experiments.py holdout`): the repair time and the job-size dispersion of a held-out storm are now
  regressed on the log peak and on the hours over which the middle 80 % of the damage arrives (`arrival_hours`),
  chosen among seven input-based candidate sets (`CANDIDATES`). With the territories, the median error of held-out customer-hours goes from 34 to 16 % in New England and from
  29 to 17 % in Pennsylvania and West Virginia. The new `holdout_select` runs
  every candidate so that the choice can itself be held out (nested selection: 16 and 17 %) and compared with a
  regression of customer-hours on the same inputs without the simulator (25 and 27 %). `holdout.csv`, `holdout_parameters.csv`, `holdout_candidates*.csv`, `macros.tex`, the validation
  figures and the README numbers are regenerated.

## 0.3.0 (2026-10-31)

First public release.

**Added**
- Reliability metrics in every run: `saidi_min` (the storm's contribution to SAIDI, major event days included),
  `caidi_h`, `crews_max` (crews on duty once aid has joined), `busy_crew_hours` and `travel_crew_hours`.
- `reactive_trigger`, the share of customers out that triggers the usual mutual-aid request, as a scenario key
  (default 0.01, unchanged behavior).
- Checks on scenario values (shares, positive sizes and speeds) next to the existing check on unknown keys.
- `docs/configuration.md`, every scenario key with type, unit, default and meaning, and a test that keeps it in step
  with the code; a comment on every key of the shipped YAML files.
- `notebooks/explore.ipynb` and `stormcrews.explore`: storms, what-if weather, crews, rules and seeds from menus and
  sliders, with a pinned baseline, synthetic storms and two rules side by side. Binder and Colab launch it without
  installing anything; `pip install "stormcrews[notebook]"` locally.
- `experiments/`, the scripts behind every number and figure of the papers, including a leave-one-storm-out check of the anchored calibration and a sensitivity to the customers per repair job.
- `scripts/make_figures.py` rebuilds every README image; `CITATION.cff`; project URLs and a fourth author in
  `pyproject.toml`.
- `DATA_LICENSE.md` ships inside the package, as a license file of the distribution and as
  `stormcrews/data/DATA_LICENSE.md`.

**Changed**
- `DATA_LICENSE.md` rewritten: EAGLE-I version 4, the ERA5 Climate Data Store entry and the variables used,
  Open-Meteo attribution, the Census ACS housing units and Gazetteer files, and NOAA Storm Events.
- README rewritten for newcomers.
- Plots use the team palette: navy for the simulator, red for observed data, gray for alternatives. County maps keep
  their ramps.
- `scripts/calibration_check.py` writes its figure to `results/` instead of `docs/figures/`.

**Removed**
- README images of the earlier description that the README no longer shows.

## 0.2.0

Proportional dispatch, replanning after aid and queue view in the engine; scheduled aid requests; anchored
calibrations, including their low and high variants.

## 0.1.0

The discrete-event restoration engine on the New England and Pennsylvania and West Virginia storms, reference rules,
scenarios, experiments with paired comparisons, the damage model with what-if knobs and synthetic storms.
