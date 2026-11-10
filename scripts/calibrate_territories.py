"""Writes the `anchored_territory` calibration of every storm into stormcrews/data/{ne,pawv}_storms.json.

It keeps the anchored fleet (line crews per customer out at the peak, from public filings) and the anchored job-size
dispersion, and refits the mean repair time so that current practice with `territories: true` (each utility territory
keeps its own crews and asks for its own aid) reproduces the storm's observed customer-hours: bisection on the log of
the repair time, seed 0, as in the anchored calibration itself.

    python scripts/calibrate_territories.py            # both regions
    python scripts/calibrate_territories.py --region ne --workers 4
"""
import argparse
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from stormcrews import Scenario, Simulator  # noqa: E402
from stormcrews.rules import CurrentPractice  # noqa: E402

DATA = ROOT / "stormcrews" / "data"
_SIMS = {}


def _sim(region):
    if region not in _SIMS:
        _SIMS[region] = Simulator(Scenario(name=region, region=region, calibration="anchored", territories=True))
    return _SIMS[region]


def _with_repair(storm, r):
    meta = dict(storm.meta); cals = dict(meta["calibrations"]); c = dict(cals["anchored"]); c["repair_mean_h"] = r
    cals["anchored"] = c; meta["calibrations"] = cals
    return replace(storm, meta=meta)


def refit(job):
    region, sid = job
    sim = _sim(region); st = sim.region.storm(sid)
    target = float(st.observed.sum()); r0 = st.calibration("anchored")["repair_mean_h"]
    lo, hi = r0 / 8, r0 * 8
    mid, ch = r0, np.nan
    for _ in range(18):
        mid = float(np.sqrt(lo * hi))
        ch = sim.run(CurrentPractice(), _with_repair(st, mid), 0).metrics["customer_hours"]
        if abs(ch / target - 1) < 0.01:
            break
        lo, hi = (mid, hi) if ch < target else (lo, mid)
    return {"region": region, "storm": sid, "repair_mean_h": mid, "repair_anchored_h": r0, "ch_ratio_fit": ch / target}


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--region", choices=("ne", "pawv", "both"), default="both")
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()
    for region in (("ne", "pawv") if a.region == "both" else (a.region,)):
        ids = _sim(region).storm_ids
        fits = []
        with ProcessPoolExecutor(max_workers=min(a.workers, 4)) as ex:
            for f in ex.map(refit, [(region, s) for s in ids]):
                fits.append(f)
                print(f"{region} storm {f['storm']}: repair {f['repair_mean_h']:.1f} h (anchored {f['repair_anchored_h']:.1f}),"
                      f" fit ratio {f['ch_ratio_fit']:.3f}", flush=True)
        path = DATA / f"{region}_storms.json"
        meta = json.loads(path.read_text(encoding="utf-8"))
        by_id = {f["storm"]: f for f in fits}
        for m in meta:
            anc = m["calibrations"]["anchored"]; f = by_id[m["id"]]
            m["calibrations"]["anchored_territory"] = {"fleet": anc["fleet"], "repair_mean_h": f["repair_mean_h"],
                                                       "size_sigma": anc["size_sigma"]}
        with open(path, "w", encoding="utf-8", newline="\n") as fh:
            fh.write(json.dumps(meta, indent=1))
        worst = max(fits, key=lambda f: abs(f["ch_ratio_fit"] - 1))
        print(f"{region}: {len(fits)} storms refitted; repair time median {np.median([f['repair_mean_h'] for f in fits]):.1f} h"
              f" (anchored {np.median([f['repair_anchored_h'] for f in fits]):.1f} h); worst fit ratio {worst['ch_ratio_fit']:.3f}"
              f" in storm {worst['storm']}", flush=True)
