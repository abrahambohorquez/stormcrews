"""Builds stormcrews/data from the research pipeline it comes from. Users do not need to run it; the data ship with the
package. It is here so every number can be traced back.

For each region it writes <region>.npz, <region>_storms.json and <region>_counties.csv:
  counties   FIPS, name, state, customers, centroid and customers per square mile
  storms     for every storm, the hourly rises of customers out above the pre-storm baseline per county (EAGLE-I,
             Oak Ridge National Laboratory, CC BY 4.0), the observed regional curve, the measured share restored
             automatically, the hour of the regional peak, and the calibrations of the crew fleet (shape of the
             restoration curve, equal jobs, fleet anchored to public crew counts at their median, lowest and highest
             ratio)
  weather    for every storm and county, the ERA5 peak gust (m/s), accumulated precipitation (mm) and mean soil
             moisture (m3/m3) over the storm window (Copernicus ERA5 via Open-Meteo, CC BY 4.0), for the damage model
  geometry   Census 2020 cartographic boundary files of the counties and states, as GeoPackages

Processing steps, from the public sources to the shipped files (steps 1 to 3 were run upstream by the research
pipeline whose products this script reads):
  1 hourly series   EAGLE-I county records every 15 minutes (Oak Ridge National Laboratory, CC BY 4.0), reduced to
                    the hourly median of the valid readings per county; customers per county rebuilt from Census
                    housing units (ACS 2017-2021, table B25001) dividing each state's EAGLE-I customer total, and
                    reports above that total clipped
  2 storm catalog   a storm is a run of hours with at least 1 % of the region's customers out; its window runs from 48
                    hours before the regional share first exceeds a tenth of the peak to the last hour it stays above a
                    tenth of the peak; windows shorter than 24 hours are dropped, peaks closer than 72 hours merged, and
                    windows failing data-quality screens (rows without values, hours without reports, sparse
                    quarter-hour sampling, isolated drops to zero) are flagged unusable; the NOAA Storm Events type is
                    attached
  3 county table    Census Gazetteer land area and interior point, and customers per square mile
  4 damage          baseline = median of the 24 hours before the window, per county; net = customers out above it;
                    rises = positive part of the hourly change of net (datos.arrivals)
  5 automatic share phi = sum over hours of min(rise in hour h, fall from h to h+1) / sum of rises
  6 weather         ERA5 via Open-Meteo, per county and hour: peak gust, summed precipitation and mean soil moisture over
                    the window
  7 calibrations    fleet, mean repair time and job-size dispersion per storm: shape (grid and bisection on customer-hours
                    and curve shape), equal (equal jobs, 3 h repairs), anchored (fleet from line crews per customer out in
                    public utility filings), and anchored_low and anchored_high at the lowest and highest ratio of the
                    filings
  8 damage model    fitted afterwards on the shipped arrays by scripts/fit_damage_model.py, which also writes the damage
                    forecast of every storm (forecast_<id> in <region>.npz)

    python scripts/build_data.py --pipeline "<research pipeline>" --geometry "<folder with the GeoPackages>"
    python scripts/fit_damage_model.py
"""
import argparse
import json
import os
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument("--pipeline", required=True, help="folder of the processing pipeline that produced the hourly series and the calibrations")
ap.add_argument("--geometry", required=True, help="folder with condados_<region>.gpkg and estados_<region>.gpkg")
ap.add_argument("--region", default="both", choices=("ne", "pawv", "both"))
A = ap.parse_args()
PIPE = Path(A.pipeline); GEO = Path(A.geometry)
OUT = Path(__file__).resolve().parents[1] / "stormcrews" / "data"; OUT.mkdir(parents=True, exist_ok=True)


def build(region):
    os.environ["TORMENTAS_DATOS"] = str(PIPE / ("datos" if region == "ne" else "replica_pawv/datos"))
    for m in [m for m in sys.modules if m.startswith("tormentas_sim")]:
        del sys.modules[m]
    sys.path.insert(0, str(PIPE / "ambiente"))
    from tormentas_sim import datos as D
    res = PIPE / ("resultados" if region == "ne" else "replica_pawv/resultados") / "v2"
    cty = D.counties(); cat = D.catalog()
    c = cty.rename(columns={"cnty_name": "name", "INTPTLAT": "lat", "INTPTLONG": "lon"})
    c[["fips", "name", "state", "customers", "lat", "lon", "density"]].to_csv(
        OUT / f"{region}_counties.csv", index=False, lineterminator="\n")
    for kind, src in (("counties", "condados"), ("states", "estados")):
        shutil.copyfile(GEO / f"{src}_{region}.gpkg", OUT / f"{region}_{kind}.gpkg")
    cals = {k: pd.read_csv(res / f).set_index("event") for k, f in
            (("shape", "calibracion.csv"), ("anchored", "calibracion_anclada.csv"),
             ("anchored_low", "calibracion_anclada_baja.csv"), ("anchored_high", "calibracion_anclada_alta.csv"))}
    for k in ("anchored_low", "anchored_high"):
        cv = cals[k]
        cals[k] = cv[cv["calibrated"]] if "calibrated" in cv else cv
    g = pd.read_csv(res / "calibracion_grilla.csv")
    cals["equal"] = g[(g["size_sigma"] == 0) & (g["repair_h"] == 3.0)].set_index("event")
    # ERA5 (Open-Meteo) per county and hour: peak gust, accumulated precipitation and mean soil moisture of every window
    wx = pd.read_parquet(D.DATA / "clima_era5.parquet", columns=["fips", "hour", "gust", "tp", "soil_moist"])
    arrays, meta = {}, []
    for e in cat["event"]:
        rises, net = D.arrivals(e)
        fall = np.clip(net[:-1] - net[1:], 0, None)
        auto = float(np.minimum(rises[:-1], fall).sum() / max(rises.sum(), 1.0))
        arrays[f"rises_{e}"] = rises            # float64, so runs match the research bit for bit
        arrays[f"observed_{e}"] = net.sum(1)                       # regional customers out above baseline, hourly
        arrays[f"observed_county_{e}"] = net.sum(0)                # observed customer-hours per county
        ev = cat.loc[e]
        w = wx[(wx["hour"] >= ev["inicio"]) & (wx["hour"] <= ev["fin"])]
        g = w.groupby("fips").agg(gust_max=("gust", "max"), precip_mm=("tp", "sum"), soil_moisture=("soil_moist", "mean"))
        arrays[f"weather_{e}"] = g.reindex(cty["fips"]).to_numpy(float)   # counties x (gust m/s, precip mm, soil m3/m3)
        arrays[f"forecast_{e}"] = np.zeros(len(cty), np.float32)       # written by scripts/fit_damage_model.py
        row = cat.loc[e]
        meta.append({"id": int(e), "start": str(row["inicio"]), "peak": str(row["pico"]), "end": str(row["fin"]),
                     "name": row["nombre_tormenta"] if isinstance(row["nombre_tormenta"], str) else "",
                     "type": row["tipo_nws"] if isinstance(row["tipo_nws"], str) else "",
                     "hours": int(rises.shape[0]), "peak_hour": int(np.argmax(net.sum(1))),
                     "observed_customer_hours": float(net.sum()), "peak_customers_out": float(net.sum(1).max()),
                     "auto_share": auto,
                     "calibrations": {k: {"fleet": int(cv.loc[e, "fleet"]), "repair_mean_h": float(cv.loc[e, "repair_h"]),
                                          "size_sigma": float(cv.loc[e, "size_sigma"]) if "size_sigma" in cv else 0.0}
                                      for k, cv in cals.items() if e in cv.index}})
    np.savez_compressed(OUT / f"{region}.npz", **arrays)
    with open(OUT / f"{region}_storms.json", "w", encoding="utf-8", newline="\n") as fh:
        fh.write(json.dumps(meta, indent=1))
    print(region, len(c), "counties", len(meta), "storms")


for r in (("ne", "pawv") if A.region == "both" else (A.region,)):
    build(r)
