"""Adds the `territory` column (the electric utility whose service area covers most of the county) to the county
tables stormcrews/data/ne_counties.csv and pawv_counties.csv, from the two CSV files produced from the public HIFLD
Electric Retail Service Territories polygons (columns fips, territory, utility_hifld_name, area_share, ...).

    python scripts/add_territories.py path/to/ne_territories_hifld.csv path/to/pawv_territories_hifld.csv
"""
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "stormcrews" / "data"

for region, src in zip(("ne", "pawv"), sys.argv[1:3]):
    t = pd.read_csv(src)[["fips", "territory"]]
    c = pd.read_csv(DATA / f"{region}_counties.csv", float_precision="round_trip")
    c = c.drop(columns=["territory"], errors="ignore").merge(t, on="fips", how="left", validate="one_to_one")
    assert c["territory"].notna().all(), f"{region}: counties without territory"
    c.to_csv(DATA / f"{region}_counties.csv", index=False, lineterminator="\n")
    print(region, c["territory"].value_counts().to_dict())
