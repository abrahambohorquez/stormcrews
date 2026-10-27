# Data licenses and attribution

The code of StormCrews is under the MIT license (see `LICENSE`). The files in `stormcrews/data/` are derived from the
public data sets listed below, each under its own terms. The derived files are released under the Creative Commons
Attribution 4.0 International license (CC BY 4.0, https://creativecommons.org/licenses/by/4.0/), the license of the
EAGLE-I and ERA5 inputs, with the attributions below. The Census and NOAA inputs are in the public domain. The
derived files are provided as is, without warranty of any kind. A copy of this file ships inside the package as
`stormcrews/data/DATA_LICENSE.md`.

| File | Content | Sources |
|---|---|---|
| `ne.npz`, `pawv.npz` | Hourly rises of customers out per county above the pre-storm baseline, observed regional curve and observed customer-hours per county for every storm; damage forecast per county; peak gust, accumulated precipitation and mean soil moisture per county and storm | EAGLE-I (ORNL); ERA5 via Open-Meteo; US Census Bureau |
| `ne_storms.json`, `pawv_storms.json` | Storm windows, NOAA event type, peak hour, measured share of automatic restorations, calibrated fleet and repair parameters | EAGLE-I (ORNL); NOAA Storm Events; line crews reported in public utility and regulator filings (numbers only) |
| `ne_counties.csv`, `pawv_counties.csv` | Customers, centroid, customers per square mile | EAGLE-I; US Census Bureau (ACS housing units, Gazetteer) |
| `*_counties.gpkg`, `*_states.gpkg` | County and state boundaries | US Census Bureau cartographic boundary files, 2020 |
| `damage_model.json` | Coefficients of the damage model fitted on the shipped arrays | Same as `ne.npz`, `pawv.npz` |

## EAGLE-I (Oak Ridge National Laboratory), CC BY 4.0

Brelsford, C., Tennille, S., Myers, A., Chinthavali, S., Tansakul, V., Denman, M., et al. *The Environment for
Analysis of Geo-Located Energy Information's Recorded Electricity Outages 2014-2025*, version 4 (25 February 2026).
figshare. Dataset. https://doi.org/10.6084/m9.figshare.24237376.v4. Described in Brelsford et al. (2024), *A dataset
of recorded electricity outages by United States county 2014–2022*, Scientific Data 11, 271,
https://doi.org/10.1038/s41597-024-03095-5. License: Creative Commons Attribution 4.0 International,
https://creativecommons.org/licenses/by/4.0/.

Modifications made for StormCrews: the 15-minute county records were aggregated to hourly values; each state's
EAGLE-I customer total was divided among its counties in proportion to Census housing units, and reports above a
county's total were clipped; storm windows were detected on the regional series; the damage of every storm was
measured as the customers out above the county's pre-storm baseline (median of the 24 hours before the window) and
turned into hourly rises; the share restored within the hour was measured; and the result was combined with ERA5 and
the Census.

**Neither Oak Ridge National Laboratory nor the US Department of Energy endorses this work.**

## ERA5 via Open-Meteo, CC BY 4.0

Peak wind gust, accumulated precipitation and mean soil moisture (0 to 7 cm) per county over every storm window, in
the `weather_*` arrays of `ne.npz` and `pawv.npz` and in the damage model fitted on them (`damage_model.json`).
Contains modified Copernicus Climate Change Service information (2026). Copernicus Climate Change Service (C3S),
*ERA5 hourly data on single levels from 1940 to present*, Climate Data Store (CDS), https://doi.org/10.24381/cds.adbb2d47;
Hersbach, H., et al. (2020), *The ERA5 global reanalysis*, Quarterly Journal of the Royal Meteorological Society
146(730), 1999–2049, https://doi.org/10.1002/qj.3803. ERA5 is distributed by the Climate Data Store under CC BY 4.0.

Weather data by Open-Meteo.com (https://open-meteo.com/), retrieved in 2026 through its Historical Weather API for
non-commercial research and offered under CC BY 4.0 (https://open-meteo.com/en/licence). Zippenfenig, P.,
*Open-Meteo.com Weather API*, Zenodo, https://doi.org/10.5281/zenodo.7970649.

Modifications made for StormCrews: the hourly ERA5 values of the 0.25° cell of each county were reduced to the
maximum gust, the summed precipitation and the mean soil moisture over each storm window, and a damage model was
fitted on them. Neither the European Commission, ECMWF nor Open-Meteo is responsible for any use that may be made of
this information, and none of them endorses this work.

## US Census Bureau (public domain)

American Community Survey 2017–2021 5-year estimates, table B25001 (housing units),
https://www2.census.gov/programs-surveys/acs/summary_file/2021/table-based-SF/, used to divide each state's EAGLE-I
customer total among its counties. Gazetteer files 2022, and 2021 for the eight Connecticut counties, for land area
and interior points, https://www.census.gov/geographies/reference-files/time-series/geo/gazetteer-files.html. Cartographic boundary files
`cb_2020_us_county_5m` and `cb_2020_us_state_5m`, clipped to the two regions and simplified,
https://www.census.gov/geographies/mapping-files/time-series/geo/carto-boundary-file.html. Source: U.S. Census Bureau.

## NOAA Storm Events Database (public domain)

Event type of every storm window. NOAA National Centers for Environmental Information, Storm Events Database,
https://www.ncei.noaa.gov/access/storm-events-database/.

## Parameters in the code (not in `stormcrews/data`)

Line crews per 1,000 customers out at the peak, read from public filings (Connecticut PURA Docket 20-08-03, New
Hampshire PUC dockets including DE 24-041, Pennsylvania PUC report on the storms of 29 April to 9 May 2025, a Central
Maine Power release of October 2019 and Eversource's presentation to the Connecticut General Assembly of 27 August
2020); only those numbers enter the anchored calibration, and no table or text of the filings is reproduced.
