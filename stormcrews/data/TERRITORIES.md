# Utility territory of every county

`ne_territories.csv` and `pawv_territories.csv` give, for each county, the electric utility territory that
`territories: true` uses (`territory`), how it was assigned (`source`), and the two utilities with the largest share
of the county's area in the HIFLD polygons (`hifld_top`, `hifld_share`, `hifld_second`, `hifld_second_share`).
`scripts/add_territories.py` copies the `territory` column into the county tables.

## Source

HIFLD Electric Retail Service Territories (U.S. Department of Homeland Security, public domain), read from the ArcGIS
FeatureServer
`https://services3.arcgis.com/OYP7N6mAJJCyH6hd/arcgis/rest/services/Electric_Retail_Service_Territories_HIFLD/FeatureServer/0/query`
on 9 November 2026 (bounding box of the two regions, 387 polygons). County polygons: `ne_counties.gpkg` and
`pawv_counties.gpkg`.

## Rule

Each county is intersected with the HIFLD polygons in an equal-area projection (EPSG:5070) and the area share of each
utility is computed, keeping only the polygons of the investor-owned operating companies (Eversource CT, MA and NH,
United Illuminating, National Grid MA, Unitil, Liberty NH, Green Mountain Power, Central Maine Power, Versant Power,
Rhode Island Energy, PPL, PECO, Duquesne Light, Penelec, Met-Ed, West Penn Power, Penn Power, Mon Power, Potomac
Edison, Appalachian Power, Wheeling Power) plus the two cooperatives that dominate whole counties (Vermont Electric
Coop, NH Electric Coop); municipal, transit and island overlays are dropped. The county goes to the largest share
(`source = hifld_iou`).

HIFLD polygons overlap heavily and several are coarse, so where the rule contradicts the utilities' published town
lists an explicit override takes precedence (`source = override`): all counties of Connecticut, Massachusetts, New
Hampshire, Rhode Island, Vermont and Maine, and in Pennsylvania and West Virginia the counties of Duquesne Light
(Allegheny, Beaver), Penn Power (Mercer, Lawrence), Met-Ed (Adams, Berks, Lebanon, Northampton, York, Monroe, Pike),
PECO (Philadelphia, Bucks, Chester, Delaware, Montgomery), Penelec (the northern and central tier), West Penn Power
(the southwest, Centre, Butler, Fulton), Potomac Edison (the eastern panhandle) and Wheeling Power (Ohio, Marshall,
Brooke, Hancock). The remaining counties of Pennsylvania (mostly PPL in the centre and northeast) and West Virginia
(Mon Power in the north, Appalachian Power in the south) keep the area rule.

## Limits

A county served by two utilities is assigned whole to one of them (New Haven County to United Illuminating, Fairfield
County to Eversource CT, for example). The area rule is weak where the top share is below 0.6 (Lycoming, Luzerne,
Cumberland, Clay, Greenbrier). Customers per utility and county are not published, so the split is by area and town
lists, not by customers.
