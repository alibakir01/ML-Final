# CEUS spatial/temporal layer

Urban (geospatial + temporal) add-ons for the STR-demand paper, for the CEUS
(Computers, Environment and Urban Systems) version of the manuscript.

## Run
```
pip install pyarrow geopandas libpysal esda mapclassify matplotlib
python inside_airbnb/ceus_spatial/ceus_spatial_temporal.py
```
Produces `figures/Figure_11..16.pdf` (vector) and prints the key numbers.

## Inputs (all existing project files)
- `inside_airbnb/outputs/oof_setupC_blend6.npz`  – blend out-of-fold predictions
- `inside_airbnb/outputs/features_setupC.parquet` – geo columns (neighbourhood, borough, lat/lon)
- `inside_airbnb/outputs/q1_panel.parquet`        – raw daily availability (for the temporal curve)
- `Internship/AirBnb_Inside/2026_Inside_Airbnb/January2026/neighbourhoods.geojson`

## Figures
| File | What | Data basis |
|---|---|---|
| Figure_11_demand_map | predicted demand choropleth by neighbourhood | model predictions |
| Figure_12_residual_map | residual (bias) map | model predictions |
| Figure_13_LISA | local spatial clusters (LISA) | model predictions |
| Figure_14_borough | mean predicted demand by borough | model predictions |
| Figure_15_demand_over_time | intra-quarter demand curve | observed (raw calendars) |
| Figure_16_borough_time | demand curve by borough | observed (raw calendars) |

## Key results (honest)
- Global Moran's I ≈ **0.066, p ≈ 0.06** → demand is only **weakly / non-significantly**
  spatially clustered.
- Borough means shallow: Manhattan 21.4 → Bronx 17.0 nights.
- Borough demand **share is stable** over the quarter (Manhattan ~50%, Brooklyn ~33%),
  i.e. **no space-time interaction** — demand scales up, does not migrate.
- Temporal demand peaks the week of **16 Mar 2026**.

## Framing note
Because the spatial signal is weak, the manuscript frames the contribution as a
**method / data-product** ("a reproducible way to turn degraded public snapshots
into fine-grained space-time demand surfaces"), NOT as a claim that demand is
strongly clustered. Do not overstate the spatial finding.
