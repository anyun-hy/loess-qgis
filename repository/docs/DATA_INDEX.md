# Loess Data Index

## Main Files

- `data/raw/clcd/CLCD_2023_shaanxi_albers.tif`
  - CLCD 2023 land-cover label raster for Shaanxi subset.
  - Single-band categorical raster.

- `data/raw/glc_fcs30d/GLC_FCS30D_2000_2022_E110N40_annual.tif`
  - GLC_FCS30D annual land-cover labels covering Suide county.
  - 23 bands: band 1 is 2000, band 23 is 2022.

- `data/processed/glc_fcs30d/GLC_FCS30D_2022_E110N40_label.tif`
  - Extracted 2022 single-band label raster from the GLC_FCS30D annual file.
  - Use this when you want the latest GLC label currently available here.

- `data/processed/glc_fcs30d/GLC_FCS30D_2022_suide_clip.tif`
  - GLC_FCS30D 2022 label raster clipped to Suide county.

- `data/processed/glc_fcs30d/GLC_FCS30D_2022_suide_vector.shp`
  - Vectorized GLC_FCS30D 2022 Suide label polygons.
  - Attribute fields follow the local interpretation Shapefile style where possible:
    `YEAR`, `COUNTY`, `COUNTYID`, `TDLYDM`, `TDLYMC`, `GLC_EN`, `AREA`, and `AREA_KM2`.

- `data/processed/glc_fcs30d/GLC_FCS30D_2022_suide_vector_smooth.shp`
  - Display-oriented smoothed copy of the GLC_FCS30D 2022 Suide polygons.
  - Uses the same GLC attributes as the unsmoothed vector file.
  - Use for visual inspection and map display, not for primary area statistics.

- `data/processed/glc_fcs30d/suide_glc_annual_area.csv`
  - Annual Suide county area statistics by GLC_FCS30D class, 2000-2022.

- `data/processed/glc_fcs30d/suide_glc_2000_2022_change.csv`
  - Class-level area change from 2000 to 2022.

- `data/processed/glc_fcs30d/suide_glc_2000_2022_transition.csv`
  - Long-form transition table: class in 2000 to class in 2022.

- `data/processed/glc_fcs30d/suide_glc_2000_2022_summary.md`
  - Short human-readable summary of the GLC_FCS30D change statistics.

- `data/boundaries/suide_county/suide_county_610826.shp`
  - Suide county boundary shapefile.
  - Keep `.shp`, `.shx`, `.dbf`, and `.prj` together.

## Derived CLCD Outputs

- `data/processed/clcd_suide/CLCD_2023_suide_clip.tif`
  - CLCD 2023 clipped to Suide county.

- `data/processed/clcd_suide/CLCD_2023_suide_vector.shp`
  - Vectorized CLCD 2023 Suide label polygons.
  - Attribute fields include `class_id`, `name_en`, `name_zh`, `area_m2`, and `area_km2`.

- `data/processed/clcd_suide/CLCD_class_mapping.csv`
  - CLCD class id to Chinese and English label mapping.

## Scratch

- `scratch/suide_test_legacy.tif`
  - Earlier tiny test file kept only for traceability.
  - Do not use as a formal label product.
