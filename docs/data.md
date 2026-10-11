# Bringing your own data

No ocean product, bed product or frontal-ablation dataset is built in as the
only route. Each enters through a neutral layout with explicit units; a named
convention is a registry entry that anyone can add.

## Conventions

| Quantity | Unit and sign |
|---|---|
| potential temperature | degC; K is converted when declared or detected |
| salinity | practical salinity, psu; absolute salinity in g/kg is converted |
| depth of the ocean | m, positive downwards |
| bed and surface elevation | m, in one vertical datum, negative below it |
| sea level | m, in that same datum (`sea_level`, `OceanParams.ocean_sea_level`, default 0) |
| water depth at the front | m, positive, `sea_level - bed`, 0 when the bed is not below sea level |
| thermal forcing | degC, temperature minus the freezing point; clipped at zero before any power |
| sea-ice concentration | 0 to 1; percent is converted |
| subglacial discharge | m3 s-1 of water |
| frontal ablation | positive for a loss of ice; a law never returns a negative flux |
| longitude, latitude | degrees east on [-180, 180), degrees north |

**Sea level.** Every function that turns a bed elevation into a water depth,
or a surface elevation into a freeboard, takes `sea_level`
(`physics.water_depth`, `physics.submerged_depth`, `physics.front_thickness`,
`physics.terminus_water_depth_from_bed`, `bed.terminus_water_depth`,
`laws.calving_law_flux`). A law object takes `sea_level` as a number or a
function of time, or reads the host's own. Use it for a bed product whose
datum is not mean sea level, or for a level that changes in time.

**Periods.** A period is `(first year, last year)`, both included
(`tidewater.periods.as_period`). `as_period(p, end='exclusive')` reads a
period that stops at the start of its last year, and `run_bounds` gives the
start and end of a run that holds a period. Functions that average a host run
(`calibration.dynamic.mean_frontal_ablation`, `flux_from_series`) or read the
end date of an observation (`calibration.targets.select_period`) take
`end='inclusive'|'exclusive'`, end-exclusive by default.

## An ocean product

`tidewater.forcing.products.open_product(path_or_ds, product=None,
**overrides)` opens a file, a dataset or several files of one product and
returns the neutral layout of `tidewater.forcing.io.normalise`: `thetao`
(degC) and `so` (psu) on `(time, level, cell)`, `depth(level)` in m positive
downwards, `latitude(cell)` and `longitude(cell)`, optionally `siconc` (0 to
1). Regular, curvilinear and unstructured grids are accepted.

- The variable names are looked up in `PRODUCTS` (the entries `cmip` and
  `nemo` are shipped), then by CF standard name. A product with other names is
  added with `register_product(name, names, temperature_units=...,
  salinity_units=..., depth_units=..., depth_positive=..., siconc_units=...)`,
  or its names are passed once as `names=` to `open_product` or `normalise`.
- Units come from the declared argument, else from the `units` attribute.
  `'auto'` tells K from degC, cm from m and percent from fraction by
  magnitude; it is a fallback, logged when used.
- Depth direction comes from `depth_positive`, else the `positive` attribute,
  else the sign.

From the normalised product to one water column per terminus:
`forcing.footprint.reduce_site` (the search radii and target area are
arguments, see [regions.md](regions.md)). From the column to the per-glacier
forcing file (`thermal_forcing(time, band)` and the variables listed in
`tidewater.forcing.io`): `forcing.bands.process_ocean_column` or
`process_ocean_data`. Bands a product does not reach at a glacier are filled
from its nearest glaciers in the same product with `forcing.fill`. An offset
onto a reference product over a reference period is
`forcing.bias.bias_correct(ds, ref_ocean, period)`.

The freezing point is linear in salinity and depth (`physics.freezing_point`)
or TEOS-10 with `teos10=True` and the `teos10` extra.

## Sea ice and subglacial discharge

Sea-ice concentration may come with the ocean product or from a separate
product: `forcing.seaice.as_fraction(siconc, units)` converts percent, and
`with_observed_siconc` puts an observed open-water series into a forcing
dataset. Subglacial discharge comes from the host's runoff:
`forcing.discharge.discharge_from_runoff` and `add_subglacial_discharge`.

## A bed product

A bed product is any `xarray.DataArray` of bed elevation on two horizontal
dimensions, projected or geographic. `tidewater` does not reproject: give
the points in the grid's coordinates.

- `bed.sample_bed(bed, x, y, method='nearest'|'bilinear')` samples it along a
  flowline.
- `bed.terminus_water_depth(bed, x, y, sea_level=..., surface_class=...,
  bed_mask=..., n_front=...)` gives the water depth at the front, the number
  of points behind it and where it came from (`bed`, `mask`, `land`, `dry`).
- A surface-type mask is read through a convention in `bed.BED_MASKS`:
  `none` (the bed and sea level decide) and `bedmachine` are shipped; add
  another with `register_bed_mask(name, ocean_classes=..., ice_classes=...,
  floating_classes=..., land_classes=...)`.
- A measured terminus depth needs no bed product: pass the water depth to
  the law or to `forcing.bands.process_ocean_data(terminus_depth=...)`
  directly.

## A frontal-ablation dataset

`tidewater.calibration.targets.read_targets(source, columns=..., flux_unit=...,
to_unit=..., ice_density=..., id_map=..., reduce=...)` reads any dataset into
one table with the columns `glacier_id`, `flux`, `flux_err`, `start_date`,
`end_date`, `source`.

- `source`: a CSV or other delimited file (pass `sep`, `skiprows` and other
  arguments of `pandas.read_csv`), a NetCDF file (`.nc`, `.nc4`, `.cdf`), a
  `pandas.DataFrame` or an xarray object.
- `columns`: the dataset's column for each neutral name; only `glacier_id`
  and `flux` are required.
- `flux_unit`: any of `tidewater.units.FLUX_UNITS`, for the flux and its
  error; `to_unit` and `ice_density` convert on reading. Volumes are volumes
  of ice.
- `id_map`: the dataset's ids to the ids of the run, as keyword arguments of
  `tidewater.ids.map_ids` (a link table such as RGI 6 to RGI 7), a mapping or
  a function. Observations without a partner are dropped.
- Periods of any length are kept as given. `select_period` keeps the
  observations of one period; `reduce='mean'|'latest'|'longest'` or a
  callable reduces several observations of one glacier to one
  (`reduce_targets`). Without `reduce`, every observation is kept, and
  `target_series` refuses a glacier with several.
- `register_target_source(name, columns=..., flux_unit=..., **read_kwargs)`
  names a dataset's layout so that `read_targets(path, preset=name)` reads
  it. `TARGET_SOURCES` is shipped empty.

The calibration functions take the observations in the unit they are told
(`flux_unit`) and convert once; see [conventions.md](conventions.md) for their defaults.
