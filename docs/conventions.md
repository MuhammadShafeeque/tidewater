# Conventions

The units, periods, model years, glacier ids and file formats that every part
of `tidewater` uses. A host and `tidewater` agree on these before any number
is exchanged; the host side is described in [hosts.md](hosts.md).

## Units

| Quantity | Unit |
|---|---|
| temperature, thermal forcing, freezing point | degC |
| salinity | psu |
| depth, thickness, width | m, depth positive downwards |
| calving constant `k` | yr-1 in parameters and files |
| frontal ablation flux | m3 ice s-1 from a law; m3 ice yr-1 on request |
| observed and calibrated frontal ablation | any of `tidewater.units.FLUX_UNITS` (`m3 s-1`, `m3 yr-1`, `km3 yr-1`, `kg s-1`, `kg yr-1`, `Gt yr-1`), stated by a `flux_unit` argument |
| subglacial discharge | m3 s-1 |
| sea-ice concentration, open-water fraction | 0 to 1 |
| ocean factor | 1 |
| longitude, latitude | degrees east, degrees north |
| time in a law | float years |
| model years in exported tables | integers, with the start month and the label convention stated in the file |

Ice density is a parameter where a volume is converted to a mass
(`ice_density`, kg m-3); there is no default for it in a conversion.
`tidewater.units.convert_flux` is the one converter.

Every calibration entry point takes `flux_unit` and `ice_density`; their
defaults keep the units each function had: `km3 yr-1` for the static fit and
the covariates, `Gt yr-1` for the dynamic fit. The returned tables carry the
unit in `attrs['flux_unit']`, and
`tidewater.calibration.column_units(flux_unit)` gives the unit of every
column.


## Periods

A period is `(first year, last year)`, both included
(`tidewater.periods`). A run that holds every year of a period starts at the
first year and ends at the last year plus one (`periods.run_bounds`). A
function that reads a period the other way says so and takes
`end='inclusive'|'exclusive'`: the mean frontal ablation of a host run
(`calibration.dynamic.mean_frontal_ablation`, `flux_from_series`) and the
end date of an observation (`calibration.targets.select_period`) are
end-exclusive by default, so `(2000, 2010)` there is the ten years 2000 to
2009.


## Time, ids and files for hosts

- `tidewater.timeconv.to_model_years` aggregates a monthly, calendar-stamped
  series (pandas or cftime, any CF calendar) to model years: calendar years,
  hydrological years with any start month, labelled by the year in which they
  end or start. The start month is never inferred from latitude unless asked
  with `start_month="auto", latitude=...`. The start month, or the latitude,
  can be given per glacier, so a table with both hemispheres converts in one
  call. Months are weighted by their days unless `weights="equal"`.
  Incomplete years are refused unless told otherwise.
- `tidewater.ids` parses RGI 6 (`RGI60-01.00570`) and RGI 7
  (`RGI2000-v7.0-G-01-00570`, also `C` complexes) ids, converts them to the id
  forms of other hosts, and maps between versions through a link table that the
  caller supplies. No link table is bundled or downloaded. Ids in any other
  format pass through unchanged.
- `tidewater.export` writes per-glacier, per-model-year text tables (one file
  per region, fixed number of header lines), a calibration CSV with a column
  mapping, and a `(time, rgi_id)` NetCDF file with CF time.
  `write_host_bundle` writes all of it for a region or the globe in one call,
  with a JSON sidecar (units, period convention, model-year convention,
  parameters, package version); `read_host_bundle` reads it back.

Regional parameters are described in [regions.md](regions.md), and
how to bring an ocean product, a bed product or a frontal-ablation dataset in
[data.md](data.md).
