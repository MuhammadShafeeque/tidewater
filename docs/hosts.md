# Coupling a host model

`tidewater` computes frontal ablation at a calving front from the state the
host gives it. The host keeps everything else: geometry, capping of the flux by
the available ice, mass removal, front migration and accounting.

Host-specific notes: [OGGM](#oggm) below, [PyGEM](hosts-pygem.md),
[GloGEM and GloGEMflow](hosts-glogem.md).

What every route needs from the host is listed in the README under "Host
integration contract" (the page of that name in the documentation). Two
conventions must be agreed before any number is
exchanged:

- **Model year.** State the start month and whether a year is labelled by the
  calendar year in which it starts or ends. `tidewater.timeconv` takes both as
  arguments and writes them into exported files. Hosts differ: calendar years,
  hydrological years starting in October or in April, or a custom start.
  The start month can differ per glacier (`start_month` as a mapping, or
  `"auto"` with a latitude per glacier), so both hemispheres convert in one
  call.
- **Period.** A period is `(first year, last year)`, both included
  (`tidewater.periods`). A host run that holds it starts at the first year and
  ends at the last year plus one (`periods.run_bounds`). Where a function
  reads a period the other way it takes `end='inclusive'|'exclusive'`; the
  mean frontal ablation of a host run is end-exclusive by default.
- **Glacier id.** State the inventory and its version. `tidewater.ids` converts
  between id forms; mapping between inventory versions needs a link table that
  the caller supplies.
- **One front per glacier.** A glacier has one calving front. A host with
  several marine fronts per glacier calls the law once per front and adds the
  fluxes.

## The yearly ocean factor

A host that computes frontal ablation once per model year multiplies its
calving constant by the yearly ocean factor. That factor is the mean of the
monthly factors over the model year,
`mean((max(TF_month, 0) / TF_ref)^gamma)`, not the factor of the yearly mean
thermal forcing. The two differ because the factor is a power of the forcing
and clipped at zero. `tidewater.timeconv.yearly_ocean_factor(tf_monthly,
tf_ref, gamma, start_month=..., weights="days")` computes it; months are
weighted by their days in the calendar of the time axis, or equally with
`weights="equal"`.

## Python hosts

### Law hook on the flowline model

A law object is built once per glacier from its forcing record (time axis in
float years, thermal forcing, optionally open-water fraction and subglacial
discharge). The host calls it at each time step.

- A host whose flowline model accepts a calving law as a callable passes the law
  object directly. The law reads thickness, water depth and width at the last
  grid point above the water level and returns a flux in m3 ice s-1. The host
  adapter in `tidewater.hosts` does the wiring and the unit handling of the
  calving constant.
- A host without such a hook calls `law.flux(h, d, w, yr)` (m3 ice s-1) or
  `law.flux_per_year(...)` (m3 ice yr-1) where it computes frontal ablation,
  with its own terminus state.
- A host that computes frontal ablation once per model year, with its own law
  and a calving constant in yr-1, can multiply that constant by the yearly
  ocean factor instead. Aggregate the monthly thermal forcing to the host's
  model years with `tidewater.timeconv.yearly_ocean_factor`, or read the
  factor from a NetCDF file written by `tidewater.export.write_netcdf`.

Calibrated constants are exchanged as a CSV keyed by glacier id
(`tidewater.export.write_calibration_csv`), with a column mapping so that the
file can carry the column names the host expects.

### OGGM

`tidewater.hosts.oggm` couples the scheme to OGGM through OGGM's own
extension points: laws are passed as the `calving_law` of the flowline
models, tasks are OGGM entity tasks, files are added to `cfg.BASENAMES` and
parameters to `cfg.PARAMS`. It runs on the OGGM branch `oggm-tidewater`,
which carries four small hooks the scheme cannot reach from outside; the
branch does not import `tidewater`.

The adapter needs the glacier settings and the data store of the OGGM
development line after release 1.6.3 (`gdir.settings`, `gdir.read_store`).
On released OGGM 1.6.3 or earlier `register()` raises and
`tidewater.hosts.oggm.supported()` is False. On OGGM master without the
branch, the forcing, the laws, the runs and the dynamic calibration work, and
the table says what is missing.

| Hook on the branch | Without it |
|---|---|
| `run_with_hydro` adds the calved mass back before the mass closure | hydro output at calving glaciers counts calving as surface melt |
| the geodetic and scalar calibration fit the surface balance to the observation plus the frontal ablation, and store it as `calving_mb` (land glaciers get no such key) | `NotImplementedError` once `inversion_calving_rate` is not zero |
| `run_dynamic_spinup` passes `is_tidewater`, `is_lake_terminating` and the calving water level to the model | calving is off in the spinup |
| `workflow.CALVING_INVERSION_TASK`, the task `inversion_tasks` runs on tidewater glaciers (default: the stock task) | `inversion_calving_from_bathymetry` is ignored by `inversion_tasks` and the Glen A calibrations |

`tidewater.hosts.oggm.register()` adds the files `ocean_data` and
`frontal_ablation_diagnostics`, the parameters of the scheme with their
defaults (`tidewater.hosts.oggm.params.DEFAULTS`: the `ocean_*` and the ocean
`calving_*` keys, `inversion_calving_from_bathymetry`, `bed_extension_*`,
`bedmachine_version`, `calving_front_width_rtol`) and sets
`workflow.CALVING_INVERSION_TASK` to `find_inversion_calving`, which picks the
bathymetry inversion per glacier when `inversion_calving_from_bathymetry` is
set and the stock task otherwise. It is idempotent and keeps values already
in `cfg.PARAMS`. It also makes every later `cfg.initialize()` set the
parameters again, read from its parameter file or `params=` when given there.
Multiprocessing workers get them with the rest of OGGM's configuration.

| Module | Holds |
|---|---|
| `laws` | the five laws as OGGM calving laws, `ocean_calving_law`, `run_with_ocean_forcing`, `write_frontal_components`, `compile_frontal_components` |
| `forcing` | `process_ocean_data`, `process_destine_ocean_data`, `write_subglacial_discharge` |
| `inversion` | `terminus_water_depth_from_bed`, `find_inversion_calving_from_bathymetry`, the static fits of `k` |
| `calibration` | `run_dynamic_calving_k_calibration` and its default pass |
| `bed`, `shop_bed` | a measured bed under and beyond the front, and the BedMachine reader |
| `tasks` | all tasks in one namespace |

The default pass of the dynamic calibration puts the measured bed under the
front with `bed.bedmachine_terminus_bed`; another bed task is passed as
`kwargs_run_function={'terminus_bed_task': ...}`.

## Hosts that cannot import Python

### Route 1: precomputed factor table

`tidewater` runs offline and writes, per region, a whitespace-delimited text
file with one row per glacier and one column per model year
(`tidewater.export.write_region_tables`). The values are the dimensionless
ocean factor, or the thermal forcing in degC. The host reads the file and
multiplies its own calving constant by the factor of the glacier and year.

- The file has a fixed number of header lines (`tidewater.export.HEADER_LINES`);
  the last one names the columns.
- The header states the variable, the units, the region, the missing value and
  the model-year convention passed by the caller.
- Row ids are the number of the glacier within its region, or the full id.
- Regions with different year conventions (for instance October and April
  starts) are aggregated with a start month per glacier, then written in one
  call.

`tidewater.export.write_host_bundle(directory, factor, calibration, params,
start_month=...)` writes the tables of every region, the calibration CSV and a
JSON sidecar (`tidewater_export.json`) with the units of every column, the
period and model-year conventions, the parameters and the package version.
`read_host_bundle(directory)` reads it back.

This route changes the least in the host: one file read and one multiplication.
The factor does not react to the modelled geometry, since it only depends on the
ocean. Laws with an explicit melt term depend on the water depth of the modelled
front and cannot be precomputed this way.

### Route 2: reimplementation from the spec

The host implements the law in its own language from the equations, and checks
its implementation against the reference vectors shipped in `tidewater.spec`
(inputs and expected outputs as plain text). The forcing still comes from
`tidewater` as a table (route 1 with the thermal forcing, and further tables for
open-water fraction or subglacial discharge where a law needs them).

This route supports every law, including those that depend on the modelled
front. It needs code in the host, and the reference vectors are the test that
the two implementations agree.

### Route 3: offline diagnostic

The host runs unchanged and writes its terminus state per glacier and model
year (thickness, water depth, width). `tidewater` then computes the frontal
ablation that its laws give for that state and compares it with what the host
removed. Nothing is fed back, so this route does not change the host's results;
it shows how much an ocean-dependent law would change them, and is the usual
first step before route 1 or 2.

## Choosing a route

| Route | Host code change | Laws supported | Feedback on geometry |
|---|---|---|---|
| Law hook (Python) | adapter only | all | yes |
| Factor table | read one file, multiply | `tf_power` | through the host's own law |
| Reimplementation from spec | the law itself | all | yes |
| Offline diagnostic | write terminus state | all | none |
