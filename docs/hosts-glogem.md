# GloGEM and GloGEMflow

GloGEM is written in IDL and cannot import Python. It computes frontal
ablation once per model year, in `procedures/processing/calving_model.pro`,
with a geometric law after Oerlemans and Nick (2005): water depth times front
height times the mean terminus slope (in degrees) times an effective width,
times one regional constant `c_calving`. Nothing in it reads the ocean.

Because `c_calving` multiplies a slope in degrees, a `k` calibrated by
`tidewater` for `q = k d h w` is not a value for GloGEM. The coupling therefore
keeps GloGEM's law and constant and multiplies the constant by the yearly
ocean factor of the glacier (route 1 of [hosts.md](hosts.md)).
`tidewater.hosts.glogem` writes the factor tables; the IDL files shipped in
`tidewater/hosts/glogem/idl/` read them inside the calving model.

## Model years

GloGEM's year `Y` runs from the start month of calendar year `Y - 1` to the
month before it in `Y`, and is labelled `Y` (label `end`). The start month is
set per region, not per glacier:

| Run | RGI regions 16 to 19 | All other regions |
|---|---|---|
| monthly | April | October |
| daily | day 121 (1 May) | day 274 (1 October) |

Region 16 (Low Latitudes) uses the southern start for all its glaciers.
`glogem.start_month(region, time_resolution)` gives the month, and
`glogem.forcing_span(first, last, region)` the calendar months the forcing
must cover for a run with `tran = [first, last]`.

## Writing the tables

```python
from tidewater.hosts import glogem

factor, info = glogem.write_ocean_factor_tables(
    "ocean_factor", tf_monthly, tf_ref, gamma=1.18,
    time_resolution="monthly", incomplete="drop", outside="stop")
```

`tf_monthly` is the monthly thermal forcing (degC), one column per RGI id.
Each model year gets the mean of the monthly factors over that year, in the
glacier's region convention. With `tf_fraction=a` below 1 the table holds
`(1 - a) + a F`. `c_calving` keeps its calibrated value when the factor
averages about 1 over the period it was calibrated on, which is the case for
`tf_ref` taken as `tidewater.laws.tf_power_mean` of the thermal forcing over
that period. `glogem.write_tables` writes any other yearly table (for
instance the yearly mean thermal forcing) in the same layout.

There is one file per RGI region, `ocean_factor_<RR>.dat`, with a JSON sidecar
(`tidewater.export.read_host_bundle` reads it all back):

```
# tidewater 0.1.0.dev0 variable=ocean_factor units=1 region=01
# start_month=10 label=end weights=days host=glogem time_resolution=monthly rgi_version=6.0 outside=stop
# missing=-9999 n_glaciers=2 n_years=4 id_style=within_region
id 2000 2001 2002 2003
00570 0.986656 0.986240 0.992651 1.003215
00001 0.789000 1.261644 1.763687 2.288921
```

Rows are keyed by the five-digit glacier number within the region, the `id`
that GloGEM reads from its `thick_<region>.dat`. The ids must all be of one
RGI version, the one the run uses (`RGIversion`).

## Switching it on in GloGEM

GloGEM needs the calving factor hook: a setting that names a function, which
its calving model calls for a factor on the frontal ablation of each glacier
and year. The hook is on the branch `glogem-tidewater` of the fork
[GloGEM-tidewater](https://github.com/MuhammadShafeeque/GloGEM-tidewater). It
holds no ocean code, and with its default (no function) GloGEM's results are
unchanged.

With that branch, ocean forcing is switched on by three lines in GloGEM's
`config.pro`:

```
!PATH = !PATH + ':<tidewater>/hosts/glogem/idl'
calving_factor_function = 'tw_table_factor'
calving_factor_dir = '<directory of the tables>'
```

The first line puts the IDL files of this package on the path
(`tidewater.hosts.glogem.IDL_DIR` gives the directory). The run is then
started as usual. The factor scales the flux before GloGEM's own cap (total
accumulation) and its split at `calv_sep`, so both still apply; where the cap
binds, the factor has no effect.

`tw_table_factor` reads a region's table once and keeps it until the region
changes. It returns 1 for a glacier that is not in the table and for a region
without a table, so land-terminating glaciers and regions without ocean data
need no entry. For a year or value the table lacks, the run stops, or uses 1
or the nearest year: `outside="stop"`, `"one"` or `"nearest"` when the tables
are written. It stops when the table's start month, label or RGI version do
not match the run. It needs `rgiregion`, which GloGEM sets when regions are
selected with `region_id_loop` from `region_batch.dat`.

In a GloGEM without the branch, the hook is two default settings in
`procedures/initialise/settings.pro`,

```
calving_factor_function = ''
calving_factor_dir = ''
```

and, in `procedures/processing/calving_model.pro`, directly after the line
that computes `F` from `c_calving`:

```
  if calving_factor_function ne '' then F = F * call_function(calving_factor_function, $
    calving_factor_dir, rgiregion, id[gg[g]], ye + tran[0], bal_month = bal_month, $
    time_resolution = time_resolution, rgi_version = RGIversion)
```

## GloGEMflow

The `GloGEMflow` branch calls the same `calving_model.pro`, so the same hook
applies. Its flow model (`procedures/flow/`) does not read the frontal
ablation: when a glacier runs with `use_flow_model`, the scaled flux is
written to the `Frontal_ablation` output, but the volume it removes (`dvol`)
only reaches the geometry through the Δh retreat model, which the flow model
replaces. Ocean forcing therefore changes the reported frontal ablation of
flow-model glaciers, not their geometry, until the flow model takes the
calving flux off its terminus.

## The laws in IDL

`tidewater_laws.pro` holds every formula of `spec/SPEC.md` (`tw_ocean_factor`,
`tw_tf_scaling`, `tw_yearly_ocean_factor`, `tw_law_flux` for the five laws,
the partition of a calibrated constant, the thermal-forcing kernels), for a
host that evaluates a law itself (route 2). `tw_spec_check, '<path to
reference_vectors.csv>'` checks them against the reference vectors. A host
that evaluates the law once per year applies it to each month and averages,
as for the factor.

## MATLAB hosts

The tables are plain headed, whitespace-delimited text: `#` comment lines,
then a line naming the columns, then one row per glacier. A MATLAB host reads
them with `readtable(path, "FileType", "text", "CommentStyle", "#")`, keeping
the id column as text if leading zeros matter. Alternatively it reimplements the
law from `spec/SPEC.md` and checks it against `spec/reference_vectors.csv`.

## Tests

`tests/hosts/glogem/` tests the writer, and runs the IDL code when an IDL or
GDL interpreter is found (`$TIDEWATER_IDL`, else `gdl` or `idl` on `PATH`):
all reference vectors, and the table lookup against the Python tables.
With `$GLOGEM_DIR` set to a GloGEM checkout that has the hook, GloGEM's own
calving routine is run on a synthetic glacier, with the hook off and on.
