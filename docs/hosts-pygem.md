# PyGEM

PyGEM (version 1.1.0) is built on released OGGM (1.6.3 or later). It computes
frontal ablation with OGGM's k-calving law, `q = k d h w`, with `calving_k` in
yr-1. Nothing in it reads the ocean. A tidewater glacier is run by one of two
dynamical models:

| `option_dynamics` | Model | Frontal ablation |
|---|---|---|
| `OGGM` | OGGM's `FluxBasedModel` | at every time step, m3 s-1 |
| `MassRedistributionCurves` | PyGEM's `MassRedistributionCurveModel` | once per model year, m3 yr-1, capped by the ice below the water level |

`tidewater.hosts.pygem` couples both, and does not import OGGM.

## Two routes, the same numbers

| Route | PyGEM it needs | What the adapter does |
|---|---|---|
| hook | the branch `pygem-tidewater` | sets `run_simulation.calving_law_for` |
| swap | released PyGEM 1.1.0 | replaces the two model names in `run_simulation` while a `with` block is open |

The branch adds a calving-law hook to PyGEM and nothing else: a module-level
`calving_law_for(gdir, option_dynamics)` in `run_simulation`, whose result is
passed to the dynamical model of each tidewater glacier, and a `calving_law`
argument of `MassRedistributionCurveModel`, called once per year as
`calving_law(model, flowline, last_above_wl, year)` and returning m3 of ice.
The branch does not import `tidewater`. With no hook set it is released PyGEM
bit for bit.

`dynamics.ocean_dynamics` takes the hook where the module has it and the swap
otherwise (`dynamics.coupling_route()` names the one in use; `route="swap"`
forces the swap). Both routes build the calving law of a glacier with the
same function and the same yearly-constant code, so their results are
identical: on PyGEM's marine-terminating sample glacier every output variable
of `run_simulation` is bit-identical between the two, for both dynamics
options.

## Conventions

- **Model year.** PyGEM's dynamics run on calendar years only
  (`sim_wateryear: calendar`; its mass balance refuses other years). Model
  year `Y` is January to December of `Y`, and the time of the flowline model
  is the calendar year plus a fraction. `tidewater.hosts.pygem.START_MONTH`
  is 1.
- **Glacier id.** RGI 6.0, or RGI 7.0 on the branch `pygem-tidewater-rgi7`
  (`rgi_version: '70G'` in PyGEM's configuration). PyGEM writes a glacier as
  a number on the command line (`1.03622`, or `05-09345` on RGI 7.0) and as
  its RGI id in its tables. `rgi_id` and `glacno` convert, and every table
  of the adapter accepts both.
- **Constant and flux.** `calving_k` in yr-1, as in `tidewater`. Observed
  frontal ablation is in Gt yr-1 (`fa_gta_obs`), ice density 900 kg m-3.

## Inside PyGEM's run scripts

```python
import sys
from pygem.bin.run import run_simulation
from tidewater.hosts.pygem import laws_from_forcing
from tidewater.hosts.pygem.dynamics import (frontal_ablation_components,
                                            ocean_dynamics)

laws = laws_from_forcing({"1.03622": "ocean/RGI60-01.03622.nc"}, "tf_power")
sys.argv = ["run_simulation", "-rgi_glac_number", "1.03622",
            "-option_dynamics", "OGGM"]
with ocean_dynamics(laws) as built:
    run_simulation.main()
calving, melt = frontal_ablation_components(built[0][1])   # m3 of ice
```

`laws` maps a glacier to a law of `tidewater.laws`, or to a table of yearly
factors for the mass-redistribution model. Each model gets its own copy of
the law; `built` lists `(rgi_id, calving)`. Glaciers without an entry, and
land-terminating glaciers, run as before, and the module is restored on
exit. On either route the change lives in the calling process: with several
cores it reaches the workers only where they are forked, not spawned.

## The two models

**FluxBasedModel.** It takes a calving law as `calving_law`, and a law of
`tidewater.laws` is called the same way and reads the same attributes. A
model built by hand takes the law as it is:

```python
from oggm.core.flowline import FluxBasedModel
from tidewater.laws import ocean_calving_law

law = ocean_calving_law(ds, "tf_power", tf_ref_period=(2000, 2019))
model = FluxBasedModel(nfls, y0=2000, mb_model=mbmod, glen_a=glen_a, fs=fs,
                       is_tidewater=True, water_level=water_level,
                       calving_law=law)
```

`ds` is the forcing dataset of the glacier (`tidewater.forcing`). A law built
without its own constant takes the model's `calving_k`.

**MassRedistributionCurveModel.** `dynamics.YearlyCalving` gives each model
year its own constant in PyGEM's product `k d h w`; PyGEM's choice of the
front, its cap and its accounting are kept. It is built from one of:

- `calving_law=law`: the constant of the year is the one at which
  `k d h w` equals the mean of the law's flux over the twelve mid-month
  times of the year, on the front at the start of the year. All five laws
  work. The months are weighted equally.
- `ocean_factor=table`: the model's `calving_k` times the factor of the year,
  from `yearly_factors(tf_monthly, tf_ref)`. This is the mean of the monthly
  factors over the calendar year, weighted by days unless
  `weights="equal"`.

On the branch a `YearlyCalving` is the `calving_law` of
`MassRedistributionCurveModel`. On released PyGEM,
`dynamics.OceanMassRedistributionCurveModel(..., calving_law=law)` is a
subclass that sets the model's `calving_k` for the year from the same
`YearlyCalving` and calls PyGEM's own frontal ablation. `components()` of a
`YearlyCalving` gives frontal ablation and its melt part per calving year.

## Calibration

PyGEM's `run_calibration_frontalablation` finds one `calving_k` per glacier
by bisection between 0.001 and 5 yr-1, so that the mean frontal ablation of
`FluxBasedModel` over `ref_startyear` to `ref_endyear` matches the
observation, lowered first where the climatic mass balance it implies is
outside the regional range. It writes `RGIId, calving_k, calving_k_nmad`,
which `run_simulation` reads; a glacier without a row takes the regional
median.

| PyGEM | tidewater |
|---|---|
| merged observations (`RGIId`, `fa_gta_obs`, `fa_gta_obs_unc`) | `calibration.read_targets(path)`, Gt yr-1 |
| the forward run of the calibration | `calibration.run_flux(build_model, period)` for `calibrate_dynamic_k` |
| `all-frontalablation_cal_ind.csv` | `calibration.write_calibration(table, calibration_file(pygem_prms))` |
| `calving_k_nmad`, sampled as a standard deviation | `calving_k_unc` |
| regional median for glaciers without data | the geometric mean, per group if given |

`build_model(rgi_id, calving_k)` returns the model PyGEM would run for that
glacier; the adapter does not build glacier directories or mass-balance
models. `tidewater` does not apply PyGEM's mass-balance bound on the target:
pass the bounded target to keep it. The hook of the branch is in
`run_simulation` only; in `run_calibration_frontalablation`, which falls back
from `FluxBasedModel` to the mass-redistribution model when a run fails,
`ocean_dynamics(laws, modules=[...])` uses the swap on both PyGEMs.

`calibration.law_from_calibration(ds, calving_k, law, period, front)` turns
a calibrated constant into an ocean-forced law. `tf_power` takes the constant
as `k0` at the power mean of the thermal forcing over the calibration
period, so the constant of the law averages to `calving_k` there. A law with
a melt term takes the residual constant that keeps the frontal ablation of
the calibrated front (`front_geometry(flowline, water_level)`). After a
change of law the dynamic calibration can be run again with the law in
`build_model`.

On the `FluxBasedModel` path, PyGEM 1.1.0's calibration converts the
modelled flux from ice to water equivalent twice, so the number it compares
with the observation is 0.9 times the frontal ablation that `run_simulation`
writes for the same constant. `run_flux(..., as_pygem=True)` reproduces that
number, to compare with PyGEM's constants; the default is the flux itself.

## Tests

`tests/hosts/pygem/test_pygem_helpers.py` runs without PyGEM.
`test_pygem_host.py` needs PyGEM and a configured `~/PyGEM/config.yaml`
(PyGEM checks its data paths on import), and skips otherwise. It runs on
released PyGEM and on the branch: the test ids name the route, and the cases
that need the hook are skipped on released PyGEM with that reason. The tests
through `run_simulation` need PyGEM's calibrated sample glacier and skip
without it. Use an environment of its own: PyGEM pins `numpy<2`.
