# OGGM

`run_oggm.py` writes the ocean forcing file of the example glacier in the
package's layout (the file OGGM reads as `ocean_data`) and prints the flux of
the `tf_power` law at the glacier's front for a calving constant of 1 yr-1.
With `--workdir` it then runs OGGM in that working directory: the dynamic
calibration of `k` on the 2000-2009 frontal ablation, then a control run
(`constant`) and an ocean-forced run (`tf_power`) to 2100.

```bash
python examples/oggm/run_oggm.py examples/results/synthetic
python examples/oggm/run_oggm.py examples/results/synthetic --workdir <dir>   # with OGGM
```

The run needs OGGM from the branch `oggm-tidewater` and a glacier directory of
RGI2000-v7.0-G-05-09345 that already holds its flowlines, its inversion and a
`gcm_data` climate file from 2000 to 2100.

## Expected numbers

On the synthetic inputs, scenario SSP1-2.6, for a front 48 m thick in 38 m of
water and 566 m wide, at `k0` = 1 yr-1:

| Year | Flux, 10^6 m3 of ice per year |
|---|---|
| 2005 | 1.304 |
| 2050 | 1.360 |
| 2095 | 2.253 |

## Results of the example glacier

From `results/05-09345/oggm.csv`: ERA5 climate, GLORYS12 ocean, IceBoost v2
thickness (Glen A fitted to its volume). Frontal ablation is the decade mean
over the observed value.

| Run | Calving constant, yr-1 | 2000-2009 | 2010-2019 |
|---|---|---|---|
| control | `k` = 3.527 | 1.01 | 0.82 |
| ocean-forced | `k0` = 2.344 | 1.03 | 0.64 |
