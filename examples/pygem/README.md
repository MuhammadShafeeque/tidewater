# PyGEM

`run_pygem.py` builds the `tf_power` law of the example glacier from its
ocean forcing and the yearly factors on PyGEM's calendar years, and prints
both. With `--run` it then runs PyGEM's `run_simulation` for the glacier
inside `ocean_dynamics`, on PyGEM's OGGM dynamics.

```bash
python examples/pygem/run_pygem.py examples/results/synthetic
python examples/pygem/run_pygem.py examples/results/synthetic --run   # with PyGEM
```

The run needs PyGEM and its inputs for the glacier where PyGEM's
configuration points. By default the glacier is RGI2000-v7.0-G-05-09345 on
its RGI 7.0 outline, which needs the branch `pygem-tidewater-rgi7` with
`rgi_version: '70G'` and OGGM's RGI 7.0 glacier directories in PyGEM's
configuration. With `--rgi6` it is RGI60-05.13043 on its RGI 6.0 outline,
for released PyGEM 1.1.0 or the branches on RGI 6.0. Give PyGEM an
environment of its own (it pins its own OGGM and numpy).

## Expected numbers

On the synthetic inputs, scenario SSP1-2.6:

| Year | `k / k0` in mid-year | mean of the year |
|---|---|---|
| 2005 | 1.263 | 0.987 |
| 2050 | 1.318 | 1.261 |
| 2095 | 2.183 | 1.816 |

The two columns differ because the forcing has a seasonal cycle: the
mid-year value falls in the warm half of the year, the yearly value is the
mean of twelve monthly factors. A law on the OGGM dynamics reads the first at
every step; the mass-redistribution model takes the second once a year.

## Results of the example glacier

From `results/05-09345/pygem.csv`: the RGI 7.0 outline, ERA5 climate,
GLORYS12 ocean, IceBoost v2 thickness (Glen A fitted to its volume), OGGM
dynamics. Frontal ablation is the decade mean over the observed value.

| Run | Calving constant, yr-1 | 2000-2009 | 2010-2019 |
|---|---|---|---|
| control | `k` = 0.2878 | 0.98 | 0.87 |
| ocean-forced | `k0` = 0.3008 | 1.01 | 1.14 |

PyGEM finds its front by its own inversion, in which the calving law carries
the observed flux: the front stands in 194 to 197 m of water in 2000, deeper than
the 38 m of BedMachine at the terminus that OGGM and GloGEM use. A front that
shallow would carry about 0.1 % of the observed flux.
