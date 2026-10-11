# GloGEM

`run_glogem.py` writes the yearly ocean factor tables of the example glacier,
one directory per scenario, and prints the three lines that switch the factor
on in GloGEM's `config.pro`. It needs no IDL: the tables are written in
Python.

```bash
python examples/glogem/run_glogem.py examples/results/synthetic glogem_tables
```

GloGEM itself runs on the branch `glogem-tidewater`, with IDL and GloGEM's own
inputs for the glacier. It keeps its calving law and its calibrated constant
and multiplies its frontal ablation by the factor of each glacier and model
year (October to September in region 05, named by the year it ends).

## Expected numbers

On the synthetic inputs of `examples/results/synthetic`:

| Quantity | Value |
|---|---|
| model years in the tables | 1996 to 2100 |
| mean factor, 2000-2009 | 1.003 |
| mean factor, 2081-2100, SSP1-2.6 | 1.593 |
| mean factor, 2081-2100, SSP5-8.5 | 3.441 |

The mean over the calibration decade is close to 1 because `TF_ref` is the
power mean of the thermal forcing over that decade; it is not exactly 1
because GloGEM's years start in October and the factor is weighted by days.

## Results of the example glacier

From `results/05-09345/glogem.csv`: ERA5 climate, GLORYS12 ocean, IceBoost v2
thickness on GloGEM's elevation bands; full model, in IDL. Frontal ablation is
the decade mean over the observed value.

| Run | GloGEM's calving constant `c_calving` | 2000-2009 | 2010-2019 |
|---|---|---|---|
| control | 0.4632 | 1.00 | 0.94 |
| ocean-forced | 0.4686 | 0.99 | 1.13 |

The 2000-2009 means include a calving event in 2007, when GloGEM removes a band
of the glacier that reaches sea level. A full GloGEM run needs an IDL licence.
