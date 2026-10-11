# The synthetic case

Runs anywhere, with the core dependencies only. It runs the chain of the
example glacier without a host model: the glacier's position and front
geometry are real, everything else is made up.

```bash
python examples/synthetic/make_inputs.py   # ocean series and map
python examples/synthetic/run.py           # calibration and runs, three routes
```

- `make_inputs.py` writes a seeded monthly thermal forcing in 0-60 m: a
  "reanalysis" for 1995-2019 and a "climate model" for 1995-2014 and two
  scenarios to 2100, with a cold bias. The model is bias corrected onto the
  reanalysis over 1995-2014 with `forcing.bias.bias_correct`, and `TF_ref` is
  the power mean of the reanalysis over 2000-2009 (`laws.tf_power_mean`). It
  also draws a fjord and finds the ocean cells of the footprint with
  `forcing.footprint.select_footprint` on a 1/12 degree grid.
- `toy_host.py` is a one-box tidewater glacier (volume, length, a bed that
  rises inland, a surface mass balance that falls with a made-up warming). It
  is run through the three coupling routes of the real hosts: a law called at
  every monthly step (as in OGGM), one constant per calendar year (as in
  PyGEM's mass-redistribution model) and one factor per October-to-September
  year read back from a GloGEM table.
- `run.py` calibrates the constant of each route and run on the 2000-2009
  target, first statically (`calibration.static.fit_calving_k`), then through
  the toy host (`calibration.dynamic.calibrate_dynamic_k`), and runs
  2000-2019 and both scenarios to 2100, with the constant-k law (control) and
  with `tf_power` (ocean).

The results are in `examples/results/synthetic/`: one CSV per route, with the
columns every host of the example case writes, plus `calibration.csv`.
