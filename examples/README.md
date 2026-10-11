# Examples

All examples work on one glacier, RGI2000-v7.0-G-05-09345 (RGI60-05.13043 in
RGI 6.0), a marine-terminating glacier of 21.09 km2 in the east of the
Greenland periphery (68.30 N, 29.52 W), whose front is 566 m wide in 38 m of
water. `case.json` holds the case: the glacier, the ocean products, the depth
range, the law, the periods, and the frontal ablation and mass change targets.

| Directory | What it shows | Needs |
|---|---|---|
| `synthetic/` | the whole chain on a toy host and synthetic inputs | the core dependencies |
| `oggm/` | the forcing file, the calibration and two runs in OGGM | OGGM from `oggm-tidewater` for the run |
| `pygem/` | the law and the yearly factors, then PyGEM's run script | PyGEM for the run |
| `glogem/` | the factor tables and the `config.pro` lines | IDL and GloGEM for the run |
| `results/` | result files that the figures read | |
| `fetch_data.py` | fetches the results and the inputs of the host runs from the data home, with checksums | an internet connection |

Every script reads a results directory with `forcing.csv` and
`forcing.json`; the default is `results/synthetic`. The documentation page
"Examples" shows the figures.

The result files of the example glacier are shipped. Its inputs (ocean
records, climate cuts, thickness, the glacier directory) and the full runs of
the three hosts are in the data home,
<https://cluster.klima.uni-bremen.de/~mshafeeque/tidewater/>, with the scripts
that made them. `python examples/fetch_data.py inputs --to DIR` fetches the
inputs; `data_files.csv` lists the files with their checksums.
