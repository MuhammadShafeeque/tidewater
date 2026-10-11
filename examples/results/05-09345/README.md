# Results of the example glacier, RGI2000-v7.0-G-05-09345

Real data, cut around one glacier of the east Greenland periphery. The files
follow the layout that `docs/figures/results.py` reads.

| File | Content | Source |
|---|---|---|
| `forcing.csv`, `forcing.json` | monthly thermal forcing, 0-60 m, at the glacier front: the reanalysis, and the climate model under two scenarios after the bias correction; the reference value, the offsets, the periods, the frontal ablation and the mass change targets | GLORYS12; MPI-ESM1-2-HR; Kochtitzky et al. (2022); Hugonnet et al. (2021) |
| `map/outline.csv` | glacier outline | RGI 7.0 |
| `map/front.csv` | terminus point and the water depth there | RGI 7.0; BedMachine Greenland v6 |
| `map/water_depth.nc` | sea level minus bed elevation, positive below sea level | BedMachine Greenland v6 |
| `map/ocean_cells.csv` | the reanalysis cell nearest to the front | GLORYS12 |
| `oggm.csv`, `pygem.csv`, `glogem.csv` | one row per run, scenario and model year: frontal ablation, volume, area, length, water depth at the front, thermal forcing, ocean factor and the calving constant in use; runs `control`, `ocean` and `ocean_gamma0` (exponent 0, which reproduces control); state at the start of the model year, so each historical run ends with a row that holds only the end state | OGGM, PyGEM and GloGEM, on ERA5, GLORYS12 and MPI-ESM1-2-HR |

Generated using E.U. Copernicus Marine Service Information (GLORYS12). The
references and licences are on the examples page of the documentation.

To draw the figures:

```bash
python docs/figures/results.py examples/results/05-09345
```
