"""Synthetic inputs of the example case: ocean series and a map.

Not observations: a stand-in with the glacier's position and front geometry,
so the examples and figures run before the real data are in. Seeded, so the
files are the same on every run.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from example_case import FORCING_COLUMNS, RESULTS, as_forcing_dataset, load_case  # noqa: E402

from tidewater.forcing.bias import bias_correct  # noqa: E402
from tidewater.forcing.footprint import select_footprint  # noqa: E402
from tidewater.forcing.io import cell_areas  # noqa: E402
from tidewater.laws import tf_power_mean  # noqa: E402

REFERENCE = "synthetic reanalysis"
MODEL = "synthetic climate model"
SEED = 1
KM_PER_DEG = 111.195


def _monthly(first, last):
    return pd.date_range(f"{first}-01-01", f"{last}-12-01", freq="MS")


def _red_noise(rng, n, sd, r=0.8):
    out = np.zeros(n)
    for i in range(1, n):
        out[i] = r * out[i - 1] + rng.normal(0, sd * np.sqrt(1 - r * r))
    return out


def ocean(rng):
    """Monthly thermal forcing: a reference product and a biased model with two scenarios."""
    ref_t = _monthly(1995, 2019)
    year = ref_t.year + (ref_t.month - 0.5) / 12
    season = np.cos(2 * np.pi * (ref_t.month - 9) / 12)
    ref = 1.25 + 0.4 * season + 0.012 * (year - 2000) + _red_noise(rng, len(ref_t), 0.25)

    hist_t, proj_t = _monthly(1995, 2014), _monthly(2015, 2100)
    model = {}
    hist_year = hist_t.year + (hist_t.month - 0.5) / 12
    model["historical"] = (0.75 + 0.3 * np.cos(2 * np.pi * (hist_t.month - 9) / 12)
                           + 0.012 * (hist_year - 2000)
                           + _red_noise(rng, len(hist_t), 0.2))
    proj_year = proj_t.year + (proj_t.month - 0.5) / 12
    ramp = np.clip((proj_year - 2015) / 85, 0, 1)
    for scenario, warming in (("ssp126", 0.6), ("ssp585", 2.6)):
        model[scenario] = (0.75 + 0.3 * np.cos(2 * np.pi * (proj_t.month - 9) / 12)
                           + 0.012 * 15 + warming * ramp ** 1.6
                           + _red_noise(rng, len(proj_t), 0.2))
    return ref_t, ref, {"historical": hist_t, "ssp126": proj_t, "ssp585": proj_t}, model


def forcing_files(case, out):
    rng = np.random.default_rng(SEED)
    ref_t, ref, times, model = ocean(rng)
    period = tuple(case["ocean"]["bias_correction_period"])
    ref_ds = as_forcing_dataset(pd.Series(ref, index=ref_t))
    hist_ds = as_forcing_dataset(pd.Series(model["historical"], index=times["historical"]))
    offset = float(bias_correct(hist_ds, ref_ds, period)["bias_offset"].values[0])

    rows = [(t.strftime("%Y-%m"), REFERENCE, "historical", 0, v) for t, v in zip(ref_t, ref)]
    for scenario, values in model.items():
        for corrected in (0, 1):
            vals = values + offset * corrected
            rows += [(t.strftime("%Y-%m"), MODEL, scenario, corrected, v)
                     for t, v in zip(times[scenario], vals)]
    table = pd.DataFrame(rows, columns=FORCING_COLUMNS)
    table.to_csv(out / "forcing.csv", index=False, float_format="%.5f")

    gamma = case["law"]["gamma"]
    y0, y1 = case["periods"]["calibration"]
    cal = pd.Series(ref, index=ref_t).loc[str(y0):str(y1)]
    meta = {
        "synthetic": True,
        "reference_product": REFERENCE,
        "model_product": MODEL,
        "band": case["ocean"]["band"],
        "band_top_m": case["ocean"]["band_top_m"],
        "band_bottom_m": case["ocean"]["band_bottom_m"],
        "gamma": gamma,
        "tf_ref_degc": tf_power_mean(cal.to_numpy(), gamma),
        "tf_ref_period": [y0, y1],
        "bias_correction_period": list(period),
        "bias_offset_degc": {case["ocean"]["band"]: offset},
        "periods": case["periods"],
        "frontal_ablation_targets": case["frontal_ablation_targets"],
    }
    with open(out / "forcing.json", "w") as f:
        json.dump(meta, f, indent=1)
    return table, meta


def _to_lonlat(x_km, y_km, lon0, lat0):
    lat = lat0 + np.asarray(y_km) / KM_PER_DEG
    lon = lon0 + np.asarray(x_km) / (KM_PER_DEG * np.cos(np.radians(lat0)))
    return lon, lat


def map_files(case, out):
    """Outline, flowline, front, water depth and ocean cells, in degrees."""
    lon0, lat0 = case["glacier"]["lon"], case["glacier"]["lat"]
    out = out / "map"
    out.mkdir(parents=True, exist_ok=True)
    w_front = case["glacier"]["front_width_m"] / 1000
    d_front = case["glacier"]["front_water_depth_m"]

    # a valley glacier flowing east, its front at x = 0
    xs = np.linspace(-10.5, 0, 40)
    half = np.where(xs < -2, 1.15, w_front / 2 + (1.15 - w_front / 2) * (-xs / 2))
    bend = 0.25 * np.sin(xs / 3)
    ox = np.r_[xs, xs[::-1]]
    oy = np.r_[bend + half, (bend - half)[::-1]]
    lon, lat = _to_lonlat(ox, oy, lon0, lat0)
    pd.DataFrame({"lon": lon, "lat": lat, "part": 0}).to_csv(
        out / "outline.csv", index=False, float_format="%.6f")
    fx = np.linspace(-9.8, 0, 30)
    lon, lat = _to_lonlat(fx, 0.25 * np.sin(fx / 3), lon0, lat0)
    pd.DataFrame({"lon": lon, "lat": lat}).to_csv(out / "flowline.csv", index=False,
                                                  float_format="%.6f")
    lon, lat = _to_lonlat([0, 0], [-w_front / 2, w_front / 2], lon0, lat0)
    pd.DataFrame({"lon": lon, "lat": lat}).to_csv(out / "front.csv", index=False,
                                                  float_format="%.6f")

    # a fjord from the front to the shelf; NaN on land and ice
    gx, gy = np.meshgrid(np.arange(-14, 42, 0.4), np.arange(-26, 26, 0.4))
    width = w_front / 2 + 0.22 * np.clip(gx, 0, None)
    in_fjord = (gx >= 0) & (np.abs(gy - 0.6 * np.sin(gx / 6)) <= width)
    shelf = gx > 14 + 3 * np.sin(gy / 5)
    depth = np.where(in_fjord, d_front + 18 * np.clip(gx, 0, None), np.nan)
    depth = np.where(shelf, 260 + 3 * (gx - 14) + 25 * np.cos(gy / 4), depth)
    lon, lat = _to_lonlat(gx, gy, lon0, lat0)
    xr.Dataset({"water_depth": (("y", "x"), depth.astype("float32"), {"units": "m"}),
                "lon": (("y", "x"), lon.astype("float32")),
                "lat": (("y", "x"), lat.astype("float32"))}).to_netcdf(
        out / "water_depth.nc",
        encoding={v: {"zlib": True} for v in ("water_depth", "lon", "lat")})

    # a 1/12 degree product grid; wet where the column reaches the band bottom
    dlon = dlat = 1 / 12
    clon = np.arange(lon.min() + dlon / 2, lon.max(), dlon)
    clat = np.arange(lat.min() + dlat / 2, lat.max(), dlat)
    clon, clat = (a.ravel() for a in np.meshgrid(clon, clat))
    cx = (clon - lon0) * KM_PER_DEG * np.cos(np.radians(lat0))
    cy = (clat - lat0) * KM_PER_DEG
    cell_depth = np.where(cx > 14 + 3 * np.sin(cy / 5), 260 + 3 * (cx - 14), 0)
    wet = (cell_depth >= case["ocean"]["band_bottom_m"])[None, :]
    area = float(cell_areas(lat0, dlat, dlon))
    n_cells = int(min(64, round(1000 / area)))
    keep, radius = select_footprint(clon, clat, wet, [0], "strict", (lon0, lat0), n_cells)
    pd.DataFrame({"lon": clon[keep], "lat": clat[keep], "dlon": dlon, "dlat": dlat}).to_csv(
        out / "ocean_cells.csv", index=False, float_format="%.6f")
    return len(keep), radius


def main(out=RESULTS / "synthetic"):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    case = load_case()
    _, meta = forcing_files(case, out)
    n, radius = map_files(case, out)
    print(f"TF_ref {meta['tf_ref_degc']:.3f} degC, offset "
          f"{meta['bias_offset_degc'][meta['band']]:+.3f} degC, "
          f"{n} ocean cells within {radius:g} km")
    return out


if __name__ == "__main__":
    main(*sys.argv[1:])
