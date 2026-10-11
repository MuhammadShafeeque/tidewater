"""The example case and the files every example reads or writes."""
import json
from pathlib import Path

import numpy as np
import pandas as pd

from tidewater.forcing.io import forcing_dataset
from tidewater.physics import freezing_point

HERE = Path(__file__).resolve().parent
CASE = HERE / "case.json"
RESULTS = HERE / "results"

# one row per run and model year, the same in every host
COLUMNS = ["host", "run", "scenario", "year", "frontal_ablation_gt_yr", "volume_km3",
           "area_km2", "length_m", "front_water_depth_m", "thermal_forcing_degc",
           "ocean_factor", "calving_k_yr"]
FORCING_COLUMNS = ["date", "product", "scenario", "corrected", "thermal_forcing_degc"]
ICE_DENSITY = 900.0


def load_case(path=CASE):
    with open(path) as f:
        return json.load(f)


def read_forcing(directory):
    """``forcing.csv`` and ``forcing.json`` of a results directory."""
    directory = Path(directory)
    table = pd.read_csv(directory / "forcing.csv")
    with open(directory / "forcing.json") as f:
        meta = json.load(f)
    return table, meta


def series(table, product, scenario, corrected):
    """One monthly thermal forcing series, indexed by month start."""
    rows = table[(table["product"] == product) & (table["scenario"] == scenario)
                 & (table["corrected"] == int(corrected))]
    return pd.Series(rows["thermal_forcing_degc"].to_numpy(dtype=float),
                     index=pd.to_datetime(rows["date"]), name="thermal_forcing")


def joined(table, meta, scenario):
    """The reference product up to the end of its record, then the corrected model."""
    ref = series(table, meta["reference_product"], "historical", False)
    model = pd.concat([series(table, meta["model_product"], "historical", True),
                       series(table, meta["model_product"], scenario, True)])
    return pd.concat([ref, model[model.index > ref.index[-1]]])


def as_forcing_dataset(tf, band="shallow_0_60", top=0.0, bottom=60.0, salinity=34.0):
    """A one-band forcing dataset of the package's file layout, from a TF series.

    Temperature is set so that it gives ``tf`` at mid-band; salinity is a constant.
    """
    values = tf.to_numpy(dtype=float)[:, None]
    sa = np.full_like(values, salinity)
    th = values + freezing_point(sa, (top + bottom) / 2)
    return forcing_dataset(pd.DatetimeIndex(tf.index).to_numpy(), [band], [top],
                           [bottom], ["uniform"], values, th, sa)


def write_results(rows, path):
    frame = pd.DataFrame(rows, columns=COLUMNS)
    frame.to_csv(path, index=False, float_format="%.6g")
    return frame
