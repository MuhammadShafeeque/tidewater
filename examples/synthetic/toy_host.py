"""A toy host: a one-box tidewater glacier, run through the three coupling routes.

The glacier has a volume, a length that follows from it, and a bed that rises
inland, so that a retreating front reaches the sea's edge and stops calving.
It keeps the capping and the mass budget; ``tidewater`` gives the flux. The
three routes differ only in how the ocean reaches the law:

- ``law each step``: a law object called at every monthly step (as in OGGM);
- ``yearly constant``: one constant per calendar year, the calibrated
  constant times the year's ocean factor (as in PyGEM's mass-redistribution
  model);
- ``factor table``: the same on October-to-September years, read from the
  text table GloGEM reads.
"""
import tempfile

import numpy as np
import pandas as pd

from tidewater.export import read_region_table
from tidewater.hosts import glogem
from tidewater.ids import parse_id
from tidewater.laws import ConstantK, TFPower
from tidewater.timeconv import yearly_ocean_factor

ROUTES = ("law each step", "yearly constant", "factor table")
ICE_DENSITY = 900.0


class ToyGlacier:
    """Volume ``V = c L^1.5``, area ``A = W L``, bed at the front rising inland."""

    def __init__(self, length_m=10500.0, area_km2=21.09, volume_km3=2.7,
                 front_width_m=566.0, front_water_depth_m=38.0, freeboard_m=10.0,
                 bed_slope=0.007):
        self.length0 = length_m
        self.width_mean = area_km2 * 1e6 / length_m
        self.c = volume_km3 * 1e9 / length_m ** 1.5
        self.front_width = front_width_m
        self.depth0 = front_water_depth_m
        self.freeboard = freeboard_m
        self.bed_slope = bed_slope
        self.volume = volume_km3 * 1e9

    def copy(self):
        new = object.__new__(ToyGlacier)
        new.__dict__.update(self.__dict__)
        return new

    @property
    def length(self):
        return (max(self.volume, 0.0) / self.c) ** (2 / 3)

    @property
    def area(self):
        return self.width_mean * self.length

    @property
    def water_depth(self):
        """Sea level minus the bed at the front, m; negative on land."""
        return self.depth0 - self.bed_slope * (self.length0 - self.length)

    @property
    def front(self):
        """``(h, d, w)`` at the front, m."""
        d = max(self.water_depth, 0.0)
        return self.freeboard + d, d, self.front_width

    def step(self, smb_m_ice_yr, calving_m3_yr, dt):
        """Apply one step; returns the calved volume, m3, capped by the ice there is."""
        self.volume += smb_m_ice_yr * self.area * dt
        calved = min(max(calving_m3_yr * dt, 0.0), max(self.volume, 0.0))
        self.volume -= calved
        return calved

    def state(self):
        return {"volume_km3": self.volume / 1e9, "area_km2": self.area / 1e6,
                "length_m": self.length, "front_water_depth_m": self.water_depth}


def air_warming(year, scenario):
    """Synthetic air temperature change since 2000, K."""
    if scenario == "historical" or year < 2020:
        return 0.02 * (year - 2000)
    ramp = min((year - 2020) / 80, 1.0)
    return 0.4 + (1.2 * ramp if scenario == "ssp126" else 5.0 * ramp ** 1.5)


def smb(year, scenario, mb0=-0.55, sensitivity=-0.45):
    """Specific surface mass balance, m of ice per year."""
    return mb0 + sensitivity * air_warming(year, scenario)


def _to_gt(m3):
    return m3 * ICE_DENSITY / 1e12


def yearly_factor(tf, tf_ref, gamma, route, glacier_id):
    """The ocean factor per model year of a yearly route."""
    if route == "yearly constant":
        return yearly_ocean_factor(tf, tf_ref, gamma, start_month=1, label="end",
                                   incomplete="drop")
    with tempfile.TemporaryDirectory() as tmp:
        glogem.write_ocean_factor_tables(tmp, tf.to_frame(glacier_id), tf_ref,
                                         gamma=gamma, incomplete="drop")
        gid = parse_id(glacier_id)
        table, _ = read_region_table(f"{tmp}/ocean_factor_{gid.region:02d}.dat")
    return table.iloc[:, 0]


def run(glacier, route, k, years, scenario, tf=None, tf_ref=None, gamma=None,
        glacier_id="RGI60-05.13043"):
    """Run model years ``years`` (first, last); ``tf`` None runs the control.

    Returns one row per model year with the columns of the example results.
    """
    rows = []
    first, last = years
    if route == "law each step":
        if tf is None:
            law = ConstantK(k=k)
        else:
            t = tf.index.year + (tf.index.month - 0.5) / 12
            t, values = t.to_numpy(), tf.to_numpy()
            law = TFPower(t, values, tf_ref=tf_ref, k0=k, gamma=gamma)
        for year in range(first, last + 1):
            calved, tfs, fac = 0.0, [], []
            for month in range(12):
                yr = year + month / 12
                h, d, w = glacier.front
                calved += glacier.step(smb(year, scenario), law.flux_per_year(h, d, w, yr),
                                       1 / 12)
                if tf is not None:
                    tfs.append(np.interp(yr, t, values))
                    fac.append(law.scaling(yr))
            rows.append(_row(glacier, year, scenario, calved, tfs, fac or [1.0], k))
        return rows

    if tf is None:
        factor = pd.Series(1.0, index=range(first, last + 1))
    else:
        factor = yearly_factor(tf, tf_ref, gamma, route, glacier_id)
    start = 1 if route == "yearly constant" else 10
    for year in range(first, last + 1):
        f = float(factor.loc[year])
        h, d, w = glacier.front
        calved = glacier.step(smb(year, scenario), ConstantK(k=k * f).flux_per_year(h, d, w),
                              1.0)
        months = _model_year_months(year, start)
        tfs = [] if tf is None else list(tf.reindex(months).to_numpy())
        rows.append(_row(glacier, year, scenario, calved, tfs, [f], k))
    return rows


def _model_year_months(year, start):
    if start == 1:
        return pd.date_range(f"{year}-01-01", periods=12, freq="MS")
    return pd.date_range(f"{year - 1}-{start:02d}-01", periods=12, freq="MS")


def _row(glacier, year, scenario, calved_m3, tfs, factors, k):
    return {"scenario": scenario, "year": year,
            "frontal_ablation_gt_yr": _to_gt(calved_m3), **glacier.state(),
            "thermal_forcing_degc": float(np.mean(tfs)) if len(tfs) else np.nan,
            "ocean_factor": float(np.mean(factors)), "calving_k_yr": k}
