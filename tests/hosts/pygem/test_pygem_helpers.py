"""The PyGEM host helpers that run without PyGEM: ids, front, years, k."""
import numpy as np
import pandas as pd
import pytest
import xarray as xr

from tidewater.config import InvalidWorkflowError
from tidewater.hosts.pygem import (by_glacier, front_cell, front_geometry,
                                   glacier_of, glacno, laws_from_forcing,
                                   rgi_id, yearly_factors)
from tidewater.hosts.pygem import calibration as cal
from tidewater.laws import ConstantK, MeltPlusCalving, TFPower
from tidewater.timeconv import yearly_ocean_factor


class Line:
    """The arrays PyGEM's calving front is read from."""

    def __init__(self, bed, thick, width=500.):
        self.bed_h = np.asarray(bed, dtype=float)
        self.thick = np.asarray(thick, dtype=float)
        self.surface_h = self.bed_h + self.thick
        self.widths_m = np.full(self.bed_h.shape, width)


def ocean(years=(1995, 2020), tf=None):
    time = pd.date_range(f'{years[0]}-01-01', f'{years[1]}-12-01', freq='MS')
    if tf is None:
        tf = 3. + np.sin(2 * np.pi * time.month / 12) + 0.05 * (time.year
                                                              - years[0])
    return xr.Dataset({'thermal_forcing': (('time', 'band'),
                                           np.asarray(tf)[:, None])},
                      coords={'time': time, 'band': ['shallow_0_60']})


def test_ids_both_ways():
    assert rgi_id('1.03622') == 'RGI60-01.03622'
    assert rgi_id(1.03622) == 'RGI60-01.03622'
    assert rgi_id('RGI60-01.03622') == 'RGI60-01.03622'
    assert glacno('RGI60-01.03622') == '1.03622'
    assert glacno(1.0362) == '1.03620'
    assert rgi_id('05-09345') == 'RGI2000-v7.0-G-05-09345'
    assert glacno('RGI2000-v7.0-G-05-09345') == '05-09345'
    assert by_glacier({'1.00570': 1, 'RGI60-15.03733': 2}) == {
        'RGI60-01.00570': 1, 'RGI60-15.03733': 2}
    mb = type('MB', (), {'glacier_rgi_table': pd.Series(
        {'RGIId': 'RGI60-01.03622'})})()
    assert glacier_of(mb) == 'RGI60-01.03622'
    with pytest.raises(InvalidWorkflowError):
        glacier_of(object())


def test_front_is_the_cell_pygem_reads():
    # the last cell with ice above water, over a bed below it
    line = Line(bed=[300., 100., -50., -150., -200.],
                thick=[100., 200., 250., 160., 0.])
    assert front_cell(line, 0.) == 3
    geo = front_geometry(line, 0.)
    assert geo == {'thickness': 160., 'water_depth': 150., 'width': 500.}
    # land-terminating: no calving
    assert front_cell(Line([300., 200.], [100., 50.]), 0.) is None
    assert front_cell(Line([300., 200.], [0., 0.]), 0.) is None
    # no ice above water: PyGEM falls back to the last ice over a deep bed
    sunk = Line(bed=[-300., -400.], thick=[100., 100.])
    assert front_cell(sunk, 0.) == 1
    assert front_geometry(sunk, 0.)['water_depth'] == 400.


def test_yearly_factors_are_the_calendar_year_means():
    ds = ocean()
    tf = ds.thermal_forcing[:, 0].to_series()
    table = pd.DataFrame({'1.03622': tf, 'RGI60-01.00001': tf + 1})
    got = yearly_factors(table, {'1.03622': 3., 'RGI60-01.00001': 4.},
                         gamma=1.18)
    assert list(got.columns) == ['RGI60-01.03622', 'RGI60-01.00001']
    want = yearly_ocean_factor(tf, 3., 1.18, start_month=1)
    np.testing.assert_allclose(got['RGI60-01.03622'], want)
    half = yearly_factors(tf, 3., gamma=1.18, tf_fraction=0.5)
    np.testing.assert_allclose(half, 0.5 + 0.5 * want)


def test_laws_from_forcing():
    ds = ocean()
    laws = laws_from_forcing({'1.03622': ds}, 'tf_power', k0=0.7)
    law = laws['RGI60-01.03622']
    assert isinstance(law, TFPower) and law.k0 == 0.7


def test_read_targets_from_pygem_columns():
    df = pd.DataFrame({'RGIId': ['RGI60-01.03622', 'RGI60-01.00001'],
                       'fa_gta_obs': [2.3, np.nan],
                       'fa_gta_obs_unc': [0.3, 0.1],
                       'start_date': [20000101, 20000101]})
    t = cal.read_targets(df)
    assert t.attrs['flux_unit'] == 'Gt yr-1'
    assert t.glacier_id.tolist() == ['RGI60-01.03622']
    assert t.flux.iloc[0] == 2.3 and t.flux_err.iloc[0] == 0.3
    km3 = cal.read_targets(df, to_unit='km3 yr-1', ice_density=900.)
    assert km3.flux.iloc[0] == pytest.approx(2.3 / 0.9)


def test_calibration_file_round_trip(tmp_path):
    prms = {'root': str(tmp_path), 'calib': {'data': {'frontalablation': {
        'frontalablation_relpath': '/frontalablation_data/',
        'frontalablation_cal_fn': 'all-frontalablation_cal_ind.csv'}}}}
    path = cal.calibration_file(prms)
    assert path.endswith('frontalablation_data/analysis/'
                         'all-frontalablation_cal_ind.csv')
    table = pd.DataFrame({'calving_k': [0.4], 'calving_k_unc': [0.1]},
                         index=pd.Index(['RGI60-01.03622'], name='glacier_id'))
    cal.write_calibration(table, path)
    raw = pd.read_csv(path)
    assert raw.columns.tolist()[:3] == ['RGIId', 'calving_k',
                                        'calving_k_nmad']
    back = cal.read_calibration(path)
    assert back.glacier_id[0] == 'RGI60-01.03622'
    assert back.calving_k[0] == 0.4 and back.calving_k_unc[0] == 0.1


class LinearModel:
    """A model whose frontal ablation is k * 1e9 m3 yr-1."""

    def __init__(self, k, y0=2000):
        self.k, self.yr, self.calving_m3_since_y0 = k, float(y0), 0.

    def run_until(self, y1):
        self.calving_m3_since_y0 += self.k * 1e9 * (y1 - self.yr)
        self.yr = float(y1)


def test_run_flux_feeds_the_dynamic_calibration():
    from tidewater.calibration.dynamic import calibrate_dynamic_k

    run = cal.run_flux(lambda gid, k: LinearModel(k), (2000, 2019),
                       flux_unit='km3 yr-1')
    assert run({'a': 0.5}) == {'a': pytest.approx(0.5)}
    gt = cal.run_flux(lambda gid, k: LinearModel(k), (2000, 2019))
    assert gt({'a': 1.})['a'] == pytest.approx(0.9)
    out = calibrate_dynamic_k(gt, {'a': 1.}, {'a': 0.45})
    assert out.loc['a', 'calving_k'] == pytest.approx(0.5)
    assert out.loc['a', 'converged']
    skip = cal.run_flux(lambda gid, k: None, (2000, 2019))
    assert np.isnan(skip({'a': 1.})['a'])
    late = cal.run_flux(lambda gid, k: LinearModel(k, 2005), (2000, 2019))
    with pytest.raises(InvalidWorkflowError):
        late({'a': 1.})


def test_law_from_calibration_keeps_the_constant():
    ds = ocean()
    law = cal.law_from_calibration(ds, 0.7, 'tf_power', period=(2000, 2019))
    yrs = law.years[(law.years > 2000) & (law.years < 2020)]
    assert np.mean([law.scaling(y) for y in yrs]) == pytest.approx(1.)
    assert law.k0 == 0.7
    assert cal.law_from_calibration(ds, 0.7, 'constant').k == 0.7
    with pytest.raises(InvalidWorkflowError):
        cal.law_from_calibration(ds, 0.7, 'melt_calving')
    front = {'thickness': 200., 'water_depth': 150., 'width': 800.}
    melt = cal.law_from_calibration(ds, 3., 'melt_calving', front=front)
    assert isinstance(melt, MeltPlusCalving) and 0 < melt.k_c < 3.
    # the total over the period is the constant-k one
    q = np.mean([melt.flux(200., 150., 800., y) for y in yrs])
    q0 = ConstantK(k=3.).flux(200., 150., 800.)
    assert q == pytest.approx(q0, rel=1e-12)
