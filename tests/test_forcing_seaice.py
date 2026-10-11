"""Sea ice around a terminus, its gaps, and observed open water in a forcing dataset."""
import numpy as np
import pandas as pd
import pytest
import xarray as xr

from tidewater.config import InvalidParamsError, InvalidWorkflowError
from tidewater.forcing import seaice
from tidewater.physics import open_water_fraction

GID = 'G-00001'


def forcing(y0, y1, siconc=True):
    t = pd.date_range(f'{y0}-01-01', f'{y1}-12-01', freq='MS')
    ds = xr.Dataset({'thetao': (('time', 'depth'), np.zeros((t.size, 2)))},
                    coords={'time': t, 'depth': [10., 50.]})
    if siconc:
        ds['siconc'] = ('time', np.full(t.size, 0.9))
    return ds


def observed(tmp_path, y0, y1):
    t = pd.date_range(f'{y0}-01-01', f'{y1}-12-01', freq='MS')
    ow = np.where(t.month.isin([7, 8, 9]), 0.95, 0.02)
    p = tmp_path / 'ow.csv'
    pd.DataFrame({'glacier_id': GID, 'time': t.strftime('%Y-%m'),
                  'open_water_frac': ow}).to_csv(p, index=False)
    return seaice.read_open_water(p)[GID], ow


@pytest.mark.parametrize('has_siconc', [True, False])
def test_siconc_replaced_by_one_minus_open_water(tmp_path, has_siconc):
    ds = forcing(2000, 2001, has_siconc)
    ow, expected = observed(tmp_path, 1999, 2002)
    out = seaice.with_observed_siconc(ds, ow, 'ow.csv')
    sic = out['siconc'].values
    assert out.attrs['siconc_source'] == 'ow.csv' and out.sizes['time'] == 24
    assert 'siconc_source' not in ds.attrs  # the input is left alone
    np.testing.assert_allclose(sic, 1 - expected[12:36])
    # the threshold transform then applies: open in summer, closed otherwise
    owf = open_water_fraction(sic, threshold=0.15)
    assert owf.max() > 0.5 and (owf[sic > 0.15] == 0).all()


def test_open_water_frac_follows_the_new_siconc(tmp_path):
    ds = forcing(2000, 2001)
    ds['open_water_frac'] = ('time', np.zeros(24))
    ow, _ = observed(tmp_path, 2000, 2001)
    with pytest.raises(InvalidParamsError, match='open_water_threshold'):
        seaice.with_observed_siconc(ds, ow)
    out = seaice.with_observed_siconc(ds, ow, open_water_threshold=0.15)
    np.testing.assert_allclose(out['open_water_frac'],
                               open_water_fraction(out['siconc'].values, threshold=0.15))


def test_a_month_without_observation_refuses(tmp_path):
    ow, _ = observed(tmp_path, 2000, 2000)
    with pytest.raises(InvalidWorkflowError, match='12 of 24 months'):
        seaice.with_observed_siconc(forcing(2000, 2001), ow)


def test_mean_concentration_around_a_terminus():
    # three cells on the dateline at 75 S, one of them land, one far away
    lon = np.array([179.8, -179.8, 180.0, 170.0])
    lat = np.array([-75.0, -75.0, -75.05, -75.0])
    sic = np.array([[0.2, 0.6, np.nan, 1.0], [0.4, np.nan, np.nan, 1.0]])
    series, n, near = seaice.mean_siconc(sic, lon, lat, (180.0, -75.0), 10.)
    np.testing.assert_allclose(series, [0.4, 0.4])
    assert n == 2 and near == pytest.approx(5.76, abs=0.01)
    series, n, _ = seaice.mean_siconc(sic, lon, lat, (180.0, -75.0), 10.,
                                      fill_is_open_water=True)
    np.testing.assert_allclose(series, [0.4, 0.2])  # the missing month of a sea cell is ice free
    series, n, near = seaice.mean_siconc(sic, lon, lat, (0.0, 0.0), 10.)
    assert n == 0 and np.isnan(series).all() and np.isnan(near)


def test_short_gaps_only():
    v = np.array([0., np.nan, 2., np.nan, np.nan, 5., np.nan])
    filled, n = seaice.fill_short_gaps(v, 1)
    assert n == 1 and filled[1] == 1. and np.isnan(filled[[3, 4, 6]]).all()
    filled, n = seaice.fill_short_gaps(v, 2)
    np.testing.assert_allclose(filled[:6], [0., 1., 2., 3., 4., 5.])
    assert n == 3 and np.isnan(filled[6])  # an open end is never extrapolated
    assert seaice.fill_short_gaps(v, 0)[1] == 0


def test_units_and_monthly_means():
    np.testing.assert_allclose(seaice.as_fraction([50., 100.], '%'), [0.5, 1.])
    np.testing.assert_allclose(seaice.as_fraction([50., 100.], 'auto'), [0.5, 1.])
    np.testing.assert_allclose(seaice.as_fraction([0.5, 1.], 'auto'), [0.5, 1.])
    with pytest.raises(InvalidParamsError):
        seaice.as_fraction([0.5], None)
    days = pd.date_range('2001-01-01', '2001-02-28', freq='D')
    vals = np.where(days.month == 1, 0.2, 0.6)[:, None] * np.ones((1, 3))
    t, m = seaice.monthly_mean(vals, days)
    assert len(t) == 2 and m.shape == (2, 3)
    np.testing.assert_allclose(m[:, 0], [0.2, 0.6])
    t, m = seaice.monthly_mean(vals[:2], pd.to_datetime(['2001-01-01', '2001-02-01']))
    assert m.shape == (2, 3)


def test_gaps_filled_from_climatology_and_a_second_record():
    t = pd.date_range('2000-01-01', '2002-12-01', freq='MS')
    truth = pd.Series(np.where(t.month.isin([7, 8, 9]), 0.8, 0.1), index=t)
    main = truth.copy()
    main['2001-08-01'] = np.nan
    main['2002-02-01'] = np.nan
    main[t.month == 5] = np.nan
    aux = truth * 0.5
    aux['2001-08-01'] = 0.5  # 0.1 above its own August mean
    aux = aux.drop(pd.Timestamp('2002-02-01'))
    filled, how = seaice.fill_gaps_climatology(main, aux)
    assert filled['2001-08-01'] == pytest.approx(0.9) and how['2001-08-01'] == 'clim_anomaly'
    assert filled['2002-02-01'] == pytest.approx(0.1) and how['2002-02-01'] == 'clim'
    assert np.isnan(filled['2000-05-01']) and how['2000-05-01'] == 'none'
    assert (how[main.notna()] == 'observed').all()
    alone, how = seaice.fill_gaps_climatology(main)
    assert alone['2001-08-01'] == pytest.approx(0.8) and how['2001-08-01'] == 'clim'
