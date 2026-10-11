"""The forcing file, the band means that fill it, its check, and product normalisation."""
import netCDF4 as nc
import numpy as np
import pandas as pd
import pytest
import xarray as xr

from tidewater.config import DEFAULTS, InvalidParamsError, InvalidWorkflowError
from tidewater.forcing import bands, bias, discharge, io
from tidewater.physics import band_names, freezing_point, open_water_fraction

DEPTH = np.array([5., 25., 55., 150., 300., 450.])


def column(y0=1995, y1=2014, kelvin=False, warm=0.0, siconc=True):
    time = pd.date_range(f'{y0}-01-01', f'{y1}-12-01', freq='MS')
    season = np.sin(2 * np.pi * (time.month.values - 1) / 12)
    t = 1.0 + warm + 0.5 * season[:, None] - 0.002 * DEPTH[None, :]
    ds = xr.Dataset(
        {'thetao': (('time', 'depth'), t + (273.15 if kelvin else 0.)),
         'so': (('time', 'depth'), 34.0 + 0.001 * DEPTH[None, :] + 0 * t)},
        coords={'time': time, 'depth': DEPTH, 'lon': 170.0, 'lat': -72.0},
        attrs={'bands': 'shallow:0:60 deep:200:500', 'n_cells': 3, 'search_radius_km': 50.6,
               'extraction': 'nearest footprint', 'source': 'synthetic', 'model': 'M'})
    if siconc:
        ds['siconc'] = ('time', np.where(np.isin(time.month, [1, 2]), 0.05, 0.9))
    return ds


def test_band_means_and_file_round_trip(tmp_path):
    col = column(kelvin=True)
    ds = bands.process_ocean_column(col, terminus_depth=80., terminus_depth_source='survey')
    assert band_names(ds) == ['shallow', 'deep']
    t = col['thetao'].values - 273.15
    tf = t - freezing_point(col['so'].values, DEPTH)
    np.testing.assert_allclose(ds['thermal_forcing'][:, 0], tf[:, :3].mean(axis=1))
    np.testing.assert_allclose(ds['thermal_forcing'][:, 1], tf[:, 4:].mean(axis=1))
    np.testing.assert_allclose(ds['thetao'][:, 0], t[:, :3].mean(axis=1))
    np.testing.assert_allclose(ds['open_water_frac'],
                               open_water_fraction(col['siconc'].values, threshold=0.15))
    assert ds['bias_offset'].values.tolist() == [0., 0.]

    f = tmp_path / 'ocean_data.nc'
    io.write_forcing(f, ds, glacier_lon=170.5, glacier_lat=-72.0)
    with nc.Dataset(f) as d:
        assert d.dimensions['time'].isunlimited() and len(d.dimensions['band']) == 2
        assert set(d.variables) == {'time', 'band_name', 'band_weighting', 'band_top',
                                    'band_bottom', 'bias_offset', 'thermal_forcing', 'thetao',
                                    'so', 'siconc', 'open_water_frac'}
        assert d['thermal_forcing'].dtype == 'f4' and d['time'].calendar == 'standard'
        assert d.ref_bathymetry_m == 80. and d.ref_bathymetry_src == 'survey'
        assert d.ocean_source == 'synthetic' and d.ocean_model == 'M' and d.n_cells == 3
        assert d.yr_0 == 1995 and d.yr_1 == 2014 and d.bias_correction_applied == 'False'
        assert d.ref_pix_dis == pytest.approx(17180, rel=0.01)  # m
    back = io.read_forcing(f)
    assert band_names(back) == ['shallow', 'deep']
    assert band_names(xr.open_dataset(f)) == ['shallow', 'deep']  # as a host reads it
    np.testing.assert_allclose(back['thermal_forcing'], ds['thermal_forcing'], atol=1e-6)
    assert back['time.year'].values[[0, -1]].tolist() == [1995, 2014]
    assert io.validate(f, bands=['shallow', 'deep']) == {
        'problem': '', 'y0': 1995, 'y1': 2014, 'filled': ''}

    # a file read back writes the same file again
    g = tmp_path / 'again.nc'
    io.write_forcing(g, back)
    again = io.read_forcing(g)
    xr.testing.assert_allclose(again, back)
    assert again.attrs['ref_pix_dis'] == back.attrs['ref_pix_dis']


def test_default_bands_and_front_depth():
    ds = bands.process_ocean_column(column(), terminus_depth=30.)
    assert band_names(ds) == ['shallow', 'deep']  # the column's own declaration wins
    col = column()
    del col.attrs['bands']
    ds = bands.process_ocean_column(col, terminus_depth=30.)
    assert band_names(ds) == [s.split(':')[0] for s in DEFAULTS.ocean_depth_bands]
    assert float(ds['band_bottom'][-1]) == 30.  # front follows the terminus depth


def test_contracts_of_the_column():
    col = column()
    with pytest.raises(InvalidParamsError, match='thetao and so'):
        bands.process_ocean_data()
    with pytest.raises(InvalidWorkflowError, match='complete calendar year'):
        bands.process_ocean_column(col.isel(time=slice(1, 10)))
    assert bands.process_ocean_column(col.isel(time=slice(3, None))).sizes['time'] == 228
    bad = col.copy(deep=True)
    bad['thetao'][3, 1] = np.nan
    with pytest.raises(InvalidWorkflowError, match='non-finite'):
        bands.process_ocean_column(bad)
    with pytest.raises(InvalidWorkflowError, match='no ocean levels'):
        bands.process_ocean_column(col, depth_bands=[('abyss', 2000., 3000.)])
    with pytest.raises(InvalidParamsError, match='built here'):
        bands.process_ocean_column(col, siconc=col['siconc'])
    k = bands.process_ocean_column(column(kelvin=True), temperature_units='K')
    c = bands.process_ocean_column(col, temperature_units='degC')
    np.testing.assert_allclose(k['thermal_forcing'], c['thermal_forcing'])


def test_bias_offset_onto_a_reference():
    ref = bands.process_ocean_column(column(warm=0.7))
    raw = bands.process_ocean_column(column(1990, 2020), ref_ocean=ref, year_range=(1995, 2014))
    np.testing.assert_allclose(raw['bias_offset'], 0.7, atol=1e-9)
    assert raw.attrs['bias_correction_applied'] == 'False'
    cor = bands.process_ocean_column(column(1990, 2020), ref_ocean=ref,
                                     apply_bias_correction=True, year_range=(1995, 2014))
    np.testing.assert_allclose(cor['thermal_forcing'] - raw['thermal_forcing'], 0.7)
    np.testing.assert_allclose(
        cor['thermal_forcing'].sel(time=slice('1995', '2014')).mean('time'),
        ref['thermal_forcing'].mean('time'))
    # the same through the dataset route, with the bands in another order
    flipped = ref.isel(band=[1, 0])
    again = bias.bias_correct(raw, flipped, (1995, 2014))
    np.testing.assert_allclose(again['thermal_forcing'], cor['thermal_forcing'])
    with pytest.raises(InvalidWorkflowError, match='already'):
        bias.bias_correct(again, ref, (1995, 2014))
    with pytest.raises(InvalidWorkflowError, match='no data'):
        bias.bias_correct(raw, ref, (1900, 1910))
    with pytest.raises(InvalidParamsError, match='no band'):
        bias.bias_correct(raw, ref.isel(band=[0]), (1995, 2014))


def test_front_band_is_floored_at_the_first_level():
    assert bands.front_band([5., 15., 25.], 40.) == (40., False)
    assert bands.front_band([5., 15., 25.], 3.) == (10., True)    # base of the first level
    assert bands.front_band([25., 5., 15.], 10.) == (10., False)
    assert bands.front_band([8.], 2.) == (8., True)               # a single level stands for itself


def write(path, y0=2000, y1=2001, names=('shallow', 'deep'), drop=None, nan=False):
    time = pd.date_range(f'{y0}-01-01', f'{y1}-12-01', freq='MS')
    if drop is not None:
        time = time.delete(drop)
    tf = np.ones((time.size, len(names)))
    if nan:
        tf[3, 0] = np.nan
    io.write_ocean_file(path, time.values, names, [0.] * len(names), [60.] * len(names),
                        ['uniform'] * len(names), tf, tf, 34 * tf, None, None)
    return path


def test_validate(tmp_path):
    assert io.validate(tmp_path / 'none.nc')['problem'] == 'missing'
    ok = write(tmp_path / 'ok.nc')
    assert io.validate(ok, bands=['shallow', 'deep'])['problem'] == ''
    assert io.validate(ok, bands=['shallow', 'mid', 'far'])['problem'] == 'band(s) absent: mid far'
    assert io.validate(write(tmp_path / 'gap.nc', drop=13))['problem'] == \
        'months not whole and consecutive'
    assert io.validate(write(tmp_path / 'short.nc', drop=23))['problem'] == \
        'months not whole and consecutive'
    assert io.validate(write(tmp_path / 'nan.nc', nan=True))['problem'] == \
        'thermal_forcing not finite'
    junk = tmp_path / 'junk.nc'
    junk.write_text('not netcdf')
    assert io.validate(junk)['problem'].startswith('unreadable')


def test_subglacial_discharge_onto_the_file(tmp_path):
    f = write(tmp_path / 'q.nc')
    area = np.array([1e6, 2e6])
    runoff = discharge.runoff_upper_bound(5.0, np.array([10., -1.]), [30., 30.], [10., 30.])
    np.testing.assert_allclose(runoff, [70., -5.])
    q1 = discharge.discharge_from_runoff(runoff, area, 2.592e6)
    assert q1 == pytest.approx(70. * 1e6 / 1000. / 2.592e6)  # the negative point adds nothing
    years = discharge.mid_month_years(np.repeat([2000], 12), np.arange(1, 13))
    q = np.linspace(0., 11., 12)
    mean = discharge.write_subglacial_discharge(f, years, q)
    ds = io.read_forcing(f)
    assert ds['subglacial_discharge'].attrs['units'] == 'm3 s-1'
    np.testing.assert_allclose(ds['subglacial_discharge'][:12], q, atol=1e-5)
    assert np.isnan(ds['subglacial_discharge'][12:]).all() and mean == pytest.approx(5.5)
    np.testing.assert_allclose(
        discharge.add_subglacial_discharge(ds.drop_vars('subglacial_discharge'), years, q)
        ['subglacial_discharge'], ds['subglacial_discharge'], atol=1e-5)
    assert io.validate(f, variables=('thermal_forcing', 'subglacial_discharge'))['problem'] \
        == 'subglacial_discharge not finite'


def product(kind):
    """One field on a regular, a curvilinear and an unstructured grid."""
    time = pd.date_range('2000-01-01', '2000-12-01', freq='MS')
    lat, lon = np.arange(-70., -67.9, 0.5), np.arange(178., 182.1, 0.5)
    z = np.array([-50., -10., -200.])  # unsorted, positive up
    la, lo = np.meshgrid(lat, lon, indexing='ij')
    t = (2.0 + 0.1 * la[None, None] + 0.01 * z[None, :, None, None]
         + np.zeros((time.size, 1, 1, 1)))
    t[:, :, 0, :] = np.nan  # a row of land
    sic = np.broadcast_to(50.0 + la, (time.size,) + la.shape)
    if kind == 'regular':
        return xr.Dataset({'T': (('time', 'z', 'lat', 'lon'), t, {'units': 'degC'}),
                           'S': (('time', 'z', 'lat', 'lon'), 0 * t + 34.),
                           'ice': (('time', 'lat', 'lon'), sic, {'units': '%'})},
                          coords={'time': time, 'z': ('z', z, {'units': 'm'}),
                                  'lat': lat, 'lon': lon}), la, lo
    if kind == 'curvilinear':
        return xr.Dataset({'T': (('time', 'z', 'j', 'i'), t, {'units': 'degC'}),
                           'S': (('time', 'z', 'j', 'i'), 0 * t + 34.),
                           'ice': (('time', 'j', 'i'), sic, {'units': '%'}),
                           'area': (('j', 'i'), np.full(la.shape, 4e8))},
                          coords={'time': time, 'z': ('z', z, {'units': 'm'}),
                                  'lat': (('j', 'i'), la), 'lon': (('j', 'i'), lo)}), la, lo
    n = la.size
    return xr.Dataset({'T': (('time', 'z', 'n'), t.reshape(time.size, 3, n), {'units': 'degC'}),
                       'S': (('time', 'z', 'n'), 0 * t.reshape(time.size, 3, n) + 34.),
                       'ice': (('time', 'n'), sic.reshape(time.size, n), {'units': '%'})},
                      coords={'time': time, 'z': ('z', z, {'units': 'm'}),
                              'lat': ('n', la.ravel()), 'lon': ('n', lo.ravel())}), la, lo


NAMES = {'thetao': 'T', 'so': 'S', 'siconc': 'ice', 'depth': 'z', 'latitude': 'lat',
         'longitude': 'lon', 'cell_area': 'area'}


@pytest.mark.parametrize('kind', ['regular', 'curvilinear', 'unstructured'])
def test_normalise_three_grid_kinds(kind):
    ds, la, lo = product(kind)
    out = io.normalise(ds, names=NAMES)
    assert out['thetao'].dims == ('time', 'level', 'cell')
    assert out['siconc'].dims == ('time', 'cell')
    np.testing.assert_array_equal(out['depth'], [10., 50., 200.])  # metres down, sorted
    assert out.sizes['cell'] == la.size - la.shape[1]  # the land row is gone
    assert float(out['longitude'].min()) == -180. and float(out['longitude'].max()) == 179.5
    np.testing.assert_allclose(out['thetao'].isel(time=0, level=0),
                               2.0 + 0.1 * out['latitude'] - 0.1)
    np.testing.assert_allclose(out['siconc'].isel(time=0), 0.5 + out['latitude'] / 100)
    assert ('cell_area_km2' in out.coords) == (kind != 'unstructured')
    if kind == 'curvilinear':
        np.testing.assert_allclose(out['cell_area_km2'], 400.)

    near = io.normalise(ds, names=NAMES, around=(-179.9, -69.0), radius_km=30., max_depth=100.)
    assert near.sizes == {'time': 12, 'level': 2, 'cell': 3}
    assert sorted(near['longitude'].values.tolist()) == [-180., -179.5, 179.5]
    np.testing.assert_allclose(near['thetao'].isel(time=0, level=1), 2.0 - 6.9 - 0.5)


def test_normalise_units_are_explicit():
    ds, _, _ = product('regular')
    ds['T'].attrs.pop('units')
    with pytest.raises(InvalidParamsError, match='temperature units'):
        io.normalise(ds, names=NAMES)
    kelvin = ds.assign(T=ds['T'] + 273.15)
    a = io.normalise(kelvin, names=NAMES, temperature_units='K')
    b = io.normalise(kelvin, names=NAMES, temperature_units='auto')
    c = io.normalise(ds, names=NAMES, temperature_units='degC')
    np.testing.assert_allclose(a['thetao'], c['thetao'])
    np.testing.assert_allclose(b['thetao'], c['thetao'])

    cm = ds.assign_coords(z=('z', ds['z'].values * 100., {'units': 'cm'}))
    np.testing.assert_array_equal(
        io.normalise(cm, names=NAMES, temperature_units='degC')['depth'], [10., 50., 200.])
    ds['z'].attrs.pop('units')
    with pytest.raises(InvalidParamsError, match='depth units'):
        io.normalise(ds, names=NAMES, temperature_units='degC')
    ds['ice'].attrs.pop('units')
    with pytest.raises(InvalidParamsError, match='sea ice units'):
        io.normalise(ds, names=NAMES, temperature_units='degC', depth_units='m')
    out = io.normalise(ds, names=NAMES, temperature_units='degC', depth_units='m',
                       siconc_units='auto')
    assert float(out['siconc'].max()) < 1.
    with pytest.raises(InvalidWorkflowError, match='no cell within'):
        io.normalise(ds, names=NAMES, temperature_units='degC', depth_units='m',
                     siconc_units='%', around=(0., 0.), radius_km=100.)
