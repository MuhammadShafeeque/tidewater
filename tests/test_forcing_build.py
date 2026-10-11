"""Product registry, column checks, explicit units and the one-call build, on synthetic products."""
import netCDF4 as nc
import numpy as np
import pandas as pd
import pytest
import xarray as xr

from tidewater.config import InvalidParamsError, InvalidWorkflowError
from tidewater.forcing import bands, build, cli, fill, io, products
from tidewater.forcing.footprint import to_180
from tidewater.physics import band_names

DEPTH = np.array([5., 25., 55., 150., 300., 450.])
TIME = pd.date_range('2000-01-01', '2001-12-01', freq='MS')
BANDS = ['shallow:0:60', 'deep:200:500']


def product(lon0=10., lat0=70., convention='-180..180', kelvin=False, cm=False,
            naming='cmip'):
    """A coast at lon0 + 1 with land east of it, a 100 m shelf, then 1000 m of water."""
    dx = np.arange(-2., 2.01, 0.25)
    lat = lat0 + np.arange(-1., 1.01, 0.25)
    lon = lon0 + dx
    lon = to_180(lon) if convention == '-180..180' else lon % 360.
    floor = np.where(dx > 1., 0., np.where(dx > 0.5, 100., 1000.))
    season = 0.5 * np.sin(2 * np.pi * (TIME.month.values - 1) / 12)
    t = (1.0 + season[:, None, None, None] - 0.002 * DEPTH[None, :, None, None]
         + 0.01 * (lat - lat0)[None, None, :, None] + 0.1 * dx[None, None, None, :])
    dry = DEPTH[:, None, None] > floor[None, None, :]
    t = np.where(dry[None], np.nan, t)
    s = np.where(dry[None], np.nan, 34.0 + 0.001 * DEPTH[None, :, None, None] + 0 * t)
    sic = np.broadcast_to(np.where(np.isin(TIME.month, [1, 2, 3]), 80., 10.)[:, None, None],
                          (TIME.size, lat.size, lon.size))
    z = DEPTH * (100. if cm else 1.)
    t = t + (273.15 if kelvin else 0.)
    tu = 'K' if kelvin else 'degC'
    if naming == 'nemo':
        la, lo = np.meshgrid(lat, lon, indexing='ij')
        return xr.Dataset(
            {'votemper': (('time_counter', 'deptht', 'y', 'x'), t, {'units': tu}),
             'vosaline': (('time_counter', 'deptht', 'y', 'x'), s),
             'siconc': (('time_counter', 'y', 'x'), sic / 100., {'units': '1'})},
            coords={'time_counter': TIME, 'deptht': ('deptht', z, {'units': 'cm' if cm else 'm'}),
                    'nav_lat': (('y', 'x'), la), 'nav_lon': (('y', 'x'), lo)})
    return xr.Dataset(
        {'thetao': (('time', 'lev', 'lat', 'lon'), t, {'units': tu}),
         'so': (('time', 'lev', 'lat', 'lon'), s, {'units': '0.001'}),
         'siconc': (('time', 'lat', 'lon'), sic, {'units': '%'})},
        coords={'time': TIME, 'lev': ('lev', z, {'units': 'cm' if cm else 'm',
                                                 'positive': 'down'}),
                'lat': lat, 'lon': lon})


def glaciers(lon0=10., lat0=70., far=False):
    rows = [{'id': 'G1', 'lon': lon0 + 1.0, 'lat': lat0, 'front_depth': 40.},
            {'id': 'G2', 'lon': lon0 + 1.0, 'lat': lat0 + 0.5, 'front_depth': np.nan}]
    if far:
        rows.append({'id': 'G3', 'lon': lon0 + 40., 'lat': lat0, 'front_depth': 50.})
    return pd.DataFrame(rows)


def test_registry():
    assert {'cmip', 'nemo'} <= set(products.PRODUCTS)
    assert products.detect_product(product()) == 'cmip'
    assert products.detect_product(product(naming='nemo')) == 'nemo'
    try:
        entry = products.register_product('mine', {'thetao': 'T', 'depth': ['z', 'depth']},
                                          temperature_units='K', grid='regular')
        assert entry['names']['thetao'] == ('T',) and entry['names']['depth'] == ('z', 'depth')
        with pytest.raises(InvalidParamsError, match='registered'):
            products.register_product('mine', {'thetao': 'T'})
        products.register_product('mine', {'thetao': 'X'}, overwrite=True)
    finally:
        products.PRODUCTS.pop('mine', None)
    with pytest.raises(InvalidParamsError, match='unknown field'):
        products.register_product('bad', {'temperature': 'T'})
    with pytest.raises(InvalidParamsError, match='grid'):
        products.register_product('bad', {'thetao': 'T'}, grid='hex')
    ds = product().rename(thetao='potemp')
    ds['potemp'].attrs['standard_name'] = 'sea_water_potential_temperature'
    assert products.resolve_names(ds, products.PRODUCTS['cmip']['names'])['thetao'] == 'potemp'
    with pytest.raises(InvalidParamsError, match='unknown product'):
        products.open_product(product(), 'nope')
    with pytest.raises(InvalidParamsError, match='no registered product'):
        products.open_product(xr.Dataset({'a': ('x', [1.])}))


def test_open_product_units_and_namings_agree():
    ref = products.open_product(product())
    assert ref.attrs['product'] == 'cmip' and ref.attrs['grid'] == 'regular'
    np.testing.assert_array_equal(ref['depth'], DEPTH)
    assert float(ref['siconc'].max()) == pytest.approx(0.8)
    k = products.open_product(product(kelvin=True, cm=True))
    np.testing.assert_allclose(k['thetao'], ref['thetao'], atol=1e-12)
    np.testing.assert_array_equal(k['depth'], DEPTH)
    nemo = products.open_product(product(naming='nemo', kelvin=True))
    assert nemo.attrs['product'] == 'nemo' and nemo.attrs['grid'] == 'curvilinear'
    np.testing.assert_allclose(nemo['thetao'], ref['thetao'], atol=1e-12)
    np.testing.assert_allclose(nemo['siconc'], ref['siconc'])
    # an override wins over the entry and over the file's attribute
    absolute = products.open_product(product(), salinity_units='g/kg')
    np.testing.assert_allclose(absolute['so'] * io.SA_PER_SP, ref['so'])
    flat = product()
    flat['thetao'].attrs.pop('units')
    with pytest.raises(InvalidParamsError, match='temperature units'):
        products.open_product(flat)
    assert products.open_product(flat, temperature_units='degC').sizes == ref.sizes
    # several files of one product
    ds = product()
    merged = products.open_product([ds[['thetao', 'so']], ds[['siconc']]])
    xr.testing.assert_allclose(merged.drop_attrs(), ref.drop_attrs())


def test_open_product_on_both_longitude_conventions_across_the_dateline():
    a = products.open_product(product(lon0=180., lat0=-70.))
    b = products.open_product(product(lon0=180., lat0=-70., convention='0..360'))
    assert float(a['longitude'].min()) >= -180. and float(a['longitude'].max()) < 180.
    np.testing.assert_allclose(np.sort(a['longitude']), np.sort(b['longitude']))
    near = products.open_product(product(lon0=180., lat0=-70., convention='0..360'),
                                 around=(-179.5, -70.), radius_km=30.)
    assert (near['longitude'] < 0).any() and (near['longitude'] > 0).any()


def test_check_column():
    ok = bands.check_column(DEPTH, depth_bands=[('shallow', 0., 60.), ('deep', 200., 500.)])
    assert ok and ok.reason == '' and ok.bands == {'shallow': '', 'deep': ''}
    shallow = bands.check_column(DEPTH[:3], depth_bands=[('shallow', 0., 60.),
                                                         ('deep', 200., 500.)])
    assert not shallow and shallow.bands['shallow'] == ''
    assert shallow.reason.startswith('deep: no level in [200, 500] m')
    # a terminus shallower than the first level still forms the floored front band
    assert bands.check_column([30., 60.], depth_bands=[('front', 0., 60.)], terminus_depth=5.)
    assert not bands.check_column([30., 60.], depth_bands=[('terminus', 0., 60.)],
                                  terminus_depth=5.)
    assert bands.check_column(-DEPTH, depth_bands=[('deep', 200., 500.)], depth_positive='up')
    assert bands.check_column(DEPTH * 100, depth_bands=[('deep', 200., 500.)], depth_units='cm')

    col = column()
    assert bands.check_column(col)
    bad = col.copy(deep=True)
    bad['thetao'][4, 4] = np.nan
    verdict = bands.check_column(bad)
    assert not verdict and 'deep: non-finite' in verdict.reason
    assert bands.process_ocean_column(bad, on_unformable='skip') is None
    with pytest.raises(InvalidWorkflowError, match='non-finite'):
        bands.process_ocean_column(bad)
    assert bands.process_ocean_column(col, depth_bands=[('abyss', 2000., 3000.)],
                                      on_unformable='skip') is None
    with pytest.raises(InvalidParamsError, match='on_unformable'):
        bands.process_ocean_column(col, on_unformable='ignore')


def column(y0=1995, y1=2004, lon=170., lat=-72.):
    time = pd.date_range(f'{y0}-01-01', f'{y1}-12-01', freq='MS')
    t = 1.0 + 0.5 * np.sin(2 * np.pi * (time.month.values - 1) / 12)[:, None] - 0.002 * DEPTH
    return xr.Dataset(
        {'thetao': (('time', 'depth'), t),
         'so': (('time', 'depth'), 34.0 + 0.001 * DEPTH[None, :] + 0 * t),
         'siconc': ('time', np.where(np.isin(time.month, [1, 2]), 0.05, 0.9))},
        coords={'time': time, 'depth': DEPTH, 'lon': lon, 'lat': lat},
        attrs={'bands': 'shallow:0:60 deep:200:500'})


def test_units_and_conventions_are_explicit():
    col = column()
    base = bands.process_ocean_column(col)
    gkg = col.assign(so=col['so'] * io.SA_PER_SP)
    np.testing.assert_allclose(
        bands.process_ocean_column(gkg, salinity_units='g/kg')['thermal_forcing'],
        base['thermal_forcing'])
    pct = col.assign(siconc=col['siconc'] * 100.)
    np.testing.assert_allclose(bands.process_ocean_column(pct, siconc_units='%')['siconc'],
                               base['siconc'])
    with pytest.raises(InvalidParamsError, match='salinity units'):
        bands.process_ocean_column(col, salinity_units='ppt')
    up = col.assign_coords(depth=('depth', -DEPTH, {'positive': 'up'}))
    np.testing.assert_allclose(bands.process_ocean_column(up)['thermal_forcing'],
                               base['thermal_forcing'])
    cm = col.assign_coords(depth=DEPTH * 100.)
    np.testing.assert_allclose(
        bands.process_ocean_column(cm, depth_units='cm')['thermal_forcing'],
        base['thermal_forcing'])
    # the column reader takes either longitude convention; the data reader is told which
    east = bands.process_ocean_column(column(lon=350.))
    assert east.attrs['ref_pix_lon'] == -10.
    th, so = col['thetao'].assign_coords(lon=350.), col['so'].assign_coords(lon=350.)
    with pytest.raises(InvalidParamsError, match='longitude 350.0 is outside'):
        bands.process_ocean_data(thetao=th, so=so)
    out = bands.process_ocean_data(thetao=th, so=so, longitude_convention='0..360')
    assert out.attrs['ref_pix_lon'] == -10.
    with pytest.raises(InvalidParamsError, match='latitude'):
        bands.check_coordinates(0., 95.)


def test_time_axis_check():
    bands.check_time_axis(TIME)
    with pytest.raises(InvalidParamsError, match='whole calendar years: it has 23 steps'):
        bands.check_time_axis(TIME[1:])
    with pytest.raises(InvalidParamsError, match='from month 1 to month 11'):
        bands.check_time_axis(TIME[:11])
    col = column()
    th = col['thetao'].isel(time=slice(2, 14))
    with pytest.raises(InvalidParamsError, match='thetao must be monthly'):
        bands.process_ocean_data(thetao=th, so=col['so'].isel(time=slice(2, 14)))


def test_periods_in_any_form():
    col = column(1990, 2010)
    a = bands.process_ocean_column(col, y0=1996, y1=2000)
    b = bands.process_ocean_column(col, years='1996-2000')
    xr.testing.assert_identical(a, b)
    assert a['time.year'].values[[0, -1]].tolist() == [1996, 2000]
    ref = bands.process_ocean_column(column(1995, 2014))
    t = bands.process_ocean_column(col, ref_ocean=ref, year_range=(1995, 2004))
    for period in ('1995-2004', '1995,2004', [1995., 2004.]):
        xr.testing.assert_identical(
            bands.process_ocean_column(col, ref_ocean=ref, period=period), t)
    series = np.arange(24.)
    days = [pd.Timestamp(d) for d in TIME]
    assert fill.ref_mean(series, days, '2001-2001') == fill.ref_mean(series, days, (2001, 2001))


def test_front_depth_from_an_elevation():
    assert bands.front_depth_from_bed(-120.) == 120.
    assert bands.front_depth_from_bed(-120., sea_level=0.5) == 120.5
    assert bands.front_depth_from_bed(3.) == 0.
    table = pd.DataFrame({'gid': ['a', 'b'], 'x': [350., -10.], 'y': [-70., 70.],
                          'depth': [np.nan, 25.], 'bed': [-80., -999.]})
    out = build.read_glaciers(table, 'gid', 'x', 'y', 'depth', bed_column='bed',
                              sea_level=1.)
    assert out['front_depth'].tolist() == [81., 25.]
    with pytest.raises(InvalidParamsError, match='without column'):
        build.read_glaciers(table)
    with pytest.raises(InvalidParamsError, match='latitude'):
        build.read_glaciers(table.assign(y=[-91., 0.]), 'gid', 'x', 'y')


def test_author_and_attrs_on_the_writers(tmp_path):
    ds = bands.process_ocean_column(column())
    io.write_forcing(tmp_path / 'a.nc', ds, glacier_lon=170.5, glacier_lat=-72.)
    io.write_forcing(tmp_path / 'b.nc', ds, author='host', author_info='a host model',
                     attrs={'project': 'test', 'author': 'ignored'})
    with nc.Dataset(tmp_path / 'a.nc') as d:
        assert d.author == 'tidewater' and 'author_info' not in d.ncattrs()
        assert d.ncattrs() == [
            'ref_pix_lon', 'ref_pix_lat', 'ref_pix_dis', 'ocean_source', 'ocean_model',
            'extraction', 'ref_bathymetry_src', 'tf_method', 'tf_lambda1', 'tf_lambda2',
            'tf_lambda3', 'bias_correction_applied', 'yr_0', 'yr_1', 'author']
        assert [v for v in d.variables] == [
            'time', 'band_name', 'band_weighting', 'band_top', 'band_bottom', 'bias_offset',
            'thermal_forcing', 'thetao', 'so', 'siconc', 'open_water_frac']
        assert d['time'].filters()['zlib'] is False and d['thetao'].filters()['zlib']
    with nc.Dataset(tmp_path / 'b.nc') as d:
        assert (d.author, d.author_info, d.project) == ('host', 'a host model', 'test')
    # a file read back keeps its author
    io.write_forcing(tmp_path / 'c.nc', io.read_forcing(tmp_path / 'b.nc'))
    with nc.Dataset(tmp_path / 'c.nc') as d:
        assert (d.author, d.author_info, d.project) == ('host', 'a host model', 'test')
    f = tmp_path / 'f.nc'
    io.write_ocean_file(f, TIME.values, ['x'], [0.], [60.], ['uniform'], np.ones((24, 1)),
                        np.ones((24, 1)), np.ones((24, 1)), None, None, author='me')
    fill.append_band(f, 'y', {v: np.ones(24) for v in io.BAND_VARS}, 0., 9., 'uniform', 0.,
                     'note', attrs={'author': 'filler'})
    with nc.Dataset(f) as d:
        assert d.author == 'filler' and d.filled_bands == 'y'


@pytest.mark.parametrize('lon0, lat0, convention', [
    (10., 70., '-180..180'), (-60., -65., '-180..180'),
    (179.5, 72., '0..360'), (180., -70., '-180..180'), (300., -65., '0..360')])
def test_build_forcing_any_region(tmp_path, lon0, lat0, convention):
    ds = product(lon0, lat0, convention)
    forcing, report = build.build_forcing(ds, glaciers(lon0, lat0, far=True), tmp_path,
                                          depth_bands=BANDS)
    assert report.loc[['G1', 'G2'], 'status'].tolist() == ['written'] * 2
    assert report.at['G3', 'status'] == 'skipped'
    assert report.at['G3', 'reason'].startswith('no cell within 113.9')
    assert sorted(forcing) == ['G1', 'G2']
    out = io.read_forcing(tmp_path / 'G1.nc')
    assert band_names(out) == ['shallow', 'deep']
    assert out.attrs['ref_bathymetry_m'] == 40. and out.attrs['author'] == 'tidewater'
    assert -180. <= out.attrs['ref_pix_lon'] < 180.
    assert out.attrs['ref_pix_dis'] < 60e3
    assert 'siconc' in out and float(out['siconc'].max()) == pytest.approx(0.8)
    assert io.validate(tmp_path / 'G1.nc', bands=['shallow', 'deep'])['problem'] == ''
    # the same product on the other longitude convention gives the same numbers
    other = '0..360' if convention == '-180..180' else '-180..180'
    again, _ = build.build_forcing(product(lon0, lat0, other), glaciers(lon0, lat0),
                                   depth_bands=BANDS)
    np.testing.assert_allclose(again['G1']['thermal_forcing'], forcing['G1']['thermal_forcing'])


def test_build_forcing_options(tmp_path):
    ds = product(kelvin=True, cm=True)
    with pytest.raises(InvalidWorkflowError, match='G3: no cell within'):
        build.build_forcing(ds, glaciers(far=True), on_unformable='raise')
    # a 12 km search reaches only the shelf: the deep band is left for a fill
    forcing, report = build.build_forcing(ds, glaciers(), depth_bands=BANDS, radii=[12.],
                                          fallback='drop')
    assert report['bands'].tolist() == ['shallow'] * 2
    # bands no level of the column reaches
    _, report = build.build_forcing(ds, glaciers(), depth_bands=['abyss:2000:3000'])
    assert report.at['G1', 'reason'] == 'no level survived the finiteness trim'
    _, report = build.build_forcing(
        ds, glaciers(), depth_bands=BANDS,
        process_options={'depth_bands': [('abyss', 2000., 3000.)]})
    assert report['status'].tolist() == ['skipped'] * 2
    assert report.at['G1', 'reason'].startswith('abyss: no level in [2000, 3000] m')
    forcing, report = build.build_forcing(ds, glaciers(), depth_bands=['shallow:0:60'],
                                          radii=[20.], years='2001-2001')
    assert report['status'].tolist() == ['built'] * 2
    assert forcing['G1'].sizes['time'] == 12 and np.isnan(report.at['G2', 'front_depth'])
    nemo = product(naming='nemo')
    with pytest.raises(InvalidParamsError, match='cell area'):
        build.build_forcing(nemo, glaciers())
    a, _ = build.build_forcing(nemo, glaciers(), depth_bands=[('shallow', 0, 60)],
                               cell_area_km2=100.)
    b, _ = build.build_forcing(product(), glaciers(), depth_bands=[('shallow', 0, 60)],
                               cell_area_km2=100.)
    np.testing.assert_allclose(a['G2']['thermal_forcing'], b['G2']['thermal_forcing'])


def test_command_line(tmp_path, capsys):
    prod = tmp_path / 'prod.nc'
    product(-60., -65., '0..360', kelvin=True).to_netcdf(prod)
    table = tmp_path / 'glaciers.csv'
    glaciers(300., -65., far=True).rename(columns={'id': 'rgi'}).to_csv(table, index=False)
    out = tmp_path / 'out'
    status = cli.main(['build', str(prod), '--glaciers', str(table), '--id-column', 'rgi',
                       '--out', str(out), '--bands', *BANDS, '--years', '2000-2001',
                       '--pattern', 'ocean_{id}.nc', '--author', 'host',
                       '--attr', 'project=demo', '--report', str(tmp_path / 'r.csv')])
    assert status == 0
    text = capsys.readouterr().out
    assert '2 of 3 glacier(s) written' in text and 'G3: no cell within' in text
    assert sorted(p.name for p in out.iterdir()) == ['ocean_G1.nc', 'ocean_G2.nc']
    with nc.Dataset(out / 'ocean_G1.nc') as d:
        assert (d.author, d.project, d.ocean_source) == ('host', 'demo', 'prod.nc')
    assert pd.read_csv(tmp_path / 'r.csv', index_col='id').at['G1', 'status'] == 'written'

    assert cli.main(['validate', str(out), '--bands', 'shallow', 'deep']) == 0
    assert '2 of 2 file(s) sound' in capsys.readouterr().out
    assert cli.main(['validate', str(out), '--bands', 'mid']) == 1
    assert 'band(s) absent: mid' in capsys.readouterr().out
    assert cli.main(['build', str(prod), '--glaciers', str(table), '--out', str(out)]) == 2
    assert 'without column' in capsys.readouterr().err
