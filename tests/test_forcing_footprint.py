"""The wet-cell footprint search on synthetic grids."""
import numpy as np
import pandas as pd
import pytest
import xarray as xr

from tidewater.config import InvalidParamsError, InvalidWorkflowError
from tidewater.forcing import footprint as fp
from tidewater.forcing.io import normalise

DEPTH = np.array([5., 25., 55., 150., 300., 450., 600., 800.])
BANDS = [('shallow', 0., 60.), ('deep', 200., 500.)]
TIME = pd.date_range('2000-01-01', '2001-12-01', freq='MS')


def field(lat, lon, bathy):
    """(time, level, cell) temperature in degC and salinity, NaN below the sea floor."""
    season = 0.5 * np.sin(2 * np.pi * (TIME.month.values - 1) / 12)
    t = (1.0 + season[:, None, None] - 0.002 * DEPTH[None, :, None]
         + 0.05 * np.asarray(lat)[None, None, :] + 0.01 * np.asarray(lon)[None, None, :])
    s = 34.0 + 0.001 * DEPTH[None, :, None] + 0 * t
    dry = DEPTH[:, None] > np.asarray(bathy)[None, :]
    return np.where(dry[None], np.nan, t), np.where(dry[None], np.nan, s)


def regular_south():
    """0.25 degree grid west of a north-south coast at 62 W, 65 S; longitudes on 0-360, in K."""
    lat = np.arange(-67., -62.99, 0.25)
    lon = np.arange(294., 302.01, 0.25)
    la, lo = np.meshgrid(lat, lon, indexing='ij')
    lo180 = fp.to_180(lo)
    bathy = np.where(lo180 > -62.0, 0., np.where(lo180 >= -62.5, 100., 1000.))
    t, s = field(la.ravel(), lo180.ravel(), bathy.ravel())
    shape = (TIME.size, DEPTH.size, lat.size, lon.size)
    ds = xr.Dataset(
        {'votemper': (('t', 'z', 'y', 'x'), t.reshape(shape) + 273.15, {'units': 'K'}),
         'vosaline': (('t', 'z', 'y', 'x'), s.reshape(shape))},
        coords={'t': TIME, 'deptht': ('z', DEPTH, {'units': 'm'}), 'nav_lat': ('y', lat),
                'nav_lon': ('x', lon)})
    names = {'thetao': 'votemper', 'so': 'vosaline', 'depth': 'deptht', 'time': 't',
             'latitude': 'nav_lat', 'longitude': 'nav_lon'}
    return normalise(ds, names=names)


def unstructured_dateline(seed=0):
    """Scattered cells from 176 E to 176 W at 70-72 N, all 500 m deep."""
    rng = np.random.default_rng(seed)
    lon = fp.to_180(rng.uniform(176., 184., 400))
    lat = rng.uniform(70., 72., 400)
    t, s = field(lat, np.zeros_like(lon), np.full(lon.size, 500.))
    return xr.Dataset(
        {'thetao': (('time', 'level', 'cell'), t, {'units': 'degC'}),
         'so': (('time', 'level', 'cell'), s)},
        coords={'time': TIME, 'depth': ('level', DEPTH), 'latitude': ('cell', lat),
                'longitude': ('cell', lon)})


def test_distances_are_dateline_and_pole_safe():
    assert fp.haversine_km(179.9, 0., -179.9, 0.) == pytest.approx(22.24, abs=0.01)
    assert fp.haversine_km(359.9, 0., 0.1, 0.) == pytest.approx(22.24, abs=0.01)
    assert fp.haversine_km(0., 89., 180., 89.) == pytest.approx(222.39, abs=0.01)
    assert fp.haversine_km(10., -90., 200., -89.) == pytest.approx(111.19, abs=0.01)
    lon, lat = fp.centroid([179., -179.], [70., 70.])
    assert abs(abs(lon) - 180.) < 1e-9 and lat > 70.
    assert fp.centroid([0., 90., 180., 270.], [89.] * 4)[1] == pytest.approx(90.)


def test_radius_schedule():
    np.testing.assert_allclose(fp.radius_schedule(),
                               [15., 22.5, 33.75, 50.625, 75.9375, 113.90625])
    assert fp.radius_schedule(max_radius=200.)[-2:] == [170.859375, 200.]
    assert fp.radius_schedule(10., 40., 2.) == [10., 20., 40.]


def test_overlapping_bands_share_levels_by_width():
    owner = fp.assign_levels(DEPTH, [('wide', 0., 700.), ('shallow', 0., 60.)])
    assert [owner[i] for i in range(7)] == ['shallow'] * 3 + ['wide'] * 4 and 7 not in owner


def test_regular_grid_southern_hemisphere():
    ds = regular_south()
    assert ds['thetao'].dims == ('time', 'level', 'cell')
    assert float(ds['longitude'].max()) < -62.0 + 1e-9  # land dropped, longitudes on +-180
    np.testing.assert_allclose(ds['cell_area_km2'].sel(cell=ds.latitude == -65.).values,
                               (6371. * np.radians(.25)) ** 2 * np.cos(np.radians(65.)),
                               rtol=1e-4)
    anchor = (-61.9, -65.0)
    col = fp.reduce_site(ds, None, anchor, None, 'nearest', bands=BANDS)

    # levels in no band are not handed on
    np.testing.assert_array_equal(col['depth'], [5., 25., 55., 300., 450.])
    assert float(col['lat']) == pytest.approx(-65.0) and float(col['lon']) == pytest.approx(-62.0)
    assert col.attrs['bands'] == 'shallow:0:60 deep:200:500'
    assert col.attrs['dropped_bands'] == 'none'

    lon, lat = ds['longitude'].values, ds['latitude'].values
    dist = fp.haversine_km(*anchor, lon, lat)
    deep_ok = np.isfinite(ds['thetao'].isel(time=0, level=[4, 5])).all('level').values
    flon, flat = fp.footprint_cells(col, 'deep')
    assert (flon <= -62.75).all()  # the shelf is 100 m deep: deep water starts farther out
    radius = [r for r in fp.radius_schedule() if (deep_ok & (dist <= r)).any()][0]
    expect = np.where(deep_ok & (dist <= radius))[0]
    expect = expect[np.argsort(dist[expect])][:3]  # 1000 km2 over ~326 km2 cells
    assert sorted(zip(flon, flat)) == sorted(zip(lon[expect], lat[expect]))
    np.testing.assert_allclose(
        col['thetao'].sel(depth=[300., 450.]),
        ds['thetao'].isel(level=[4, 5], cell=expect).mean('cell'))
    slon, slat = fp.footprint_cells(col, 'shallow')
    assert (slon == -62.0).all() and slat.tolist() == [-65.0]


def test_coverage_rule_and_fallback():
    ds = regular_south()
    anchor = (-61.9, -65.0)
    # partial: the 100 m shelf cells hold no deep level at all, so the answer is unchanged
    part = fp.reduce_site(ds, None, anchor, None, 'nearest', bands=BANDS,
                          coverage={'deep': 'partial'})
    assert (fp.footprint_cells(part, 'deep')[0] <= -62.75).all()
    # one radius of 20 km reaches no deep water
    col = fp.reduce_site(ds, None, anchor, None, 'nearest', bands=BANDS, radii=[20.])
    assert col.attrs['band_footprint_deep'].startswith('shelf fallback')
    assert fp.footprint_cells(col, 'deep')[0].size == int(
        np.isfinite(ds['thetao'].isel(time=0, level=[4, 5])).all('level').sum())
    col = fp.reduce_site(ds, None, anchor, None, 'nearest', bands=BANDS, radii=[20.],
                         fallback='drop')
    assert col.attrs['bands'] == 'shallow:0:60'
    assert 'deep: no wet cell in reach' in col.attrs['dropped_bands']
    with pytest.raises(InvalidWorkflowError, match='finiteness'):
        fp.reduce_site(ds, None, (-40., -65.), None, 'nearest', bands=BANDS, fallback='drop')


def test_column_scheme_uses_one_footprint():
    ds = regular_south()
    # from 40 km offshore the nearest column is deep, and serves both bands
    col = fp.reduce_site(ds, None, (-63.0, -65.0), None, 'column', bands=BANDS)
    assert col.attrs['footprint_cells_shallow'] == col.attrs['footprint_cells_deep']
    # at the coast the column is the 100 m shelf, which has no deep level
    col = fp.reduce_site(ds, None, (-61.9, -65.0), None, 'column', bands=BANDS,
                         fallback='drop')
    assert col.attrs['bands'] == 'shallow:0:60'


def test_shelf_scheme_radius():
    ds = regular_south()
    col = fp.reduce_site(ds, None, (-61.9, -65.0), None, 'shelf', bands=BANDS,
                         shelf_radius_km=60.)
    lon, lat = fp.footprint_cells(col, 'shallow')
    d = fp.haversine_km(-61.9, -65.0, lon, lat)
    n = int((fp.haversine_km(-61.9, -65.0, ds.longitude.values, ds.latitude.values) <= 60.).sum())
    assert d.max() <= 60. and lon.size == n


def test_terminus_depth_sets_the_front_band():
    ds = regular_south()
    col = fp.reduce_site(ds, None, (-61.9, -65.0), 30., 'nearest',
                         bands=[('front', 0., 60.)])
    assert col.attrs['bands'] == 'front:0:30' and col.attrs['terminus_depth_m'] == 30.
    np.testing.assert_array_equal(col['depth'], [5., 25.])
    col = fp.reduce_site(ds, None, (-61.9, -65.0), 2., 'nearest', bands=[('front', 0., 60.)])
    assert col.attrs['bands'] == 'front:0:5'  # floored at the first level


def test_unstructured_cells_across_the_dateline():
    ds = unstructured_dateline()
    anchor = (179.95, 71.0)
    col = fp.reduce_site(ds, None, anchor, None, 'nearest', bands=BANDS,
                         cell_area_km2=100., r0_km=40.)
    assert col.attrs['n_cells'] == 10 and col.attrs['footprint_area_km2'] == 1000.
    lon, lat = fp.footprint_cells(col, 'shallow')
    assert (lon > 0).any() and (lon < 0).any()  # both sides of the dateline
    d = fp.haversine_km(*anchor, lon, lat)
    assert d.max() <= 40.
    all_d = np.sort(fp.haversine_km(*anchor, ds.longitude.values, ds.latitude.values))
    np.testing.assert_allclose(np.sort(d), all_d[:10], atol=0.05)
    assert abs(abs(float(col['lon'])) - 180.) < 1.0
    # the same search with the anchor written on 0-360
    again = fp.reduce_site(ds, None, (-180.05 + 360., 71.0), None, 'nearest', bands=BANDS,
                           cell_area_km2=100., r0_km=40.)
    assert again.attrs['footprint_cells_shallow'] == col.attrs['footprint_cells_shallow']


def test_cell_area_is_required():
    with pytest.raises(InvalidParamsError, match='cell area'):
        fp.reduce_site(unstructured_dateline(), None, (179.95, 71.0), None, 'nearest',
                       bands=BANDS)


def test_incomplete_years_are_trimmed():
    ds = unstructured_dateline().isel(time=slice(3, None))
    col = fp.reduce_site(ds, None, (179.95, 71.0), None, 'nearest', bands=BANDS,
                         cell_area_km2=100., r0_km=40.)
    assert col.sizes['time'] == 12 and int(col['time.year'][0]) == 2001


def test_sea_ice_over_the_footprint():
    ds = unstructured_dateline()
    sic = np.full((TIME.size, ds.sizes['cell']), 0.8)
    sic[5] = np.nan
    ds2 = xr.Dataset({'siconc': (('time', 'cell'), sic)},
                     coords={'time': TIME, 'latitude': ds.latitude, 'longitude': ds.longitude})
    kw = dict(bands=BANDS, cell_area_km2=100., r0_km=40.)
    col = fp.reduce_site(ds, ds2, (179.95, 71.0), None, 'nearest', **kw)
    assert 'siconc' not in col and 'siconc: dropped' in col.attrs['extraction']
    col = fp.reduce_site(ds, ds2, (179.95, 71.0), None, 'nearest', siconc_max_gap=1, **kw)
    np.testing.assert_allclose(col['siconc'], 0.8)
    assert col['siconc'].attrs['months_interpolated'] == 1
    col = fp.reduce_site(ds, ds2, (179.95, 71.0), None, 'nearest',
                         siconc_fill_is_open_water=True, **kw)
    assert col['siconc'].values[5] == 0. and col['siconc'].values[4] == pytest.approx(0.8)
