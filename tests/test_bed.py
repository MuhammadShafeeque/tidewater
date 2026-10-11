import numpy as np
import pytest
import xarray as xr

from tidewater.bed import (BED_MASKS, BedMask, bed_mask_convention,
                           register_bed_mask, sample_bed, terminus_water_depth)
from tidewater.config import InvalidParamsError, InvalidWorkflowError


def _plane(x_name='x', y_name='y', descending_y=True, name='bed'):
    """A bed that is linear in x and y: bilinear sampling is exact on it."""
    x = np.arange(0., 1000., 100.)
    y = np.arange(0., 800., 100.)
    if descending_y:
        y = y[::-1]
    z = 50. - 0.2 * x[None, :] - 0.1 * y[:, None]
    return xr.DataArray(z, coords={y_name: y, x_name: x}, dims=(y_name, x_name),
                        name=name)


def test_the_conventions_are_data():
    bm = BED_MASKS['bedmachine']
    assert bm.water_classes == (0, 3) and bm.ice_classes == (2,)
    assert bm.land_classes == (1,) and bm.has_classes
    assert not BED_MASKS['none'].has_classes
    assert bed_mask_convention(None) is BED_MASKS['none']
    assert bed_mask_convention(' BedMachine ') is bm
    assert bed_mask_convention(bm) is bm
    with pytest.raises(InvalidParamsError, match='register_bed_mask'):
        bed_mask_convention('ibcao')
    try:
        entry = register_bed_mask('Mine', ocean_classes=[10], ice_classes=[20],
                                  note='a test')
        assert isinstance(entry, BedMask) and BED_MASKS['mine'] is entry
        assert entry.water_classes == (10,) and entry.note == 'a test'
    finally:
        del BED_MASKS['mine']


def test_nearest_and_bilinear_on_a_projected_grid():
    bed = _plane()
    x, y = np.array([0., 150., 420., 899.]), np.array([0., 260., 700., 333.])
    exact = 50. - 0.2 * x - 0.1 * y
    np.testing.assert_allclose(sample_bed(bed, x, y, method='bilinear'), exact,
                               rtol=1e-12, atol=1e-12)
    near = sample_bed(bed, x, y)
    xs, ys = np.array([0., 200., 400., 900.]), np.array([0., 300., 700., 300.])
    np.testing.assert_allclose(near, 50. - 0.2 * xs - 0.1 * ys)
    # on a grid point both are the grid value
    assert sample_bed(bed, 300., 400.) == sample_bed(bed, 300., 400.,
                                                     method='bilinear') == \
        float(bed.sel(x=300., y=400.))
    # the shape of the points is kept
    assert sample_bed(bed, np.zeros((2, 3)), np.zeros((2, 3))).shape == (2, 3)
    with pytest.raises(InvalidParamsError, match='method'):
        sample_bed(bed, 0., 0., method='cubic')


def test_the_order_and_names_of_the_grid_do_not_matter():
    up = _plane(descending_y=False)
    down = _plane(descending_y=True)
    x, y = np.array([120., 640.]), np.array([380., 55.])
    for method in ('nearest', 'bilinear'):
        np.testing.assert_allclose(sample_bed(up, x, y, method=method),
                                   sample_bed(down, x, y, method=method))
        np.testing.assert_allclose(sample_bed(down.T, x, y, method=method),
                                   sample_bed(down, x, y, method=method))
    named = down.rename(x='easting', y='northing')
    np.testing.assert_array_equal(sample_bed(named, x, y), sample_bed(down, x, y))
    other = down.rename(x='col', y='row')
    np.testing.assert_array_equal(sample_bed(other, x, y), sample_bed(down, x, y))
    np.testing.assert_array_equal(
        sample_bed(other.T, x, y, x_dim='col', y_dim='row'), sample_bed(down, x, y))
    ds = xr.Dataset({'topg': down, 'mask': down * 0})
    np.testing.assert_array_equal(sample_bed(ds, x, y, var='topg'),
                                  sample_bed(down, x, y))
    with pytest.raises(InvalidParamsError, match='var'):
        sample_bed(ds, x, y)
    with pytest.raises(InvalidParamsError, match='dimension'):
        sample_bed(down, x, y, x_dim='lon')
    # a singleton extra dimension is dropped, a real one is refused
    np.testing.assert_array_equal(sample_bed(down.expand_dims(time=1), x, y),
                                  sample_bed(down, x, y))
    with pytest.raises(InvalidParamsError, match='more than two'):
        sample_bed(down.expand_dims(time=2), x, y)


def test_points_outside_the_grid_are_nan():
    bed = _plane()
    out = sample_bed(bed, [-49., -51., 949., 951., 100., np.nan],
                     [100., 100., 100., 100., 900., 100.])
    assert np.isfinite(out[[0, 2]]).all()
    assert np.isnan(out[[1, 3, 4, 5]]).all()
    holed = bed.copy()
    holed[3, 3] = np.nan
    assert np.isnan(sample_bed(holed, 330., 400., method='bilinear'))
    assert np.isfinite(sample_bed(holed, 330., 300., method='nearest'))


def test_longitude_conventions():
    lon = np.arange(-180., 180., 10.)
    lat = np.arange(60., 85., 5.)
    z = -lon[None, :] - lat[:, None]
    bed = xr.DataArray(z, coords={'lat': lat, 'lon': lon}, dims=('lat', 'lon'))
    ref = sample_bed(bed, -40., 70., method='bilinear')
    assert ref == pytest.approx(40. - 70.)
    assert sample_bed(bed, 320., 70., method='bilinear') == pytest.approx(ref)
    east = bed.assign_coords(lon=bed.lon % 360.).sortby('lon')
    assert sample_bed(east, -40., 70., method='bilinear') == pytest.approx(ref)
    assert sample_bed(east, 320., 70., method='bilinear') == pytest.approx(ref)
    named = bed.rename(lon='longitude', lat='latitude')
    assert sample_bed(named, 320., 70.) == sample_bed(bed, -40., 70.)


def _fjord():
    """Bed falling towards x, a land strip at x >= 700 m."""
    x = np.arange(0., 1000., 100.)
    y = np.arange(0., 500., 100.)
    bed = xr.DataArray(np.tile(100. - 0.4 * x, (y.size, 1)),
                       coords={'y': y, 'x': x}, dims=('y', 'x'))
    cls = xr.full_like(bed, 2.)
    cls[:, 5:7] = 0.
    cls[:, 7:] = 1.
    return bed, cls


def test_the_depth_at_the_front():
    bed, cls = _fjord()
    x, y = np.array([100., 300., 500.]), np.full(3, 200.)
    # the last point: 100 - 200 = -100 m
    assert terminus_water_depth(bed, x, y) == (100., 1, 'bed')
    assert terminus_water_depth(bed, x, y, n_front=2) == (60., 2, 'bed')
    assert terminus_water_depth(bed, x, y, sea_level=2.5) == (102.5, 1, 'bed')
    assert terminus_water_depth(bed + 420., x, y, sea_level=420.) == (100., 1, 'bed')
    assert terminus_water_depth(bed, x[:1], y[:1]) == (0., 1, 'dry')
    assert terminus_water_depth(bed, [450.], [200.], method='bilinear')[0] == \
        pytest.approx(80.)
    with pytest.raises(InvalidWorkflowError, match='no value'):
        terminus_water_depth(bed, [5000.], [200.])


def test_the_mask_convention_at_the_front():
    bed, cls = _fjord()
    y = [200.]
    assert terminus_water_depth(bed, [500.], y, surface_class=cls) == \
        (100., 1, 'mask')
    # grounded ice in water counts as marine
    assert terminus_water_depth(bed, [400.], y, surface_class=cls) == \
        (60., 1, 'mask')
    # the bed is below sea level, but the mask shows ice-free land
    assert terminus_water_depth(bed, [800.], y, surface_class=cls) == (0., 1, 'land')
    # no convention: the bed alone decides
    for none in (None, 'none'):
        assert terminus_water_depth(bed, [800.], y, surface_class=cls,
                                    bed_mask=none) == (220., 1, 'bed')
    try:
        register_bed_mask('flipped', ocean_classes=(1,), land_classes=(0,))
        assert terminus_water_depth(bed, [800.], y, surface_class=cls,
                                    bed_mask='flipped') == (220., 1, 'mask')
        assert terminus_water_depth(bed, [500.], y, surface_class=cls,
                                    bed_mask='flipped')[2] == 'land'
    finally:
        del BED_MASKS['flipped']
