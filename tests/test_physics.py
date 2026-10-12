import numpy as np
import pytest
import xarray as xr

from tidewater import units
from tidewater.bed import register_bed_mask
from tidewater.config import (DEFAULTS, REGIONAL_FIELDS, REGIONS,
                              InvalidParamsError, InvalidWorkflowError,
                              OceanParams, register_region)
from tidewater.physics import (LAMBDA1, LAMBDA2, LAMBDA3, band_index,
                               band_names, flotation_thickness, freezing_point,
                               front_thickness, ocean_cells,
                               open_water_fraction, parse_depth_bands,
                               submerged_depth, terminus_water_depth_from_bed,
                               thermal_forcing, thermal_forcing_bands,
                               to_celsius, water_depth)


def test_freezing_point_matches_slater_form():
    s, z = 34.8, 300.
    assert freezing_point(s, z) == pytest.approx(LAMBDA1 * s + LAMBDA2 - LAMBDA3 * z)
    # colder with depth, and colder with salinity
    assert freezing_point(34.8, 500.) < freezing_point(34.8, 100.)
    assert freezing_point(35.5, 300.) < freezing_point(33.0, 300.)


def test_hand_computed_values():
    # -0.0573 * 34.8 + 0.0832 - 0.000761 * 300
    assert float(freezing_point(34.8, 300.)) == -0.0573 * 34.8 + 0.0832 - 0.000761 * 300.
    assert float(freezing_point(34.8, 300.)) == pytest.approx(-2.13914, abs=1e-12)
    assert float(thermal_forcing(1.5, 34.8, 300.)) == pytest.approx(3.63914, abs=1e-12)
    assert float(open_water_fraction(0.05)) == pytest.approx(2. / 3.)
    assert flotation_thickness(90.) == 90. * 1028. / 900.
    assert flotation_thickness(90., rho_ocean=1000., rho_ice=900.) == 100.
    assert front_thickness(40., 150.) == 190.
    assert front_thickness(-5., 150.) == 150.


def test_thermal_forcing_is_temperature_minus_freezing_point():
    th = np.array([[-1., 0.5], [2., 3.]])
    so = np.full_like(th, 34.)
    z = np.array([10., 200.])
    np.testing.assert_array_equal(thermal_forcing(th, so, z),
                                  th - freezing_point(so, z))


def test_teos10_needs_a_latitude():
    pytest.importorskip('gsw')
    with pytest.raises(InvalidParamsError, match='lat'):
        freezing_point(34.8, 100., teos10=True)
    assert freezing_point(34.8, 100., teos10=True, lat=70.) == pytest.approx(
        freezing_point(34.8, 100.), abs=0.1)


def test_open_water_fraction_is_bounded():
    owf = open_water_fraction([0., 0.05, 0.15, 0.9, 1.])
    assert owf[0] == 1.
    assert owf[-1] == 0.
    assert np.all((owf >= 0) & (owf <= 1))
    assert np.all(np.diff(owf) <= 0)
    assert open_water_fraction(0.25, threshold=0.5) == 0.5


def test_to_celsius_tells_kelvin_by_magnitude():
    np.testing.assert_allclose(to_celsius([274.15, 275.15]), [1., 2.])
    np.testing.assert_array_equal(to_celsius([1., 2.]), [1., 2.])


def test_to_celsius_is_the_one_in_units():
    assert to_celsius is units.to_celsius
    # a cold Kelvin record and a hot Celsius one are not guessed
    np.testing.assert_allclose(to_celsius([90., 95.], units='K'), [-183.15, -178.15])
    np.testing.assert_array_equal(to_celsius([274.15], units='degC'), [274.15])
    with pytest.raises(InvalidParamsError, match='units'):
        to_celsius([1.], units='F')


def test_sea_level_is_an_argument():
    assert water_depth(-150.) == 150. and water_depth(30.) == -30.
    assert water_depth(-150., sea_level=2.5) == 152.5
    assert water_depth(415., sea_level=420.) == 5.
    np.testing.assert_array_equal(water_depth([-10., 5.], sea_level=1.), [11., -4.])
    assert submerged_depth(200., 100.) == 100.
    assert submerged_depth(200., 100., sea_level=20.) == 120.
    assert front_thickness(40., 150., sea_level=0.) == front_thickness(40., 150.)
    assert front_thickness(460., 35., sea_level=420.) == 75.
    assert front_thickness(1., 150., sea_level=2.5) == 150.


def test_ocean_cells_follow_the_sea_level_and_the_mask_convention():
    bed, mask = _depth_grid()
    bed[25:27, 15:25] = -120.
    assert ocean_cells(bed, mask, sea_level=0.)[0].sum() == ocean_cells(bed, mask)[0].sum()
    # a lower sea level leaves only the deep cells wet
    assert ocean_cells(bed, mask, sea_level=-100.)[0].sum() == 20 + 44
    # a higher one floods the land around the glacier, never the glacier
    flooded, source = ocean_cells(bed, mask, sea_level=150.)
    assert source == 'bed_below_sea_level'
    assert flooded.sum() == bed.size - mask.sum()
    assert ocean_cells(bed, sea_level=150.)[0].all()

    cls = np.full(bed.shape, 2.)
    cls[26, 15:25] = 0.
    cls[25, 15:20] = 3.
    cls[25, 20:25] = 7.
    default = ocean_cells(bed, mask, surface_class=cls)
    named = ocean_cells(bed, mask, surface_class=cls, bed_mask='bedmachine')
    np.testing.assert_array_equal(default[0], named[0])
    assert named[0].sum() == 15 and named[1] == 'mask'
    # no convention: the bed and the sea level decide
    for none in (None, 'none'):
        cells, source = ocean_cells(bed, mask, surface_class=cls, bed_mask=none)
        assert source == 'bed_below_sea_level'
        np.testing.assert_array_equal(cells, ocean_cells(bed, mask)[0])
    try:
        entry = register_bed_mask('sevens', ocean_classes=(7,))
        assert ocean_cells(bed, mask, surface_class=cls, bed_mask='sevens')[0].sum() == 5
        assert ocean_cells(bed, mask, surface_class=cls, bed_mask=entry)[0].sum() == 5
    finally:
        from tidewater.bed import BED_MASKS
        del BED_MASKS['sevens']
    # explicit classes win over the convention
    assert ocean_cells(bed, mask, surface_class=cls, ocean_classes=(7,),
                       bed_mask='bedmachine')[0].sum() == 5
    with pytest.raises(InvalidParamsError, match='bed mask'):
        ocean_cells(bed, mask, surface_class=cls, bed_mask='unheard_of')


def test_depth_from_the_map_takes_the_sea_level_and_the_convention():
    bed, mask = _depth_grid()
    bed[25:27, 15:25] = -120.
    assert terminus_water_depth_from_bed(bed, mask, sea_level=0.) == \
        terminus_water_depth_from_bed(bed, mask) == (120., 20, 'ring')
    assert terminus_water_depth_from_bed(bed, mask, sea_level=2.5) == \
        (122.5, 20, 'ring')
    # a bed product on a datum 30 m higher gives the same depth
    assert terminus_water_depth_from_bed(bed + 30., mask, sea_level=30.) == \
        (120., 20, 'ring')
    cls = np.full(bed.shape, 2.)
    cls[26, 15:25] = 0.
    cls[25, 15:20] = 3.
    assert terminus_water_depth_from_bed(bed, mask, surface_class=cls)[1:] == \
        (15, 'ring')
    assert terminus_water_depth_from_bed(
        bed, mask, surface_class=cls, ocean_classes=(0,))[1:] == (10, 'ring')
    assert terminus_water_depth_from_bed(
        bed, mask, surface_class=cls, bed_mask=None)[1:] == (20, 'ring')


def test_front_band_follows_bathymetry():
    """A measured front depth sets the front band's bottom, floored at the first level."""
    z = np.arange(5., 705., 10.)
    t = np.full((12, len(z)), 1.5)
    s = np.full_like(t, 34.8)
    bands = parse_depth_bands(DEFAULTS.ocean_depth_bands)
    names, tops, bots, _, tf, _, _ = thermal_forcing_bands(
        t, s, z, *bands, terminus_depth=120.)
    assert names == ['shallow_0_60', 'ismip6', 'front']
    assert bots == [60., 500., 120.]
    assert tf.shape == (12, 3)
    *_, bots, _, _, _, _ = thermal_forcing_bands(t, s, z, *bands, terminus_depth=2.)
    assert bots[2] == 5.
    # the band of earlier versions still takes the depth
    *_, bots, _, _, _, _ = thermal_forcing_bands(t, s, z, [('terminus', 0., 60.)],
                                                 {}, terminus_depth=120.)
    assert bots == [120.]


def test_the_front_band_is_named_by_the_caller():
    z = np.arange(5., 705., 10.)
    t = np.full((2, len(z)), 1.5)
    s = np.full_like(t, 34.8)
    bands = [('grounding_line', 0., 60.), ('front', 0., 60.)]
    *_, bots, _, _, _, _ = thermal_forcing_bands(
        t, s, z, bands, {}, terminus_depth=300., front_bands=('grounding_line',))
    assert bots == [300., 60.]


def test_band_average_by_hand():
    z = np.array([10., 30., 100., 300.])
    th = np.array([[0., 1., 2., 3.]])
    so = np.full((1, 4), 34.)
    tf_z = th[0] - (LAMBDA1 * 34. + LAMBDA2 - LAMBDA3 * z)
    _, _, _, _, tf, th_b, so_b = thermal_forcing_bands(
        th, so, z, [('u', 0., 100.), ('d', 0., 100.)],
        {'u': 'uniform', 'd': 'depth_weighted'})
    np.testing.assert_allclose(tf[0, 0], tf_z[:3].mean(), rtol=1e-14)
    g = np.array([20., 45., 135.])  # index-centred differences of z
    np.testing.assert_allclose(tf[0, 1], (tf_z[:3] * g).sum() / g.sum(), rtol=1e-14)
    np.testing.assert_allclose(th_b[0], [1., (0. * 20 + 1. * 45 + 2. * 135) / 200.])
    np.testing.assert_allclose(so_b[0], [34., 34.])


def test_terminus_is_read_as_the_fixed_shallow_band():
    names = ['terminus', 'ismip6']
    assert band_index(names, 'shallow_0_60') == band_index(names, 'terminus') == 0
    with pytest.raises(InvalidParamsError):
        band_index(names, 'front')
    assert band_index(['a', 'b'], 'c', aliases={'c': 'b'}) == 1


def test_empty_band_raises():
    """The 200-500 m band at a site with 60 m of water must not return nan."""
    z = np.arange(5., 65., 10.)  # nothing below 60 m
    t = np.full((12, len(z)), 1.5)
    with pytest.raises(InvalidWorkflowError, match='no ocean levels'):
        thermal_forcing_bands(t, np.full_like(t, 34.8), z,
                              *parse_depth_bands(DEFAULTS.ocean_depth_bands))


def test_nan_in_a_band_raises():
    """A band whose levels are below the sea floor must not average to a nan column."""
    z = np.array([10., 50., 300.])
    thetao = np.array([[1., 1., np.nan]])
    so = np.full((1, 3), 34.8)
    with pytest.raises(InvalidWorkflowError, match='non-finite'):
        thermal_forcing_bands(thetao, so, z, [('ismip6', 200., 500.)],
                              {'ismip6': 'uniform'})


def test_band_weighting_differs():
    """Depth weighting and uniform weighting are not the same number."""
    z = np.concatenate([np.arange(5., 100., 5.), np.arange(100., 705., 50.)])
    t = np.tile(np.linspace(-1.5, 2.5, len(z)), (12, 1))
    s = np.full_like(t, 34.8)
    bands = [('column', 0., 700.)]
    _, _, _, _, tf_u, _, _ = thermal_forcing_bands(t, s, z, bands,
                                                   {'column': 'uniform'})
    _, _, _, _, tf_d, _, _ = thermal_forcing_bands(t, s, z, bands,
                                                   {'column': 'depth_weighted'})
    assert not np.isclose(tf_u[0, 0], tf_d[0, 0])


def test_unknown_weighting_raises():
    z = np.arange(5., 705., 10.)
    t = np.full((12, len(z)), 1.5)
    with pytest.raises(InvalidParamsError, match='weighting'):
        thermal_forcing_bands(t, np.full_like(t, 34.8), z, [('b', 0., 700.)],
                              {'b': 'inverse_distance'})


def test_depth_bands_are_parsed():
    bands, weighting = parse_depth_bands(DEFAULTS.ocean_depth_bands)
    assert bands == [('shallow_0_60', 0., 60.), ('ismip6', 200., 500.),
                     ('front', 0., 60.)]
    assert weighting == {'shallow_0_60': 'uniform', 'ismip6': 'uniform',
                         'front': 'uniform'}
    assert parse_depth_bands(['a:0:50']) == ([('a', 0., 50.)], {'a': 'uniform'})
    assert parse_depth_bands('a:0:50, b:0:700:depth_weighted') == (
        [('a', 0., 50.), ('b', 0., 700.)],
        {'a': 'uniform', 'b': 'depth_weighted'})
    with pytest.raises(InvalidParamsError):
        parse_depth_bands(['a:0'])


def test_band_names_from_a_dataset():
    ds = xr.Dataset({'thermal_forcing': (('time', 'band'), np.zeros((2, 2)))})
    assert band_names(ds) == ['0', '1']
    ds['band_name'] = ('band', [b'shallow\x00\x00', b'deep'])
    assert band_names(ds) == ['shallow', 'deep']
    ds['band_name'] = ('band', ['a', 'b'])
    assert band_names(ds) == ['a', 'b']
    chars = np.array([list('ab'), list('cd')], dtype='S1')
    ds2 = xr.Dataset({'band_name': (('band', 'nchar'), chars)})
    assert band_names(ds2) == ['ab', 'cd']


def _depth_grid():
    """No water at the outline, shallow water inside a 5-cell trim, deep water only past it."""
    bed = np.full((40, 40), 100.)
    mask = np.zeros((40, 40))
    mask[15:25, 15:25] = 1
    bed[30:34, 15:21] = -50.
    bed[0:4, 0:11] = -900.
    return bed, mask


def test_depth_fallback_is_trimmed():
    bed, mask = _depth_grid()
    assert terminus_water_depth_from_bed(bed, mask) == (900., 68, 'fallback')
    assert terminus_water_depth_from_bed(bed, mask, fallback_trim=5) == \
        (50., 24, 'fallback')
    with pytest.raises(InvalidWorkflowError, match='ocean cells'):
        terminus_water_depth_from_bed(np.full((40, 40), 100.), mask)


def test_depth_from_the_ring_and_the_surface_class():
    bed, mask = _depth_grid()
    bed[25:27, 15:25] = -120.
    assert terminus_water_depth_from_bed(bed, mask) == (120., 20, 'ring')
    # the class of the bed product decides what is ocean
    cls = np.full(bed.shape, 2.)
    cls[26, 15:25] = 0.
    cls[25, 15:20] = 3.
    cls[25, 20:25] = 7.
    ocean, source = ocean_cells(bed, mask, surface_class=cls)
    assert source == 'mask' and ocean.sum() == 15
    assert ocean_cells(bed, mask, surface_class=cls,
                       ocean_classes=(7,))[0].sum() == 5
    assert ocean_cells(bed, mask)[1] == 'bed_below_sea_level'
    assert terminus_water_depth_from_bed(bed, mask, ocean=ocean)[1:] == (15, 'ring')


def test_params_hold_the_defaults_and_read_a_host_mapping():
    p = OceanParams()
    assert (p.ocean_tf_band, p.calving_law) == ('shallow_0_60', 'constant')
    assert (p.calving_tf_exponent, p.calving_tf_fraction) == (1.18, 1.0)
    assert (p.calving_melt_A, p.calving_melt_B) == (3e-4, 0.15)
    assert (p.calving_melt_alpha, p.calving_melt_beta) == (0.39, 1.18)
    assert (p.ocean_water_density, p.ice_density) == (1028., 900.)
    assert p.ocean_tf_ref is None and p.ocean_tf_ref_period == (1995, 2014)
    with pytest.raises(Exception):
        p.calving_law = 'tf_power'

    host = {'ocean_depth_bands': ['column:0:700:depth_weighted', 'top:0:20'],
            'ocean_tf_band': 'column', 'ocean_tf_ref_period': [2000, 2010],
            'ocean_tf_ref': '', 'ocean_use_teos10': 'True',
            'calving_melt_beta': '1.61', 'ice_density': 917,
            'unrelated_host_key': 3}
    q = OceanParams.from_mapping(host, calving_law='sea_ice')
    assert q.depth_bands() == ([('column', 0., 700.), ('top', 0., 20.)],
                               {'column': 'depth_weighted', 'top': 'uniform'})
    assert q.ocean_tf_ref is None and q.ocean_tf_ref_period == (2000, 2010)
    assert q.ocean_use_teos10 is True and q.ice_density == 917.
    assert q.calving_melt_alpha == 0.39
    kw = q.law_kwargs()
    assert kw['beta'] == 1.61 and kw['delta'] == 1.0 and 'gamma' not in kw
    assert q.law_kwargs('tf_power') == {'gamma': 1.18, 'tf_fraction': 1.0,
                                        'sec_in_year': 31536000.}
    assert OceanParams.from_mapping(
        {'ocean_depth_bands': 'a:0:50, b:10:20'}).ocean_depth_bands == \
        ('a:0:50', 'b:10:20')
    assert q.replace(ocean_tf_ref=1.2).ocean_tf_ref == 1.2
    with pytest.raises(InvalidParamsError):
        q.law_kwargs('nope')


def test_periods_in_the_parameters_are_read_in_one_way():
    for period in ('2000,2010', '2000-2010', [2000., 2010.], (2000, 2010)):
        assert OceanParams.from_mapping(
            {'ocean_tf_ref_period': period}).ocean_tf_ref_period == (2000, 2010)
    with pytest.raises(InvalidParamsError, match='period'):
        OceanParams.from_mapping({'ocean_tf_ref_period': (2010, 2000)})


def test_the_sea_level_is_a_parameter():
    assert DEFAULTS.ocean_sea_level == 0.
    assert OceanParams.from_mapping({'ocean_sea_level': '2.5'}).ocean_sea_level == 2.5
    assert 'sea_level' not in DEFAULTS.law_kwargs('sea_ice')


def test_parameters_by_region():
    assert OceanParams.for_region('greenland') == DEFAULTS
    assert OceanParams.for_region(' Greenland ') == DEFAULTS
    assert list(REGIONS) == ['greenland', 'svalbard']
    assert set(REGIONS['greenland']) == set(REGIONAL_FIELDS)
    sv = OceanParams.for_region('svalbard')
    assert (sv.calving_tf_exponent, sv.ocean_tf_ref_period) == (0.9, (2016, 2024))
    assert sv.replace(calving_tf_exponent=1.18, ocean_tf_ref_period=(1995, 2014)) == DEFAULTS
    assert OceanParams.for_region('greenland', calving_law='tf_power',
                                  calving_melt_beta=1.61) == DEFAULTS.replace(
        calving_law='tf_power', calving_melt_beta=1.61)
    with pytest.raises(InvalidParamsError, match='register_region'):
        OceanParams.for_region('atlantis')
    try:
        register_region('Atlantis', calving_tf_exponent=1.5,
                        ocean_depth_bands='deep:300:600', ocean_tf_band='deep')
        p = OceanParams.for_region('atlantis', ice_density=917.)
        assert (p.calving_tf_exponent, p.ocean_tf_band, p.ice_density) == \
            (1.5, 'deep', 917.)
        assert p.depth_bands()[0] == [('deep', 300., 600.)]
        # what the region does not set keeps the default
        assert p.calving_melt_beta == DEFAULTS.calving_melt_beta
    finally:
        del REGIONS['atlantis']
    with pytest.raises(InvalidParamsError, match='not parameters'):
        register_region('atlantis', calving_gamma=1.5)
    with pytest.raises(ValueError):
        register_region('atlantis', calving_tf_exponent='steep')
    assert list(REGIONS) == ['greenland', 'svalbard']
