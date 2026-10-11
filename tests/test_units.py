import numpy as np
import pytest

from tidewater.config import SEC_IN_YEAR, InvalidParamsError
from tidewater.units import (FLUX_UNITS, MASS_UNITS, VOLUME_UNITS, convert_flux,
                             temperature_unit, to_celsius)


def test_the_six_units():
    assert FLUX_UNITS == VOLUME_UNITS + MASS_UNITS
    assert set(FLUX_UNITS) == {'m3 s-1', 'm3 yr-1', 'km3 yr-1', 'kg s-1',
                               'kg yr-1', 'Gt yr-1'}


def test_hand_computed_conversions():
    q = 2.  # m3 s-1 of ice
    assert convert_flux(q, 'm3 s-1', 'm3 yr-1') == 2. * SEC_IN_YEAR
    assert convert_flux(q, 'm3 s-1', 'km3 yr-1') == 2. * SEC_IN_YEAR / 1e9
    assert convert_flux(q, 'm3 s-1', 'kg s-1', ice_density=900.) == 1800.
    assert convert_flux(q, 'm3 s-1', 'kg yr-1', ice_density=900.) == \
        pytest.approx(1800. * SEC_IN_YEAR)
    assert convert_flux(q, 'm3 s-1', 'Gt yr-1', ice_density=900.) == \
        pytest.approx(2. * SEC_IN_YEAR * 900. / 1e12)
    assert convert_flux(1., 'Gt yr-1', 'km3 yr-1', ice_density=900.) == 1. / 0.9
    assert convert_flux(1., 'km3 yr-1', 'Gt yr-1', ice_density=917.) == 0.917
    assert convert_flux(1., 'Gt yr-1', 'kg yr-1') == 1e12
    assert convert_flux(1e12, 'kg yr-1', 'Gt yr-1') == pytest.approx(1.)


def test_every_pair_round_trips():
    q = np.array([0.3, 2., 55.])
    for a in FLUX_UNITS:
        for b in FLUX_UNITS:
            there = convert_flux(q, a, b, ice_density=917.)
            np.testing.assert_allclose(
                convert_flux(there, b, a, ice_density=917.), q, rtol=1e-14)


def test_the_same_unit_is_returned_untouched():
    q = np.array([1., 2.])
    for unit in FLUX_UNITS:
        assert convert_flux(q, unit, unit) is q
    assert convert_flux(3., 'Gt yr-1', 'Gt yr-1') == 3.


def test_the_density_is_required_between_mass_and_volume():
    for a in VOLUME_UNITS:
        for b in MASS_UNITS:
            for pair in ((a, b), (b, a)):
                with pytest.raises(InvalidParamsError, match='ice density'):
                    convert_flux(1., *pair)
                with pytest.raises(InvalidParamsError, match='ice density'):
                    convert_flux(1., *pair, ice_density=0.)
    # and is not needed within one basis
    assert convert_flux(1., 'km3 yr-1', 'm3 yr-1') == 1e9
    assert convert_flux(SEC_IN_YEAR, 'kg yr-1', 'kg s-1') == pytest.approx(1.)


def test_the_density_scales_the_mass():
    a = convert_flux(1., 'km3 yr-1', 'Gt yr-1', ice_density=900.)
    b = convert_flux(1., 'km3 yr-1', 'Gt yr-1', ice_density=917.)
    assert b / a == pytest.approx(917. / 900.)


def test_the_year_length_is_an_argument():
    julian = 365.25 * 86400.
    assert convert_flux(1., 'm3 s-1', 'm3 yr-1', sec_in_year=julian) == \
        pytest.approx(julian, rel=1e-15)
    assert convert_flux(1., 'm3 s-1', 'km3 yr-1', sec_in_year=julian) == \
        pytest.approx(julian / 1e9, rel=1e-15)
    assert convert_flux(1., 'km3 yr-1', 'Gt yr-1', ice_density=900.,
                        sec_in_year=julian) == pytest.approx(0.9, rel=1e-15)


def test_unknown_units_raise():
    for pair in (('mm w.e.', 'Gt yr-1'), ('Gt yr-1', 'Gt a-1')):
        with pytest.raises(InvalidParamsError, match='unknown flux unit'):
            convert_flux(1., *pair, ice_density=900.)


def test_lists_and_arrays_convert():
    out = convert_flux([1., 2.], 'km3 yr-1', 'Gt yr-1', ice_density=900.)
    np.testing.assert_allclose(out, [0.9, 1.8])
    out = convert_flux([1., 2.], 'm3 s-1', 'kg s-1', ice_density=900.)
    np.testing.assert_allclose(out, [900., 1800.])


def test_temperature_unit_names():
    for name in ('degC', 'C', 'celsius', 'degree_Celsius', ' degrees_celsius '):
        assert temperature_unit(name) == 'degC'
    for name in ('K', 'kelvin', 'degK'):
        assert temperature_unit(name) == 'K'
    with pytest.raises(InvalidParamsError, match='not understood'):
        temperature_unit('F')


def test_auto_tells_the_unit_by_magnitude():
    assert temperature_unit('auto', [274., 276.]) == 'K'
    assert temperature_unit(None, [1., 3., np.nan]) == 'degC'
    with pytest.raises(InvalidParamsError, match='need finite values'):
        temperature_unit('auto')
    with pytest.raises(InvalidParamsError, match='need finite values'):
        temperature_unit('auto', [np.nan, np.nan])


def test_to_celsius():
    np.testing.assert_allclose(to_celsius([274.15, 275.15]), [1., 2.])
    np.testing.assert_array_equal(to_celsius([1., 2.]), [1., 2.])
    np.testing.assert_allclose(to_celsius([274.15], units='K'), [1.])
    # an explicit unit is not second-guessed
    np.testing.assert_array_equal(to_celsius([274.15], units='degC'), [274.15])
    np.testing.assert_allclose(to_celsius([1.], units='kelvin'), [-272.15])
    assert to_celsius(275.15).shape == ()
    assert float(to_celsius(275.15)) == pytest.approx(2.)
    with pytest.raises(InvalidParamsError):
        to_celsius([1.], units='F')
