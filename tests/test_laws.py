import pickle

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from tidewater.config import (DEFAULTS, SEC_IN_YEAR, InvalidParamsError,
                              InvalidWorkflowError, OceanParams)
from tidewater import units
from tidewater.laws import (HOST_ATTRS, LAWS, CalvingLaw, ConstantK, FrontFlux,
                            LargerOf, MeltPlusCalving, SeaIceModulated,
                            TFPower, calving_law_flux,
                            calving_law_from_series, convert_flux,
                            discharge_per_front_area,
                            frontal_ablation_components,
                            frontal_ablation_specific_mb,
                            frontal_component_series, inversion_k_from_ocean,
                            larger_of_calving_constant, melt_share,
                            ocean_calving_law, ocean_factor, ocean_tf_mean,
                            partition_calving_constant, partition_law_k,
                            register_law, submarine_melt_rate, tf_power_mean,
                            tf_scaling)
from tidewater.physics import open_water_fraction


class FakeFlowline:
    """The three cell arrays a law reads from a host, and nothing else."""

    def __init__(self, thick=200., surface_h=100., width=1000.):
        self.thick = np.array([thick])
        self.surface_h = np.array([surface_h])
        self.widths_m = np.array([width])


class FakeModel:
    def __init__(self, calving_k=0.6, water_level=0., yr=2005.):
        self.calving_k = calving_k / SEC_IN_YEAR  # a host holds it in s-1
        self.water_level = water_level
        self.yr = yr


def control_flux(host, line, front):
    """The control written out from SPEC.md, sections 2 and 6."""
    thick, width = line.thick[front], line.widths_m[front]
    depth = thick - (line.surface_h[front] - host.water_level)
    return host.calving_k * depth * thick * width


@pytest.fixture
def state():
    return FakeModel(), FakeFlowline(), 0


@pytest.fixture
def years():
    return np.arange(2000, 2022, 1 / 12)


# --- the kernels ---------------------------------------------------------------

def test_hand_computed_values():
    assert ocean_factor(3., 1.5, 2.) == 4.
    assert ocean_factor(-1., 1.5, 1.18) == 0.
    assert ocean_factor(1.8, 1.5) == (1.8 / 1.5) ** 1.18
    assert tf_scaling(3., 1.5, 2., 0.5) == 2.5
    assert tf_scaling(3., 1.5, 2., 1.) == 4.
    # 0.6 yr-1 on a 200 m front in 100 m of water, 1 km wide
    q = ConstantK(k=0.6).flux(200., 100., 1000.)
    assert q == 0.6 / 31536000 * 100. * 200. * 1000.
    assert q == pytest.approx(0.380517503805175, rel=1e-14)
    assert ConstantK(k=0.6).flux_per_year(200., 100., 1000.) == pytest.approx(1.2e7)
    # B * TF**beta, m d-1
    assert submarine_melt_rate(100., 0., 1.8) == 0.15 * 1.8 ** 1.18
    assert submarine_melt_rate(100., 0., 1.8) == pytest.approx(0.30013235, rel=1e-7)
    assert submarine_melt_rate(100., 2., 1.) == 3e-4 * 100. * 2. ** 0.39 + 0.15
    assert np.isnan(submarine_melt_rate(100., 2., 0.))
    assert submarine_melt_rate(100., np.nan, 1.) == 0.15
    assert discharge_per_front_area(10., 100., 1000.) == 10. * 86400. / 1e5
    assert discharge_per_front_area(-10., 100., 1000.) == 0.


def test_ocean_factor_scalar_and_array_agree():
    tf = np.array([-1., 0., 0.4, 1.5, 3.7])
    for gamma in (0., 0.5, 1.18, 2.):
        arr = ocean_factor(tf, 1.5, gamma)
        np.testing.assert_allclose(
            arr, [ocean_factor(float(t), 1.5, gamma) for t in tf], rtol=1e-15)
        np.testing.assert_allclose(
            tf_scaling(tf, 1.5, gamma, 0.3), 0.7 + 0.3 * arr, rtol=1e-15)
    assert np.all(np.diff(ocean_factor(tf, 1.5)) >= 0)


def test_flux_units():
    q = 2.  # m3 s-1
    assert convert_flux(q) == 2. * SEC_IN_YEAR
    assert convert_flux(q, to_unit='km3 yr-1') == pytest.approx(2. * SEC_IN_YEAR / 1e9)
    assert convert_flux(q, to_unit='Gt yr-1') == pytest.approx(
        2. * SEC_IN_YEAR * 900. / 1e12)
    assert convert_flux(q, to_unit='Gt yr-1', ice_density=917.) == pytest.approx(
        2. * SEC_IN_YEAR * 917. / 1e12)
    assert convert_flux(1., 'Gt yr-1', 'km3 yr-1') == pytest.approx(1. / 0.9)
    assert convert_flux(q, to_unit='kg s-1') == 1800.
    assert convert_flux(q, to_unit='m3 yr-1', sec_in_year=360 * 86400.) == \
        pytest.approx(2. * 360 * 86400.)
    for unit in ('m3 yr-1', 'km3 yr-1', 'kg s-1', 'kg yr-1', 'Gt yr-1'):
        assert convert_flux(convert_flux(q, to_unit=unit), unit, 'm3 s-1') == \
            pytest.approx(q)
    with pytest.raises(InvalidParamsError, match='unit'):
        convert_flux(q, to_unit='mm w.e.')
    # 1 km3 yr-1 of ice off 1000 km2
    assert frontal_ablation_specific_mb(1., 1e9) == 900.
    assert frontal_ablation_specific_mb(1., 1e9, ice_density=917.) == 917.


def test_calving_law_flux_takes_the_depth_and_the_free_board():
    out = calving_law_flux(150., 0.6, width=2000., surface_h=40.)
    assert isinstance(out, FrontFlux)
    assert out.thickness == 190. and out.freeboard == 40.
    assert out.flux == 0.6 * 190. * 150. * 2000. / 1e9
    assert out.flotation_thickness == 150. * 1028. / 900.
    assert (out.width, out.water_depth, out.calving_k, out.sea_level) == \
        (2000., 150., 0.6, 0.)
    assert calving_law_flux(150., 0.6, thick=300., width=2000.).flux == \
        0.6 * 300. * 150. * 2000. / 1e9
    assert calving_law_flux(150., -1., thick=300., width=2000.).flux == 0.
    assert calving_law_flux(150., 1., thick=300., width=2000., rho_ocean=1000.,
                            rho_ice=1000.).flotation_thickness == 150.
    with pytest.raises(InvalidParamsError):
        calving_law_flux(150., 0.6, thick=300.)
    with pytest.raises(InvalidParamsError):
        calving_law_flux(150., 0.6, width=2000.)


def test_front_flux_is_a_neutral_record():
    out = calving_law_flux(150., 0.6, width=2000., surface_h=40.)
    assert out['flux'] == out.flux and out['thickness'] == 190.
    with pytest.raises(Exception):
        out.flux = 1.
    assert set(out.as_dict()) == {'flux', 'width', 'thickness', 'water_depth',
                                  'freeboard', 'flotation_thickness',
                                  'calving_k', 'sea_level'}
    host = out.as_dict({'thickness': 'h_front', 'calving_k': 'k'})
    assert host['h_front'] == 190. and host['k'] == 0.6 and 'thickness' not in host


def test_calving_law_flux_takes_the_sea_level():
    # a lake at 420 m, or a bed product on another datum
    out = calving_law_flux(35., 1.1, width=600., surface_h=460., sea_level=420.)
    assert out.thickness == 75. and out.freeboard == 40.
    assert out.sea_level == 420.
    ref = calving_law_flux(35., 1.1, width=600., surface_h=40.)
    assert out.flux == ref.flux
    # a surface below the water has no freeboard
    assert calving_law_flux(35., 1.1, width=600., surface_h=1.,
                            sea_level=2.5).thickness == 35.


def test_one_converter():
    q = np.array([0.5, 2., 40.])
    for a in units.FLUX_UNITS:
        for b in units.FLUX_UNITS:
            np.testing.assert_array_equal(
                convert_flux(q, a, b),
                units.convert_flux(q, a, b, ice_density=DEFAULTS.ice_density))
    assert convert_flux(2., to_unit='kg s-1', ice_density=917.) == \
        units.convert_flux(2., 'm3 s-1', 'kg s-1', ice_density=917.)


def test_melt_share():
    assert melt_share(3., total=12.) == 0.25
    assert melt_share(3., calving=9.) == 0.25
    assert melt_share(3., total=0.) == 0. and melt_share(0., calving=0.) == 0.
    np.testing.assert_array_equal(
        melt_share([0., 1., 3.], total=[0., 4., 4.]), [0., 0.25, 0.75])
    np.testing.assert_array_equal(
        melt_share([0., 1., 3.], calving=[0., 3., 1.]), [0., 0.25, 0.75])
    for kw in ({}, {'total': 1., 'calving': 1.}):
        with pytest.raises(InvalidParamsError, match='melt_share'):
            melt_share(1., **kw)


def test_inversion_k_scales_with_the_period_mean_forcing():
    np.testing.assert_allclose(inversion_k_from_ocean(0.6, 1.4, tf_ref=1.),
                               0.6 * 1.4 ** 1.18)
    # the period mean is its own reference: k is left as it is
    assert inversion_k_from_ocean(0.6, 1.4) == 0.6
    with pytest.raises(InvalidParamsError):
        inversion_k_from_ocean(0.6, 1.4, tf_ref=0.)
    with pytest.raises(InvalidParamsError):
        inversion_k_from_ocean(0.6, -0.2)
    kw = dict(tf_ref=1.)
    full = inversion_k_from_ocean(0.6, 1.4, **kw)
    assert inversion_k_from_ocean(0.6, 1.4, tf_fraction=0., **kw) == pytest.approx(0.6)
    assert inversion_k_from_ocean(0.6, 1.4, tf_fraction=1., **kw) == full
    assert inversion_k_from_ocean(0.6, 1.4, tf_fraction=0.5, **kw) == \
        pytest.approx(0.5 * (0.6 + full))
    assert inversion_k_from_ocean(0.6, -1., **kw) == 0.


# --- the laws ------------------------------------------------------------------

def test_constant_k_is_the_control(state):
    model, fl, i = state
    assert ConstantK()(model, fl, i) == control_flux(model, fl, i)
    # and an explicit k in yr-1 must be converted, not used raw
    assert ConstantK(k=0.6)(model, fl, i) == pytest.approx(control_flux(model, fl, i))


def test_tf_power_reduces_to_the_control(state, years):
    """TF == TF_ref for all t  =>  identical to the control. The nesting check."""
    model, fl, i = state
    law = TFPower(years, np.full_like(years, 1.5), tf_ref=1.5, gamma=1.18)
    assert law(model, fl, i) == control_flux(model, fl, i)


def test_units_are_m3_per_second(state, years):
    """k in yr-1, d, h, w in m: the flux must be m3 s-1, not m3 yr-1."""
    model, fl, i = state
    q = ConstantK(k=0.6)(model, fl, i)
    h, w = 200., 1000.
    d = h - (100. - 0.)
    assert q == pytest.approx(0.6 / SEC_IN_YEAR * d * h * w)
    assert q * SEC_IN_YEAR == pytest.approx(0.6 * d * h * w)


def test_the_plain_call_is_the_model_call(state, years):
    """flux(h, d, w, yr) and law(model, flowline, i) are the same number."""
    model, fl, i = state
    h, w = fl.thick[i], fl.widths_m[i]
    d = h - (fl.surface_h[i] - model.water_level)
    tf = 1.5 + np.sin(2 * np.pi * years)
    ow = np.linspace(0.1, 0.9, len(years))
    q_sg = np.full_like(years, 30.)
    for name in LAWS:
        own = calving_law_from_series(name, years, tf, open_water=ow, q_sg=q_sg,
                                      tf_ref=1.5)
        assert own.flux(h, d, w, model.yr, k=0.6) == own(model, fl, i)
        q_c, q_m = own.flux_components(h, d, w, model.yr, k=0.6)
        assert q_c + q_m == pytest.approx(own(model, fl, i))
        assert own.flux_per_year(h, d, w, model.yr, k=0.6) == \
            own(model, fl, i) * SEC_IN_YEAR
    with pytest.raises(InvalidWorkflowError, match='calving constant'):
        TFPower(years, tf, tf_ref=1.5).flux(h, d, w, 2005.)


def test_the_year_length_is_an_argument(years):
    tf = np.full_like(years, 1.5)
    julian = 365.25 * 86400.
    for law in (ConstantK(k=0.6, sec_in_year=julian),
                TFPower(years, tf, tf_ref=1.5, k0=0.6, sec_in_year=julian)):
        assert law.flux(200., 100., 1000., 2005.) == pytest.approx(
            0.6 / julian * 100. * 200. * 1000.)
        assert law.flux_per_year(200., 100., 1000., 2005.) == pytest.approx(1.2e7)


def test_negative_tf_never_nan(state, years):
    """A freezing-point ocean gives zero, not a NaN from a fractional power."""
    model, fl, i = state
    law = TFPower(years, np.full_like(years, -1.8), tf_ref=1.5, gamma=1.18)
    q = law(model, fl, i)
    assert np.isfinite(q)
    assert q == 0.


def test_tf_ref_must_be_positive(years):
    with pytest.raises(InvalidParamsError, match='TF_ref'):
        TFPower(years, np.full_like(years, 0.05), tf_ref=0.)
    with pytest.raises(InvalidParamsError, match='TF_ref'):
        TFPower(years, np.full_like(years, -0.2))


def test_monotone_in_tf(state, years):
    """gamma > 0 => the flux is non-decreasing in TF. The physical contract."""
    model, fl, i = state
    q = [TFPower(years, np.full_like(years, tf), tf_ref=1.5, gamma=1.18)(model, fl, i)
         for tf in [0., 0.5, 1.0, 1.5, 2.0, 3.0]]
    assert np.all(np.diff(q) >= 0)


def test_gamma_zero_is_the_control(state, years):
    model, fl, i = state
    law = TFPower(years, np.full_like(years, 4.2), tf_ref=1.5, gamma=0.)
    assert law(model, fl, i) == control_flux(model, fl, i)


def test_gamma_zero_and_a_zero_are_the_control_to_the_bit(years):
    """Over fronts and times, the same float as the control, not just close."""
    rng = np.random.default_rng(0)
    tf = 1.5 + 2.5 * np.sin(2 * np.pi * years / 3.)
    n = 2000
    surface = rng.uniform(1., 80., n)
    thick = surface + rng.uniform(1., 800., n)
    width = rng.uniform(50., 6000., n)
    k = rng.uniform(0.05, 8., n)
    level = rng.uniform(-1., 1., n)
    yr = rng.uniform(years[0], years[-1], n)
    laws = [TFPower(years, tf, tf_ref=1.7, gamma=0.),
            TFPower(years, tf, tf_ref=1.7, gamma=1.18, tf_fraction=0.)]
    control = ConstantK()
    for j in range(n):
        model = FakeModel(calving_k=k[j], water_level=level[j], yr=yr[j])
        fl = FakeFlowline(thick[j], surface[j], width[j])
        want = control(model, fl, 0)
        assert want == control_flux(model, fl, 0)
        h, d, w = thick[j], thick[j] - (surface[j] - level[j]), width[j]
        for law in laws:
            assert law(model, fl, 0) == want
            assert law.flux(h, d, w, yr[j], k=k[j]) == \
                ConstantK(k=k[j]).flux(h, d, w)
    d = thick - (surface - level)
    for law in laws:
        np.testing.assert_array_equal(law.fluxes(thick, d, width, yr, k=k),
                                      ConstantK().fluxes(thick, d, width, k=k))


def test_tf_fraction_nests_the_control_and_the_power_law(state, years):
    """a = 0 is constant k, a = 1 is the law without a background term."""
    model, fl, i = state
    tf = np.full_like(years, 4.2)
    stock = control_flux(model, fl, i)
    assert TFPower(years, tf, tf_ref=1.5, tf_fraction=0.)(model, fl, i) == stock
    full = TFPower(years, tf, tf_ref=1.5)(model, fl, i)
    assert TFPower(years, tf, tf_ref=1.5, tf_fraction=1.)(model, fl, i) == full
    half = TFPower(years, tf, tf_ref=1.5, tf_fraction=0.5)(model, fl, i)
    assert half == pytest.approx(0.5 * (stock + full))
    with pytest.raises(InvalidParamsError, match='tf_fraction'):
        TFPower(years, tf, tf_ref=1.5, tf_fraction=1.2)


def test_tf_fraction_keeps_the_mean_scaling_of_one(years):
    """At the power-mean reference the law averages to k0 for any a."""
    tf = 1.5 + np.sin(2 * np.pi * years)
    for gamma in (0.5, 1.18, 2.):
        ref = tf_power_mean(tf, gamma)
        for a in (0., 0.5, 1.):
            law = TFPower(years, tf, tf_ref=ref, gamma=gamma, tf_fraction=a)
            np.testing.assert_allclose(
                np.mean([law.scaling(y) for y in years]), 1., rtol=2e-3)


def test_larger_of_reduces_to_the_control(state, years):
    """lam = 0 is constant k; a large melt speed takes over the whole flux."""
    model, fl, i = state
    tf = np.full_like(years, 1.8)
    assert LargerOf(years, tf, lam=0.)(model, fl, i) == \
        pytest.approx(control_flux(model, fl, i))
    h, w = fl.thick[i], fl.widths_m[i]
    d = h - (fl.surface_h[i] - model.water_level)
    small = LargerOf(years, tf, k_c=0.6, lam=1e-3)
    u_c, u_m = small.frontal_speed_components(h, d, w, model.yr)
    assert u_m == 0. and u_c == pytest.approx(0.6 / SEC_IN_YEAR * h)
    big = LargerOf(years, tf, k_c=0.6, lam=1e3)
    u_c, u_m = big.frontal_speed_components(h, d, w, model.yr)
    assert u_c == 0. and u_m == pytest.approx(big.melt_speed(d, w, model.yr))
    assert (u_c + u_m) * d * w == pytest.approx(big(model, fl, i))
    assert big(model, fl, i) > small(model, fl, i)


def test_no_extrapolation_past_the_forcing(state):
    """Running past the end of the forcing freezes it instead of trending off."""
    yrs = np.array([2000., 2001., 2002.])
    law = TFPower(yrs, np.array([1.0, 2.0, 3.0]), tf_ref=1.5)
    assert law._at(law.tf, 1990.) == 1.0
    assert law._at(law.tf, 2050.) == 3.0


def test_melt_calving_reduces_to_rignot_b_limit(state, years):
    """With no subglacial discharge the melt term is exactly B * TF**beta."""
    tf = 1.8
    law = MeltPlusCalving(years, np.full_like(years, tf), k_c=0.6)
    expected = (DEFAULTS.calving_melt_B * tf ** DEFAULTS.calving_melt_beta
                / 86400.)
    assert law.melt_rate(100., 1000., 2005.) == pytest.approx(expected)


def test_melt_rate_is_the_stand_alone_one_with_discharge(years):
    tf, q_sg = 1.8, 40.
    law = MeltPlusCalving(years, np.full_like(years, tf),
                          q_sg=np.full_like(years, q_sg), k_c=0.6)
    q = discharge_per_front_area(q_sg, 100., 1000.)
    assert law.melt_rate(100., 1000., 2005.) == \
        submarine_melt_rate(100., q, tf) / 86400.
    fixed = MeltPlusCalving(years, np.full_like(years, tf),
                            q_sg=np.full_like(years, q_sg), k_c=0.6,
                            water_depth=250.)
    assert fixed.melt_rate(100., 1000., 2005.) == submarine_melt_rate(
        250., discharge_per_front_area(q_sg, 250., 1000.), tf) / 86400.


def test_melt_calving_adds_to_the_control(state, years):
    """With k_c = k the law is the control plus a strictly positive melt term."""
    model, fl, i = state
    law = MeltPlusCalving(years, np.full_like(years, 1.5), k_c=0.6)
    assert law(model, fl, i) > control_flux(model, fl, i)


def test_delta_zero_recovers_melt_calving(state, years):
    """SeaIceModulated(delta=0) == MeltPlusCalving, exactly. The nesting check."""
    model, fl, i = state
    tf = np.full_like(years, 1.8)
    ow = np.linspace(0.1, 0.9, len(years))
    a = MeltPlusCalving(years, tf, open_water=ow, k_c=0.6)
    b = SeaIceModulated(years, tf, open_water=ow, k_ice=0.6, delta=0.)
    assert b(model, fl, i) == pytest.approx(a(model, fl, i), rel=1e-12)


def test_sea_ice_needs_open_water(state, years):
    model, fl, i = state
    law = SeaIceModulated(years, np.full_like(years, 1.8), k_ice=0.6)
    with pytest.raises(InvalidWorkflowError, match='open_water'):
        law(model, fl, i)


def test_sea_ice_can_fall_while_tf_rises(state, years):
    """A falling open-water fraction can outweigh a rising thermal forcing,
    which the power law cannot do with gamma > 0."""
    model, fl, i = state
    tf = np.linspace(1.0, 2.0, len(years))
    ow = np.linspace(0.9, 0.1, len(years))
    law = SeaIceModulated(years, tf, open_water=ow, k_ice=0., delta=2.)
    early = law(FakeModel(yr=2001.), fl, i)
    late = law(FakeModel(yr=2020.), fl, i)
    assert late < early

    rising = TFPower(years, tf, tf_ref=1.5, k0=0.6, gamma=1.18)
    assert rising(FakeModel(yr=2020.), fl, i) > rising(FakeModel(yr=2001.), fl, i)


def test_no_calving_above_water(years):
    """A front sitting above the water level gives zero, not a negative flux."""
    fl = FakeFlowline(thick=50., surface_h=100.)  # d = 50 - 100 < 0
    law = TFPower(years, np.full_like(years, 1.5), tf_ref=1.5)
    assert law(FakeModel(), fl, 0) == 0.
    assert law.flux(50., -50., 1000., 2005., k=0.6) == 0.
    assert law.flux_components(50., -50., 1000., 2005.) == (0., 0.)


def test_laws_are_picklable(state, years):
    model, fl, i = state
    for law in (ConstantK(k=0.6),
                TFPower(years, np.full_like(years, 1.5), tf_ref=1.5),
                MeltPlusCalving(years, np.full_like(years, 1.5), k_c=0.6),
                SeaIceModulated(years, np.full_like(years, 1.5),
                                open_water=np.full_like(years, 0.5),
                                k_ice=0.6),
                LargerOf(years, np.full_like(years, 1.5), k_c=0.6)):
        again = pickle.loads(pickle.dumps(law))
        assert again(model, fl, i) == law(model, fl, i)


def test_beta_is_a_law_parameter(state, years):
    """The low-discharge branch (beta = 1.61) has to be reachable per law."""
    tf = 1.8
    law = MeltPlusCalving(years, np.full_like(years, tf), k_c=0.6,
                          beta=1.61)
    assert law.beta == 1.61
    assert DEFAULTS.calving_melt_beta == 1.18
    assert law.melt_rate(100., 1000., 2005.) == pytest.approx(
        DEFAULTS.calving_melt_B * tf ** 1.61 / 86400.)
    ice = SeaIceModulated(years, np.full_like(years, tf), open_water=np.ones_like(years),
                          k_ice=0.6, beta=1.61)
    assert ice.beta == 1.61


def test_tf_power_mean_is_where_the_law_averages_to_one():
    tf = np.array([-0.5, 0.5, 1., 2., 4., np.nan])
    for gamma in (0.5, 1., 1.18, 2.):
        ref = tf_power_mean(tf, gamma)
        ok = np.isfinite(tf)
        scaling = (np.clip(tf[ok], 0, None) / ref) ** gamma
        np.testing.assert_allclose(scaling.mean(), 1.)
        np.testing.assert_allclose(ocean_factor(tf[ok], ref, gamma).mean(), 1.)
    np.testing.assert_allclose(tf_power_mean(tf, 1.), 7.5 / 5)
    assert np.isnan(tf_power_mean([np.nan], 1.18))


# --- from an ocean dataset -----------------------------------------------------

def _ocean_dataset(y0=2000, y1=2021, names=('terminus', 'ismip6', 'column')):
    """A rising thermal forcing and a shrinking ice season, stored as float32."""
    time = pd.date_range(f'{y0}-01-01', f'{y1}-12-01', freq='MS')
    n = len(time)
    tf = np.stack([np.linspace(0.2, 0.6, n), np.linspace(1.0, 2.0, n),
                   np.linspace(0.5, 1.2, n)], axis=1)
    sic = np.linspace(0.9, 0.2, n)
    ds = xr.Dataset(
        {'thermal_forcing': (('time', 'band'), tf.astype('f4')),
         'band_name': ('band', list(names)),
         'siconc': ('time', sic.astype('f4')),
         'open_water_frac': ('time', open_water_fraction(sic).astype('f4'))},
        coords={'time': time})
    return ds, tf


@pytest.fixture
def ocean_ds():
    return _ocean_dataset()[0]


def test_law_from_dataset_reads_the_named_band(ocean_ds, state):
    model, fl, i = state
    q = {}
    for band in ('terminus', 'ismip6'):
        law = ocean_calving_law(ocean_ds, calving_law='tf_power', band=band,
                                tf_ref=1.0, k0=0.6)
        q[band] = law(model, fl, i)
    # the ismip6 band is the warmer one in this record, so it must calve more
    assert q['ismip6'] > q['terminus']
    # the default band is found under its earlier name
    assert ocean_calving_law(ocean_ds, 'tf_power', tf_ref=1., k0=0.6)(
        model, fl, i) == q['terminus']

    with pytest.raises(InvalidParamsError, match='band'):
        ocean_calving_law(ocean_ds, calving_law='tf_power', band='not_a_band')
    with pytest.raises(InvalidParamsError, match='unknown calving law'):
        ocean_calving_law(ocean_ds, calving_law='nope')
    with pytest.raises(InvalidWorkflowError):
        ocean_calving_law(None, calving_law='tf_power')
    assert isinstance(ocean_calving_law(None), ConstantK)


def test_law_from_dataset_is_time_varying(ocean_ds, state):
    _, fl, i = state
    law = ocean_calving_law(ocean_ds, calving_law='tf_power', band='terminus',
                            tf_ref=0.2, k0=0.6)
    assert law(FakeModel(yr=2020.5), fl, i) > law(FakeModel(yr=2000.5), fl, i)
    assert law.years[0] == 2000 + 0.5 / 12


def test_variable_names_are_arguments(ocean_ds, state):
    ds = ocean_ds.rename(thermal_forcing='tf', open_water_frac='owf')
    law = ocean_calving_law(ds, 'sea_ice', band='ismip6', tf_var='tf',
                            open_water_var='owf', k_ice=0.6)
    ref = ocean_calving_law(ocean_ds, 'sea_ice', band='ismip6', k_ice=0.6)
    assert law(*state) == ref(*state)
    assert ocean_tf_mean(ds, band='ismip6', period=(2000, 2010), tf_var='tf') == \
        ocean_tf_mean(ocean_ds, band='ismip6', period=(2000, 2010))


def test_tf_power_takes_its_own_reference_by_default(ocean_ds):
    """Without a tf_ref the law returns k on average over the reference period."""
    law = ocean_calving_law(ocean_ds, 'tf_power', band='ismip6',
                            tf_ref_period=(2000, 2010))
    sel = (law.years >= 2000) & (law.years < 2011)
    np.testing.assert_allclose(law.tf_ref, tf_power_mean(law.tf[sel], law.gamma))
    np.testing.assert_allclose(
        np.mean((law.tf[sel] / law.tf_ref) ** law.gamma), 1.)
    # the argument takes precedence
    assert ocean_calving_law(ocean_ds, 'tf_power', band='ismip6',
                             tf_ref=1.5).tf_ref == 1.5
    # the default period is the one of the parameters
    full = ocean_calving_law(ocean_ds, 'tf_power', band='ismip6')
    sel = (full.years >= 1995) & (full.years < 2015)
    np.testing.assert_allclose(full.tf_ref, tf_power_mean(full.tf[sel], 1.18))

    # a record outside the reference period has no reference
    with pytest.raises(InvalidWorkflowError, match='does not cover'):
        ocean_calving_law(ocean_ds, 'tf_power', band='ismip6',
                          tf_ref_period=(1950, 1960))
    with pytest.raises(InvalidParamsError):
        TFPower(law.years, law.tf)


def test_law_parameters_come_from_the_host_parameters(ocean_ds):
    params = OceanParams.from_mapping({'calving_tf_exponent': 2.,
                                       'calving_melt_beta': 1.61,
                                       'calving_openwater_exponent': 0.5})
    assert ocean_calving_law(ocean_ds, 'tf_power', band='ismip6', tf_ref=1.,
                             **params.law_kwargs('tf_power')).gamma == 2.
    law = ocean_calving_law(ocean_ds, 'sea_ice', band='ismip6',
                            **params.law_kwargs('sea_ice'))
    assert (law.beta, law.delta) == (1.61, 0.5)
    assert law.B == DEFAULTS.calving_melt_B
    # an explicit argument wins
    kw = dict(params.law_kwargs('tf_power'), gamma=1.)
    assert ocean_calving_law(ocean_ds, 'tf_power', band='ismip6', tf_ref=1.,
                             **kw).gamma == 1.
    # a None falls back on the default
    assert calving_law_from_series('tf_power', [0., 1.], [1., 1.], tf_ref=1.,
                                   gamma=None).gamma == 1.18


def test_period_mean_thermal_forcing():
    ds, tf = _ocean_dataset(1990, 2012)
    mean = ocean_tf_mean(ds, band='ismip6', period=(2000, 2010))
    np.testing.assert_allclose(mean, tf[120:252, 1].mean(), rtol=1e-6)
    np.testing.assert_allclose(
        ocean_tf_mean(ds, band='ismip6', period=(2000, 2010), gamma=1.18),
        tf_power_mean(tf[120:252, 1], 1.18), rtol=1e-6)
    # the default band and period
    np.testing.assert_allclose(ocean_tf_mean(ds), tf[60:, 0].mean(), rtol=1e-6)


FRONT = {'calving_k': 0.6, 'calving_front_thick': 260.,
         'terminus_water_depth': 150., 'calving_front_width': 3000.}


def test_partition_keeps_the_calibrated_total():
    ds, _ = _ocean_dataset(1990, 2012)
    with pytest.raises(InvalidWorkflowError, match='partitioning needs'):
        ocean_calving_law(ds, 'melt_calving', band='terminus', partition=True)
    with pytest.raises(InvalidWorkflowError, match='does not cover'):
        ocean_calving_law(ds, 'melt_calving', band='terminus', partition=True,
                          front=FRONT, partition_period=(1950, 1960))

    law = ocean_calving_law(ds, 'melt_calving', band='terminus',
                            partition=True, partition_period=(2000, 2010),
                            front=FRONT)
    k, h, w = 0.6, 260., 3000.
    sel = (law.years >= 2000) & (law.years < 2011)
    assert sel.sum() == 132
    mdot = np.mean([law.melt_rate(150., w, y) for y in law.years[sel]])
    assert 0 < law.k_c < k
    np.testing.assert_allclose(law.k_c / SEC_IN_YEAR * h + law.lam * mdot,
                               k / SEC_IN_YEAR * h)
    np.testing.assert_allclose(law.melt_fraction_reference,
                               mdot / (k / SEC_IN_YEAR * h))
    # a law without a melt term is left alone
    plain = ocean_calving_law(ds, 'tf_power', band='terminus', partition=True,
                              tf_ref=1., front=FRONT)
    assert partition_law_k(plain, 0.6, h, 150., w) == (None, 0.)


def test_partition_averages_the_gated_melt():
    """The open-water law's residual constant keeps the total with its gate on."""
    ds, _ = _ocean_dataset(1990, 2012)
    kw = dict(band='terminus', partition=True, partition_period=(2000, 2010),
              front=FRONT)
    law = ocean_calving_law(ds, 'sea_ice', **kw)
    free = ocean_calving_law(ds, 'melt_calving', **kw)
    k, h, w = 0.6, 260., 3000.
    sel = (law.years >= 2000) & (law.years < 2011)
    u_melt = np.mean([law.melt_speed(150., w, y) for y in law.years[sel]])
    np.testing.assert_allclose(law.k_c / SEC_IN_YEAR * h + u_melt,
                               k / SEC_IN_YEAR * h)
    assert law.k_c > free.k_c  # the gate leaves less melt to carve out


def test_partition_calibrates_the_larger_of_law():
    """With partition, the larger-of law returns the calibrated total over the period."""
    ds, _ = _ocean_dataset(1990, 2012)
    k, h, w = 0.6, 260., 3000.
    unit = ocean_calving_law(ds, 'larger_of', band='terminus', lam=1.)
    sel = (unit.years >= 2000) & (unit.years < 2011)
    u1 = np.array([unit.melt_speed(150., w, y) for y in unit.years[sel]])
    # mean melt at 95 % of the total, so that the last months exceed it
    lam = 0.95 * k / SEC_IN_YEAR * h / u1.mean()
    law = ocean_calving_law(ds, 'larger_of', band='terminus', partition=True,
                            partition_period=(2000, 2010), lam=lam, front=FRONT)
    u = lam * u1
    assert u.max() > k / SEC_IN_YEAR * h
    c = law.k_c / SEC_IN_YEAR * h
    assert 0 < law.k_c < k
    np.testing.assert_allclose(np.mean(np.maximum(c, u)), k / SEC_IN_YEAR * h)


# --- the melt/calving split ----------------------------------------------------

def _laws(years):
    tf = np.full_like(years, 1.8)
    ow = np.full_like(years, 0.5)
    return [ConstantK(k=0.6),
            TFPower(years, tf, tf_ref=1.5, k0=0.6),
            MeltPlusCalving(years, tf, k_c=0.6),
            SeaIceModulated(years, tf, open_water=ow, k_ice=0.6),
            LargerOf(years, tf, k_c=0.6, lam=1e3)]


def test_components_sum_to_the_flux(state, years):
    """Whatever the law splits, the total it returns must not change."""
    model, fl, i = state
    for law in _laws(years):
        h, w = fl.thick[i], fl.widths_m[i]
        d = h - (fl.surface_h[i] - model.water_level)
        u_c, u_m = law.frontal_speed_components(h, d, w, model.yr)
        assert u_c + u_m == pytest.approx(law.frontal_speed(h, d, w, model.yr))
        assert (u_c + u_m) * d * w == pytest.approx(law(model, fl, i))


def test_only_the_melt_laws_report_melt(state, years):
    model, fl, i = state
    h, w = fl.thick[i], fl.widths_m[i]
    d = h - (fl.surface_h[i] - model.water_level)
    for law in _laws(years)[:2]:
        assert law.frontal_speed_components(h, d, w, model.yr)[1] == 0.
        assert law.flux_components(h, d, w, model.yr)[1] == 0.
    for law in _laws(years)[2:]:
        assert law.frontal_speed_components(h, d, w, model.yr)[1] > 0.
        assert law.flux_components(h, d, w, model.yr)[1] > 0.


def test_delta_zero_recovers_the_split_not_only_the_total(state, years):
    model, fl, i = state
    tf = np.full_like(years, 1.8)
    ow = np.linspace(0.1, 0.9, len(years))
    a = MeltPlusCalving(years, tf, open_water=ow, k_c=0.6)
    b = SeaIceModulated(years, tf, open_water=ow, k_ice=0.6, delta=0.)
    h, w = fl.thick[i], fl.widths_m[i]
    d = h - (fl.surface_h[i] - model.water_level)
    assert (b.frontal_speed_components(h, d, w, 2005.) ==
            pytest.approx(a.frontal_speed_components(h, d, w, 2005.)))


def test_the_sea_ice_gate_moves_the_split(state, years):
    """Closing the ice season lowers the melt term and leaves the calving term."""
    model, fl, i = state
    tf = np.full_like(years, 1.8)
    h, w = fl.thick[i], fl.widths_m[i]
    d = h - (fl.surface_h[i] - model.water_level)
    open_sea = SeaIceModulated(years, tf, open_water=np.full_like(years, 0.9),
                               k_ice=0.6, delta=2.)
    icy = SeaIceModulated(years, tf, open_water=np.full_like(years, 0.1),
                          k_ice=0.6, delta=2.)
    c_open, m_open = open_sea.frontal_speed_components(h, d, w, 2005.)
    c_icy, m_icy = icy.frontal_speed_components(h, d, w, 2005.)
    assert m_icy < m_open
    assert c_icy == pytest.approx(c_open)


def test_a_bare_call_does_not_accumulate(state, years):
    """A law called without a model clock is an inspection, not a run."""
    model, fl, i = state
    law = MeltPlusCalving(years, np.full_like(years, 1.8), k_c=0.6)
    law(model, fl, i)
    law.flux(200., 100., 1000., 2005.)
    assert law.frontal_ablation_m3 == 0.
    assert law.components_m3() == (0., 0.)


def test_accounting_survives_pickle(state, years):
    law = MeltPlusCalving(years, np.full_like(years, 1.8), k_c=0.6)
    law.frontal_ablation_m3, law.submarine_melt_m3 = 10., 3.
    again = pickle.loads(pickle.dumps(law))
    assert again.components_m3() == (7., 3.)


class ClockModel(FakeModel):
    """A model with a clock, stepping as a flowline model does."""

    def __init__(self, law, fl, dt=86400. * 5, **kwargs):
        super().__init__(**kwargs)
        self.calving_law, self.fl, self.dt = law, fl, dt
        self.t, self.calving_m3_since_y0 = 0., 0.
        self.yr0 = self.yr
        self.cum = [(self.yr, 0.)]

    def run(self, steps):
        for _ in range(steps):
            q = self.calving_law(self, self.fl, 0)
            self.calving_m3_since_y0 += q * self.dt
            self.t += self.dt
            self.yr = self.yr0 + self.t / SEC_IN_YEAR
            self.cum.append((self.yr, self.calving_m3_since_y0))
        return self


def test_the_split_sums_to_the_model_total(years):
    tf = np.linspace(0.5, 3., len(years))
    ow = np.linspace(0.9, 0.1, len(years))
    for law in (ConstantK(), TFPower(years, tf, tf_ref=1.5),
                MeltPlusCalving(years, tf, k_c=0.3),
                SeaIceModulated(years, tf, open_water=ow, k_ice=0.3),
                LargerOf(years, tf, k_c=0.02)):
        model = ClockModel(law, FakeFlowline()).run(400)
        calving, melt = frontal_ablation_components(model)
        assert calving + melt == pytest.approx(model.calving_m3_since_y0)
        assert (melt > 0) == hasattr(law, 'melt_rate')

        t, cum = np.array(model.cum).T
        melt_t, calving_t, frac = frontal_component_series(law, t, cum, model)
        np.testing.assert_allclose(melt_t + calving_t, cum)
        assert np.all(np.diff(melt_t) >= -1e-6)
        assert frac == pytest.approx(melt / (calving + melt))
        np.testing.assert_allclose(melt_t[-1], melt, rtol=1e-6)

        law.reset_accounting()
        assert law.components_m3() == (0., 0.)
        np.testing.assert_array_equal(
            frontal_component_series(law, t, cum)[0], 0.)

    stock = FakeModel()
    stock.calving_m3_since_y0 = 5.
    assert frontal_ablation_components(stock) == (5., 0.)


# --- keeping a calibrated total ------------------------------------------------

def test_partition_calving_constant_preserves_the_total():
    """k_c + the melt term must reproduce exactly what the calibrated k gave."""
    h, k = 200., 0.6
    mdot = 0.1 / 86400.  # 0.1 m per day, well inside the calibrated total
    k_c, frac = partition_calving_constant(k, mdot, h, lam=1.)
    assert k_c < k
    assert (k_c / SEC_IN_YEAR * h + mdot ==
            pytest.approx(k / SEC_IN_YEAR * h))
    assert frac == pytest.approx(mdot / (k / SEC_IN_YEAR * h))
    assert partition_calving_constant(k, mdot, h) == (k_c, frac)


def test_partition_calving_constant_refuses_to_go_negative():
    """Melt exceeding the observed total is a result to report, not a negative k."""
    k_c, frac = partition_calving_constant(0.01, 1.0, 200., lam=1.)
    assert (k_c, frac) == (0., 1.)
    with pytest.raises(InvalidParamsError, match='thickness'):
        partition_calving_constant(0.6, 1e-7, 0.)


def test_larger_of_calving_constant_keeps_the_total():
    """The period mean of max(k_c h, u_melt) must equal what the calibrated k gave."""
    h, k = 200., 0.6
    u_total = k / SEC_IN_YEAR * h
    u = u_total * np.array([0.1, 0.5, 1.5, 0.2, 0.8, 0.3])
    k_c, frac = larger_of_calving_constant(k, u, h)
    c = k_c / SEC_IN_YEAR * h
    assert 0 < k_c < k
    assert np.mean(np.maximum(c, u)) == pytest.approx(u_total)
    assert frac == pytest.approx(np.where(u > c, u, 0.).mean() / u_total)


def test_larger_of_calving_constant_limits():
    """Melt never above calving keeps k; melt above the total on average gives zero."""
    h, k = 200., 0.6
    u_total = k / SEC_IN_YEAR * h
    assert larger_of_calving_constant(k, [0., 0.5 * u_total], h) == \
        (pytest.approx(k), 0.)
    assert larger_of_calving_constant(k, [0.5, 2.0] * np.array(u_total) * 2,
                                      h) == (0., 1.)
    with pytest.raises(InvalidParamsError, match='thickness'):
        larger_of_calving_constant(k, [0.], 0.)


# --- registry, description, arrays, sea level -------------------------------------

class Doubled(TFPower):
    name = 'doubled'

    def frontal_speed_components(self, h, d, w, yr):
        u_c, u_m = super().frontal_speed_components(h, d, w, yr)
        return 2 * u_c, u_m


def test_a_law_registry_is_an_argument(ocean_ds, years):
    mine = dict(LAWS)
    assert register_law('doubled', Doubled, laws=mine) is Doubled
    assert 'doubled' not in LAWS
    tf = np.full_like(years, 1.5)
    law = calving_law_from_series('doubled', years, tf, k0=0.6, laws=mine)
    ref = calving_law_from_series('tf_power', years, tf, k0=0.6)
    assert isinstance(law, Doubled) and law.gamma == 1.18
    assert law.tf_ref == ref.tf_ref
    assert law.flux(200., 100., 1000., 2005.) == \
        2 * ref.flux(200., 100., 1000., 2005.)
    from_ds = ocean_calving_law(ocean_ds, 'doubled', band='ismip6', tf_ref=1.,
                                k0=0.6, laws=mine)
    assert isinstance(from_ds, Doubled)
    assert DEFAULTS.law_kwargs('doubled', laws=mine) == \
        DEFAULTS.law_kwargs('tf_power')
    with pytest.raises(InvalidParamsError, match='unknown calving law'):
        calving_law_from_series('doubled', years, tf)
    with pytest.raises(InvalidParamsError, match='unknown calving law'):
        ocean_calving_law(ocean_ds, 'doubled')
    with pytest.raises(InvalidParamsError, match='unknown calving law'):
        DEFAULTS.law_kwargs('doubled')
    with pytest.raises(InvalidParamsError, match='CalvingLaw'):
        register_law('nope', dict, laws=mine)

    class Control(ConstantK):
        name = 'control'

    register_law('control', Control, laws=mine)
    assert type(calving_law_from_series('control', k=0.6, laws=mine)) is Control
    assert type(ocean_calving_law(None, 'control', laws=mine)) is Control
    try:
        register_law('doubled', Doubled)
        assert isinstance(calving_law_from_series('doubled', years, tf), Doubled)
    finally:
        del LAWS['doubled']
    assert issubclass(Doubled, CalvingLaw)


def test_sea_ice_keeps_k_ice_when_k_c_is_forwarded_as_none(years):
    tf = np.full_like(years, 1.8)
    ow = np.full_like(years, 0.5)

    class Forwarding(SeaIceModulated):
        def __init__(self, *args, k_c=None, **kwargs):
            super().__init__(*args, k_c=k_c, **kwargs)

    for law in (Forwarding(years, tf, open_water=ow, k_ice=0.6),
                SeaIceModulated(years, tf, open_water=ow, k_ice=0.6, k_c=None),
                calving_law_from_series('sea_ice', years, tf, open_water=ow,
                                        k_ice=0.6, k_c=None)):
        assert law.k_c == law.k_ice == 0.6
        assert law.flux(200., 100., 1000., 2005.) == SeaIceModulated(
            years, tf, open_water=ow, k_c=0.6).flux(200., 100., 1000., 2005.)
    # an explicit k_c still wins
    assert SeaIceModulated(years, tf, open_water=ow, k_ice=0.6, k_c=0.2).k_c == 0.2


def test_describe_lists_the_parameters(years):
    tf = np.full_like(years, 1.8)
    assert ConstantK(k=0.6).describe() == {
        'law': 'constant', 'sec_in_year': SEC_IN_YEAR, 'sea_level': None,
        'k': 0.6, 'record': None}
    d = TFPower(years, tf, tf_ref=1.5, k0=0.6).describe()
    assert (d['law'], d['k0'], d['tf_ref'], d['gamma'], d['tf_fraction']) == \
        ('tf_power', 0.6, 1.5, 1.18, 1.0)
    assert d['record'] == (years[0], years[-1])
    d = SeaIceModulated(years, tf, open_water=tf, k_ice=0.3, delta=2.,
                        sea_level=1.5).describe()
    assert (d['k_c'], d['delta'], d['lam'], d['beta'], d['sea_level']) == \
        (0.3, 2., 1.0, 1.18, 1.5)
    assert LargerOf(years, tf).describe()['law'] == 'larger_of'


def test_fluxes_is_flux_over_arrays(years):
    tf = 1.5 + np.sin(2 * np.pi * years)
    ow = np.linspace(0.1, 0.9, len(years))
    h = np.array([200., 350., 50., 80.])
    d = np.array([100., 300., 60., -10.])
    w = np.array([1000., 4000., 800., 800.])
    yr = np.array([2001.2, 2005.5, 2010.9, 2030.])
    for name in LAWS:
        law = calving_law_from_series(name, years, tf, open_water=ow,
                                      q_sg=np.full_like(years, 30.), tf_ref=1.5)
        out = law.fluxes(h, d, w, yr, k=0.6)
        assert out.shape == (4,)
        assert list(out) == [law.flux(*c, k=0.6) for c in zip(h, d, w, yr)]
        grid = law.fluxes(h[:, None], d[:, None], w[:, None], yr[None, :], k=0.6)
        assert grid.shape == (4, 4)
        assert grid[1, 2] == law.flux(h[1], d[1], w[1], yr[2], k=0.6)
        per_front = law.fluxes(h, d, w, 2005.5, k=np.array([0.1, 0.6, 1., 2.]))
        assert per_front[3] == law.flux(h[3], d[3], w[3], 2005.5, k=2.)
        assert law.fluxes(200., 100., 1000., 2005.5, k=0.6) == \
            law.flux(200., 100., 1000., 2005.5, k=0.6)
    assert ConstantK(k=0.6).fluxes(h, d, w)[0] == ConstantK(k=0.6).flux(200., 100., 1000.)


def test_the_sea_level_of_a_host_call(years):
    fl = FakeFlowline(thick=200., surface_h=100., width=1000.)
    tf = np.full_like(years, 1.8)
    for law_cls, kw in ((ConstantK, {}), (MeltPlusCalving, {'k_c': 0.6})):
        args = () if law_cls is ConstantK else (years, tf)
        base = law_cls(*args, **kw)(FakeModel(), fl, 0)
        # the host's level is read by default
        raised = law_cls(*args, **kw)(FakeModel(water_level=20.), fl, 0)
        assert raised > base
        # the law's own level overrules it, as a number or a function of time
        assert law_cls(*args, sea_level=20., **kw)(FakeModel(), fl, 0) == raised
        assert law_cls(*args, sea_level=0., **kw)(
            FakeModel(water_level=20.), fl, 0) == base
        rising = law_cls(*args, sea_level=lambda yr: (yr - 2005.) * 20., **kw)
        assert rising(FakeModel(yr=2005.), fl, 0) == base
        assert rising(FakeModel(yr=2006.), fl, 0) == raised
    # depth = thickness - (surface - sea level)
    assert ConstantK(k=0.6, sea_level=20.)(FakeModel(), fl, 0) == \
        ConstantK(k=0.6).flux(200., 120., 1000.)


def test_the_host_attributes_are_data(years):
    class Line:
        ice = np.array([200.])
        top = np.array([100.])
        breadth = np.array([1000.])

    class Host:
        sea = 0.
        rate = 0.6 / SEC_IN_YEAR
        when = 2005.

    class Mine(ConstantK):
        host_attrs = dict(HOST_ATTRS, thickness='ice', surface='top',
                          width='breadth', sea_level='sea', calving_k='rate',
                          year='when')

    assert Mine()(Host(), Line(), 0) == ConstantK()(FakeModel(), FakeFlowline(), 0)


def test_periods_are_read_in_one_way(ocean_ds):
    ref = ocean_calving_law(ocean_ds, 'tf_power', band='ismip6',
                            tf_ref_period=(2000, 2010)).tf_ref
    for period in ([2000, 2010], '2000-2010', '2000,2010', (2000., 2010.)):
        assert ocean_calving_law(ocean_ds, 'tf_power', band='ismip6',
                                 tf_ref_period=period).tf_ref == ref
        assert ocean_tf_mean(ocean_ds, period=period) == \
            ocean_tf_mean(ocean_ds, period=(2000, 2010))
    a = ocean_calving_law(ocean_ds, 'melt_calving', partition=True, front=FRONT,
                          partition_period='2000-2010')
    b = ocean_calving_law(ocean_ds, 'melt_calving', partition=True, front=FRONT,
                          partition_period=(2000, 2010))
    assert a.k_c == b.k_c and a.melt_fraction_reference == b.melt_fraction_reference
    with pytest.raises(InvalidParamsError, match='period'):
        ocean_tf_mean(ocean_ds, period=(2010, 2000))


def test_the_partition_front_may_be_a_front_flux(ocean_ds):
    front = calving_law_flux(150., 0.6, thick=260., width=3000.)
    a = ocean_calving_law(ocean_ds, 'melt_calving', partition=True, front=front,
                          partition_period=(2000, 2010))
    b = ocean_calving_law(ocean_ds, 'melt_calving', partition=True, front=FRONT,
                          partition_period=(2000, 2010))
    c = ocean_calving_law(ocean_ds, 'melt_calving', partition=True,
                          front={'calving_k': 0.6, 'thickness': 260.,
                                 'water_depth': 150., 'width': 3000.},
                          partition_period=(2000, 2010))
    assert a.k_c == b.k_c == c.k_c


def test_the_melt_share_of_a_run(years):
    tf = np.linspace(0.5, 3., len(years))
    law = MeltPlusCalving(years, tf, k_c=0.3)
    model = ClockModel(law, FakeFlowline()).run(50)
    calving, melt = law.components_m3(model)
    assert law.melt_share(model) == melt / (calving + melt)
    assert 0 < law.melt_share(model) < 1
    assert ConstantK(k=0.6).melt_share() == 0.
