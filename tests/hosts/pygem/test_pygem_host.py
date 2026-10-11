"""The PyGEM host on synthetic tidewater glaciers and, where PyGEM has its
calibrated sample glacier, through ``run_simulation``.

The ocean factor 1 cases must give PyGEM's own k-calving, and the two routes
into PyGEM (its ``calving_law_for`` hook, or the replaced model names of a
PyGEM without it) must give the same numbers. The test ids name the route.
"""
import copy
import glob
import inspect
import os
import sys
import types

import numpy as np
import pandas as pd
import pytest

pytest.importorskip('pygem')
pytest.importorskip('oggm')

try:
    from pygem.glacierdynamics import MassRedistributionCurveModel
except Exception as err:  # noqa: BLE001
    # PyGEM reads and checks its configuration file on import
    pytest.skip(f'PyGEM is not configured: {err}', allow_module_level=True)

import xarray as xr  # noqa: E402
from oggm.core.flowline import FluxBasedModel  # noqa: E402
from oggm.core.massbalance import ScalarMassBalance  # noqa: E402
from oggm.tests.funcs import bu_tidewater_bed  # noqa: E402

from tidewater.config import (InvalidParamsError,  # noqa: E402
                              InvalidWorkflowError)
from tidewater.hosts.pygem import calibration as cal  # noqa: E402
from tidewater.hosts.pygem import front_cell, yearly_factors  # noqa: E402
from tidewater.hosts.pygem.dynamics import (  # noqa: E402
    HOOK, MONTHS, OceanMassRedistributionCurveModel, StepCalving,
    YearlyCalving, coupling_route, frontal_ablation_components,
    ocean_dynamics)
from tidewater.laws import ConstantK, MeltPlusCalving, TFPower  # noqa: E402

RGI = 'RGI60-01.00001'
K = 0.2

# what this PyGEM offers: the hook of the branch, or only the released names
ROUTE = coupling_route()
HAS_ARGUMENT = 'calving_law' in inspect.signature(
    MassRedistributionCurveModel.__init__).parameters
NO_HOOK = 'this PyGEM has no calving-law hook (released 1.1.0): swap only'


def routes():
    """Both routes where PyGEM has the hook, else the swap and a named skip."""
    hook = pytest.param('hook', id='route=hook', marks=pytest.mark.skipif(
        ROUTE != 'hook', reason=NO_HOOK))
    return [pytest.param('swap', id='route=swap'), hook]


class TaggedMB(ScalarMassBalance):
    """OGGM's scalar balance with the glacier table PyGEM's models carry."""

    glacier_rgi_table = pd.Series({'RGIId': RGI})


class AnnualMB:
    """What PyGEM's mass-redistribution model reads from its balance model."""

    hemisphere = 'nh'
    glacier_rgi_table = pd.Series({'RGIId': RGI})

    def __init__(self, nbins, y0=2000, ny=10, mb=0.):
        self.y0, self.mb = y0, mb
        self.years = list(range(y0, y0 + ny))
        self.dates_table = pd.DataFrame({'year': np.repeat(self.years, 12),
                                         'days_in_step': 30})
        self.glac_bin_frontalablation = np.zeros((nbins, 12 * ny + 1))
        for name in ('area', 'icethickness', 'width'):
            setattr(self, f'glac_bin_{name}_annual', np.zeros((nbins, ny + 1)))
        self.glac_wide_area_annual = np.zeros(ny + 1)
        self.glac_wide_volume_annual = np.zeros(ny + 1)

    def get_year_index(self, year):
        return int(year) - self.y0

    def get_step_inds(self, year):
        i = 12 * (int(year) - self.y0)
        return i, i + 11

    def get_annual_mb(self, heights, year=None, fls=None, fl_id=None,
                      **kwargs):
        return np.full(len(heights), self.mb)


@pytest.fixture(scope='module')
def glacier():
    """A tidewater glacier grown to the water by OGGM's stock model."""
    model = FluxBasedModel(bu_tidewater_bed(), mb_model=TaggedMB(),
                           is_tidewater=True, flux_gate=0.12,
                           do_kcalving=True, calving_k=K)
    model.run_until(1100)
    assert model.calving_m3_since_y0 > 0
    return model.fls


def flux_run(fls, law=None, y1=2100, cls=FluxBasedModel, **kwargs):
    if law is not None:
        kwargs['calving_law'] = law
    model = cls(copy.deepcopy(fls), mb_model=TaggedMB(), y0=2000,
                is_tidewater=True, flux_gate=0.12, do_kcalving=True,
                calving_k=K, **kwargs)
    model.run_until(y1)
    return model


def record(tf=2., years=(1990, 2200)):
    yrs = np.arange(*years) + 0.5
    return yrs, np.full(yrs.shape, tf)


def seasonal(year=2000):
    time = pd.date_range(f'{year}-01-01', f'{year}-12-01', freq='MS')
    return pd.Series(2. + np.cos(2 * np.pi * time.month / 12), index=time)


@pytest.fixture(scope='module')
def stock(glacier):
    return flux_run(glacier)


def redistribution(fls, cls=MassRedistributionCurveModel, ny=10, **kwargs):
    fls = copy.deepcopy(fls)
    model = cls(fls, mb_model=AnnualMB(fls[0].nx, ny=ny), y0=2000,
                is_tidewater=True, water_level=0., **kwargs)
    model.calving_k = K
    return model


def yearly_model(fls, how, ny=10, **kwargs):
    """A mass-redistribution model with a YearlyCalving, by either class."""
    if how == 'argument':
        return redistribution(fls, ny=ny, calving_law=YearlyCalving(**kwargs))
    return redistribution(fls, OceanMassRedistributionCurveModel, ny=ny,
                          **kwargs)


def hows():
    """PyGEM's own ``calving_law`` argument where it has it; the subclass."""
    argument = pytest.param('argument', id='pygem-argument',
                            marks=pytest.mark.skipif(
                                not HAS_ARGUMENT, reason=NO_HOOK))
    return [pytest.param('subclass', id='subclass'), argument]


@pytest.mark.parametrize('route', [ROUTE], ids=[f'pygem-route={ROUTE}'])
def test_route_of_this_pygem(route):
    from pygem.bin.run import run_simulation
    assert hasattr(run_simulation, HOOK) == (route == 'hook') == HAS_ARGUMENT


def test_the_flux_model_takes_the_law_on_released_oggm(glacier, stock):
    assert stock.calving_m3_since_y0 > 0
    for law in (ConstantK(), StepCalving(ConstantK())):
        same = flux_run(glacier, law)
        assert same.calving_m3_since_y0 == stock.calving_m3_since_y0
        assert same.volume_m3 == stock.volume_m3
    at_ref = flux_run(glacier, TFPower(*record(2.), tf_ref=2.))
    np.testing.assert_allclose(at_ref.calving_m3_since_y0,
                               stock.calving_m3_since_y0, rtol=1e-9)
    np.testing.assert_allclose(at_ref.volume_m3, stock.volume_m3, rtol=1e-9)
    warm = flux_run(glacier, TFPower(*record(3.), tf_ref=2.))
    assert warm.calving_m3_since_y0 > 1.2 * stock.calving_m3_since_y0
    calving, melt = frontal_ablation_components(warm)
    assert melt == 0 and calving == pytest.approx(warm.calving_m3_since_y0)


def test_melt_law_books_its_share(glacier):
    step = StepCalving(MeltPlusCalving(*record(3.), k_c=0.1))
    model = flux_run(glacier, step, y1=2030)
    assert step.model is model
    for run in (model, step):
        calving, melt = frontal_ablation_components(run)
        assert 0 < melt < calving + melt
        assert calving + melt == pytest.approx(model.calving_m3_since_y0)


@pytest.mark.parametrize('how', hows())
def test_redistribution_unchanged_at_factor_one(glacier, how):
    ref = redistribution(glacier)
    ref.run_until(2010)
    assert ref.calving_m3_since_y0 > 0
    ones = pd.Series(1., index=range(2000, 2010))
    for kwargs in ({'calving_law': ConstantK()},
                   {'calving_law': TFPower(*record(2.), tf_ref=2.)},
                   {'ocean_factor': ones}):
        model = yearly_model(glacier, how, **kwargs)
        model.run_until(2010)
        np.testing.assert_allclose(model.calving_m3_since_y0,
                                   ref.calving_m3_since_y0, rtol=1e-12)
        np.testing.assert_allclose(model.mb_model.glac_bin_frontalablation,
                                   ref.mb_model.glac_bin_frontalablation,
                                   rtol=1e-12)
        np.testing.assert_allclose(model.fls[0].section, ref.fls[0].section,
                                   rtol=1e-12)
        assert model.calving_k == K
        out = frontal_ablation_components(model)
        assert sum(out) == pytest.approx(model.calving_m3_since_y0)
    assert np.array_equal(model.fls[0].section, ref.fls[0].section)


@pytest.mark.skipif(not HAS_ARGUMENT, reason=NO_HOOK)
def test_argument_and_subclass_agree_bit_for_bit(glacier):
    law = MeltPlusCalving(2000 + MONTHS, seasonal().to_numpy() + 1, k_c=K)
    runs = []
    for how in ('subclass', 'argument'):
        model = yearly_model(glacier, how, calving_law=copy.deepcopy(law))
        model.run_until(2010)
        runs.append(model)
    a, b = runs
    assert a.calving_m3_since_y0 == b.calving_m3_since_y0 > 0
    assert np.array_equal(a.fls[0].section, b.fls[0].section)
    assert np.array_equal(a.mb_model.glac_bin_frontalablation,
                          b.mb_model.glac_bin_frontalablation)
    pd.testing.assert_frame_equal(a.frontal_components(),
                                  b.calving_law.components())


def test_yearly_calving(glacier):
    fl, wl = glacier[0], 0.
    front = front_cell(fl, wl)
    table = YearlyCalving(ocean_factor={2000: 2.})
    assert table.constant(fl, front, wl, 2000., K) == (2 * K, 0.)
    with pytest.raises(InvalidWorkflowError):
        table.constant(fl, front, wl, 2001., K)
    with pytest.raises(InvalidParamsError):
        YearlyCalving()
    with pytest.raises(InvalidParamsError):
        redistribution(glacier, OceanMassRedistributionCurveModel)
    # the law's constant is the equal-weight mean of its monthly factors
    tf = seasonal()
    yearly = YearlyCalving(TFPower(2000 + MONTHS, tf.to_numpy(), tf_ref=2.))
    k, share = yearly.constant(fl, front, wl, 2000., K)
    want = yearly_factors(tf, 2., weights='equal').loc[2000]
    assert k == pytest.approx(K * want, rel=1e-12) and share == 0.
    # called as PyGEM's yearly law it returns k d h w, and keeps its books
    model = redistribution(glacier)
    q = yearly(model, model.fls[0], front, 2000.)
    h = fl.thick[front]
    d = h - (fl.surface_h[front] - wl)
    assert q == k * d * h * fl.widths_m[front]
    model.calving_m3_since_y0 += q
    out = yearly.components()
    assert out.index.tolist() == [2000]
    assert out.frontal_ablation_m3.iloc[0] == pytest.approx(q)


@pytest.mark.parametrize('how', hows())
def test_redistribution_with_a_melt_law(glacier, how):
    ref = redistribution(glacier, ny=1)
    ref.run_until(2001)
    law = MeltPlusCalving(*record(3.), k_c=K)
    model = yearly_model(glacier, how, ny=1, calving_law=law)
    model.run_until(2001)
    calving, melt = frontal_ablation_components(model)
    assert melt > 0 and calving + melt > ref.calving_m3_since_y0
    assert calving + melt == pytest.approx(model.calving_m3_since_y0)


def fake_module(route):
    """The names ocean_dynamics acts on, in a PyGEM with or without a hook."""
    mod = types.SimpleNamespace(
        FluxBasedModel=FluxBasedModel,
        MassRedistributionCurveModel=MassRedistributionCurveModel)
    if route == 'hook':
        mod.calving_law_for = None
    return mod


def build_as_pygem(mod, fls, dynamics):
    """Build the model of one tidewater glacier the way run_simulation does."""
    gdir = types.SimpleNamespace(rgi_id=RGI, is_tidewater=True)
    kwargs = {}
    hook = getattr(mod, HOOK, None)
    calving = None if hook is None else hook(gdir, dynamics)
    if calving is not None:
        kwargs['calving_law'] = calving
    if dynamics == 'OGGM':
        return mod.FluxBasedModel(copy.deepcopy(fls), mb_model=TaggedMB(),
                                  y0=2000, is_tidewater=True, flux_gate=0.12,
                                  do_kcalving=True, calving_k=K, **kwargs)
    fls = copy.deepcopy(fls)
    model = mod.MassRedistributionCurveModel(
        fls, mb_model=AnnualMB(fls[0].nx), y0=2000, is_tidewater=True,
        water_level=0., **kwargs)
    model.calving_k = K
    return model


def test_ocean_dynamics_picks_the_route_and_restores():
    law = TFPower(*record(3.), tf_ref=2.)
    for route in ('swap', 'hook'):
        mod = fake_module(route)
        before = dict(vars(mod))
        assert coupling_route(mod) == route
        with ocean_dynamics({'1.00001': law}, modules=[mod]):
            changed = {k for k, v in vars(mod).items() if v is not before[k]}
        assert changed == ({HOOK} if route == 'hook' else
                           {'FluxBasedModel', 'MassRedistributionCurveModel'})
        assert vars(mod) == before
    with pytest.raises(InvalidWorkflowError):
        with ocean_dynamics({'1.00001': law}, modules=[fake_module('swap')],
                            route='hook'):
            pass
    with pytest.raises(InvalidParamsError):
        with ocean_dynamics({}, modules=[], route='patch'):
            pass


@pytest.mark.parametrize('dynamics', ['OGGM', pytest.param(
    'MassRedistributionCurves', marks=pytest.mark.skipif(
        not HAS_ARGUMENT, reason=NO_HOOK + ', for this model'))])
def test_hook_and_swap_give_the_same_run(glacier, dynamics):
    law = MeltPlusCalving(*record(3.), k_c=K)
    out = {}
    for route in ('swap', 'hook'):
        mod = fake_module(route)
        with ocean_dynamics({RGI: law}, modules=[mod]) as built:
            model = build_as_pygem(mod, glacier, dynamics)
            land = mod.FluxBasedModel(bu_tidewater_bed(),
                                      mb_model=TaggedMB())
        model.run_until(2010)
        (gid, calving), = built
        assert gid == RGI and calving.calving_law is not law
        assert not isinstance(land.calving_law, StepCalving)
        out[route] = (model.calving_m3_since_y0, model.fls[0].section.copy(),
                      frontal_ablation_components(calving))
    assert out['swap'][0] == out['hook'][0] > 0
    assert np.array_equal(out['swap'][1], out['hook'][1])
    assert out['swap'][2] == out['hook'][2] and out['swap'][2][1] > 0


def test_a_factor_table_is_for_the_redistribution_model_only(glacier):
    for route in ('swap', 'hook'):
        mod = fake_module(route)
        with ocean_dynamics(lambda gid: {2000: 1.}, modules=[mod]):
            with pytest.raises(InvalidWorkflowError):
                build_as_pygem(mod, glacier, 'OGGM')


def test_run_flux_as_pygem_reports(glacier):
    def build(gid, k):
        model = redistribution(glacier, ny=20)
        model.calving_k = k
        return model

    run = cal.run_flux(build, (2000, 2019))
    mr = run({RGI: K})[RGI]
    assert mr > 0
    assert cal.run_flux(build, (2000, 2019), as_pygem=True)({RGI: K})[RGI] \
        == mr

    def build_flux(gid, k):
        return FluxBasedModel(copy.deepcopy(glacier), mb_model=TaggedMB(),
                              y0=2000, is_tidewater=True, flux_gate=0.12,
                              do_kcalving=True, calving_k=k)

    q = cal.run_flux(build_flux, (2000, 2019))({RGI: K})[RGI]
    q_pygem = cal.run_flux(build_flux, (2000, 2019),
                           as_pygem=True)({RGI: K})[RGI]
    assert q > 0 and q_pygem == pytest.approx(0.9 * q)


# --- PyGEM's own run_simulation, on its calibrated sample glacier -----------

SAMPLE = '1.03622'


def simulate(rs, dynamics, tag, laws=None, route=None):
    """One run of PyGEM's run_simulation; every output variable, by name."""
    root = rs.pygem_prms['root']
    argv = ['run_simulation', '-rgi_glac_number', SAMPLE,
            '-sim_climate_name', 'ERA5', '-sim_startyear', '2000',
            '-sim_endyear', '2019', '-option_calibration', 'emulator',
            '-nsims', '1', '-option_dynamics', dynamics,
            '-export_extra_vars', '-outputfn_sfix', f'tw{tag}_']
    old, sys.argv = sys.argv, argv
    try:
        if laws is None:
            rs.main()
        else:
            with ocean_dynamics(laws, route=route) as built:
                rs.main()
            assert built, 'no model ran with the law'
    finally:
        sys.argv = old
    files = glob.glob(os.path.join(root, 'Output', 'simulations', '**',
                                   f'*_tw{tag}_all.nc'), recursive=True)
    assert len(files) == 1, files
    try:
        with xr.open_dataset(files[0]) as ds:
            return {name: ds[name].values.copy() for name in ds.data_vars
                    if ds[name].dtype.kind == 'f'}
    finally:
        os.remove(files[0])


@pytest.fixture(scope='module')
def run_simulation():
    from pygem.bin.run import run_simulation as rs
    prms = rs.pygem_prms
    reg = SAMPLE.split('.')[0].zfill(2)
    need = [cal.calibration_file(prms),
            os.path.join(prms['root'], 'Output', 'calibration', reg,
                         f'{SAMPLE}-modelprms_dict.json')]
    if not all(os.path.isfile(p) for p in need):
        pytest.skip(f'PyGEM has no calibrated sample glacier {SAMPLE} under '
                    'its root: run its frontal-ablation calibration first')
    return rs


@pytest.mark.parametrize('dynamics', ['OGGM', 'MassRedistributionCurves'])
@pytest.mark.parametrize('route', routes())
def test_run_simulation_keeps_pygem_at_factor_one(run_simulation, route,
                                                  dynamics):
    rs = run_simulation
    ref = simulate(rs, dynamics, 'ref')
    assert ref['glac_frontalablation'].sum() > 0
    exact = (ConstantK() if dynamics == 'OGGM'
             else pd.Series(1., index=range(2000, 2020)))
    got = simulate(rs, dynamics, 'one', {SAMPLE: exact}, route)
    assert got.keys() == ref.keys()
    for name in ref:
        assert np.array_equal(got[name], ref[name], equal_nan=True), name
    at_ref = TFPower(np.arange(1990, 2030) + 0.5, np.full(40, 4.), tf_ref=4.)
    got = simulate(rs, dynamics, 'ref4', {SAMPLE: at_ref}, route)
    for name in ref:
        np.testing.assert_allclose(got[name], ref[name], rtol=1e-11,
                                   atol=0, err_msg=name)


@pytest.mark.skipif(ROUTE != 'hook', reason=NO_HOOK)
@pytest.mark.parametrize('dynamics', ['OGGM', 'MassRedistributionCurves'])
def test_run_simulation_hook_equals_swap(run_simulation, dynamics):
    warm = TFPower(np.arange(1990, 2030) + 0.5, np.linspace(3., 6., 40),
                   tf_ref=4.)
    runs = [simulate(run_simulation, dynamics, route, {SAMPLE: warm}, route)
            for route in ('hook', 'swap')]
    ref = simulate(run_simulation, dynamics, 'ref')
    assert not np.array_equal(runs[0]['glac_frontalablation'],
                              ref['glac_frontalablation'])
    for name in runs[0]:
        assert np.array_equal(runs[0][name], runs[1][name],
                              equal_nan=True), name
