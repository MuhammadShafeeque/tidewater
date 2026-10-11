"""Targets, the static fit of k, its fill rules and its scaling by the ocean."""
import numpy as np
import pandas as pd
import pytest

from tidewater.calibration import static, targets
from tidewater.config import SEC_IN_YEAR, InvalidParamsError, InvalidWorkflowError


# --- targets ----------------------------------------------------------------------

def test_units_convert_with_an_explicit_ice_density():
    assert targets.convert_flux(0.9, 'Gt yr-1', 'km3 yr-1', rho_ice=900.) == 0.9 / 0.9
    assert targets.convert_flux(1., 'km3 yr-1', 'Gt yr-1', rho_ice=917.) == pytest.approx(0.917)
    assert targets.convert_flux(1., 'km3 yr-1', 'm3 s-1') == pytest.approx(1e9 / SEC_IN_YEAR)
    back = targets.convert_flux(targets.convert_flux(np.array([0.3, 2.]), 'Gt yr-1', 'm3 s-1',
                                                     rho_ice=917.),
                                'm3 s-1', 'Gt yr-1', rho_ice=917.)
    np.testing.assert_allclose(back, [0.3, 2.])
    assert targets.convert_flux(5., 'Gt yr-1', 'Gt yr-1') == 5.
    with pytest.raises(InvalidParamsError, match='ice density'):
        targets.convert_flux(1., 'Gt yr-1', 'km3 yr-1')
    with pytest.raises(InvalidParamsError, match='unknown flux unit'):
        targets.convert_flux(1., 'Gt yr-1', 'Sv')


def test_any_dataset_reads_into_the_neutral_table(tmp_path):
    f = tmp_path / 'obs.csv'
    f.write_text('ID,FA,FA_unc,t0,t1\nA-1,0.9,0.09,2000,2010\nA-2,,0.1,2000,2010\n'
                 'A-3,1.8,,2010,2020\n')
    cols = dict(glacier_id='ID', flux='FA', flux_err='FA_unc', start_date='t0',
                end_date='t1')
    t = targets.read_targets(f, cols, to_unit='km3 yr-1', rho_ice=900., source_name='inv')
    assert list(t.columns) == targets.TARGET_COLUMNS
    assert t.glacier_id.tolist() == ['A-1', 'A-3']            # no flux: no target
    np.testing.assert_allclose(t.flux, [1., 2.])
    assert t.flux_err[0] == pytest.approx(0.1) and np.isnan(t.flux_err[1])
    assert t.start_date.tolist() == [2000, 2010] and set(t.source) == {'inv'}
    assert t.attrs['flux_unit'] == 'km3 yr-1'
    assert targets.target_series(t)['A-3'] == pytest.approx(2.)

    gt = targets.convert_targets(t, 'Gt yr-1', rho_ice=900.)
    np.testing.assert_allclose(gt.flux, [0.9, 1.8])

    # a frame with the neutral names needs no mapping, and no density without a conversion
    frame = pd.DataFrame({'glacier_id': ['x', 'x'], 'flux': [1., 2.]})
    t = targets.read_targets(frame)
    assert t.attrs['flux_unit'] == 'Gt yr-1' and t.source.isna().all()
    with pytest.raises(InvalidParamsError, match='several observations'):
        targets.target_series(t)
    with pytest.raises(InvalidParamsError, match='no column'):
        targets.read_targets(frame, dict(flux='nope'))
    with pytest.raises(InvalidParamsError, match='not target columns'):
        targets.read_targets(frame, dict(ident='glacier_id'))


def test_there_is_one_converter():
    from tidewater import units
    for pair in (('Gt yr-1', 'km3 yr-1'), ('km3 yr-1', 'm3 s-1'), ('m3 s-1', 'Gt yr-1'),
                 ('kg s-1', 'm3 yr-1')):
        assert (targets.convert_flux(0.37, *pair, rho_ice=900.)
                == units.convert_flux(0.37, *pair, ice_density=900.))
    assert targets.convert_flux(1., 'km3 yr-1', 'Gt yr-1', ice_density=917.) == 0.917
    assert targets.UNITS == units.FLUX_UNITS


def test_targets_read_any_table_unit_and_id_convention(tmp_path):
    f = tmp_path / 'obs.txt'
    f.write_text('ID;FA;t0;t1\nold-1;2e9;2000-01-01;2009-12-31\nold-2;1e9;2000;2010\n'
                 'old-9;1e9;2000;2010\n')
    cols = dict(glacier_id='ID', flux='FA', start_date='t0', end_date='t1')
    link = pd.DataFrame({'old': ['old-1', 'old-2'], 'new': ['new-1', 'new-2']})
    t = targets.read_targets(f, cols, flux_unit='m3 yr-1', to_unit='Gt yr-1',
                             ice_density=900., sep=';',
                             id_map=dict(link=link, source='old', target='new',
                                         missing='none'))
    assert t.glacier_id.tolist() == ['new-1', 'new-2']           # old-9 has no partner
    np.testing.assert_allclose(t.flux, [1.8, 0.9])
    assert targets.read_targets(f, cols, sep=';', id_map=str.upper).glacier_id[0] == 'OLD-1'
    assert targets.read_targets(f, cols, sep=';',
                                id_map={'old-1': 'x'}).glacier_id.tolist() == ['x']

    frame = pd.DataFrame({'glacier_id': ['a', 'b'], 'flux': [1., 2.],
                          'flux_err': [0.1, 0.2]})
    nc = tmp_path / 'obs.nc'
    frame.set_index('glacier_id').to_xarray().to_netcdf(nc)
    for source in (nc, frame.set_index('glacier_id').to_xarray()):
        t = targets.read_targets(source, flux_unit='km3 yr-1')
        assert t.glacier_id.tolist() == ['a', 'b'] and t.flux_err.tolist() == [0.1, 0.2]

    assert targets.TARGET_SOURCES == {}
    targets.register_target_source('demo', columns=cols, flux_unit='m3 yr-1', sep=';')
    try:
        t = targets.read_targets(f, preset='demo')
        assert t.attrs['flux_unit'] == 'm3 yr-1' and set(t.source) == {'demo'}
        with pytest.raises(InvalidParamsError, match='unknown target source'):
            targets.read_targets(f, preset='nope')
        with pytest.raises(InvalidParamsError, match='unknown flux unit'):
            targets.register_target_source('bad', flux_unit='Sv')
    finally:
        targets.TARGET_SOURCES.clear()


def test_several_observations_reduce_to_one_explicitly():
    frame = pd.DataFrame({
        'glacier_id': ['a', 'a', 'a', 'b'], 'flux': [1., 3., 5., 7.],
        'flux_err': [0.1, 0.3, 0.2, 0.7],
        'start_date': [2000, 2010, '2000-01-01', 2000],
        'end_date': [2010, 2020, '2020-01-01', 2010], 'source': ['x', 'x', 'y', 'x']})
    t = targets.read_targets(frame)
    assert len(t) == 4                                            # nothing is reduced unasked
    mean = targets.read_targets(frame, reduce='mean').set_index('glacier_id')
    assert mean.flux['a'] == 3. and mean.flux_err['a'] == pytest.approx(0.2)
    assert mean.start_date['a'] == 2000 and mean.source['a'] == 'x+y'
    assert mean.flux['b'] == 7.
    latest = targets.reduce_targets(t, 'latest').set_index('glacier_id')
    assert latest.flux['a'] == 5. and latest.attrs['flux_unit'] == 'Gt yr-1'
    longest = targets.reduce_targets(t, 'longest').set_index('glacier_id')
    assert longest.flux['a'] == 5. and longest.end_date['a'] == '2020-01-01'
    top = targets.reduce_targets(t, lambda g: g.loc[g.flux.idxmax()])
    assert targets.target_series(top).to_dict() == {'a': 5., 'b': 7.}
    with pytest.raises(InvalidParamsError, match='unknown reduction'):
        targets.reduce_targets(t, 'first')
    with pytest.raises(InvalidParamsError, match='needs the dates'):
        targets.reduce_targets(t.assign(end_date=None), 'latest')

    # a period is (first year, last year); the end date of a table is read as told
    ten = targets.select_period(t, (2000, 2009))
    assert ten.flux.tolist() == [1., 7.]
    assert targets.select_period(t, (2000, 2010), end='inclusive').flux.tolist() == [1., 7.]
    assert targets.select_period(t, (2000, 2019)).flux.tolist() == [5.]
    assert targets.decimal_year('2001-07-02') == pytest.approx(2001.5, abs=2e-3)


# --- the static fit ------------------------------------------------------------------

@pytest.fixture
def fronts():
    geom = pd.DataFrame({'thickness_m': [200., 300., 150., 250., 100.],
                         'water_depth_m': [150., 250., 100., 200., 0.],
                         'width_m': [2000., 4000., 1000., 3000., 500.]},
                        index=pd.Index(list('abcde'), name='glacier_id'))
    return geom, static.shape_flux(geom)


def test_the_shape_is_the_law_at_unit_k(fronts):
    from tidewater.laws import calving_law_flux
    geom, shape = fronts
    for gid, r in geom.iterrows():
        assert shape[gid] == calving_law_flux(r.water_depth_m, 1., thick=r.thickness_m,
                                              width=r.width_m)['flux']
    assert shape['e'] == 0.
    other = geom.rename(columns={'thickness_m': 'h', 'water_depth_m': 'd', 'width_m': 'w'})
    pd.testing.assert_series_equal(static.shape_flux(other, 'h', 'd', 'w'), shape)


def test_the_pooled_k_is_flux_weighted(fronts):
    _, shape = fronts
    obs = {'a': 0.03, 'b': 0.6, 'e': 1., 'z': 9., 'c': np.nan}
    k, per = static.fit_calving_k(shape, obs)
    assert per == {'a': 0.03 / shape['a'], 'b': 0.6 / shape['b']}   # e has no shape
    assert k == (0.03 + 0.6) / (shape['a'] + shape['b'])
    with pytest.raises(InvalidWorkflowError):
        static.fit_calving_k(shape, {})


def test_every_glacier_gets_a_k_and_the_flux_it_implies(fronts):
    _, shape = fronts
    obs = {'a': 0.03, 'b': 0.6, 'c': 0.03}
    out = static.calving_k_table(shape, obs, observed_err={'a': 0.003})
    assert list(out.columns) == static.K_TABLE_COLUMNS and out.index.name == 'glacier_id'
    assert out.source.to_dict() == dict(a='observed', b='observed', c='observed',
                                        d='modelled', e='modelled')
    np.testing.assert_allclose(out.flux[list('abc')], [0.03, 0.6, 0.03])
    ks = out.calving_k[list('abc')]
    gm = np.exp(np.log(ks).mean())
    assert out.calving_k['d'] == pytest.approx(gm)                 # intercept alone
    assert out.flux['d'] == pytest.approx(gm * shape['d']) and out.flux['e'] == 0.
    assert out.calving_k_unc['a'] == pytest.approx(0.003 / shape['a'])
    assert np.isnan(out.calving_k_unc['b']) and out.calving_k_unc['d'] > 0

    for how, want in (('min', ks.min()), ('max', ks.max()), ('median', ks.median()),
                      ('geometric_mean', gm),
                      ('pooled', sum(obs.values()) / shape[list('abc')].sum())):
        out = static.calving_k_table(shape, obs, fill=how)
        assert out.calving_k['d'] == pytest.approx(want) and out.source['d'] == how

    # an excluded glacier keeps its own k and informs no other
    out = static.calving_k_table(shape, obs, fill='max', exclude=[ks.idxmax()])
    assert out.calving_k['d'] == pytest.approx(ks.drop(ks.idxmax()).max())
    assert out.calving_k[ks.idxmax()] == ks.max()
    with pytest.raises(InvalidParamsError):
        static.calving_k_table(shape, obs, fill='mode')


def test_the_static_fit_takes_any_flux_unit(fronts):
    geom, shape = fronts
    obs = {'a': 0.03, 'b': 0.6, 'c': 0.03}
    ref = static.calving_k_table(shape, obs, observed_err={'a': 0.003}, fill='median')
    assert ref.attrs['flux_unit'] == 'km3 yr-1'
    gt = {g: 0.9 * v for g, v in obs.items()}
    # observations in Gt yr-1 against a shape in km3 yr-1, or both in Gt yr-1
    for kwargs in (dict(shape=shape, shape_unit='km3 yr-1'),
                   dict(shape=static.shape_flux(geom, flux_unit='Gt yr-1', ice_density=900.))):
        out = static.calving_k_table(observed_flux=gt, observed_err={'a': 0.0027},
                                     fill='median', flux_unit='Gt yr-1', ice_density=900.,
                                     **kwargs)
        np.testing.assert_allclose(out.calving_k, ref.calving_k)
        np.testing.assert_allclose(out.calving_k_unc, ref.calving_k_unc)
        np.testing.assert_allclose(out.flux, 0.9 * ref.flux)
        np.testing.assert_allclose(out['shape'], 0.9 * ref['shape'])
        assert out.attrs['flux_unit'] == 'Gt yr-1'
    k, per = static.fit_calving_k(shape, gt, flux_unit='Gt yr-1', shape_unit='km3 yr-1',
                                  ice_density=900.)
    assert k == pytest.approx(static.fit_calving_k(shape, obs)[0])
    with pytest.raises(InvalidParamsError, match='ice density'):
        static.fit_calving_k(shape, gt, flux_unit='Gt yr-1', shape_unit='km3 yr-1')


def test_the_spread_of_a_filled_k_is_std_or_nmad(fronts):
    _, shape = fronts
    obs = {'a': 0.03, 'b': 0.6, 'c': 0.03}
    std = static.calving_k_table(shape, obs, fill='geometric_mean')
    logs = np.log(std.calving_k[list('abc')])
    assert std.calving_k_unc['d'] == std.calving_k['d'] * np.expm1(logs.std(ddof=1))
    nmad = static.calving_k_table(shape, obs, fill='geometric_mean', k_unc='nmad')
    want = 1.4826 * np.median(np.abs(logs - np.median(logs)))
    assert nmad.calving_k_unc['d'] == pytest.approx(nmad.calving_k['d'] * np.expm1(want))
    assert nmad.calving_k['d'] == std.calving_k['d']
    with pytest.raises(InvalidParamsError, match='k_unc'):
        static.calving_k_table(shape, obs, fill='median', k_unc='iqr')


def test_a_fill_can_be_regional(fronts):
    _, shape = fronts
    obs = {'a': 0.03, 'b': 0.6, 'c': 0.03}
    groups = dict(a='north', b='south', c='south', d='south', e='west')
    out = static.calving_k_table(shape, obs, fill='median', groups=groups)
    ks = out.calving_k
    assert ks['d'] == pytest.approx(np.median([ks['b'], ks['c']]))
    assert ks['e'] == pytest.approx(np.median([ks['a'], ks['b'], ks['c']]))  # no observed west


def test_a_k_table_overrides_and_carries_the_flux_of_its_run(fronts):
    _, shape = fronts
    obs = {'a': 0.03, 'b': 0.6, 'c': 0.03}
    tab = pd.DataFrame({'calving_k': [0.4, 0.7], 'flux': [0.9, np.nan]}, index=['a', 'd'])
    out = static.calving_k_table(shape, obs, k_table=tab)
    assert out.source['a'] == out.source['d'] == 'table'
    assert out.calving_k['a'] == 0.4 and out.flux['a'] == 0.9      # not k * shape
    assert out.flux['d'] == pytest.approx(0.7 * shape['d'])


def test_a_covariate_model_fills_the_rest(fronts):
    geom, shape = fronts
    # k proportional to the width: one log term recovers it
    obs = {g: geom.width_m[g] / 1000. * shape[g] for g in 'abc'}
    cov = geom.rename(columns={'width_m': 'width'}).reset_index()
    out = static.calving_k_table(shape, obs, terms=('width',), covariates=cov)
    assert out.calving_k['d'] == pytest.approx(3.)
    with pytest.raises(InvalidParamsError, match='covariates'):
        static.calving_k_table(shape, obs, terms=('width',))


# --- the ocean ---------------------------------------------------------------------

def test_k_scales_with_the_period_mean_forcing():
    tf = pd.Series({'a': 2., 'b': 4., 'c': np.nan, 'd': -1.})
    out = static.scale_k_from_ocean(0.6, tf, tf_ref=2., gamma=1.18)
    assert out.calving_k['a'] == pytest.approx(0.6)
    assert out.calving_k['b'] == pytest.approx(0.6 * 2 ** 1.18)
    assert out.calving_k['c'] == 0.6 and not out.scaled['c']       # no record: listed
    assert out.calving_k['d'] == 0.                                 # water below freezing
    # its own mean as the reference leaves k alone
    out = static.scale_k_from_ocean({'a': 0.5, 'b': 0.7}, tf[['a', 'b']])
    assert out.calving_k.to_dict() == {'a': 0.5, 'b': 0.7} and out.tf_ref['b'] == 4.
    # a background term that does not follow the ocean
    out = static.scale_k_from_ocean(0.6, tf[['b']], tf_ref=2., gamma=1., tf_fraction=0.5)
    assert out.calving_k['b'] == pytest.approx(0.6 * 1.5)
    # a reference per glacier, which every glacier with a record must have
    out = static.scale_k_from_ocean(0.6, tf[['a', 'b', 'c']], tf_ref={'a': 1., 'b': 4.},
                                    gamma=1.)
    assert out.calving_k.tolist() == pytest.approx([1.2, 0.6, 0.6])
    with pytest.raises(InvalidParamsError, match='no tf_ref'):
        static.scale_k_from_ocean(0.6, tf[['a', 'b']], tf_ref={'a': 1.})
    with pytest.raises(InvalidParamsError):
        static.scale_k_from_ocean(0.6, tf[['a']], tf_ref=0.)


def test_the_own_reference_is_the_power_mean_of_a_window(tmp_path):
    from tidewater.laws import tf_power_mean
    years = np.repeat(np.arange(1990, 2021), 12)
    tf = np.linspace(0.5, 3., years.size)
    assert static.parse_own('own:2000-2010') == (2000, 2010, None)
    assert static.parse_own('own:2000-2010:_ref') == (2000, 2010, '_ref')
    assert static.parse_own('1.5') is None
    win = tf[(years >= 2000) & (years <= 2010)]
    assert static.own_tf_ref(tf, years, 2000, 2010, 1.18) == tf_power_mean(win, 1.18)
    assert np.isnan(static.own_tf_ref(tf, years, 2030, 2040, 1.18))

    assert static.resolve_tf_ref('1.5') == 1.5 and static.resolve_tf_ref(2) == 2.
    assert static.resolve_tf_ref('') is None and static.resolve_tf_ref(None) is None
    assert static.resolve_tf_ref(np.nan) is None
    assert static.resolve_tf_ref('own:2000-2010', (years, tf), 1.18) == \
        tf_power_mean(win, 1.18)
    # a labelled record is fetched by its label
    records = {None: (years, tf), '_ref': (years, 2 * tf)}
    assert static.resolve_tf_ref('own:2000-2010:_ref', records.get, 1.) == \
        pytest.approx(2 * win.mean())
    with pytest.raises(InvalidParamsError):
        static.resolve_tf_ref('own:2000-2010')

    f = tmp_path / 'ref.csv'
    f.write_text('glacier_id,tf_ref\na,1.5\nb,2\n')
    assert static.read_tf_ref_map(f) == {'a': 1.5, 'b': 2.}


def test_the_own_window_includes_both_years():
    years = np.repeat([1999, 2000, 2001, 2002], 12) + (np.tile(np.arange(12), 4) + 0.5) / 12
    tf = np.repeat([9., 1., 3., 9.], 12)
    assert static.parse_own('own:2000-2001') == (2000, 2001, None)
    assert static.own_tf_ref(tf, years, 2000, 2001, 1.) == pytest.approx(2.)
    assert static.own_tf_ref(tf, np.floor(years), 2000, 2001, 1.) == pytest.approx(2.)


def test_the_larger_of_constant_keeps_the_total():
    h, k = 200., 0.6
    u_total = k / SEC_IN_YEAR * h
    u = u_total * np.array([0.1, 0.5, 1.5, 0.2, 0.8, 0.3])
    k_c, frac = static.larger_of_calving_constant(k, u, h)
    c = k_c / SEC_IN_YEAR * h
    assert 0 < k_c < k
    assert np.mean(np.maximum(c, u)) == pytest.approx(u_total)
    assert frac == pytest.approx(np.where(u > c, u, 0.).mean() / u_total)
