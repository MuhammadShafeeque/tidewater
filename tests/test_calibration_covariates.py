"""Covariate models of k: the fit, the search and its permutation test."""
import numpy as np
import pandas as pd
import pytest

from tidewater.calibration import covariates as cv
from tidewater.calibration.permutation import permutation_test
from tidewater.config import SEC_IN_DAY, InvalidWorkflowError


def test_calving_k_model_is_the_geometric_mean_without_terms():
    rows = [dict(glacier_id=i, k=k, width=w)
            for i, (k, w) in enumerate([(0.5, 100.), (2., 400.), (1., 200.), (4., 800.)])]
    model = cv.fit_calving_k_model(rows)
    np.testing.assert_allclose(cv.calving_k_from_model(model, {}),
                               np.exp(np.mean(np.log([0.5, 2., 1., 4.]))))
    # k = width / 200 here, which one log term recovers exactly
    model = cv.fit_calving_k_model(rows, terms=('width',))
    np.testing.assert_allclose(cv.calving_k_from_model(model, {'width': 1000.}), 5.)
    np.testing.assert_allclose(model['residual'], 0., atol=1e-10)
    assert model['glacier_ids'] == [0, 1, 2, 3] and model['n'] == 4
    with pytest.raises(InvalidWorkflowError):
        cv.fit_calving_k_model(rows[:2], terms=('width',))


def test_which_terms_enter_as_logs_is_the_callers_choice():
    rows = [dict(glacier_id=i, k=np.exp(0.01 * s), my_slope=s)
            for i, s in enumerate([1., 5., 9., 20.])]
    model = cv.fit_calving_k_model(rows, terms=('my_slope',))          # linear by default
    np.testing.assert_allclose(model['coefficients'], [0., 0.01], atol=1e-12)
    rows = [dict(r, k=r['my_slope'] ** 2) for r in rows]
    model = cv.fit_calving_k_model(rows, terms=('my_slope',), log_terms={'my_slope'})
    np.testing.assert_allclose(model['coefficients'], [0., 2.], atol=1e-12)
    np.testing.assert_allclose(cv.calving_k_from_model(model, {'my_slope': 3.}), 9.)


def test_rows_without_a_k_or_a_term_are_left_out_and_frames_are_rows():
    df = pd.DataFrame({'k': [0.5, 2., 1., 4., np.nan, 3.],
                       'width': [100., 400., 200., 800., 300., np.nan]},
                      index=pd.Index(list('abcdef'), name='glacier_id'))
    model = cv.fit_calving_k_model(df, terms=('width',))
    assert model['glacier_ids'] == list('abcd')
    assert set(cv.fit_calving_k_model(df)['glacier_ids']) == set('abcdf')
    np.testing.assert_allclose(model['loo_residual'], 0., atol=1e-10)
    assert model['loo_sd'] == pytest.approx(0., abs=1e-10)


def test_the_covariates_of_a_front():
    row = cv.calving_covariates('G1', 150., thick=200., width=2000., observed_flux=0.03,
                                q_sg=10., tf=3., area_km2=40., my_slope=0.1)
    assert row['glacier_id'] == 'G1'
    assert row['shape_km3'] == 200. * 150. * 2000. / 1e9
    assert row['k'] == 0.03 / row['shape_km3'] and row['obs_flux_km3'] == 0.03
    assert row['free_board'] == 50. and row['depth_fraction'] == 0.75
    assert row['aspect_ratio'] == 10.
    assert row['thick_flotation'] == pytest.approx(150. * 1028. / 900.)
    assert row['flotation_ratio'] == pytest.approx(200. / row['thick_flotation'])
    assert row['q_sg'] == pytest.approx(10. * SEC_IN_DAY / (2000. * 200.))
    assert row['melt_rate'] == pytest.approx(
        (3e-4 * 150. * row['q_sg'] ** 0.39 + 0.15) * 3. ** 1.18)
    assert row['area_km2'] == 40. and row['my_slope'] == 0.1         # the host's own

    # the thickness from the free board; no ocean and no observation: no such keys
    row = cv.calving_covariates('G2', 150., surface_h=50., width=2000., rho_ice=917.)
    assert row['front_thickness'] == 200. and np.isnan(row['q_sg'])
    assert row['thick_flotation'] == pytest.approx(150. * 1028. / 917.)
    assert not {'tf', 'melt_rate', 'k'} & set(row)


@pytest.fixture
def sample():
    """Thirty fronts whose k follows the width, with noise; two terms carry nothing."""
    rng = np.random.default_rng(7)
    n = 30
    df = pd.DataFrame({'glacier_id': [f'G{i}' for i in range(n)],
                       'width': np.exp(rng.normal(7.5, 0.6, n)),
                       'water_depth': np.exp(rng.normal(5., 0.4, n)),
                       'noise': rng.normal(0., 1., n),
                       'flat': 1.,
                       'gappy': np.r_[np.nan, rng.normal(0., 1., n - 1)]})
    df['shape_km3'] = 250. * df.water_depth * df.width / 1e9
    df['k'] = 0.002 * df.width ** 0.8 * np.exp(rng.normal(0., 0.15, n))
    df['obs_flux_km3'] = df.k * df.shape_km3
    return df


TERMS = ('width', 'water_depth', 'noise', 'flat', 'gappy')


def test_only_terms_that_are_finite_and_vary_are_searched(sample):
    assert cv.usable_terms(sample, TERMS) == ['width', 'water_depth', 'noise']
    assert cv.term_sets(['a', 'b', 'c'], 2) == [(), ('a',), ('b',), ('c',), ('a', 'b'),
                                               ('a', 'c'), ('b', 'c')]
    assert cv.term_sets(['a', 'b'], 0) == [()]


def test_the_pooled_baseline_leaves_each_glacier_out():
    shape, flux = np.array([1., 2., 4.]), np.array([2., 2., 2.])
    res = cv.pooled_loo_residuals(shape, flux)
    np.testing.assert_allclose(res[0], np.log((4. / 6.) / 2.))
    np.testing.assert_allclose(res[2], np.log((4. / 3.) / 0.5))
    # one k everywhere: nothing to miss
    np.testing.assert_allclose(cv.pooled_loo_residuals(shape, 0.7 * shape), 0., atol=1e-12)


def test_the_search_ranks_the_true_term_first_and_against_the_pooled_k(sample):
    out = cv.search_calving_k_models(sample, TERMS, max_terms=2)
    assert len(out) == 1 + 7                                   # pooled + the term sets
    assert out.loo_sd.is_monotonic_increasing
    assert 'width' in out.terms[0]
    one = out[out.terms == 'width'].iloc[0]
    assert one.loo_sd == cv.fit_calving_k_model(sample, ('width',))['loo_sd']
    assert one.loo_sd < 0.2 and one.vs_pooled_pct < 0 and len(one.coefficients) == 2
    assert one.coefficients[1] == pytest.approx(0.8, abs=0.15)
    pooled = out[out.terms == '(pooled k)'].iloc[0]
    assert pooled.vs_pooled_pct == 0.
    assert pooled.loo_sd == pytest.approx(
        cv.pooled_loo_residuals(sample.shape_km3, sample.obs_flux_km3).std())
    assert out.loo_sd.min() == cv.best_score(sample, cv.usable_terms(sample, TERMS), 2)
    # a baseline given is the baseline used
    out = cv.search_calving_k_models(sample, ('width',), max_terms=1, pooled_sd=1.)
    assert out[out.terms == 'width'].vs_pooled_pct.iloc[0] == pytest.approx(
        100 * (one.loo_sd - 1.))


def test_the_permutation_test_separates_signal_from_search_luck(sample):
    got = permutation_test(sample, ['width', 'water_depth', 'noise'], seed=3, n=60)
    assert got['n'] == 30 and got['n_terms'] == 3 and got['null'].shape == (60,)
    assert got['best'] == cv.best_score(sample, ['width', 'water_depth', 'noise'], 2)
    assert got['p_value'] == pytest.approx(1 / 61)             # no shuffle does as well
    # without the term that carries the signal the search finds nothing
    flat = permutation_test(sample, ['noise'], seed=3, n=60, max_terms=1)
    assert flat['p_value'] > 0.1
    assert flat['p_value'] == (np.sum(flat['null'] <= flat['best']) + 1) / 61


def test_the_permutation_test_follows_its_seed(sample):
    a = permutation_test(sample, ['width', 'noise'], seed=11, n=20, max_terms=1)
    b = permutation_test(sample, ['width', 'noise'], seed=11, n=20, max_terms=1)
    c = permutation_test(sample, ['width', 'noise'], seed=12, n=20, max_terms=1)
    np.testing.assert_array_equal(a['null'], b['null'])
    assert not np.array_equal(a['null'], c['null'])
    # a generator is drawn from, so two tests share one stream
    rng = np.random.default_rng(11)
    d = permutation_test(sample, ['width', 'noise'], seed=rng, n=20, max_terms=1)
    e = permutation_test(sample, ['width', 'noise'], seed=rng, n=20, max_terms=1)
    np.testing.assert_array_equal(a['null'], d['null'])
    assert not np.array_equal(d['null'], e['null'])
    # rows without a k take no part
    more = pd.concat([sample, sample.head(3).assign(k=np.nan)])
    assert permutation_test(more, ['width'], seed=1, n=5)['n'] == 30


def test_the_observed_flux_of_a_front_takes_any_unit_and_law_result(monkeypatch):
    ref = cv.calving_covariates('a', 150., thick=200., width=2000., observed_flux=0.03)
    gt = cv.calving_covariates('a', 150., thick=200., width=2000., observed_flux=0.027,
                               flux_unit='Gt yr-1', ice_density=900.)
    assert gt['k'] == pytest.approx(ref['k'])
    assert gt['obs_flux_km3'] == pytest.approx(0.03)
    raised = cv.calving_covariates('a', 150., surface_h=60., width=2000., sea_level=10.)
    assert raised['front_thickness'] == 200.

    # the law result is read as a mapping or as an object, under either naming
    class Result:
        flux, width, thickness, water_depth = 0.06, 2000., 200., 150.
        freeboard, flotation_thickness = 50., ref['thick_flotation']

    as_dict = dict(flux=0.06, width=2000., thick=200., water_depth=150., free_board=50.,
                   thick_flotation=ref['thick_flotation'])
    for result in (Result(), as_dict):
        monkeypatch.setattr(cv, 'calving_law_flux', lambda *a, **k: result)
        assert cv.calving_covariates('a', 150., thick=200., width=2000.,
                                     observed_flux=0.03) == ref
