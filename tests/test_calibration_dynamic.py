"""The k correction returns the target, bounds its steps, and lists what it cannot correct."""
import numpy as np
import pandas as pd
import pytest

from tidewater.calibration import dynamic as dyn
from tidewater.calibration.dynamic import (best_calving_k, calibrate_dynamic_k,
                                           calving_k_for_glaciers_without_target,
                                           calving_k_step, propose_calving_k)
from tidewater.config import InvalidParamsError


def _step_all(prev, q_dyn, targets, **kwargs):
    return {gid: calving_k_step(k, q_dyn.get(gid), *targets.get(gid, (None, None)),
                                **kwargs)
            for gid, k in prev.items()}


@pytest.fixture
def k_case():
    prev = dict.fromkeys('abcdefg', 1.)
    q_dyn = {'a': 0.2, 'b': 0.103, 'c': 1.0, 'd': 0., 'f': 0.3, 'g': 0.12}
    targets = dict.fromkeys('abcde', (0.1, 0.001))
    return prev, q_dyn, targets


def test_a_k_step_is_the_ratio_bounded_and_listed(k_case):
    out = _step_all(*k_case)
    assert out['a']['calving_k'] == pytest.approx(0.5)  # target over control
    assert out['a']['rule'] == 'ratio' and out['a']['reason'] == ''
    assert out['b']['converged'] and out['b']['calving_k'] == 1.  # within 5 %
    assert out['c']['calving_k'] == pytest.approx(0.25)
    assert out['c']['reason'] == 'step_bounded'
    assert out['d']['reason'] == 'control_flux_zero' and out['d']['calving_k'] == 1.
    assert out['e']['reason'] == 'no_control_run' and out['e']['calving_k'] == 1.
    assert out['f']['reason'] == out['g']['reason'] == 'no_target'
    assert {r for r, o in out.items() if o['moved']} == {'a', 'c'}


def test_a_front_within_its_error_bar_is_still_corrected(k_case):
    prev, q_dyn, targets = k_case
    targets['a'] = (0.1, 0.11)
    out = _step_all(prev, q_dyn, targets)['a']
    assert out['within_error'] and not out['converged']
    assert out['calving_k'] == pytest.approx(0.5)


def test_a_target_that_is_not_positive_is_listed():
    out = calving_k_step(1., 0.2, 0.)
    assert out['reason'] == 'target_not_positive' and not out['moved']


def _k_loop(q_of_k, k0=3., target=0.1, passes=6):
    hist, k = [], k0
    for n in range(passes):
        q = q_of_k(k)
        out = calving_k_step(k, q, target, 0., hist)
        hist.append((k, q))
        if not out['moved']:
            break
        k = out['calving_k']
    return out, hist, n


def test_the_k_loop_converges_on_a_front_that_thickens():
    out, hist, n = _k_loop(lambda k: 0.2 * k ** 0.7, passes=5)
    assert out['converged'] and n < 5


@pytest.mark.parametrize('power', [0.4, 1.8, 2.5])
def test_a_bracket_closes_a_front_that_over_or_under_reacts(power):
    out, hist, n = _k_loop(lambda k: 0.2 * k ** power)
    assert out['converged'] and n <= 4
    out = calving_k_step(3., 0.05, 0.1, 0., [(9., 0.4)])
    assert out['rule'] == 'bracket' and 3. < out['calving_k'] < 9.


def test_the_best_pass_is_the_closest_in_log_space():
    out, hist, _ = _k_loop(lambda k: 0.03 if k < 2.5 else 0.3, passes=5)
    assert not out['converged'] and out['moved']
    k, q = best_calving_k(hist, 0.1)
    assert (k, q) in hist
    assert abs(np.log(q / 0.1)) == min(abs(np.log(qq / 0.1)) for _, qq in hist)
    assert best_calving_k(hist, None) is None
    assert best_calving_k([(1., 0.)], 0.1) is None
    # equally close passes: the earliest
    assert best_calving_k([(2., 0.2), (1., 0.2)], 0.1) == (2., 0.2)


def test_a_bracket_holds_when_the_response_is_not_monotonic():
    seen = [(1.077000, 0.0072322), (1.208033, 0.0065828), (1.262477, 0.0128236)]
    k, rule = propose_calving_k(1.208033, 0.0065828, 0.0109396, seen, 4.)
    assert rule == 'bracket' and 1.208033 < k < 1.262477


def test_a_bracket_that_repeats_a_constant_takes_the_midpoint():
    k, rule = propose_calving_k(2., 0.2, 0.1, [(1., 0.05), (1.0005, 0.2)], 4.)
    assert rule == 'bracket' and k == pytest.approx(np.sqrt(1.0005))


def test_the_response_exponent_is_used_only_where_it_is_sane():
    # Q = 0.2 k**2 measured twice above the target: the step is a square root
    k, rule = propose_calving_k(2., 0.8, 0.1, [(4., 3.2), (2., 0.8)], 100.)
    assert rule == 'response' and k == pytest.approx(2. * (0.1 / 0.8) ** 0.5)
    assert dyn.response_exponent([(4., 3.2), (2., 0.8)], 0.1) == pytest.approx(2.)
    # a response of 10 is not believed: the plain ratio
    k, rule = propose_calving_k(2., 0.8, 0.1, [(2.2, 0.8 * 1.1 ** 10), (2., 0.8)], 100.)
    assert rule == 'ratio' and k == pytest.approx(2. * 0.1 / 0.8)


def test_passes_at_the_same_constant_keep_their_order():
    seen = [(19.87, 0.00285), (19.87, 0.00345), (16.92, 0.00341), (19.33, 0.00337)]
    k, rule = propose_calving_k(19.33, 0.00337, 0.00294, seen)
    assert rule == 'bracket' and 19.33 < k < 19.87
    k, rule = propose_calving_k(19.33, 0.00337, 0.00294, [seen[i] for i in (1, 0, 2, 3)])
    assert rule == 'ratio' and k == pytest.approx(19.33 * 0.00294 / 0.00337)


def test_a_long_history_with_ties_is_sorted_stably():
    """Beyond 16 passes an unstable sort would reorder passes at one constant."""
    seen = [(1., 0.05)] * 20 + [(1., 0.2), (2., 0.3)]
    k, rule = propose_calving_k(2., 0.3, 0.1, seen)
    # the last pass at k = 1 is the neighbour of k = 2, and it is above the target
    assert rule == 'ratio'


def test_mean_frontal_ablation_reads_the_window():
    t = np.arange(1990, 2021)
    c = (t - 1990) * 1e9
    assert dyn.mean_frontal_ablation(t, c, (2000, 2010)) == pytest.approx(1.)
    assert dyn.mean_frontal_ablation(t, c, (2000, 2010), rho_ice=917.) == pytest.approx(0.917)
    assert np.isnan(dyn.mean_frontal_ablation(t, c, (2000, 2030)))
    # the default period is end-exclusive: ten years, read at 2000 and 2010
    c = np.cumsum(np.arange(31.)) * 1e9
    assert (dyn.mean_frontal_ablation(t, c, (2000, 2009), end='inclusive')
            == dyn.mean_frontal_ablation(t, c, (2000, 2010)))
    assert (dyn.mean_frontal_ablation(t, c, (2000, 2010), end='inclusive')
            == (c[21] - c[10]) / 11 / 1e9)
    assert (dyn.mean_frontal_ablation(t, c, (2000, 2010), flux_unit='Gt yr-1',
                                      ice_density=917.)
            == dyn.mean_frontal_ablation(t, c, (2000, 2010), rho_ice=917.))
    assert dyn.mean_frontal_ablation(t, c, (2000, 2010), flux_unit='m3 yr-1') == pytest.approx(
        1e9 * dyn.mean_frontal_ablation(t, c, (2000, 2010)))
    with pytest.raises(InvalidParamsError):
        dyn.mean_frontal_ablation(t, c, (2000, 2010), end='open')


def test_flux_from_series_reads_the_window():
    s = pd.DataFrame({'glacier_id': 'a', 'year': range(1990, 2021),
                      'calving_m3': np.arange(31) * 1e9})
    assert dyn.flux_from_series(s, 2000, 2010, rho_ice=900.)['a'] == pytest.approx(0.9)
    assert dyn.flux_from_series(s, 2000, 2010)['a'] == pytest.approx(1.)
    assert dyn.flux_from_series(s[s.year < 2005], 2000, 2010).empty
    other = s.rename(columns={'glacier_id': 'gid', 'year': 't', 'calving_m3': 'v'})
    q = dyn.flux_from_series(other, 2000, 2010, id_column='gid', time_column='t',
                             volume_column='v')
    assert q['a'] == pytest.approx(1.)
    assert dyn.flux_from_series(s, 2000, 2009, end='inclusive')['a'] == pytest.approx(1.)
    assert dyn.flux_from_series(s, 2000, 2010, flux_unit='Gt yr-1',
                                ice_density=900.)['a'] == pytest.approx(0.9)


# --- the same rule on tables, one pass per call ---------------------------------

def table(**cols):
    return pd.DataFrame(cols).set_index('glacier_id')


@pytest.fixture
def case():
    prev = table(glacier_id=list('abcdefg'), calving_k=[1.] * 7)
    q_dyn = pd.Series({'a': 0.2, 'b': 0.103, 'c': 1.0, 'd': 0., 'f': 0.3, 'g': 0.12})
    tgt = table(glacier_id=list('abcde'), target=[0.1] * 5, err=[0.001] * 5)
    return prev, q_dyn, tgt


def test_a_table_step_is_the_ratio_bounded_and_listed(case):
    df = dyn.step(*case)
    assert list(df.columns) == dyn.COLS
    df = df.set_index('glacier_id')
    assert df.calving_k['a'] == pytest.approx(0.5)
    assert df.converged['b'] and df.calving_k['b'] == 1.
    assert df.calving_k['c'] == pytest.approx(0.25) and df.reason['c'] == 'step_bounded'
    assert df.reason['d'] == 'control_flux_zero' and df.calving_k['d'] == 1.
    assert df.reason['e'] == 'no_control_run' and df.calving_k['e'] == 1.
    assert df.passes.to_dict() == dict(a=1, b=0, c=1, d=0, e=0, f=0, g=0)


def test_the_table_step_is_the_scalar_step(case):
    prev, q_dyn, tgt = case
    df = dyn.step(prev, q_dyn, tgt, k_fit_exclude=list('abcde')).set_index('glacier_id')
    for gid in 'abcde':
        one = calving_k_step(1., q_dyn.get(gid), 0.1, 0.001)
        assert df.calving_k[gid] == one['calving_k'] and df.rule[gid] == one['rule']
        assert df.reason[gid] == one['reason']


def test_ungated_fronts_take_the_geometric_mean_of_the_corrected(case):
    prev, q_dyn, tgt = case
    df = dyn.step(prev, q_dyn, tgt, k_fit_exclude=['d', 'e']).set_index('glacier_id')
    mean = np.exp(np.log([0.5, 1., 0.25]).mean())
    assert df.calving_k['f'] == pytest.approx(mean) and df.reason['f'] == 'no_target'
    # the target, else the control flux scaled to the new k
    assert df.flux['a'] == 0.1
    assert df.flux['f'] == pytest.approx(0.3 * mean)


def test_ungated_fronts_take_the_mean_of_their_group(case):
    prev, q_dyn, tgt = case
    groups = dict(a='north', b='south', c='south', d='north', e='north', f='north',
                  g='elsewhere')
    df = dyn.step(prev, q_dyn, tgt, k_fit_exclude=['d', 'e'],
                  groups=groups).set_index('glacier_id')
    assert df.calving_k['f'] == pytest.approx(0.5)          # only 'a' informs the north
    # a group without a calibrated constant takes the mean of all
    assert df.calving_k['g'] == pytest.approx(np.exp(np.log([0.5, 1., 0.25]).mean()))


def run_loop(q_of_k, k0=3., target=0.1, passes=6):
    prev, hist = table(glacier_id=['a'], calving_k=[k0]), None
    tgt = table(glacier_id=['a'], target=[target], err=[0.])
    for n in range(passes):
        df = dyn.step(prev, pd.Series({'a': q_of_k(prev.calving_k['a'])}), tgt, history=hist)
        hist = pd.concat([hist, dyn.points(df)])
        if df.converged[0]:
            break
        prev = df.set_index('glacier_id')
    return df, hist, n


def test_the_table_loop_converges_and_keeps_its_record():
    df, hist, n = run_loop(lambda k: 0.2 * k ** 0.7, passes=5)
    assert df.converged[0] and n < 5
    assert df.k_static[0] == 3. and df.passes[0] == n


@pytest.mark.parametrize('power', [0.4, 1.8, 2.5])
def test_the_table_loop_is_the_scalar_loop(power):
    df, hist, n = run_loop(lambda k: 0.2 * k ** power)
    out, shist, sn = _k_loop(lambda k: 0.2 * k ** power)
    assert df.converged[0] and n == sn
    assert hist.calving_k.tolist() == [k for k, _ in shist]


def test_the_best_pass_is_kept_for_a_front_that_does_not_close():
    df, hist, _ = run_loop(lambda k: 0.03 if k < 2.5 else 0.3, passes=5)
    assert not df.converged[0]
    final = dyn.best(df, hist).set_index('glacier_id')
    assert final.rule['a'] == 'best_pass'
    assert final.calving_k['a'] in set(hist.calving_k)
    q = hist.set_index('calving_k').q_dyn[final.calving_k['a']]
    assert abs(np.log(q / 0.1)) == np.abs(np.log(hist.q_dyn / 0.1)).min()
    assert 'within tolerance' in dyn.summary(final.reset_index())


def test_glaciers_without_a_target_take_the_geometric_mean():
    k = {'G0': 0.5, 'G1': 1., 'G2': 0.25, 'G3': 4., 'G4': 1., 'G5': 1.}
    tgt = {'G0': 0.1, 'G1': 0.1, 'G2': 0.1, 'G3': 0.1, 'G5': np.nan}
    mean = calving_k_for_glaciers_without_target(k, tgt, exclude=['G3'])
    assert mean == pytest.approx(np.exp(np.log([0.5, 1., 0.25]).mean()))
    fill = dyn.fill_without_target(k, tgt, exclude=['G3'])
    assert fill.to_dict() == {'G4': mean, 'G5': mean}
    assert calving_k_for_glaciers_without_target({'G4': 1.}, {}) is None
    assert dyn.fill_without_target({'G4': 1.}, {}).empty


def test_only_calibrated_fronts_inform_the_mean_when_asked(case):
    prev, q_dyn, tgt = case
    tgt.loc['c', 'target'] = 0.                    # not positive; e has no control run
    df = dyn.step(prev, q_dyn, tgt, max_step=4.).set_index('glacier_id')
    assert df.reason[list('cde')].tolist() == ['target_not_positive', 'control_flux_zero',
                                               'no_control_run']
    ks = df.calving_k
    assert ks['f'] == pytest.approx(np.exp(np.log(ks[list('abcde')]).mean()))
    df = dyn.step(prev, q_dyn, tgt, max_step=4., fill_from='calibrated').set_index('glacier_id')
    assert df.calving_k['f'] == pytest.approx(np.sqrt(ks['a'] * ks['b']))
    k = {'a': 0.5, 'b': 2., 'c': 8., 'd': 1.}
    t = {'a': 0.1, 'b': 0.1, 'c': -1.}
    assert calving_k_for_glaciers_without_target(k, t) == pytest.approx(2.)
    assert calving_k_for_glaciers_without_target(k, t, fill_from='calibrated') == pytest.approx(1.)
    assert calving_k_for_glaciers_without_target(
        k, t, fill_from='calibrated', reasons={'b': 'no_control_run'}) == 0.5
    with pytest.raises(InvalidParamsError, match='fill_from'):
        dyn.fill_without_target(k, t, fill_from='some')


# --- the whole search over a host's run -------------------------------------------

class Host:
    """A synthetic host whose control flux is `q_of_k(glacier, k)`, recording each call."""

    def __init__(self, q_of_k, fail_at=None):
        self.q_of_k, self.fail_at, self.calls = q_of_k, fail_at, []

    def __call__(self, ks):
        self.calls.append(dict(ks))
        if self.fail_at == len(self.calls):
            raise RuntimeError('the host stops')
        return {g: self.q_of_k(g, k) for g, k in ks.items()}


POWERS = {'a': 0.4, 'b': 0.7, 'c': 1., 'd': 1.8, 'e': 2.5}


def test_the_calibration_closes_power_law_fronts():
    host = Host(lambda g, k: 0.2 * k ** POWERS[g])
    out = calibrate_dynamic_k(host, dict.fromkeys(POWERS, 3.), dict.fromkeys(POWERS, 0.1),
                              dict.fromkeys(POWERS, 0.02))
    assert list(out.columns) == dyn.RESULT_COLS and out.index.name == 'glacier_id'
    assert out.converged.all() and out.within_error.all()
    assert (out.reason == '').all() and (out.rule == '').all()
    for g, p in POWERS.items():
        assert out.flux[g] == 0.2 * out.calving_k[g] ** p
        assert abs(out.flux[g] / 0.1 - 1) <= 0.05
        # each front follows the rule on its own, whatever the others do
        _, hist, _ = _k_loop(lambda k: 0.2 * k ** p, passes=10)
        assert [k for k, _ in out.history[g]] == [k for k, _ in hist]
        assert out.passes[g] == len(hist) - 1 == len(out.rules[g])
        # the target's 20 % error over the response of the front
        e = dyn.response_exponent(out.history[g], 0.1)
        assert out.calving_k_unc[g] == pytest.approx(out.calving_k[g] * 0.2 / e)
    over = calibrate_dynamic_k(Host(lambda g, k: 0.2 * k ** 2.5), {'e': 1.5}, {'e': 0.1})
    assert over.converged['e'] and 'bracket' in over.rules['e']
    # a converged front is not run again
    assert all(len(c) <= 5 for c in host.calls) and len(host.calls[-1]) < 5
    assert (out.k_static == 3.).all()


def test_the_calibration_ends_on_the_best_pass():
    host = Host(lambda g, k: 0.03 if k < 2.5 else 0.3)
    out = calibrate_dynamic_k(host, {'a': 3.}, {'a': 0.1}, maxiter=5).loc['a']
    assert out['rule'] == 'best_pass' and not out['converged']
    assert out['passes'] == 5
    hist = out['history']
    assert len(hist) == 6 and host.calls[-1] == {'a': out['calving_k']}
    assert abs(np.log(hist[-1][1] / 0.1)) == min(abs(np.log(q / 0.1)) for _, q in hist)
    assert out['flux'] == hist[-1][1]


def test_the_calibration_bounds_its_step():
    host = Host(lambda g, k: 4. * k)
    out = calibrate_dynamic_k(host, {'a': 1.}, {'a': 0.1}, maxiter=1).loc['a']
    assert host.calls == [{'a': 1.}]  # the closest pass is the one run: no rerun
    assert out['reason'] == 'step_bounded'
    assert out['rules'] == ['ratio'] and out['passes'] == 1
    assert out['rule'] == 'best_pass' and out['calving_k'] == 1.


def test_fronts_without_a_target_are_filled_not_run():
    host = Host(lambda g, k: 0.2 * k)
    k0 = dict(a=1., b=2., c=3., d=4., e=5.)
    out = calibrate_dynamic_k(host, k0, dict(a=0.1, b=0.3, c=np.nan, e=0.))
    assert all(set(c) <= {'a', 'b'} for c in host.calls)
    assert out.reason.to_dict() == dict(a='', b='', c='no_target', d='no_target',
                                        e='target_not_positive')
    # a target that is not positive keeps its first guess, and informs the mean with it
    mean = np.exp(np.log([out.calving_k['a'], out.calving_k['b'], 5.]).mean())
    assert out.calving_k['c'] == out.calving_k['d'] == pytest.approx(mean)
    assert out.rule['c'] == 'geometric_mean' and np.isnan(out.flux['c'])
    assert out.calving_k_unc['c'] > 0
    assert out.calving_k['e'] == 5. and out.k_static['c'] == 3.

    out = calibrate_dynamic_k(host, k0, dict(a=0.1, b=0.3), exclude=['b'], run_filled=True)
    assert out.calving_k['c'] == out.calving_k['a']
    assert out.flux['c'] == pytest.approx(0.2 * out.calving_k['a'])
    assert host.calls[-1] == dict.fromkeys('cde', out.calving_k['a'])

    out = calibrate_dynamic_k(host, k0, dict(a=0.1, b=0.3), fill=False)
    assert out.calving_k['c'] == 3. and out.rule['c'] == ''

    groups = dict(a='n', b='s', c='n', d='s', e='x')
    out = calibrate_dynamic_k(host, k0, dict(a=0.1, b=0.3), groups=groups)
    assert out.calving_k['c'] == out.calving_k['a']
    assert out.calving_k['d'] == out.calving_k['b']
    assert out.calving_k['e'] == pytest.approx(
        np.sqrt(out.calving_k['a'] * out.calving_k['b']))


def test_a_front_the_host_leaves_out_is_listed():
    out = calibrate_dynamic_k(lambda ks: {'a': 0.2 * ks['a']}, dict(a=1., b=1.),
                              dict(a=0.1, b=0.1), fill=False)
    assert out.converged['a'] and out.reason['b'] == 'no_control_run'
    assert out.calving_k['b'] == 1. and np.isnan(out.flux['b'])


def test_a_failed_pass_ends_the_calibration():
    with pytest.raises(RuntimeError, match='host stops'):
        calibrate_dynamic_k(Host(lambda g, k: 0.2 * k, fail_at=2), {'a': 3.}, {'a': 0.1})
    host = Host(lambda g, k: 0.2 * k, fail_at=2)
    out = calibrate_dynamic_k(host, {'a': 3.}, {'a': 0.1}, ignore_errors=True).loc['a']
    assert out['reason'] == 'no_control_run' and 'host stops' in out['error']
    # the constant that failed, its flux unknown
    assert host.calls[-1] == {'a': out['calving_k']} and np.isnan(out['history'][-1][1])
    with pytest.raises(InvalidParamsError):
        calibrate_dynamic_k(host, {'a': 3.}, {'a': 0.1}, maxiter=0)


def test_the_fill_rule_and_the_spread_are_options():
    host = Host(lambda g, k: 0.2 * k)
    k0 = dict(a=1., b=2., c=3., d=4., e=5., f=6.)
    tgt = dict(a=0.1, b=0.3, d=0.2, e=0.)
    out = calibrate_dynamic_k(host, k0, tgt)
    fit = out.calving_k[list('abde')]
    assert out.calving_k['c'] == pytest.approx(np.exp(np.log(fit).mean()))
    assert out.calving_k_unc['c'] == out.calving_k['c'] * np.expm1(np.log(fit).std())

    cal = calibrate_dynamic_k(host, k0, tgt, fill_from='calibrated', k_unc='nmad')
    fit = cal.calving_k[list('abd')]
    assert cal.calving_k['c'] == cal.calving_k['f'] == pytest.approx(np.exp(np.log(fit).mean()))
    logs = np.log(fit)
    nmad = 1.4826 * np.median(np.abs(logs - np.median(logs)))
    assert cal.calving_k_unc['c'] == pytest.approx(cal.calving_k['c'] * np.expm1(nmad))
    assert cal.calving_k['e'] == 5.                       # it keeps its first guess
    assert dyn.log_spread([1., np.e]) == pytest.approx(np.sqrt(0.5))
    assert np.isnan(dyn.log_spread([1.], 'nmad'))
    with pytest.raises(InvalidParamsError, match='k_unc'):
        calibrate_dynamic_k(host, k0, tgt, k_unc='iqr')

    # a front the host did not run does not inform the calibrated mean either
    part = calibrate_dynamic_k(lambda ks: {g: 0.2 * k for g, k in ks.items() if g != 'd'},
                               k0, tgt, fill_from='calibrated')
    assert part.reason['d'] == 'no_control_run'
    assert part.calving_k['c'] == pytest.approx(np.sqrt(part.calving_k['a'] * part.calving_k['b']))


def test_the_dynamic_fit_takes_any_flux_unit():
    k0, tgt, err = dict(a=1., b=2., c=3.), dict(a=0.1, b=0.3), dict(a=0.02)
    ref = calibrate_dynamic_k(Host(lambda g, k: 0.2 * k), k0, tgt, err)
    assert ref.attrs['flux_unit'] == 'Gt yr-1'
    # the host returns m3 s-1 of ice, the targets are km3 yr-1
    rho, sec = 900., 365 * 86400.
    host = Host(lambda g, k: 0.2 * k / 0.9 * 1e9 / sec)
    out = calibrate_dynamic_k(host, k0, {g: v / 0.9 for g, v in tgt.items()},
                              {g: v / 0.9 for g, v in err.items()}, flux_unit='km3 yr-1',
                              run_unit='m3 s-1')
    np.testing.assert_allclose(out.calving_k, ref.calving_k)
    np.testing.assert_allclose(out.calving_k_unc, ref.calving_k_unc)
    np.testing.assert_allclose(out.flux[['a', 'b']], ref.flux[['a', 'b']] / 0.9)
    assert out.attrs['flux_unit'] == 'km3 yr-1'
    mass = calibrate_dynamic_k(host, k0, tgt, err, run_unit='m3 s-1', ice_density=rho)
    np.testing.assert_allclose(mass.calving_k, ref.calving_k)
    with pytest.raises(InvalidParamsError, match='ice density'):
        calibrate_dynamic_k(host, k0, tgt, run_unit='m3 s-1')
    with pytest.raises(InvalidParamsError, match='unknown flux unit'):
        calibrate_dynamic_k(host, k0, tgt, flux_unit='Sv')

    prev = table(glacier_id=['a', 'b'], calving_k=[1., 1.])
    tgts = table(glacier_id=['a', 'b'], target=[0.1, 0.1], err=[0.01, 0.01])
    q = pd.Series({'a': 0.2, 'b': 0.05})
    one = dyn.step(prev, q, tgts)
    other = dyn.step(prev, q / 0.9, tgts, run_unit='km3 yr-1', ice_density=rho)
    pd.testing.assert_frame_equal(one, other)
