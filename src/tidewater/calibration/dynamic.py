"""Calibration of the calving constant through a host's dynamic model.

A constant fitted on the inverted front, ``k = Q_obs / flux(k=1)``, is not the
constant at which a dynamic model returns ``Q_obs``, because the run reshapes
the front. The update rule (:func:`propose_calving_k`):

- ``k <- k * (Q_target / Q_dyn) ** (1 / e)`` while every pass is on one side
  of the target, with ``e`` the response ``d ln Q / d ln k`` of the two passes
  closest to the target where it lies in [0.2, 3] (``rule = 'response'``),
  else 1 (``'ratio'``), and the factor bounded by ``max_step``;
- log-interpolated between the narrowest pair of neighbouring constants that
  straddles the target, once one exists (``'bracket'``).

A front that does not close ends on the constant of its closest pass
(``'best_pass'``). Glaciers without a target take the geometric mean of the
calibrated constants. :func:`calibrate_dynamic_k` runs the whole search over
a callable the host provides; :func:`step`, :func:`best` and :func:`finish`
are the same rule one pass at a time on tables, for hosts that run each pass
as a separate job.
"""
import numpy as np
import pandas as pd

from tidewater.config import InvalidParamsError
from tidewater.periods import as_period, run_bounds
from tidewater.units import FLUX_UNITS, GT_PER_YR, KM3_PER_YR, convert_flux

# The fronts best_calving_k may still move
OPEN_REASONS = ('', 'step_bounded')

# The fronts whose constant no pass corrected
NOT_CALIBRATED = ('target_not_positive', 'no_control_run', 'control_flux_zero')

FILL_FROM = ('all', 'calibrated')
K_UNC = ('std', 'nmad')

COLS = ['glacier_id', 'k_static', 'k_prev', 'calving_k', 'passes', 'q_target',
        'q_err', 'q_dyn', 'ratio', 'converged', 'within_error', 'reason',
        'rule', 'flux']

RESULT_COLS = ['calving_k', 'calving_k_unc', 'flux', 'k_static', 'q_target',
               'q_err', 'ratio', 'converged', 'within_error', 'reason', 'rule',
               'rules', 'passes', 'history', 'error']


def _num(v):
    return np.nan if v is None else float(v)


def _positive(history):
    """(k, q) pairs with both > 0 as an array, in the order they were measured."""
    a = np.asarray(history, dtype=float).reshape(-1, 2)
    return a[(a[:, 1] > 0) & (a[:, 0] > 0)]


def response_exponent(history, target, bounds=(0.2, 3.)):
    """The response d ln Q / d ln k of the two passes closest to the target.

    Parameters
    ----------
    history : list of (k, q) pairs
        k in yr-1, q in the unit of ``target``
    target : float
        the frontal ablation to match, > 0
    bounds : (float, float)
        the range a measured response is believed in

    Returns
    -------
    float
        the exponent, 1 when fewer than two distinct passes measured one
        inside ``bounds``
    """
    seen = _positive(history)
    near = seen[np.argsort(np.abs(np.log(seen[:, 1] / target)), kind='stable')]
    if len(near) > 1:
        (k0, q0), (k1, q1) = near[0].tolist(), near[1].tolist()
        if k0 != k1 and q0 != q1:
            slope = np.log(q1 / q0) / np.log(k1 / k0)
            if bounds[0] <= slope <= bounds[1]:
                return slope
    return 1.


def propose_calving_k(k_now, q_now, target, history, max_step=4.):
    """The next calving constant of one front, and the rule that gave it.

    Parameters
    ----------
    k_now : float
        the constant of the last control run, yr-1
    q_now : float
        the frontal ablation that run returned, > 0
    target : float
        the frontal ablation to match, > 0, in the unit of ``q_now``
    history : list of (k, q) pairs
        every pass measured on the front, the last one included
    max_step : float
        the largest factor between two constants outside a bracket

    Returns
    -------
    (float, str)
        the constant in yr-1 and ``'bracket'``, ``'response'`` or ``'ratio'``
    """
    seen = _positive(history)
    # The narrowest straddling pair of neighbours, so that a response that is
    # not monotonic never steps out of a bracket already found. Stable sorts:
    # passes at the same constant stay in the order they were measured.
    s = seen[np.argsort(seen[:, 0], kind='stable')]
    pairs = [(a, b) for a, b in zip(s[:-1], s[1:])
             if a[0] < b[0] and (a[1] - target) * (b[1] - target) < 0]
    if pairs:
        (k0, q0), (k1, q1) = min(pairs, key=lambda p: p[1][0] / p[0][0])
        w = np.log(target / q0) / np.log(q1 / q0)
        k = float(np.exp(np.log(k0) + w * (np.log(k1) - np.log(k0))))
        if np.isclose(seen[:, 0], k, rtol=1e-3).any():
            k = float(np.sqrt(k0 * k1))
        return k, 'bracket'
    two_sided = (seen[:, 1] < target).any() and (seen[:, 1] > target).any()
    e = 1. if two_sided else response_exponent(seen, target)
    factor = float(np.clip((target / q_now) ** (1 / e), 1 / max_step,
                           max_step))
    return k_now * factor, 'ratio' if e == 1 else 'response'


def calving_k_step(k_prev, q_dyn, target, target_err=None, history=(),
                   tolerance=0.05, max_step=4.):
    """One correction pass of one front.

    Parameters
    ----------
    k_prev : float
        the constant the control ran with, yr-1
    q_dyn : float or None
        the frontal ablation the control returned; None or NaN when it
        did not run
    target : float or None
        the frontal ablation to match, in the unit of ``q_dyn``; None or NaN
        for no target
    target_err : float or None
        the error of the target. A front within the larger of
        ``tolerance`` and ``target_err / target`` is ``within_error``.
    history : list of (k, q) pairs
        the passes measured on the front before this one
    tolerance : float
        relative mismatch within which the front is converged
    max_step : float
        see :func:`propose_calving_k`

    Returns
    -------
    dict
        ``calving_k`` (the next constant), ``rule``, ``reason`` (``''`` or
        one of ``no_target``, ``target_not_positive``, ``no_control_run``,
        ``control_flux_zero``, ``step_bounded``), ``ratio``, ``converged``,
        ``within_error`` and ``moved``
    """
    tgt, dyn, err = _num(target), _num(q_dyn), _num(target_err)
    gated = not np.isnan(tgt)
    ratio = dyn / tgt if tgt > 0 else np.nan
    rel_err = err / tgt if tgt > 0 else np.nan
    wide = max(tolerance, 0. if np.isnan(rel_err) else rel_err)
    converged = bool(abs(ratio - 1) <= tolerance)
    within_error = bool(abs(ratio - 1) <= wide)
    reason = ''
    if not gated:
        reason = 'no_target'
    elif not tgt > 0:
        reason = 'target_not_positive'
    elif np.isnan(dyn):
        reason = 'no_control_run'
    elif dyn <= 0:
        reason = 'control_flux_zero'
    moved = reason == '' and not converged
    k, rule = float(k_prev), ''
    if moved:
        k, rule = propose_calving_k(k_prev, dyn, tgt,
                                    list(history) + [(k_prev, dyn)],
                                    max_step=max_step)
        if (rule != 'bracket' and
                not 1 / max_step < k / k_prev < max_step):
            reason = 'step_bounded'
    return dict(calving_k=k, rule=rule, reason=reason, ratio=ratio,
                converged=converged, within_error=within_error, moved=moved)


def best_calving_k(history, target):
    """The pass whose control came closest to the target, in log space.

    Parameters
    ----------
    history : list of (k, q) pairs
        every pass measured on the front, k in yr-1
    target : float
        the frontal ablation to match, in the unit of q

    Returns
    -------
    (float, float) or None
        the (k, q) of that pass, the earliest of equally close ones, or None
        without a target > 0 or a positive control
    """
    h = np.asarray(history, dtype=float).reshape(-1, 2)
    h = h[h[:, 1] > 0]
    if not (target is not None and target > 0) or not len(h):
        return None
    i = int(np.argmin(np.abs(np.log(h[:, 1] / target))))
    return float(h[i, 0]), float(h[i, 1])


def geometric_mean(values):
    """Geometric mean of the positive finite values; None without any."""
    v = pd.Series(values, dtype=float)
    v = v[np.isfinite(v) & (v > 0)]
    if v.empty:
        return None
    return float(np.exp(np.log(v).mean()))


def log_spread(calving_k, how='std'):
    """Spread of the positive constants in log units; NaN below two.

    ``how``: 'std' (standard deviation, one degree of freedom removed) or
    'nmad' (1.4826 times the median absolute deviation from the median).
    """
    if how not in K_UNC:
        raise InvalidParamsError(f'unknown k_unc {how!r}; one of {K_UNC}')
    k = pd.Series(calving_k, dtype=float)
    if not (k > 0).sum() > 1:
        return np.nan
    logs = np.log(k[k > 0])
    if how == 'std':
        return logs.std()
    return float(1.4826 * (logs - logs.median()).abs().median())


def _informs(k, tgt, exclude=(), fill_from='all', reasons=None):
    """Mask of the fronts whose constant informs the others."""
    if fill_from not in FILL_FROM:
        raise InvalidParamsError(f'unknown fill_from {fill_from!r}; one of '
                                 f'{FILL_FROM}')
    keep = tgt.notna() & ~k.index.isin(list(exclude))
    if fill_from == 'calibrated':
        keep &= (tgt > 0)
        if reasons is not None:
            keep &= ~pd.Series(reasons).reindex(k.index).isin(
                NOT_CALIBRATED).to_numpy()
    return keep


def calving_k_for_glaciers_without_target(calving_k, targets, exclude=(),
                                          fill_from='all', reasons=None):
    """The geometric mean of the calibrated constants, for the other glaciers.

    Parameters
    ----------
    calving_k : mapping or :py:class:`pandas.Series`
        glacier id -> the constant the calibration left, yr-1
    targets : mapping or :py:class:`pandas.Series`
        glacier id -> the target; absent, None or NaN for no target
    exclude : sequence
        ids that keep their own constant but do not inform the mean
    fill_from : str
        'all' (default): every glacier with a target informs the mean, one
        that was not corrected with the constant it kept. 'calibrated': only
        those with a target > 0 and none of the ``NOT_CALIBRATED`` reasons.
    reasons : mapping, optional
        glacier id -> the ``reason`` of its last pass, for 'calibrated'

    Returns
    -------
    float
        the mean in yr-1, or None if no glacier has a calibrated constant
    """
    k = pd.Series(calving_k, dtype=float)
    tgt = pd.Series(targets, dtype=float).reindex(k.index)
    return geometric_mean(k[_informs(k, tgt, exclude, fill_from, reasons)])


def fill_without_target(calving_k, targets, exclude=(), groups=None,
                        fill_from='all', reasons=None):
    """The constant of each glacier without a target.

    The geometric mean of the calibrated constants of its group, or of all
    glaciers for a group without one, or when ``groups`` is None.

    Parameters
    ----------
    calving_k, targets, exclude, fill_from, reasons :
        see :func:`calving_k_for_glaciers_without_target`
    groups : mapping, optional
        glacier id -> group label (a region, a basin, ...)

    Returns
    -------
    :py:class:`pandas.Series`
        glacier id -> yr-1, for the glaciers without a target; empty if no
        glacier has a calibrated constant
    """
    k = pd.Series(calving_k, dtype=float)
    tgt = pd.Series(targets, dtype=float).reindex(k.index)
    k_all = calving_k_for_glaciers_without_target(k, tgt, exclude, fill_from,
                                                  reasons)
    rest = k.index[tgt.isna().to_numpy()]
    if k_all is None:
        return pd.Series(dtype=float)
    out = pd.Series(k_all, index=rest, dtype=float)
    if groups is not None:
        grp = pd.Series(groups).reindex(k.index)
        for label in grp[rest].dropna().unique():
            k_grp = calving_k_for_glaciers_without_target(
                k[(grp == label).to_numpy()], tgt, exclude, fill_from, reasons)
            if k_grp is not None:
                out[rest[(grp[rest] == label).to_numpy()]] = k_grp
    return out


def _convert(q, from_unit, to_unit, ice_density):
    """:func:`tidewater.units.convert_flux` that keeps a Series a Series."""
    out = convert_flux(q, from_unit, to_unit, ice_density=ice_density)
    if isinstance(q, pd.Series):
        return pd.Series(out, index=q.index, name=q.name)
    return out


def _volume_rate(q, rho_ice=None, flux_unit=None, ice_density=None):
    """km3 yr-1 of ice in the unit asked for."""
    if flux_unit is None:
        return q if rho_ice is None else q * (rho_ice / 1000.)
    return _convert(q, KM3_PER_YR, flux_unit,
                    rho_ice if ice_density is None else ice_density)


def mean_frontal_ablation(time, calving_m3, period, rho_ice=None,
                          end='exclusive', flux_unit=None, ice_density=None):
    """Mean frontal ablation of a run over ``period`` from its cumulative volume.

    ``period`` is end-exclusive by default: (2000, 2010) is the ten years
    2000 to 2009, read at the model times 2000 and 2010.

    Parameters
    ----------
    time : array
        the time axis of the run, model years
    calving_m3 : array
        cumulative frontal ablation, m3 of ice
    period : (y0, y1)
        the years to average over (:func:`tidewater.periods.as_period`)
    rho_ice : float, optional
        kg m-3. Given: Gt yr-1. None: km3 yr-1 of ice.
    end : str
        'exclusive' (default) or 'inclusive': whether ``y1`` is the model
        time the average stops at, or the last year it holds
    flux_unit : str, optional
        one of ``FLUX_UNITS``; overrides the unit rule of ``rho_ice``
    ice_density : float, optional
        kg m-3, for a mass ``flux_unit``. Default: ``rho_ice``.

    Returns
    -------
    float
        NaN when the run does not cover the period
    """
    t0, t1 = run_bounds(as_period(period, end))
    time = np.asarray(time)
    c = np.asarray(calving_m3, dtype=float)
    i0, i1 = np.flatnonzero(time == t0), np.flatnonzero(time == t1)
    if not (len(i0) and len(i1)):
        return np.nan
    q = float(c[i1[0]] - c[i0[0]]) / (t1 - t0) / 1e9
    return _volume_rate(q, rho_ice, flux_unit, ice_density)


def flux_from_series(series, y0, y1, rho_ice=None, id_column='glacier_id',
                     time_column='year', volume_column='calving_m3',
                     end='exclusive', flux_unit=None, ice_density=None):
    """:func:`mean_frontal_ablation` per glacier from a long table.

    Parameters
    ----------
    series : :py:class:`pandas.DataFrame`
        rows of glacier id, time and cumulative frontal ablation in m3 of ice
    y0, y1 :
        the period, on the time column; end-exclusive by default
    rho_ice : float, optional
        kg m-3. Given: Gt yr-1. None: km3 yr-1 of ice.
    id_column, time_column, volume_column : str
        the columns of ``series``
    end, flux_unit, ice_density :
        see :func:`mean_frontal_ablation`

    Returns
    -------
    :py:class:`pandas.Series`
        ``q_dyn`` by glacier id, in Gt yr-1, km3 yr-1 or ``flux_unit``; a
        glacier that misses either end is left out
    """
    y0, y1 = run_bounds(as_period((y0, y1), end))
    s = series[series[time_column].isin([y0, y1])].pivot_table(
        index=id_column, columns=time_column, values=volume_column)
    s = s.reindex(columns=[y0, y1]).dropna()
    q = (s[y1] - s[y0]) / (y1 - y0) / 1e9
    if rho_ice is not None or flux_unit is not None:
        q = _volume_rate(q, rho_ice, flux_unit, ice_density)
    return q.rename('q_dyn')


def _in_flux_unit(q, run_unit, flux_unit, ice_density):
    """The control flux of a host in the unit of the targets."""
    for unit in (flux_unit, run_unit or flux_unit):
        if unit not in FLUX_UNITS:
            raise InvalidParamsError(f'unknown flux unit {unit!r}; one of '
                                     f'{FLUX_UNITS}')
    if run_unit is None or run_unit == flux_unit:
        return q
    return _convert(q, run_unit, flux_unit, ice_density)


def points(table):
    """The (constant, control flux) a pass table measured, as history rows."""
    if 'glacier_id' not in table.columns:
        table = table.rename_axis('glacier_id').reset_index()
    return table[['glacier_id', 'k_prev', 'q_dyn']].rename(
        columns={'k_prev': 'calving_k'}).dropna()


def _histories(history):
    """glacier id -> list of (k, q), in the order of the rows."""
    if history is None or not len(history):
        return {}
    return {gid: list(zip(g['calving_k'], g['q_dyn']))
            for gid, g in history.groupby('glacier_id', sort=False)}


def step(prev, q_dyn, targets, max_step=4., tolerance=0.05, k_fit_exclude=(),
         history=None, groups=None, fill_from='all', flux_unit=GT_PER_YR,
         run_unit=None, ice_density=None):
    """One correction pass over a table of fronts.

    Parameters
    ----------
    prev : :py:class:`pandas.DataFrame`
        by glacier id: ``calving_k`` in yr-1 the control ran with, and
        ``k_static`` and ``passes`` when an earlier pass wrote them
    q_dyn : :py:class:`pandas.Series`
        glacier id -> the control's frontal ablation, in ``run_unit``
    targets : :py:class:`pandas.DataFrame`
        by glacier id: ``target`` and ``err``, in ``flux_unit``
    max_step, tolerance : float
        see :func:`calving_k_step`
    k_fit_exclude : sequence
        ids that do not inform the constant of the glaciers without a target
    history : :py:class:`pandas.DataFrame`, optional
        the rows of :func:`points` of the passes before
    groups, fill_from :
        see :func:`fill_without_target`
    flux_unit : str
        the unit of the targets and of the table, one of ``FLUX_UNITS``.
        Default: 'Gt yr-1'.
    run_unit : str, optional
        the unit of ``q_dyn``. Default: ``flux_unit``, which ``history``
        is always in.
    ice_density : float, optional
        kg m-3, needed when the two units differ in basis (mass, volume)

    Returns
    -------
    :py:class:`pandas.DataFrame`
        ``COLS``, one row per glacier of ``prev``: ``glacier_id``;
        ``k_static``, ``k_prev`` and ``calving_k`` in yr-1; ``q_target``,
        ``q_err``, ``q_dyn`` and ``flux`` in ``flux_unit``; ``ratio``
        (``q_dyn / q_target``) dimensionless; ``passes`` a count;
        ``converged`` and ``within_error`` booleans; ``reason`` and ``rule``
        text
    """
    df = pd.DataFrame(index=prev.index)
    df['k_static'] = prev['k_static'] if 'k_static' in prev else prev['calving_k']
    df['k_prev'] = prev['calving_k']
    df['passes'] = prev['passes'] if 'passes' in prev else 0
    df['q_target'] = targets['target'].reindex(df.index)
    df['q_err'] = targets['err'].reindex(df.index)
    df['q_dyn'] = _in_flux_unit(pd.Series(q_dyn, dtype=float), run_unit,
                                flux_unit, ice_density).reindex(df.index)
    seen = _histories(history)
    rows = [calving_k_step(df.at[gid, 'k_prev'], df.at[gid, 'q_dyn'],
                           df.at[gid, 'q_target'], df.at[gid, 'q_err'],
                           seen.get(gid, ()), tolerance=tolerance,
                           max_step=max_step) for gid in df.index]
    out = pd.DataFrame(rows, index=df.index, columns=[
        'calving_k', 'rule', 'reason', 'ratio', 'converged', 'within_error',
        'moved'])
    for col in ('ratio', 'converged', 'within_error', 'reason', 'calving_k',
                'rule'):
        df[col] = out[col]
    df.loc[out['moved'].to_numpy(dtype=bool), 'passes'] += 1
    return finish(df, k_fit_exclude, groups=groups, fill_from=fill_from)


def finish(df, k_fit_exclude=(), groups=None, fill_from='all'):
    """Fill the constants of the glaciers without a target and the flux column.

    ``flux`` is the target, else the control's scaled to the new constant.
    Returns ``COLS`` in order, in the units of ``df``.
    """
    tgt, dyn = df['q_target'], df['q_dyn']
    gated = tgt.notna()
    fill = fill_without_target(df['calving_k'], tgt, k_fit_exclude, groups,
                               fill_from, df.get('reason'))
    df.loc[fill.index, 'calving_k'] = fill
    df['flux'] = tgt.where(gated & (tgt > 0)).fillna(
        dyn * df['calving_k'] / df['k_prev'])
    return df.rename_axis('glacier_id').reset_index()[COLS]


def best(table, history, k_fit_exclude=(), groups=None, fill_from='all'):
    """``table`` with every front outside the tolerance back on its closest pass.

    Parameters
    ----------
    table : :py:class:`pandas.DataFrame`
        as returned by :func:`step`
    history : :py:class:`pandas.DataFrame`
        the rows of :func:`points` of every pass, in the unit of ``table``

    Returns
    -------
    :py:class:`pandas.DataFrame`
        ``COLS``; the moved fronts have ``rule = 'best_pass'``
    """
    df = table.set_index('glacier_id')
    seen = _histories(history)
    for gid in df.index:
        row = df.loc[gid]
        if (row['reason'] not in OPEN_REASONS or row['converged']
                or pd.isna(row['q_target'])):
            continue
        pick = best_calving_k(seen.get(gid, ()), row['q_target'])
        if pick is not None:
            df.at[gid, 'calving_k'] = pick[0]
            df.at[gid, 'rule'] = 'best_pass'
    return finish(df, k_fit_exclude, groups=groups, fill_from=fill_from)


def summary(df):
    """One line on a pass table: how many fronts closed, and why not."""
    g = df[df['q_target'].notna()]
    ok = g['converged']
    return (f'{int(ok.sum())} of {len(g)} fronts with a target within '
            f'tolerance, {int(g["within_error"].sum())} within their error; '
            f'sum ratio {g["q_dyn"].sum() / g["q_target"].sum():.3f}, median '
            f'{g["ratio"].median():.3f}; not converged by reason: '
            f'{g.loc[~ok, "reason"].replace("", "moved").value_counts().to_dict()}')


def calving_k_uncertainty(calving_k, target, target_err, history=()):
    """The error of a calibrated constant from the error of its target, yr-1.

    ``k * (target_err / target) / e`` with ``e`` the measured response
    (:func:`response_exponent`); NaN without a target error.
    """
    tgt, err = _num(target), _num(target_err)
    if not tgt > 0 or np.isnan(err):
        return np.nan
    return float(calving_k * (err / tgt) / response_exponent(history, tgt))


def calibrate_dynamic_k(run_flux, k0, targets, target_err=None,
                        tolerance=0.05, max_step=4., maxiter=10,
                        ignore_errors=False, exclude=(), groups=None,
                        fill=True, run_filled=False, fill_from='all',
                        k_unc='std', flux_unit=GT_PER_YR, run_unit=None,
                        ice_density=None):
    """Calibrate the calving constants so that a dynamic model returns the
    observed frontal ablation.

    Each pass calls ``run_flux`` with the constants of the fronts still open
    and takes the next ones from :func:`calving_k_step`. A front stops within
    ``tolerance`` of its target, when a pass cannot be corrected, or after
    ``maxiter`` passes; one still outside the tolerance then goes back to its
    closest pass (:func:`best_calving_k`), run once more so that the host
    holds that run. A glacier without a target, or with one not > 0, is not
    run.

    Parameters
    ----------
    run_flux : callable
        ``run_flux(k_by_glacier)`` runs the host's control with these
        constants (dict glacier id -> yr-1) and returns glacier id -> mean
        frontal ablation over the period of the targets, in ``run_unit``; a
        glacier it leaves out, or returns NaN for, did not run
    k0 : mapping or :py:class:`pandas.Series`
        glacier id -> first guess, yr-1
    targets : mapping or :py:class:`pandas.Series`
        glacier id -> observed frontal ablation, in ``flux_unit``; absent or
        NaN: no target
    target_err : mapping or :py:class:`pandas.Series`, optional
        glacier id -> its error, same unit
    tolerance : float
        relative mismatch within which a front is converged
    max_step : float
        the largest factor between two constants outside a bracket
    maxiter : int
        the largest number of passes, before the rerun of the closest one
    ignore_errors : bool
        if True, a pass in which ``run_flux`` raises counts as not run for
        its fronts (``no_control_run``); else the error is raised
    exclude : sequence
        ids that do not inform the constant of the glaciers without a target
    groups : mapping, optional
        glacier id -> group label; glaciers without a target then take the
        geometric mean of their group (:func:`fill_without_target`)
    fill : bool
        give the glaciers without a target that mean (``rule =
        'geometric_mean'``)
    run_filled : bool
        run those glaciers once at their constant, for the ``flux`` column
    fill_from : str
        which glaciers inform that mean and its spread. 'all' (default):
        every glacier with a target, one with a target not > 0 or without a
        control run with the constant it kept. 'calibrated': only those a
        pass could correct (none of the ``NOT_CALIBRATED`` reasons).
    k_unc : str
        the spread behind ``calving_k_unc`` of a filled glacier, in log
        units: 'std' (default) or 'nmad' (:func:`log_spread`)
    flux_unit : str
        the unit of the targets, their errors and the table, one of
        ``FLUX_UNITS``. Default: 'Gt yr-1'.
    run_unit : str, optional
        the unit ``run_flux`` returns. Default: ``flux_unit``.
    ice_density : float, optional
        kg m-3, needed when the two units differ in basis (mass, volume)

    Returns
    -------
    :py:class:`pandas.DataFrame`
        by glacier id, with ``attrs['flux_unit']``: ``calving_k`` (yr-1),
        ``calving_k_unc`` (yr-1; :func:`calving_k_uncertainty`, or the mean
        times ``expm1`` of the log spread for a filled glacier), ``flux``
        (``flux_unit``, the control at the final constant), ``k_static``
        (yr-1), ``q_target`` and ``q_err`` (``flux_unit``), ``ratio``
        (``flux / q_target``, dimensionless), ``converged`` and
        ``within_error`` (booleans), ``reason`` (text), ``rule``
        (``'best_pass'``, ``'geometric_mean'``, else the rule of the last
        correction, ``''`` if the front converged), ``rules`` (list),
        ``passes`` (count), ``history`` (every ``[k, q]`` run, yr-1 and
        ``flux_unit``) and ``error`` (text)
    """
    if maxiter < 1:
        raise InvalidParamsError('maxiter must be at least 1')
    _in_flux_unit(np.nan, run_unit, flux_unit, ice_density)
    log_spread((), k_unc)
    k0 = pd.Series(k0, dtype=float)
    tgt = pd.Series(targets, dtype=float).reindex(k0.index)
    err = (pd.Series(target_err, dtype=float).reindex(k0.index)
           if target_err is not None else pd.Series(np.nan, index=k0.index))
    ids = list(k0.index)
    history = {g: [] for g in ids}
    rules = {g: [] for g in ids}
    errors = {g: [] for g in ids}
    last = {}

    def one_pass(ks):
        try:
            got = run_flux(dict(ks))
            got = {g: _in_flux_unit(_num(got.get(g)), run_unit, flux_unit,
                                    ice_density) for g in ks}
        except Exception as e:  # noqa: BLE001
            if not ignore_errors:
                raise
            for g in ks:
                errors[g].append(f'{type(e).__name__}: {e}'[:200])
            got = dict.fromkeys(ks, np.nan)
        for g, k in ks.items():
            history[g].append([float(k), got[g]])
        return got

    def one_step(g, k, q, **kwargs):
        return calving_k_step(k, q, tgt[g], err[g], tolerance=tolerance,
                              **kwargs)

    for g in ids:
        if not tgt[g] > 0:
            last[g] = one_step(g, k0[g], None)
    k_now = {g: float(k0[g]) for g in ids if g not in last}
    for _ in range(maxiter):
        if not k_now:
            break
        q = one_pass(k_now)
        nxt = {}
        for g, k in k_now.items():
            last[g] = one_step(g, k, q[g], history=history[g][:-1],
                               max_step=max_step)
            if last[g]['moved']:
                rules[g].append(last[g]['rule'])
                nxt[g] = last[g]['calving_k']
        k_now = nxt

    final, rule, rerun = {}, {}, {}
    for g in ids:
        final[g] = history[g][-1] if history[g] else [float(k0[g]), np.nan]
        rule[g] = last[g]['rule']
        if (history[g] and last[g]['reason'] in OPEN_REASONS
                and not last[g]['converged']):
            final[g] = list(best_calving_k(history[g], tgt[g]))
            rule[g] = 'best_pass'
            if final[g][0] != history[g][-1][0]:
                rerun[g] = final[g][0]
    if rerun:
        q = one_pass(rerun)
        for g in rerun:
            final[g][1] = q[g]

    rows = {}
    for g in ids:
        k, q = final[g]
        end = one_step(g, k, q)
        reason = last[g]['reason']
        if errors[g] and np.isnan(q):
            reason = 'no_control_run'
        rows[g] = dict(
            calving_k=k, flux=q, k_static=float(k0[g]), q_target=tgt[g],
            q_err=err[g], ratio=end['ratio'], converged=end['converged'],
            within_error=end['within_error'], reason=reason, rule=rule[g],
            rules=rules[g], passes=len(rules[g]), history=history[g],
            error='; '.join(errors[g]),
            calving_k_unc=calving_k_uncertainty(k, tgt[g], err[g],
                                                history[g]))
    out = pd.DataFrame.from_dict(rows, orient='index')[RESULT_COLS]
    out.index.name = k0.index.name or 'glacier_id'
    out.attrs['flux_unit'] = flux_unit

    if fill:
        filled = fill_without_target(out['calving_k'], tgt, exclude, groups,
                                     fill_from, out['reason'])
        if len(filled):
            fit = out['calving_k'][_informs(out['calving_k'], tgt, exclude,
                                            fill_from,
                                            out['reason']).to_numpy()]
            spread = log_spread(fit, k_unc)
            out.loc[filled.index, 'calving_k'] = filled
            out.loc[filled.index, 'calving_k_unc'] = filled * np.expm1(spread)
            out.loc[filled.index, 'rule'] = 'geometric_mean'
            if run_filled:
                ks = filled.to_dict()
                q = one_pass(ks)
                for g in ks:
                    out.at[g, 'flux'] = q[g]
                    out.at[g, 'history'] = history[g]
                    out.at[g, 'error'] = '; '.join(errors[g])
    return out
