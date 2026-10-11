"""Covariate models of the calving constant, scored by leave-one-out."""
import itertools
from collections.abc import Mapping

import numpy as np
import pandas as pd

from tidewater.config import SEC_IN_DAY, InvalidWorkflowError
from tidewater.laws import calving_law_flux, submarine_melt_rate
from tidewater.units import KM3_PER_YR, convert_flux

# Covariates that enter a model of log k as logs, unless the caller says otherwise
LOG_TERMS = frozenset(('water_depth', 'front_thickness', 'width', 'area_km2',
                       'free_board', 'aspect_ratio', 'flowline_length', 'q_sg',
                       'tf', 'melt_rate', 'shape_km3'))


def _field(result, *names):
    """One value of a law result, a mapping or an object, under any name."""
    for name in names:
        if isinstance(result, Mapping):
            if name in result:
                return result[name]
        elif hasattr(result, name):
            return getattr(result, name)
    raise KeyError(names[0])


def calving_covariates(glacier_id, water_depth, thick=None, width=None,
                       surface_h=None, observed_flux=None, q_sg=None, tf=None,
                       rho_ocean=None, rho_ice=None, melt_kwargs=None,
                       flux_unit=KM3_PER_YR, ice_density=None,
                       sea_level=None, **extra):
    """Candidate predictors of one glacier's calving constant.

    Parameters
    ----------
    glacier_id : str
        the glacier
    water_depth : float
        at the front, m
    thick : float, optional
        the terminus thickness, m. Default: free board plus depth, from
        ``surface_h``.
    width : float
        of the front, m
    surface_h : float, optional
        surface elevation of the terminus, m, in the datum of ``sea_level``
    observed_flux : float, optional
        observed frontal ablation, in ``flux_unit``; adds the constant ``k``
        that reproduces it
    q_sg : float, optional
        mean subglacial discharge, m3 s-1 of water
    tf : float, optional
        mean thermal forcing, degC; adds ``melt_rate`` in m d-1
    rho_ocean, rho_ice : float, optional
        kg m-3. Default: 1028 and 900.
    melt_kwargs : dict, optional
        passed to :func:`tidewater.laws.submarine_melt_rate`
    flux_unit : str
        the unit of ``observed_flux``, one of
        ``tidewater.units.FLUX_UNITS``. Default: 'km3 yr-1'.
    ice_density : float, optional
        kg m-3, for a mass ``flux_unit``. Default: ``rho_ice``.
    sea_level : float, optional
        m; passed to :func:`tidewater.laws.calving_law_flux` when given
    **extra
        any further covariate the host knows (area, slope, length, ...)

    Returns
    -------
    dict
        one row: ``glacier_id``; ``shape_km3`` (the flux at k = 1, km3 yr-1
        of ice per yr-1); ``water_depth``, ``front_thickness``,
        ``free_board``, ``thick_flotation`` and ``width`` in m;
        ``flotation_ratio``, ``depth_fraction`` and ``aspect_ratio``
        dimensionless; ``q_sg`` in m d-1 of water over the front area;
        ``tf`` in degC and ``melt_rate`` in m d-1 with ``tf``;
        ``obs_flux_km3`` (km3 yr-1 of ice) and ``k`` (yr-1) with
        ``observed_flux``; ``extra`` as given
    """
    level = {} if sea_level is None else {'sea_level': sea_level}
    shape = calving_law_flux(water_depth, 1., thick=thick, width=width,
                             surface_h=surface_h, rho_ocean=rho_ocean,
                             rho_ice=rho_ice, **level)
    shape_km3 = _field(shape, 'flux')
    out = dict(glacier_id=glacier_id, shape_km3=shape_km3,
               water_depth=_field(shape, 'water_depth'),
               front_thickness=_field(shape, 'thickness', 'thick'),
               free_board=_field(shape, 'freeboard', 'free_board'),
               thick_flotation=_field(shape, 'flotation_thickness',
                                      'thick_flotation'),
               width=_field(shape, 'width'), **extra)
    # below one, the front is thinner than flotation
    out['flotation_ratio'] = (out['front_thickness'] / out['thick_flotation']
                              if out['thick_flotation'] > 0 else np.nan)
    out['depth_fraction'] = out['water_depth'] / out['front_thickness']
    out['aspect_ratio'] = out['width'] / out['front_thickness']

    front_area = out['width'] * out['front_thickness']
    out['q_sg'] = (float(q_sg) * SEC_IN_DAY / front_area
                   if q_sg is not None and front_area > 0 else np.nan)
    if tf is not None:
        out['tf'] = float(tf)
        out['melt_rate'] = submarine_melt_rate(out['water_depth'], out['q_sg'],
                                               out['tf'], **(melt_kwargs or {}))
    if observed_flux is not None:
        observed_flux = convert_flux(
            observed_flux, flux_unit, KM3_PER_YR,
            ice_density=rho_ice if ice_density is None else ice_density)
        out['obs_flux_km3'] = observed_flux
        out['k'] = observed_flux / shape_km3 if shape_km3 > 0 else np.nan
    return out


def _rows(covariates):
    if isinstance(covariates, pd.DataFrame):
        if 'glacier_id' not in covariates.columns:
            covariates = covariates.rename_axis('glacier_id').reset_index()
        return covariates.to_dict('records')
    return list(covariates)


def _has_k(row):
    k = row.get('k')
    return k is not None and np.isfinite(k)


def _design(rows, terms, log_terms=LOG_TERMS):
    cols = [np.ones(len(rows))]
    for t in terms:
        v = np.array([r[t] for r in rows], dtype=float)
        cols.append(np.log(v) if t in log_terms else v)
    return np.column_stack(cols)


def calving_k_from_model(model, row):
    """The calving constant a fitted model implies for one row of covariates.

    Parameters
    ----------
    model : dict
        as returned by :func:`fit_calving_k_model`
    row : dict
        the covariates of one glacier, in the units the model was fitted on

    Returns
    -------
    float
        the constant in yr-1
    """
    return float(np.exp(_design([row], model['terms'],
                                model.get('log_terms', LOG_TERMS))
                        @ model['coefficients'])[0])


def fit_calving_k_model(covariates, terms=(), log_terms=LOG_TERMS):
    """Regress ``log k`` on covariates across a set of glaciers.

    With no ``terms`` the model is the intercept alone, i.e. the geometric
    mean of the constants.

    Parameters
    ----------
    covariates : list of dict or :py:class:`pandas.DataFrame`
        one row per glacier with ``glacier_id``, ``k`` in yr-1 and the
        terms; rows without a finite ``k`` or term are left out
    terms : sequence of str
        the keys to regress on
    log_terms : set of str
        the terms that enter as logs

    Returns
    -------
    dict
        ``terms``, ``log_terms``, ``coefficients`` (of ``log k``, the
        intercept first), ``glacier_ids``, ``k_model`` (glacier id -> yr-1),
        ``residual`` and ``loo_residual`` (in-sample and leave-one-out, log
        units), ``loo_sd`` (log units) and ``n`` (glaciers)
    """
    rows = [r for r in _rows(covariates) if _has_k(r)
            and all(np.isfinite(r.get(t, np.nan)) for t in terms)]
    if len(rows) <= len(terms) + 1:
        raise InvalidWorkflowError(
            f'{len(rows)} usable glaciers against {len(terms) + 1} '
            'coefficients: the model would interpolate rather than fit.')

    y = np.log(np.array([r['k'] for r in rows], dtype=float))
    X = _design(rows, terms, log_terms)
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)

    loo = np.full(len(rows), np.nan)
    for i in range(len(rows)):
        keep = np.arange(len(rows)) != i
        if np.linalg.matrix_rank(X[keep]) < X.shape[1]:
            continue
        b, *_ = np.linalg.lstsq(X[keep], y[keep], rcond=None)
        loo[i] = X[i] @ b - y[i]

    return dict(terms=list(terms), log_terms=frozenset(log_terms),
                coefficients=beta,
                glacier_ids=[r['glacier_id'] for r in rows],
                k_model={r['glacier_id']: float(np.exp(v))
                         for r, v in zip(rows, X @ beta)},
                residual=X @ beta - y, loo_residual=loo,
                loo_sd=float(np.nanstd(loo)),
                n=len(rows))


def usable_terms(covariates, terms):
    """The terms that are finite and vary across the glaciers with a ``k``."""
    rows = [r for r in _rows(covariates) if _has_k(r)]
    return [t for t in terms
            if all(np.isfinite(r.get(t, np.nan)) for r in rows)
            and np.std([r[t] for r in rows]) > 0]


def term_sets(terms, max_terms=2):
    """Every combination of up to ``max_terms`` terms, the empty one first."""
    return [c for n in range(max_terms + 1)
            for c in itertools.combinations(terms, n)]


def pooled_loo_residuals(shape, flux):
    """Leave-one-out log residuals of the pooled, flux-weighted constant.

    Parameters
    ----------
    shape : array
        the flux of each glacier at k = 1
    flux : array
        the observed frontal ablation, same unit

    Returns
    -------
    array
        ``log(k_pooled without i / k_i)``
    """
    shape = np.asarray(shape, dtype=float)
    flux = np.asarray(flux, dtype=float)
    out = np.empty(len(shape))
    for i in range(len(shape)):
        keep = np.arange(len(shape)) != i
        out[i] = np.log(flux[keep].sum() / shape[keep].sum()
                        * shape[i] / flux[i])
    return out


def search_calving_k_models(covariates, terms, max_terms=2, pooled_sd=None,
                            log_terms=LOG_TERMS):
    """Fit every combination of terms and rank them by leave-one-out spread.

    Parameters
    ----------
    covariates : list of dict or :py:class:`pandas.DataFrame`
        see :func:`fit_calving_k_model`
    terms : sequence of str
        the candidates; those not usable are dropped (:func:`usable_terms`)
    max_terms : int
        the largest number of terms in one model
    pooled_sd : float, optional
        the leave-one-out spread of the pooled constant. Default: from
        ``shape_km3`` and ``obs_flux_km3`` of the rows with a ``k``.
    log_terms : set of str
        the terms that enter as logs

    Returns
    -------
    :py:class:`pandas.DataFrame`
        one row per model, the pooled constant included, sorted by
        ``loo_sd``: ``terms`` (text), ``n_terms`` and ``n`` (counts),
        ``loo_sd`` (log units), ``vs_pooled_pct`` (percent) and
        ``coefficients`` (of ``log k``)
    """
    rows = _rows(covariates)
    gated = [r for r in rows if _has_k(r)]
    if pooled_sd is None:
        pooled_sd = float(pooled_loo_residuals(
            [r['shape_km3'] for r in gated],
            [r['obs_flux_km3'] for r in gated]).std())
    models = [dict(terms='(pooled k)', n_terms=0, n=len(gated),
                   loo_sd=pooled_sd, vs_pooled_pct=0., coefficients=())]
    for combo in term_sets(usable_terms(rows, terms), max_terms):
        try:
            m = fit_calving_k_model(rows, combo, log_terms)
        except (InvalidWorkflowError, np.linalg.LinAlgError):
            continue
        models.append(dict(
            terms=' + '.join(combo) or '(intercept only)', n_terms=len(combo),
            n=m['n'], loo_sd=m['loo_sd'],
            vs_pooled_pct=100 * (m['loo_sd'] / pooled_sd - 1),
            coefficients=tuple(float(c) for c in m['coefficients'])))
    return pd.DataFrame(models).sort_values(
        'loo_sd', kind='stable').reset_index(drop=True)


def best_score(covariates, terms, max_terms=2, log_terms=LOG_TERMS):
    """The smallest leave-one-out spread over every combination of terms."""
    rows = _rows(covariates)
    top = np.inf
    for combo in term_sets(terms, max_terms):
        try:
            top = min(top, fit_calving_k_model(rows, combo,
                                               log_terms)['loo_sd'])
        except (InvalidWorkflowError, np.linalg.LinAlgError):
            continue
    return top
