"""The calving constant on the inverted front: fit, fill and ocean scaling.

The shape of the calving law at a prescribed front (thickness, water depth,
width) does not depend on the climate, so ``k = observed / shape`` is exact
before any mass balance is calibrated. The host supplies the shape, or the
geometry it follows from.
"""
import numpy as np
import pandas as pd

from tidewater.calibration.covariates import (LOG_TERMS, _rows,
                                              calving_k_from_model,
                                              fit_calving_k_model)
from tidewater.calibration.dynamic import geometric_mean, log_spread
from tidewater.config import InvalidParamsError, InvalidWorkflowError
from tidewater.laws import (inversion_k_from_ocean,  # noqa: F401
                            larger_of_calving_constant, tf_power_mean)
from tidewater.periods import as_period, year_mask
from tidewater.units import KM3_PER_YR, convert_flux

FILLS = ('model', 'geometric_mean', 'median', 'min', 'max', 'pooled')

K_TABLE_COLUMNS = ['calving_k', 'calving_k_unc', 'flux', 'source', 'shape']


def shape_flux(geometry, thickness='thickness_m', water_depth='water_depth_m',
               width='width_m', flux_unit=KM3_PER_YR, ice_density=None):
    """The flux of the calving law at k = 1 for a table of fronts.

    Parameters
    ----------
    geometry : :py:class:`pandas.DataFrame`
        by glacier id: terminus thickness, water depth and width, all in m
    thickness, water_depth, width : str
        the columns of ``geometry``
    flux_unit : str
        one of ``tidewater.units.FLUX_UNITS``. Default: 'km3 yr-1'.
    ice_density : float, optional
        kg m-3, for a mass ``flux_unit``

    Returns
    -------
    :py:class:`pandas.Series`
        glacier id -> ``flux_unit`` per unit of k (yr-1)
    """
    q = (1. * geometry[thickness] * geometry[water_depth] * geometry[width]
         / 1e9)
    q = q.clip(lower=0.).rename('shape')
    if flux_unit == KM3_PER_YR:
        return q
    return pd.Series(convert_flux(q.to_numpy(dtype=float), KM3_PER_YR,
                                  flux_unit, ice_density=ice_density),
                     index=q.index, name='shape')


def _shape_in(shape, shape_unit, flux_unit, ice_density):
    """The shape in the unit of the observations."""
    if shape_unit is None or shape_unit == flux_unit:
        return shape
    shape = pd.Series(shape, dtype=float)
    return pd.Series(convert_flux(shape.to_numpy(), shape_unit, flux_unit,
                                  ice_density=ice_density),
                     index=shape.index, name=shape.name)


def fit_calving_k(shape, observed_flux, flux_unit=KM3_PER_YR,
                  shape_unit=None, ice_density=None):
    """Pooled calving constant matching the summed observed frontal ablation.

    Parameters
    ----------
    shape : mapping or :py:class:`pandas.Series`
        glacier id -> the flux of the calving law at k = 1, in
        ``shape_unit`` (:func:`shape_flux`)
    observed_flux : mapping
        glacier id -> observed frontal ablation, in ``flux_unit``. Glaciers
        absent from it, or with a shape not > 0, take no part in the fit.
    flux_unit : str
        one of ``tidewater.units.FLUX_UNITS``. Default: 'km3 yr-1'.
    shape_unit : str, optional
        default: ``flux_unit``
    ice_density : float, optional
        kg m-3, needed when the two units differ in basis (mass, volume)

    Returns
    -------
    (k, k_per_glacier) : the pooled, flux-weighted constant in yr-1, and a
        dict of the constants (yr-1) that reproduce each glacier's own
        observation
    """
    shape = _shape_in(shape, shape_unit, flux_unit, ice_density)
    num, den, per = 0., 0., {}
    for gid, s in dict(shape).items():
        obs = observed_flux.get(gid)
        if obs is None or not obs == obs or not s > 0:
            continue
        num += obs
        den += s
        per[gid] = obs / s
    if den <= 0:
        raise InvalidWorkflowError('no glacier contributed to the k fit')
    return num / den, per


def fill_calving_k(k_fit, how='geometric_mean', observed_flux=None,
                   shape=None):
    """One constant for glaciers without a target, from those with one.

    Parameters
    ----------
    k_fit : mapping or :py:class:`pandas.Series`
        glacier id -> fitted constant, yr-1
    how : str
        'geometric_mean', 'median', 'min', 'max' or 'pooled' (the
        flux-weighted constant, which needs the next two)
    observed_flux, shape : mapping, optional
        glacier id -> flux, both in one unit, as in :func:`fit_calving_k`

    Returns
    -------
    float
        yr-1, or None when ``k_fit`` is empty
    """
    k = pd.Series(k_fit, dtype=float)
    k = k[np.isfinite(k) & (k > 0)]
    if k.empty:
        return None
    if how == 'geometric_mean':
        return geometric_mean(k)
    if how == 'median':
        return float(k.median())
    if how == 'min':
        return float(k.min())
    if how == 'max':
        return float(k.max())
    if how == 'pooled':
        if observed_flux is None or shape is None:
            raise InvalidParamsError("fill 'pooled' needs observed_flux and "
                                     'shape')
        return (sum(observed_flux[g] for g in k.index)
                / sum(shape[g] for g in k.index))
    raise InvalidParamsError(f'unknown fill {how!r}; one of {FILLS}')


def _spread(k, value, how='std'):
    """Multiplicative spread of the fitted constants as an error of `value`."""
    k = np.asarray(k, dtype=float)
    k = k[np.isfinite(k) & (k > 0)]
    if how != 'std':
        return float(value * np.expm1(log_spread(k, how)))
    if k.size < 2:
        return np.nan
    return float(value * np.expm1(np.log(k).std(ddof=1)))


def calving_k_table(shape, observed_flux, observed_err=None, fill='model',
                    terms=(), covariates=None, exclude=(), k_table=None,
                    groups=None, log_terms=LOG_TERMS, k_unc='std',
                    flux_unit=KM3_PER_YR, shape_unit=None, ice_density=None):
    """The constant of every glacier and the frontal flux it implies.

    A glacier with an observation takes the constant that reproduces it; one
    in ``k_table`` takes that constant; the others are filled.

    Parameters
    ----------
    shape : mapping or :py:class:`pandas.Series`
        glacier id -> the flux of the calving law at k = 1, in
        ``shape_unit`` (:func:`shape_flux`), for every glacier to give a
        constant
    observed_flux : mapping
        glacier id -> observed frontal ablation, in ``flux_unit``
    observed_err : mapping, optional
        glacier id -> its error, in ``flux_unit``
    fill : str
        the constant of a glacier without an observation: 'model' (the
        covariate model on ``terms``; the geometric mean without terms),
        'geometric_mean', 'median', 'min', 'max' or 'pooled'
    terms : sequence of str
        the covariates of the model, columns of ``covariates``
    covariates : list of dict or :py:class:`pandas.DataFrame`, optional
        one row per glacier with ``glacier_id`` and the terms
    exclude : sequence
        ids that keep their own constant but inform no other
    k_table : :py:class:`pandas.DataFrame`, optional
        by glacier id: ``calving_k`` (yr-1) and optionally ``calving_k_unc``
        (yr-1) and ``flux`` (``flux_unit``, the flux a run returns at that
        constant)
    groups : mapping, optional
        glacier id -> group label (a region, a basin, ...): any fill but
        'model' is then taken within the group, and over all glaciers for a
        group without an observation
    log_terms : set of str
        the terms that enter the model as logs
    k_unc : str
        the spread behind ``calving_k_unc`` of a filled glacier, in log
        units: 'std' (default) or 'nmad'
        (:func:`tidewater.calibration.dynamic.log_spread`)
    flux_unit : str
        the unit of the observations and of the table, one of
        ``tidewater.units.FLUX_UNITS``. Default: 'km3 yr-1'.
    shape_unit : str, optional
        default: ``flux_unit``
    ice_density : float, optional
        kg m-3, needed when the two units differ in basis (mass, volume)

    Returns
    -------
    :py:class:`pandas.DataFrame`
        by glacier id, with ``attrs['flux_unit']``: ``calving_k`` (yr-1),
        ``calving_k_unc`` (yr-1: the observation's error over the shape, the
        constant times ``expm1`` of the log spread of the fitted constants
        for a filled glacier, or of the leave-one-out spread for a modelled
        one), ``flux`` (``flux_unit``), ``source`` (text: 'table',
        'observed', 'modelled' or the fill) and ``shape`` (``flux_unit`` per
        yr-1)
    """
    if fill not in FILLS:
        raise InvalidParamsError(f'unknown fill {fill!r}; one of {FILLS}')
    shape = pd.Series(_shape_in(shape, shape_unit, flux_unit, ice_density),
                      dtype=float)
    observed_err = {} if observed_err is None else observed_err
    exclude = set(exclude)
    k_obs = fit_calving_k(shape, observed_flux)[1] if len(shape) else {}
    k_fit = pd.Series({g: k for g, k in k_obs.items() if g not in exclude},
                      dtype=float)

    model = None
    if fill == 'model':
        if terms and covariates is None:
            raise InvalidParamsError('a k model on terms needs covariates')
        cov = {} if covariates is None else {
            r['glacier_id']: r for r in _rows(covariates)}
        rows = {g: dict(cov.get(g, {}), glacier_id=g) for g in shape.index}
        model = fit_calving_k_model(
            [dict(rows[g], k=k_fit[g]) for g in k_fit.index], terms, log_terms)
        model_unc = np.expm1(model['loo_sd'])

    grp = None if groups is None else pd.Series(groups).reindex(shape.index)
    fills = {}

    def filled(gid):
        label = None if grp is None or pd.isna(grp[gid]) else grp[gid]
        if label not in fills:
            members = (k_fit if label is None else
                       k_fit[(grp.reindex(k_fit.index) == label).to_numpy()])
            if members.empty:
                members = k_fit
            value = fill_calving_k(members, fill, observed_flux, shape)
            if value is None:
                raise InvalidWorkflowError('no glacier contributed to the k '
                                           'fit')
            fills[label] = (value, _spread(members, value, k_unc))
        return fills[label]

    out = {}
    for gid in shape.index:
        q_run = None
        if k_table is not None and gid in k_table.index:
            row = k_table.loc[gid]
            k, how = float(row['calving_k']), 'table'
            unc = float(row.get('calving_k_unc', np.nan))
            q = row.get('flux', np.nan)
            q_run = None if pd.isna(q) else float(q)
        elif gid in k_obs:
            k, how = k_obs[gid], 'observed'
            unc = float(observed_err.get(gid, np.nan)) / shape[gid]
        elif model is not None:
            k, how = calving_k_from_model(model, rows[gid]), 'modelled'
            unc = k * model_unc
        else:
            (k, unc), how = filled(gid), fill
        out[gid] = dict(calving_k=k, calving_k_unc=unc,
                        flux=k * shape[gid] if q_run is None else q_run,
                        source=how, shape=shape[gid])
    out = pd.DataFrame.from_dict(out, orient='index', columns=K_TABLE_COLUMNS)
    out.index.name = shape.index.name or 'glacier_id'
    out.attrs['flux_unit'] = flux_unit
    return out


def parse_own(spec):
    """``own:Y0-Y1[:<label>]`` as (Y0, Y1, label or None), else None.

    Both years are included (:mod:`tidewater.periods`). The label names the ocean record the mean is taken from, when it is not
    the one the run is forced with.
    """
    parts = str(spec).split(':')
    if parts[0] != 'own':
        return None
    y0, y1 = as_period(parts[1])
    return y0, y1, (parts[2] if len(parts) > 2 else None)


def own_tf_ref(tf, years, y0, y1, gamma):
    """The reference at which the law's mean scaling over Y0 to Y1 is 1.

    Parameters
    ----------
    tf : array
        thermal forcing of one glacier in one band, degC
    years : array
        the calendar or decimal year of each value
    y0, y1 : int
        first and last year, both included
    gamma : float
        the exponent of the law

    Returns
    -------
    float
        degC; nan without a record in the window
    """
    return tf_power_mean(np.asarray(tf)[year_mask(years, (y0, y1))], gamma)


def resolve_tf_ref(spec, series=None, gamma=None):
    """The reference thermal forcing a specification stands for on one glacier.

    Parameters
    ----------
    spec : float, str or None
        a number in degC; ``own:Y0-Y1[:<label>]`` for the glacier's own power
        mean over those years; '' or None for no reference
    series : (years, tf) or callable, optional
        the glacier's record for the ``own`` form, or ``series(label)``
        returning it, called with None for the record the run is forced with
    gamma : float
        the exponent of the law, for the ``own`` form

    Returns
    -------
    float or None
        degC
    """
    if spec is None or (isinstance(spec, float) and np.isnan(spec)):
        return None
    own = parse_own(spec)
    if own is None:
        return float(spec) if str(spec) != '' else None
    y0, y1, label = own
    if series is None or gamma is None:
        raise InvalidParamsError(f'{spec!r} needs the record of the glacier '
                                 'and the exponent of the law')
    years, tf = series(label) if callable(series) else series
    return own_tf_ref(tf, years, y0, y1, float(gamma))


def read_tf_ref_map(path, id_column='glacier_id', column='tf_ref'):
    """glacier id -> reference thermal forcing in degC, from a CSV."""
    df = pd.read_csv(path)
    return dict(zip(df[id_column], df[column].astype(float)))


def scale_k_from_ocean(k_ref, tf_mean, tf_ref=None, gamma=None,
                       tf_fraction=None):
    """Scale each glacier's constant by its period-mean thermal forcing.

    ``k = k_ref * ((1 - a) + a * (max(TF_mean, 0) / TF_ref) ** gamma)``
    (:func:`tidewater.laws.inversion_k_from_ocean`).

    Parameters
    ----------
    k_ref : float, mapping or :py:class:`pandas.Series`
        the constant at the reference forcing, yr-1: one value or per glacier
    tf_mean : mapping or :py:class:`pandas.Series`
        glacier id -> period-mean thermal forcing, degC; NaN for a glacier
        without a record, which keeps ``k_ref``
    tf_ref : float, mapping or :py:class:`pandas.Series`, optional
        degC: one value, or per glacier (every glacier with a record must
        have one). Default: each glacier's own mean, which leaves ``k_ref``
        unchanged.
    gamma, tf_fraction : float, optional
        default: 1.18 and 1

    Returns
    -------
    :py:class:`pandas.DataFrame`
        by glacier id: ``calving_k`` and ``k_ref`` (yr-1), ``tf_mean`` and
        ``tf_ref`` (degC) and ``scaled`` (boolean, False where the glacier
        had no record)
    """
    tf_mean = pd.Series(tf_mean, dtype=float)
    if np.ndim(k_ref) == 0 and not isinstance(k_ref, dict):
        k_ref = pd.Series(float(k_ref), index=tf_mean.index)
    k_ref = pd.Series(k_ref, dtype=float).reindex(tf_mean.index)
    per_glacier = isinstance(tf_ref, (dict, pd.Series))
    if per_glacier:
        tf_ref = pd.Series(tf_ref, dtype=float)
        missing = [g for g in tf_mean.index[tf_mean.notna().to_numpy()]
                   if g not in tf_ref.index or np.isnan(tf_ref[g])]
        if missing:
            raise InvalidParamsError(
                f'no tf_ref for {len(missing)} glaciers, e.g. {missing[0]}')
    rows = {}
    for gid in tf_mean.index:
        mean = float(tf_mean[gid])
        ref = float(tf_ref[gid]) if per_glacier and gid in tf_ref.index else (
            None if per_glacier or tf_ref is None else float(tf_ref))
        scaled = not np.isnan(mean)
        k = (inversion_k_from_ocean(k_ref[gid], mean, tf_ref=ref, gamma=gamma,
                                    tf_fraction=tf_fraction)
             if scaled else float(k_ref[gid]))
        rows[gid] = dict(calving_k=k, k_ref=float(k_ref[gid]), tf_mean=mean,
                         tf_ref=(mean if ref is None else ref) if scaled
                         else np.nan, scaled=scaled)
    out = pd.DataFrame.from_dict(rows, orient='index', columns=[
        'calving_k', 'k_ref', 'tf_mean', 'tf_ref', 'scaled'])
    out.index.name = tf_mean.index.name or 'glacier_id'
    return out
