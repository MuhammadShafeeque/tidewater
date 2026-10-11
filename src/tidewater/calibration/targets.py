"""Observed frontal ablation as a neutral table, and its units."""
from pathlib import Path

import numpy as np
import pandas as pd

from tidewater import units as _units
from tidewater.config import SEC_IN_YEAR, InvalidParamsError
from tidewater.ids import map_ids
from tidewater.periods import as_period
from tidewater.units import (FLUX_UNITS, GT_PER_YR, KM3_PER_YR,  # noqa: F401
                             M3_PER_S)

UNITS = FLUX_UNITS

TARGET_COLUMNS = ['glacier_id', 'flux', 'flux_err', 'start_date', 'end_date',
                  'source']

REDUCTIONS = ('mean', 'latest', 'longest')

# name -> the defaults of read_targets for one dataset; none is shipped
TARGET_SOURCES = {}

_NETCDF = ('.nc', '.nc4', '.cdf', '.netcdf')


def convert_flux(values, from_unit, to_unit, rho_ice=None,
                 sec_in_year=SEC_IN_YEAR, ice_density=None):
    """:func:`tidewater.units.convert_flux`; ``rho_ice`` is ``ice_density``."""
    return _units.convert_flux(
        values, from_unit, to_unit,
        ice_density=rho_ice if ice_density is None else ice_density,
        sec_in_year=sec_in_year)


def register_target_source(name, columns=None, flux_unit=None, **read_kwargs):
    """Name the layout of one dataset, for ``read_targets(..., preset=name)``.

    Parameters
    ----------
    name : str
    columns : dict, optional
        neutral name -> the dataset's column, as in :func:`read_targets`
    flux_unit : str, optional
        one of ``FLUX_UNITS``
    **read_kwargs
        defaults of the reader, for instance ``sep`` or ``skiprows``
    """
    if flux_unit is not None and flux_unit not in FLUX_UNITS:
        raise InvalidParamsError(f'unknown flux unit {flux_unit!r}; one of '
                                 f'{FLUX_UNITS}')
    _check_columns(columns)
    TARGET_SOURCES[name] = dict(columns=dict(columns or {}),
                                flux_unit=flux_unit, read_kwargs=read_kwargs)
    return TARGET_SOURCES[name]


def _check_columns(columns):
    unknown = set(columns or {}) - set(TARGET_COLUMNS)
    if unknown:
        raise InvalidParamsError(f'{sorted(unknown)} are not target columns; '
                                 f'use {TARGET_COLUMNS}')


def _frame(source, read_kwargs):
    """Any table as a DataFrame: a frame, an xarray object or a file."""
    if isinstance(source, pd.DataFrame):
        return source
    if hasattr(source, 'to_dataframe'):
        return source.to_dataframe().reset_index()
    if Path(str(source)).suffix.lower() in _NETCDF:
        import xarray as xr
        with xr.open_dataset(source, **read_kwargs) as ds:
            return ds.load().to_dataframe().reset_index()
    return pd.read_csv(source, **read_kwargs)


def _mapped_ids(glacier_ids, id_map):
    """Ids in the caller's convention; None where the map has no partner."""
    if id_map is None:
        return list(glacier_ids)
    if callable(id_map):
        return [id_map(g) for g in glacier_ids]
    if isinstance(id_map, dict) and 'link' in id_map:
        return map_ids(glacier_ids, **id_map)
    lookup = dict(id_map)
    return [lookup.get(g, lookup.get(str(g))) for g in glacier_ids]


def decimal_year(value):
    """A year, a date or a date string as a float year; NaN when missing."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return np.nan
    if isinstance(value, (int, float, np.integer, np.floating)):
        return float(value)
    try:
        return float(value)
    except (TypeError, ValueError):
        pass
    if hasattr(value, 'dayofyr'):
        return value.year + (value.dayofyr - 1) / 365.
    stamp = pd.Timestamp(value)
    if pd.isna(stamp):
        return np.nan
    return stamp.year + (stamp.dayofyear - 1) / (366. if stamp.is_leap_year
                                                else 365.)


def _years(column):
    return np.array([decimal_year(v) for v in column], dtype=float)


def read_targets(source, columns=None, flux_unit=None, to_unit=None,
                 rho_ice=None, source_name=None, sec_in_year=SEC_IN_YEAR,
                 ice_density=None, id_map=None, reduce=None, preset=None,
                 **read_kwargs):
    """Observed frontal ablation of any dataset as one neutral table.

    Parameters
    ----------
    source : path, :py:class:`pandas.DataFrame` or xarray object
        the dataset. A path ending in .nc, .nc4 or .cdf is opened with
        xarray, any other with :func:`pandas.read_csv` (pass ``sep`` for
        another separator).
    columns : dict, optional
        neutral name (one of ``TARGET_COLUMNS``) -> the dataset's column.
        Default: the neutral names. Only ``glacier_id`` and ``flux`` must
        exist.
    flux_unit : str, optional
        the unit of the dataset's flux and its uncertainty, one of
        ``FLUX_UNITS``. Default: 'Gt yr-1'.
    to_unit : str, optional
        the unit of the table. Default: ``flux_unit``.
    rho_ice, ice_density : float
        kg m-3, needed between mass and volume; two names of one argument
    source_name : str, optional
        fills ``source`` when the dataset has no such column
    sec_in_year : float
        s
    id_map : dict, mapping or callable, optional
        brings the dataset's ids to the caller's convention: the keyword
        arguments of :func:`tidewater.ids.map_ids` (``link``, ``source``,
        ``target``, ...), a mapping id -> id, or a function of one id.
        Observations whose id maps to None are left out.
    reduce : str or callable, optional
        one observation per glacier, see :func:`reduce_targets`. Default:
        every observation is kept.
    preset : str, optional
        a name of ``TARGET_SOURCES``, whose columns, unit and reader
        arguments are the defaults of this call
    **read_kwargs
        passed to the reader

    Returns
    -------
    :py:class:`pandas.DataFrame`
        ``TARGET_COLUMNS``, one row per observation with a finite flux:
        ``glacier_id``, ``flux`` and ``flux_err`` in ``to_unit`` (also in
        ``attrs['flux_unit']``), ``start_date`` and ``end_date`` as given
        (any length of period), ``source``
    """
    if preset is not None:
        if preset not in TARGET_SOURCES:
            raise InvalidParamsError(
                f'unknown target source {preset!r}; registered: '
                f'{sorted(TARGET_SOURCES)}')
        known = TARGET_SOURCES[preset]
        columns = {**known['columns'], **(columns or {})}
        flux_unit = flux_unit or known['flux_unit']
        read_kwargs = {**known['read_kwargs'], **read_kwargs}
        source_name = source_name or preset
    flux_unit = flux_unit or GT_PER_YR
    df = _frame(source, read_kwargs)
    names = {c: c for c in TARGET_COLUMNS}
    _check_columns(columns)
    names.update(columns or {})
    for key in ('glacier_id', 'flux'):
        if names[key] not in df.columns:
            raise InvalidParamsError(f'no column {names[key]!r} for {key}')

    to_unit = to_unit or flux_unit
    rho = rho_ice if ice_density is None else ice_density
    out = pd.DataFrame({'glacier_id': df[names['glacier_id']].to_numpy()})
    if id_map is not None:
        out['glacier_id'] = _mapped_ids(out['glacier_id'].tolist(), id_map)
    for key in ('flux', 'flux_err'):
        if names[key] in df.columns:
            vals = pd.to_numeric(df[names[key]], errors='coerce').to_numpy(
                dtype=float)
            out[key] = convert_flux(vals, flux_unit, to_unit, rho_ice=rho,
                                    sec_in_year=sec_in_year)
        else:
            out[key] = np.nan
    for key in ('start_date', 'end_date'):
        out[key] = (df[names[key]].to_numpy() if names[key] in df.columns
                    else None)
    out['source'] = (df[names['source']].to_numpy()
                     if names['source'] in df.columns else source_name)
    keep = out['flux'].notna()
    if id_map is not None:
        keep &= out['glacier_id'].notna()
    out = out[keep].reset_index(drop=True)[TARGET_COLUMNS]
    out.attrs['flux_unit'] = to_unit
    return out if reduce is None else reduce_targets(out, reduce)


def reduce_targets(table, reduce='mean'):
    """One observation per glacier from a table with several.

    Parameters
    ----------
    table : :py:class:`pandas.DataFrame`
        as returned by :func:`read_targets`
    reduce : str or callable
        'mean': the mean flux and the mean error, over the earliest start to
        the latest end. 'latest': the observation that ends last. 'longest':
        the one with the longest period. A callable takes the rows of one
        glacier and returns one row (a Series or a dict of
        ``TARGET_COLUMNS``).

    Returns
    -------
    :py:class:`pandas.DataFrame`
        ``TARGET_COLUMNS`` in the unit of ``table``, one row per glacier in
        the order of first appearance
    """
    if not callable(reduce) and reduce not in REDUCTIONS:
        raise InvalidParamsError(f'unknown reduction {reduce!r}; one of '
                                 f'{REDUCTIONS} or a callable')
    rows = []
    for gid, g in table.groupby('glacier_id', sort=False):
        if callable(reduce):
            row = dict(reduce(g))
        elif len(g) == 1:
            row = g.iloc[0].to_dict()
        elif reduce == 'mean':
            t0, t1 = _years(g['start_date']), _years(g['end_date'])
            src = pd.unique(g['source'].dropna())
            row = dict(
                flux=g['flux'].mean(), flux_err=g['flux_err'].mean(),
                start_date=(g['start_date'].iloc[int(np.nanargmin(t0))]
                            if np.isfinite(t0).any() else None),
                end_date=(g['end_date'].iloc[int(np.nanargmax(t1))]
                          if np.isfinite(t1).any() else None),
                source=(src[0] if len(src) == 1 else
                        '+'.join(str(s) for s in src) if len(src) else None))
        else:
            t0, t1 = _years(g['start_date']), _years(g['end_date'])
            rank = t1 if reduce == 'latest' else t1 - t0
            if not np.isfinite(rank).all():
                raise InvalidParamsError(
                    f'reduce={reduce!r} needs the dates of every observation '
                    f'of {gid}')
            # ties go to the observation listed last
            row = g.iloc[len(rank) - 1 - int(np.argmax(rank[::-1]))].to_dict()
        rows.append(dict(row, glacier_id=gid))
    out = pd.DataFrame(rows, columns=TARGET_COLUMNS)
    out.attrs.update(table.attrs)
    return out


def select_period(table, period, end='exclusive'):
    """The observations of ``table`` that span exactly ``period``.

    Parameters
    ----------
    table : :py:class:`pandas.DataFrame`
        as returned by :func:`read_targets`
    period : (first year, last year)
        both included (:mod:`tidewater.periods`)
    end : str
        how ``end_date`` is read. 'exclusive' (default): the observation
        stops at the start of that year, so 2000 to 2010 is the ten years
        2000 to 2009. 'inclusive': it holds that year to its end.
    """
    if end not in ('inclusive', 'exclusive'):
        raise InvalidParamsError(f'end {end!r} is not inclusive or exclusive')
    y0, y1 = as_period(period)
    t0, t1 = _years(table['start_date']), _years(table['end_date'])
    last = np.floor(t1) - (1 if end == 'exclusive' else 0)
    out = table[(np.floor(t0) == y0) & (last == y1)].reset_index(drop=True)
    out.attrs.update(table.attrs)
    return out


def convert_targets(table, to_unit, rho_ice=None, sec_in_year=SEC_IN_YEAR,
                    ice_density=None):
    """A target table in another unit (see :func:`convert_flux`)."""
    out = table.copy()
    for key in ('flux', 'flux_err'):
        out[key] = convert_flux(table[key].to_numpy(dtype=float),
                                table.attrs['flux_unit'], to_unit,
                                rho_ice=rho_ice, sec_in_year=sec_in_year,
                                ice_density=ice_density)
    out.attrs['flux_unit'] = to_unit
    return out


def target_series(table, column='flux'):
    """One column of a target table by glacier id, in the table's unit.

    Raises when a glacier has several observations: select one period or
    source first, or reduce them (:func:`reduce_targets`).
    """
    dup = table['glacier_id'][table['glacier_id'].duplicated()].unique()
    if len(dup):
        raise InvalidParamsError(
            f'{len(dup)} glaciers have several observations, e.g. {dup[0]}; '
            'select one period or source first.')
    return table.set_index('glacier_id')[column]
