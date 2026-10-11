"""Sea ice concentration at a terminus and the open water that follows from it."""
import logging
import warnings

import numpy as np
import pandas as pd

from tidewater.config import InvalidParamsError, InvalidWorkflowError
from tidewater.forcing.footprint import haversine_km

log = logging.getLogger(__name__)


def as_fraction(siconc, units=None):
    """Concentration as a 0-1 fraction.

    ``units``: '1' or '%'; 'auto' reads values above 1.5 as a percentage.
    """
    vals = np.asarray(siconc, dtype=float)
    u = str(units).strip().lower()
    if u == 'auto':
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', RuntimeWarning)
            u = '%' if np.nanmax(vals) > 1.5 else '1'
        log.info('sea ice units "auto": read as %s', u)
    if u in ('%', 'percent'):
        return vals / 100.0
    if u in ('1', 'fraction', '(0 - 1)', '0-1'):
        return vals
    raise InvalidParamsError(f'sea ice units {units!r}: give "1", "%" or "auto"')


def monthly_mean(vals, time):
    """(time, values) of monthly means of a daily (time, cell) series; monthly passes through."""
    idx = pd.DatetimeIndex(time)
    if len(idx) == len(idx.to_period('M').unique()):
        return idx.values, np.asarray(vals)
    g = pd.DataFrame(np.asarray(vals), index=idx).groupby(idx.to_period('M')).mean()
    return g.index.to_timestamp().values, g.values


def mean_siconc(siconc, lon, lat, centre, radius_km, fill_is_open_water=False,
                wet_only=True):
    """Mean concentration of the cells within ``radius_km`` of ``centre``.

    Parameters
    ----------
    siconc : (time, cell) array, 0-1
    lon, lat : arrays over cells, degrees
    centre : (lon, lat), degrees
    radius_km : float
    fill_is_open_water : bool
        a missing month of a cell that has data in other months is ice free
    wet_only : bool
        count only cells that carry data in some month

    Returns
    -------
    (series over time, number of cells, km to the nearest of them); the series
    is NaN where no cell has data and the count 0 when no cell is in reach
    """
    vals = np.asarray(siconc, dtype=float)
    d = haversine_km(centre[0], centre[1], lon, lat)
    ok = d <= radius_km
    if wet_only:
        ok = ok & np.isfinite(vals).any(axis=0)
    sel = np.where(ok)[0]
    if not sel.size:
        return np.full(vals.shape[0], np.nan), 0, np.nan
    cells = vals[:, sel]
    if fill_is_open_water:
        cells = np.where(np.isnan(cells) & np.isfinite(cells).any(axis=0), 0.0, cells)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        series = np.nanmean(cells, axis=1)
    return series, int(sel.size), float(d[sel].min())


def fill_short_gaps(vals, max_gap):
    """Interpolate runs of at most ``max_gap`` missing steps; returns (values, n filled)."""
    v = np.asarray(vals, dtype=float)
    bad = ~np.isfinite(v)
    if not bad.any() or max_gap < 1:
        return v, 0
    idx = np.arange(v.size)
    filled, n = v.copy(), 0
    start = None
    for i in range(v.size + 1):
        if i < v.size and bad[i]:
            start = i if start is None else start
            continue
        if start is not None:
            if i - start <= max_gap and start > 0 and i < v.size:
                filled[start:i] = np.interp(idx[start:i], [start - 1, i], [v[start - 1], v[i]])
                n += i - start
            start = None
    return filled, n


def fill_gaps_climatology(series, aux=None):
    """Fill missing months with the calendar-month mean, plus an anomaly where known.

    Parameters
    ----------
    series : :py:class:`pandas.Series`
        monthly values (0-1) on a DatetimeIndex, NaN where missing
    aux : :py:class:`pandas.Series`, optional
        a second record of the same quantity; its anomaly from its own mean
        over the months ``series`` observed is added to the climatology

    Returns
    -------
    (filled series clipped to [0, 1], series of 'observed', 'clim_anomaly',
    'clim' or 'none')
    """
    obs = series.dropna()
    out = series.astype(float).copy()
    how = pd.Series('observed', index=series.index, dtype=object)
    for t in series.index[series.isna()]:
        same = obs[obs.index.month == t.month]
        if same.empty:
            how[t] = 'none'
            continue
        clim = same.mean()
        ref = aux.reindex(same.index).dropna() if aux is not None else []
        o = aux.get(t, np.nan) if aux is not None else np.nan
        if np.isfinite(o) and len(ref):
            out[t], how[t] = float(np.clip(clim + o - ref.mean(), 0, 1)), 'clim_anomaly'
        else:
            out[t], how[t] = clim, 'clim'
    return out, how


def with_observed_siconc(ds, open_water, source='', open_water_threshold=None):
    """A copy of ``ds`` whose siconc is 1 - the observed open water, on its own months.

    Parameters
    ----------
    ds : :py:class:`xarray.Dataset`
        a water column or a forcing dataset, with a monthly ``time``
    open_water : :py:class:`pandas.Series`
        observed open-water fraction (0-1) indexed by 'YYYY-MM'; every month
        of ``ds`` must be there
    source : str
        provenance, written as ``siconc_source``
    open_water_threshold : float, optional
        recompute ``open_water_frac`` from the new siconc with this threshold;
        needed when ``ds`` carries that variable
    """
    months = ds['time'].dt.strftime('%Y-%m').values
    ow = open_water.reindex(months)
    if ow.isna().any():
        raise InvalidWorkflowError(
            f'{int(ow.isna().sum())} of {len(months)} months without observed '
            f'open water, first {ow.index[ow.isna()][0]}')
    out = ds.copy()
    out['siconc'] = ('time', 1.0 - ow.to_numpy(float),
                     {'units': '1', 'long_name': 'sea ice area fraction',
                      'source': source})
    if 'open_water_frac' in out:
        if open_water_threshold is None:
            raise InvalidParamsError('open_water_frac is in the dataset: pass '
                                     'open_water_threshold to recompute it')
        from tidewater.physics import open_water_fraction
        out['open_water_frac'] = ('time', open_water_fraction(
            out['siconc'].values, threshold=open_water_threshold),
            dict(ds['open_water_frac'].attrs))
    out.attrs['siconc_source'] = source
    return out


def read_open_water(path, id_column='glacier_id', time_column='time',
                    value_column='open_water_frac'):
    """glacier id -> open-water series indexed by 'YYYY-MM', from a CSV."""
    d = pd.read_csv(path, dtype={time_column: str})
    return {gid: g.set_index(time_column)[value_column]
            for gid, g in d.groupby(id_column)}
