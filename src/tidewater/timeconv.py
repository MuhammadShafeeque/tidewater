"""Monthly calendar-stamped series to host model years.

A yearly ocean factor is the mean of the monthly factors over the model year
(:func:`yearly_ocean_factor`), not the factor of the yearly mean thermal
forcing: the factor is a power of the forcing, clipped at zero.
"""

import calendar as _calendar
from collections.abc import Mapping

import numpy as np
import pandas as pd
import xarray as xr

from tidewater.laws import ocean_factor

# hydrological-year start months, as in OGGM (Maussion et al., 2019)
HYDRO_START_NORTH = 10
HYDRO_START_SOUTH = 4

_NOLEAP = (31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31)
_NAMED_STARTS = {
    "calendar": 1,
    "hydro_north": HYDRO_START_NORTH,
    "hydro_south": HYDRO_START_SOUTH,
}


def resolve_start_month(start_month=1, latitude=None):
    """First month (1 to 12) of the model year.

    ``start_month`` is an integer, "calendar", "hydro_north" (October),
    "hydro_south" (April) or "auto", which picks the hydrological start from
    the sign of ``latitude`` (degrees north).
    """
    if isinstance(start_month, str):
        if start_month == "auto":
            if latitude is None or not np.isfinite(latitude):
                raise ValueError('start_month="auto" needs a latitude')
            return HYDRO_START_NORTH if latitude >= 0 else HYDRO_START_SOUTH
        if start_month not in _NAMED_STARTS:
            raise ValueError(f"unknown start_month {start_month!r}")
        return _NAMED_STARTS[start_month]
    month = int(start_month)
    if month != start_month or not 1 <= month <= 12:
        raise ValueError(f"start_month must be in 1..12, got {start_month!r}")
    return month


def _per_key(value):
    """Whether a start month or latitude is given per glacier."""
    return isinstance(value, (Mapping, pd.Series)) or (
        not isinstance(value, str) and np.ndim(value) > 0)


def _spread_over(value, keys, what):
    if isinstance(value, (Mapping, pd.Series)):
        lacking = [k for k in keys if k not in value]
        if lacking:
            raise ValueError(f"no {what} for {lacking[0]!r}")
        return [value[k] for k in keys]
    if not _per_key(value):
        return [value] * len(keys)
    values = list(np.asarray(value).tolist())
    if len(values) != len(keys):
        raise ValueError(f"{len(values)} values of {what} for {len(keys)} "
                         "glaciers")
    return values


def resolve_start_months(keys, start_month=1, latitude=None):
    """First month of the model year of each glacier, as an integer array.

    ``start_month`` and ``latitude`` are one value for all glaciers (see
    :func:`resolve_start_month`), a mapping glacier -> value, or one value
    per glacier in the order of ``keys``.
    """
    keys = list(keys)
    months = _spread_over(start_month, keys, "start_month")
    lats = _spread_over(latitude, keys, "latitude")
    return np.array([resolve_start_month(m, lat)
                     for m, lat in zip(months, lats)], dtype=int)


def _check_label(label):
    if label not in ("end", "start"):
        raise ValueError(f'label must be "end" or "start", got {label!r}')


def model_year(year, month, start_month=1, label="end"):
    """Model year of each calendar (year, month).

    ``label="end"`` names a model year by the calendar year in which it ends,
    ``label="start"`` by the one in which it starts.
    """
    _check_label(label)
    start_month = resolve_start_month(start_month)
    year = np.asarray(year, dtype=int)
    month = np.asarray(month, dtype=int)
    if start_month == 1:
        return year.copy()
    first = np.where(month >= start_month, year, year - 1)
    return first + 1 if label == "end" else first


def model_year_span(year, start_month=1, label="end"):
    """First and last calendar (year, month) of a model year."""
    _check_label(label)
    start_month = resolve_start_month(start_month)
    year = int(year)
    if start_month == 1:
        return (year, 1), (year, 12)
    first = year - 1 if label == "end" else year
    return (first, start_month), (first + 1, start_month - 1)


def days_in_month(year, month, calendar="standard"):
    """Days in each (year, month) for a CF calendar name."""
    year = np.atleast_1d(np.asarray(year, dtype=int))
    month = np.atleast_1d(np.asarray(month, dtype=int))
    if calendar == "360_day":
        return np.full(month.shape, 30)
    days = np.array(_NOLEAP)[month - 1]
    if calendar in ("noleap", "365_day"):
        return days
    if calendar in ("all_leap", "366_day"):
        leap = np.ones(year.shape, dtype=bool)
    elif calendar == "julian":
        leap = year % 4 == 0
    elif calendar in ("standard", "gregorian", "proleptic_gregorian"):
        leap = np.array([_calendar.isleap(y) for y in year], dtype=bool)
    else:
        raise ValueError(f"unknown calendar {calendar!r}")
    return np.where(leap & (month == 2), 29, days)


def time_fields(time, calendar=None):
    """Year, month and calendar name of a pandas, numpy or cftime time axis."""
    if isinstance(time, pd.PeriodIndex):
        time = time.to_timestamp()
    values = np.asarray(time)
    if values.ndim != 1:
        raise ValueError("the time axis must be one-dimensional")
    if np.issubdtype(values.dtype, np.datetime64):
        index = pd.DatetimeIndex(values)
        return (index.year.to_numpy(), index.month.to_numpy(),
                calendar or "proleptic_gregorian")
    try:
        year = np.array([t.year for t in values], dtype=int)
        month = np.array([t.month for t in values], dtype=int)
    except AttributeError as err:
        raise TypeError("the time axis must hold dates (pandas, numpy "
                        "datetime64 or cftime)") from err
    if calendar is None:
        calendar = getattr(values[0], "calendar", None) if len(values) else None
    return year, month, calendar or "standard"


def _aggregate(values, time, start_month, label, how, weights, incomplete,
               calendar):
    """Aggregate a (time, n) array; returns the year labels and (year, n)."""
    _check_label(label)
    if how not in ("mean", "sum"):
        raise ValueError(f'how must be "mean" or "sum", got {how!r}')
    if weights not in ("days", "equal"):
        raise ValueError(f'weights must be "days" or "equal", got {weights!r}')
    if incomplete not in ("raise", "drop", "keep"):
        raise ValueError('incomplete must be "raise", "drop" or "keep", '
                         f"got {incomplete!r}")
    year, month, calendar = time_fields(time, calendar)
    labels = model_year(year, month, start_month, label)
    stamps = pd.MultiIndex.from_arrays([year, month])
    if stamps.has_duplicates:
        raise ValueError("the series is not monthly: a calendar month occurs "
                         "more than once")
    counts = pd.Series(1, index=labels).groupby(level=0).sum()
    partial = counts.index[counts != 12].to_numpy()
    if len(partial) and incomplete == "raise":
        raise ValueError(
            f"incomplete model years {partial.tolist()} (start month "
            f'{start_month}, label "{label}"); pass incomplete="drop" or '
            '"keep" to aggregate anyway')
    frame = pd.DataFrame(np.asarray(values, dtype=float))
    missing = frame.isna().groupby(labels).any()
    if how == "sum":
        out = frame.groupby(labels).sum()
    else:
        w = (days_in_month(year, month, calendar).astype(float)
             if weights == "days" else np.ones(len(year)))
        total = frame.mul(w, axis=0).groupby(labels).sum()
        out = total.div(pd.Series(w).groupby(labels).sum(), axis=0)
    out = out.mask(missing)
    if len(partial) and incomplete == "drop":
        out = out.drop(index=partial)
    return out.index.to_numpy(dtype=int), out.to_numpy()


def _by_start_month(data, start_month, latitude, glacier_dim, time_dim,
                    **kwargs):
    """:func:`to_model_years` with one start month per glacier."""
    if isinstance(data, pd.DataFrame):
        keys = list(data.columns)
    elif isinstance(data, xr.DataArray):
        if glacier_dim is None:
            others = [d for d in data.dims if d != time_dim]
            if len(others) != 1:
                raise ValueError("start months per glacier need glacier_dim")
            glacier_dim = others[0]
        keys = (data[glacier_dim].values.tolist()
                if glacier_dim in data.coords else range(data.sizes[glacier_dim]))
    else:
        raise TypeError("start months per glacier need a DataFrame or a "
                        "DataArray")
    months = resolve_start_months(keys, start_month, latitude)
    groups = [np.flatnonzero(months == m) for m in np.unique(months)]
    back = np.argsort(np.concatenate(groups), kind="stable")
    if isinstance(data, pd.DataFrame):
        parts = [to_model_years(data.iloc[:, idx], int(months[idx[0]]),
                                **kwargs) for idx in groups]
        out = pd.concat(parts, axis=1).sort_index().iloc[:, back]
        return out.rename_axis("year")
    parts = [to_model_years(data.isel({glacier_dim: idx}), int(months[idx[0]]),
                            time_dim=time_dim, **kwargs) for idx in groups]
    out = xr.concat(parts, dim=glacier_dim, join="outer")
    return out.isel({glacier_dim: back}).sortby("year")


def to_model_years(data, start_month=1, label="end", how="mean",
                   weights="days", incomplete="raise", latitude=None,
                   calendar=None, time_dim="time", glacier_dim=None):
    """Aggregate a monthly series to model years.

    Parameters
    ----------
    data : pandas Series or DataFrame indexed by time, or xarray DataArray
        monthly values stamped with calendar dates (pandas or cftime)
    start_month : int, str, mapping or sequence
        see :func:`resolve_start_month`; 1 is the calendar year. A mapping
        glacier -> start month, or one value per glacier, converts a table
        whose glaciers have different model years in one call (the columns
        of a DataFrame, ``glacier_dim`` of a DataArray).
    label : "end" or "start"
        which calendar year names a model year
    how : "mean" or "sum"
        "mean" keeps the units, "sum" adds monthly amounts
    weights : "days" or "equal"
        weighting of the months in the mean: by their number of days in the
        calendar of the time axis (default), or all months alike
    incomplete : "raise", "drop" or "keep"
        what to do with model years that do not have 12 months
    latitude : float, mapping or sequence, optional
        degrees north, only read with ``start_month="auto"``; per glacier
        like ``start_month``
    calendar : str, optional
        CF calendar, overrides the one of the time axis
    glacier_dim : str, optional
        the glacier dimension of a DataArray, for start months per glacier.
        Default: its only dimension besides time.

    Returns
    -------
    the same kind of object, indexed by the integer model year ("year").
    With start months per glacier, a model year that only some glaciers
    have is NaN for the others.
    """
    if _per_key(start_month) or (isinstance(start_month, str)
                                 and start_month == "auto"
                                 and _per_key(latitude)):
        return _by_start_month(
            data, start_month, latitude, glacier_dim, time_dim, label=label,
            how=how, weights=weights, incomplete=incomplete,
            calendar=calendar)
    start_month = resolve_start_month(start_month, latitude)
    args = (start_month, label, how, weights, incomplete, calendar)
    if isinstance(data, xr.DataArray):
        moved = data.transpose(time_dim, ...)
        flat = moved.values.reshape(moved.shape[0], -1)
        years, out = _aggregate(flat, moved[time_dim].values, *args)
        coords = {k: v for k, v in moved.coords.items()
                  if time_dim not in v.dims}
        coords["year"] = years
        return xr.DataArray(
            out.reshape((len(years),) + moved.shape[1:]),
            dims=("year",) + moved.dims[1:], coords=coords, name=data.name,
            attrs=dict(data.attrs)).transpose(
                *[("year" if d == time_dim else d) for d in data.dims])
    if isinstance(data, pd.Series):
        years, out = _aggregate(data.to_numpy()[:, None], data.index, *args)
        return pd.Series(out[:, 0], index=pd.Index(years, name="year"),
                         name=data.name)
    if isinstance(data, pd.DataFrame):
        years, out = _aggregate(data.to_numpy(), data.index, *args)
        return pd.DataFrame(out, index=pd.Index(years, name="year"),
                            columns=data.columns)
    raise TypeError("data must be a pandas Series or DataFrame or an xarray "
                    "DataArray")


def yearly_ocean_factor(tf_monthly, tf_ref, gamma=None, start_month=1,
                        label="end", weights="days", incomplete="raise",
                        latitude=None, calendar=None, time_dim="time",
                        glacier_dim=None):
    """Ocean factor per model year: the mean of the monthly factors.

    ``mean(ocean_factor(TF_month))`` over each model year, which is not
    ``ocean_factor(mean(TF_month))``.

    Parameters
    ----------
    tf_monthly : pandas Series or DataFrame indexed by time, or DataArray
        monthly thermal forcing, degC; the columns of a DataFrame are
        glaciers
    tf_ref : float, mapping, Series or DataArray
        reference thermal forcing, degC, > 0: one value or per glacier
    gamma : float, optional
        the exponent, default of :func:`tidewater.laws.ocean_factor`
    start_month, label, weights, incomplete, latitude, calendar, time_dim, \
glacier_dim :
        see :func:`to_model_years`; ``weights`` is "days" by default

    Returns
    -------
    the same kind of object, dimensionless, indexed by the model year
    """
    if isinstance(tf_monthly, xr.DataArray):
        ref = (tf_ref.broadcast_like(tf_monthly).transpose(
            *tf_monthly.dims).values if isinstance(tf_ref, xr.DataArray)
            else tf_ref)
        factor = tf_monthly.copy(
            data=ocean_factor(tf_monthly.values, ref, gamma))
        factor.attrs["units"] = "1"
    elif isinstance(tf_monthly, pd.DataFrame):
        ref = (np.array([tf_ref[c] for c in tf_monthly.columns], dtype=float)
               if isinstance(tf_ref, (Mapping, pd.Series)) else tf_ref)
        factor = pd.DataFrame(
            ocean_factor(tf_monthly.to_numpy(dtype=float), ref, gamma),
            index=tf_monthly.index, columns=tf_monthly.columns)
    elif isinstance(tf_monthly, pd.Series):
        factor = pd.Series(
            ocean_factor(tf_monthly.to_numpy(dtype=float), tf_ref, gamma),
            index=tf_monthly.index, name=tf_monthly.name)
    else:
        raise TypeError("tf_monthly must be a pandas Series or DataFrame or "
                        "an xarray DataArray")
    return to_model_years(
        factor, start_month=start_month, label=label, how="mean",
        weights=weights, incomplete=incomplete, latitude=latitude,
        calendar=calendar, time_dim=time_dim, glacier_dim=glacier_dim)
