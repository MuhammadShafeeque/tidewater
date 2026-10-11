"""Subglacial discharge from a monthly runoff series, onto the forcing's time axis.

The runoff itself comes from the host's mass-balance model: per month and per
grid point, the melt and the liquid precipitation in kg m-2, and the area of
each point. Everything from there on is here.
"""
import numpy as np

from tidewater.config import InvalidWorkflowError


def runoff_upper_bound(melt_f, tmelt, prcp, prcpsol):
    """Melt plus liquid precipitation, no snow bucket and no refreezing, kg m-2 month-1.

    ``melt_f`` in kg m-2 month-1 K-1, ``tmelt`` in K above the melt threshold,
    ``prcp`` and ``prcpsol`` (total and solid precipitation) in kg m-2 month-1.
    """
    return melt_f * np.asarray(tmelt) + (np.asarray(prcp) - np.asarray(prcpsol))


def discharge_from_runoff(runoff, area, sec_in_month):
    """Discharge of one month, m3 s-1 of water.

    Parameters
    ----------
    runoff : array
        kg m-2 month-1 per grid point; negative values count as zero
    area : array
        m2 per grid point
    sec_in_month : float
        s
    """
    total_kg = float(np.sum(np.clip(runoff, 0., None) * area))
    return total_kg / 1000. / sec_in_month


def mid_month_years(years, months):
    """Decimal year of the middle of each month."""
    return np.asarray(years) + (np.asarray(months) - 0.5) / 12


def interp_discharge(target_years, years, q):
    """``q`` (m3 s-1, at decimal ``years``) on ``target_years``; NaN outside the record."""
    return np.interp(target_years, years, q, left=np.nan, right=np.nan)


def add_subglacial_discharge(ds, years, q):
    """A copy of a forcing dataset with ``subglacial_discharge`` (m3 s-1) on its time axis.

    ``years`` are decimal years (mid-month) and ``q`` the discharge there.
    """
    tgt = mid_month_years(ds['time.year'].values, ds['time.month'].values)
    out = ds.copy()
    out['subglacial_discharge'] = (
        'time', interp_discharge(tgt, years, q),
        {'units': 'm3 s-1', 'long_name': 'subglacial discharge'})
    return out


def write_subglacial_discharge(fpath, years, q):
    """Add ``subglacial_discharge`` to a forcing file in place; returns its mean, m3 s-1.

    ``years`` are decimal years (mid-month) and ``q`` the discharge there.
    """
    import cftime
    import netCDF4

    if years is None or q is None:
        raise InvalidWorkflowError('no discharge series to write')
    with netCDF4.Dataset(fpath, 'a') as nc:
        t = nc.variables['time']
        dates = cftime.num2date(t[:].astype(float), t.units,
                                getattr(t, 'calendar', 'standard'))
        tgt_yrs = np.array([d.year + (d.month - 0.5) / 12 for d in dates])
        vals = interp_discharge(tgt_yrs, years, q)
        if 'subglacial_discharge' in nc.variables:
            v = nc.variables['subglacial_discharge']
        else:
            v = nc.createVariable('subglacial_discharge', 'f4', ('time',))
            v.units = 'm3 s-1'
            v.long_name = 'subglacial discharge'
        v[:] = vals
    return float(np.nanmean(vals))
