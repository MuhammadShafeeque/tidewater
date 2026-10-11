"""Offset of the thermal forcing onto a reference product (Slater et al. 2020, Eq. 5)."""
import numpy as np
import xarray as xr

from tidewater.config import InvalidParamsError, InvalidWorkflowError
from tidewater.periods import time_slice


def band_offsets(tf, time, names, ref_ocean, period):
    """Per-band offset that aligns the period mean of ``tf`` with the reference, degC.

    Parameters
    ----------
    tf : (time, band) array
        thermal forcing, degC
    time : the time axis of ``tf``
    names : list of str
        the band names of ``tf``
    ref_ocean : :py:class:`xarray.Dataset`
        the reference, with ``thermal_forcing(time, band)`` and, to align the
        bands by name, ``band_name``
    period : (first year, last year)
        the reference period, both years included; see
        :func:`tidewater.periods.as_period`
    """
    from tidewater.physics import band_names

    years = time_slice(period)
    ref = ref_ocean.sel(time=years).thermal_forcing.mean('time')
    # `band` is a bare dimension: align by name, never by index
    ref_names = band_names(ref_ocean) if 'band_name' in ref_ocean else None
    if ref_names is not None:
        missing = [n for n in names if n not in ref_names]
        if missing:
            raise InvalidParamsError(
                f'ref_ocean has no band(s) {", ".join(missing)}; it '
                f'carries {", ".join(ref_names)}')
        ref = ref.isel(band=[ref_names.index(n) for n in names])
    elif ref.sizes['band'] != len(names):
        raise InvalidParamsError(
            f'ref_ocean has {ref.sizes["band"]} bands against '
            f'{len(names)} here, and no band_name to align them by')
    if not np.isfinite(ref.values).all():
        raise InvalidWorkflowError('ref_ocean has no data in the '
                                   'reference period')
    mod = xr.DataArray(
        tf, dims=('time', 'band'),
        coords={'time': np.asarray(time), 'band': list(names)},
    ).sel(time=years).mean('time')
    return np.asarray(ref.values - mod.values, dtype=float)


def bias_correct(ds, ref_ocean, period, apply=True):
    """A forcing dataset with ``bias_offset`` set and, with ``apply``, added.

    ``ds`` must be uncorrected; ``period`` as in :func:`band_offsets`.
    """
    from tidewater.physics import band_names

    if str(ds.attrs.get('bias_correction_applied', 'False')) == 'True':
        raise InvalidWorkflowError('the dataset is already bias corrected')
    offsets = band_offsets(ds['thermal_forcing'].values, ds['time'].values,
                           band_names(ds), ref_ocean, period)
    out = ds.copy()
    out['bias_offset'] = ('band', offsets, dict(ds['bias_offset'].attrs)
                          if 'bias_offset' in ds else {'units': 'degC'})
    if apply:
        out['thermal_forcing'] = ds['thermal_forcing'] + xr.DataArray(
            offsets, dims='band')
        out['thermal_forcing'].attrs = dict(ds['thermal_forcing'].attrs)
    out.attrs['bias_correction_applied'] = str(bool(apply))
    return out
