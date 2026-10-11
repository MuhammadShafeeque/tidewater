"""Ocean boundary conditions at the glacier terminus, for frontal ablation.

Writes one ``ocean_data`` file per glacier directory: thermal forcing in one
or more depth bands, sea-ice concentration and open-water fraction, and
optionally subglacial discharge. The calendar and longitude conventions are
those of :func:`oggm.shop.gcm_climate.process_gcm_data`.
"""
import logging
import os

import numpy as np
import xarray as xr
from netCDF4 import Dataset as ncDataset
from oggm import cfg
from oggm import entity_task
from oggm.exceptions import InvalidParamsError, InvalidWorkflowError
from oggm.shop.gcm_climate import _get_xr_cftime_kwargs

from tidewater import physics as _physics
from tidewater.forcing import bands as _bands
from tidewater.forcing import discharge as _discharge
from tidewater.forcing import io as _io
from tidewater.forcing.footprint import whole_years as _whole_years  # noqa
from tidewater.forcing.io import _as_datetimes, _pad  # noqa: F401
from tidewater.hosts.oggm.errors import oggm_errors, translated
from tidewater.hosts.oggm.params import ocean_params, param
from tidewater.physics import LAMBDA1, LAMBDA2, LAMBDA3  # noqa: F401

log = logging.getLogger(__name__)

# The settings the forcing file is built from
_FORCING_KEYS = ('ocean_depth_bands', 'ocean_tf_ref_period',
                 'ocean_use_teos10', 'ocean_bias_correct',
                 'ocean_open_water_threshold')

parse_depth_bands = translated(_physics.parse_depth_bands)
freezing_point = translated(_physics.freezing_point)
thermal_forcing_bands = translated(_physics.thermal_forcing_bands)


def open_water_fraction(siconc, threshold=None):
    """Open-water fraction from sea-ice concentration, clipped to [0, 1].

    Parameters
    ----------
    siconc : array
        sea-ice area fraction, 0-1
    threshold : float, optional
        the concentration at and above which no water is open. Default:
        ``cfg.PARAMS['ocean_open_water_threshold']``.
    """
    return _physics.open_water_fraction(
        siconc, threshold=param('ocean_open_water_threshold', threshold))


@entity_task(log, writes=['ocean_data'])
def process_ocean_data(gdir, thetao=None, so=None, siconc=None,
                       subglacial_discharge=None,
                       depth_bands=None, band_weighting=None,
                       terminus_depth=None,
                       year_range=None,
                       apply_bias_correction=None, ref_ocean=None,
                       teos10=None, source='', ocean_model='',
                       extraction='', n_cells=None, search_radius_km=None,
                       terminus_depth_source='',
                       output_filesuffix=''):
    """Write the ocean boundary conditions of one glacier directory.

    Nothing is written for a glacier that is not tidewater.

    Parameters
    ----------
    gdir : :py:class:`oggm.GlacierDirectory`
        the glacier directory to process
    thetao, so : :py:class:`xarray.DataArray`
        monthly sea-water potential temperature (degC or K, told apart by
        magnitude) and salinity (psu), with a ``depth`` coordinate in m
        positive downwards and scalar ``lon`` and ``lat``, i.e. one water
        column, already reduced over the cells it stands for. Whole calendar
        years only.
    siconc : :py:class:`xarray.DataArray`, optional
        monthly sea-ice area fraction, 0-1, on the same time axis
    subglacial_discharge : :py:class:`xarray.DataArray`, optional
        monthly subglacial discharge in m3 s-1 (see also
        :func:`tidewater.hosts.oggm.forcing.write_subglacial_discharge`)
    depth_bands : list of (name, top, bottom), optional
        default: ``gdir.settings['ocean_depth_bands']``
    band_weighting : dict, optional
        band name -> 'uniform' or 'depth_weighted'. Default: as given in
        ``gdir.settings['ocean_depth_bands']``.
    terminus_depth : float, optional
        water depth at the terminus, m; the bottom of the 'front' band
    year_range : tuple of two years, optional
        reference period of the bias correction. Default:
        ``gdir.settings['ocean_tf_ref_period']``.
    apply_bias_correction : bool, optional
        add one offset per band so that the ``year_range`` mean matches
        ``ref_ocean`` (Slater et al. 2020, Eq. 5). When False the offset is
        written to the file but not applied. Default:
        ``gdir.settings['ocean_bias_correct']``.
    ref_ocean : :py:class:`xarray.Dataset`, optional
        the reference, with a ``thermal_forcing(time, band)`` variable
    teos10 : bool, optional
        default: ``gdir.settings['ocean_use_teos10']``
    source, ocean_model, extraction, terminus_depth_source : str
        provenance, written as global attributes
    n_cells : int, optional
        the number of ocean cells the column was reduced over (provenance)
    search_radius_km : float, optional
        the radius they were searched in (provenance)
    output_filesuffix : str
        the filesuffix of the written file
    """
    with oggm_errors():
        ds = _bands.process_ocean_data(
            thetao=thetao, so=so, siconc=siconc,
            subglacial_discharge=subglacial_discharge,
            depth_bands=depth_bands, band_weighting=band_weighting,
            terminus_depth=terminus_depth, year_range=year_range,
            apply_bias_correction=apply_bias_correction, ref_ocean=ref_ocean,
            teos10=teos10, source=source, ocean_model=ocean_model,
            extraction=extraction, n_cells=n_cells,
            search_radius_km=search_radius_km,
            terminus_depth_source=terminus_depth_source,
            params=ocean_params(gdir.settings, keys=_FORCING_KEYS))

    if not gdir.is_tidewater:
        log.warning('(%s) not tidewater, no ocean data written', gdir.rgi_id)
        return
    _write_forcing(gdir, ds, filesuffix=output_filesuffix)


@entity_task(log, writes=['ocean_data'])
def process_destine_ocean_data(gdir, fpath=None, y0=None, y1=None,
                               terminus_depth=None, terminus_depth_source='',
                               thetao_var='thetao', so_var='so',
                               siconc_var='siconc', depth_var='depth',
                               output_filesuffix='', source=None,
                               ocean_model='', **kwargs):
    """Read one ocean water column from a file and write the ocean_data file.

    The file holds the monthly water column that stands for one glacier, as
    extracted from the Destination Earth Climate DT ocean or from any other
    ocean product: ``thetao(time, depth)`` and ``so(time, depth)``, optionally
    ``siconc(time)``, and scalar ``lon`` and ``lat``. Its global attributes
    ``bands`` (``name:top:bottom`` separated by spaces), ``extraction``,
    ``n_cells``, ``search_radius_km``, ``terminus_depth_m``, ``source`` and
    ``model`` are used when the matching argument is not given.

    Parameters
    ----------
    gdir : :py:class:`oggm.GlacierDirectory`
        the glacier directory to process
    fpath : str, optional
        path to the file. Default: ``cfg.PATHS['destine_ocean_file']``.
    y0, y1 : int, optional
        clip to these years
    terminus_depth : float, optional
        water depth at the terminus, m
    terminus_depth_source : str
        provenance of that depth
    thetao_var, so_var, siconc_var, depth_var : str
        the variable names in the file
    output_filesuffix : str
        the filesuffix of the written file
    source, ocean_model : str, optional
        provenance. Default: the file's ``source`` and ``model`` attributes.
    **kwargs
        any other argument of :func:`process_ocean_data`
    """
    for key in ('thetao', 'so', 'siconc', 'subglacial_discharge'):
        if key in kwargs:
            raise InvalidParamsError(f'{key} is built by this task, not '
                                     'passed to it')

    if not gdir.is_tidewater:
        log.warning('(%s) not tidewater, no ocean data written', gdir.rgi_id)
        return

    fpath = fpath or cfg.PATHS.get('destine_ocean_file')
    if not fpath:
        raise InvalidParamsError("Need to set cfg.PATHS['destine_ocean_file']")

    with xr.open_dataset(fpath, **_get_xr_cftime_kwargs()) as column:
        source = (source or column.attrs.get('source')
                  or os.path.basename(fpath))
        with oggm_errors():
            ds = _bands.process_ocean_column(
                column, y0=y0, y1=y1, terminus_depth=terminus_depth,
                terminus_depth_source=terminus_depth_source,
                thetao_var=thetao_var, so_var=so_var, siconc_var=siconc_var,
                depth_var=depth_var, source=source, ocean_model=ocean_model,
                params=ocean_params(gdir.settings, keys=_FORCING_KEYS),
                **kwargs)
    _write_forcing(gdir, ds, filesuffix=output_filesuffix)


def _sign(fpath):
    with ncDataset(fpath, 'a') as nc:
        nc.author = 'OGGM'
        nc.author_info = 'Open Global Glacier Model'


def _write_forcing(gdir, ds, filesuffix=''):
    """Write a forcing dataset of tidewater as the ocean_data file."""
    fpath = gdir.get_filepath('ocean_data', filesuffix=filesuffix, delete=True)
    _io.write_forcing(fpath, ds, glacier_lon=gdir.cenlon,
                      glacier_lat=gdir.cenlat,
                      zlib=cfg.PARAMS['compress_climate_netcdf'])
    _sign(fpath)


def _write_ocean_file(gdir, time, names, tops, bots, weights, tf, th, sa,
                      siconc, open_water, q_sg=None, lon=None, lat=None,
                      terminus_depth=None, terminus_depth_source='',
                      source='', ocean_model='', extraction='', n_cells=None,
                      search_radius_km=None, offsets=None, teos10=False,
                      bias_applied=False, filesuffix=''):
    """Write the ocean_data file, as write_monthly_climate_file does."""
    fpath = gdir.get_filepath('ocean_data', filesuffix=filesuffix, delete=True)
    _io.write_ocean_file(
        fpath, time, names, tops, bots, weights, tf, th, sa, siconc,
        open_water, q_sg=q_sg, lon=lon, lat=lat, terminus_depth=terminus_depth,
        terminus_depth_source=terminus_depth_source, source=source,
        ocean_model=ocean_model, extraction=extraction, n_cells=n_cells,
        search_radius_km=search_radius_km, offsets=offsets, teos10=teos10,
        bias_applied=bias_applied, glacier_lon=gdir.cenlon,
        glacier_lat=gdir.cenlat, zlib=cfg.PARAMS['compress_climate_netcdf'])
    _sign(fpath)


def subglacial_discharge_from_mb(gdir, period=None, mb_model=None,
                                 input_filesuffix=''):
    """Monthly runoff leaving the glacier, from the calibrated mass balance.

    Melt (``melt_f * tmelt``) and liquid precipitation on the fixed geometry
    of the inversion flowlines, with no snow bucket and no refreezing: an
    upper bound of the runoff, meant as the subglacial discharge of the melt
    parameterisation before any dynamical run exists.

    Parameters
    ----------
    gdir : :py:class:`oggm.GlacierDirectory`
        the glacier directory to process
    period : tuple of two years, optional
        default: the period of the baseline climate
    mb_model : :py:class:`oggm.core.massbalance.MassBalanceModel`, optional
        default: a ``MultipleFlowlineMassBalance`` of ``MonthlyTIModel``
    input_filesuffix : str
        the filesuffix of the inversion flowlines

    Returns
    -------
    (years, discharge) : float years and m3 s-1 of water, monthly, or
        ``(None, None)`` when the climate does not cover the period
    """
    from oggm.core.massbalance import (MultipleFlowlineMassBalance,
                                       MonthlyTIModel)
    from oggm.utils import date_to_floatyear

    fls = gdir.read_store('inversion_flowlines', filesuffix=input_filesuffix)
    if mb_model is None:
        mb_model = MultipleFlowlineMassBalance(gdir, fls=fls,
                                               mb_model_class=MonthlyTIModel)
    with xr.open_dataset(gdir.get_filepath('climate_historical')) as ds:
        y0c, y1c = int(ds.time.dt.year[0]), int(ds.time.dt.year[-1])
    y0, y1 = (int(period[0]), int(period[1])) if period else (y0c, y1c)
    y0, y1 = max(y0, y0c + 1), min(y1, y1c)
    if y1 < y0:
        return None, None

    years, q = [], []
    for yr in range(y0, y1 + 1):
        for m in range(1, 13):
            t = date_to_floatyear(yr, m)
            runoff, area = [], []
            for fl, mbm in zip(fls, mb_model.flowline_mb_models):
                _, _, tmelt, prcp, prcpsol = mbm.get_monthly_mb(
                    fl.surface_h, year=t, add_climate=True)
                runoff.append(_discharge.runoff_upper_bound(
                    mbm.melt_f, tmelt, prcp, prcpsol))
                area.append(fl.widths_m * fl.dx_meter)
            years.append(float(_discharge.mid_month_years(yr, m)))
            q.append(_discharge.discharge_from_runoff(
                np.concatenate(runoff), np.concatenate(area),
                mbm.sec_in_month(year=t)))
    return np.asarray(years), np.asarray(q)


@entity_task(log, writes=['ocean_data'])
def write_subglacial_discharge(gdir, period=None, ocean_filesuffix='',
                               input_filesuffix=''):
    """Add ``subglacial_discharge`` to the ocean_data file.

    The discharge of :func:`subglacial_discharge_from_mb`, interpolated onto
    the time axis of the ocean file. Without it
    :class:`~tidewater.hosts.oggm.laws.MeltPlusCalving` runs in the
    no-discharge limit of the melt parameterisation.

    Parameters
    ----------
    gdir : :py:class:`oggm.GlacierDirectory`
        the glacier directory to process
    period : tuple of two years, optional
        default: the period of the baseline climate
    ocean_filesuffix : str
        the filesuffix of the ``ocean_data`` file
    input_filesuffix : str
        the filesuffix of the inversion flowlines

    Returns
    -------
    float
        the mean discharge in m3 s-1, or None for a glacier that is not
        tidewater
    """

    if not gdir.is_tidewater:
        return None
    if not gdir.has_file('ocean_data', filesuffix=ocean_filesuffix):
        raise InvalidWorkflowError(f'({gdir.rgi_id}) no ocean_data file')

    yrs, q = subglacial_discharge_from_mb(gdir, period=period,
                                          input_filesuffix=input_filesuffix)
    if yrs is None:
        raise InvalidWorkflowError(f'({gdir.rgi_id}) the climate does not '
                                   f'cover {period}')

    path = gdir.get_filepath('ocean_data', filesuffix=ocean_filesuffix)
    mean = _discharge.write_subglacial_discharge(path, yrs, q)
    gdir.add_to_diagnostics('subglacial_discharge_mean', mean)
    return mean
