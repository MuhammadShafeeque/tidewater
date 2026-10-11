"""From one water column to the band means of the forcing file."""
import logging
import os
from dataclasses import dataclass, field

import numpy as np
import xarray as xr

from tidewater.config import DEFAULTS, InvalidParamsError, InvalidWorkflowError
from tidewater.forcing.bias import band_offsets
from tidewater.forcing.footprint import resolve_bands, to_180, whole_years
from tidewater.forcing.io import depth_metres, forcing_dataset, salinity_to_psu
from tidewater.forcing.seaice import as_fraction
from tidewater.periods import as_period, time_slice
from tidewater.units import temperature_unit

log = logging.getLogger(__name__)

FRONT = 'front'
LONGITUDE_CONVENTIONS = ('-180..180', '0..360', 'any')
ON_UNFORMABLE = ('raise', 'skip')


def front_band(depths, front_depth):
    """The `front` band of a column with levels at `depths` (m): (bottom, floored).

    The bottom is never above the base of the first level, so the band always
    holds a level.
    """
    z = np.sort(np.asarray(depths, dtype=float))
    floor = z[0] if z.size == 1 else 0.5 * (z[0] + z[1])
    return max(float(front_depth), float(floor)), bool(front_depth < floor)


def front_depth_from_bed(bed_elevation, sea_level=0.):
    """Water depth at a front whose bed is at ``bed_elevation``, m; 0 when the bed is dry.

    Both in m on the same vertical datum.
    """
    return max(float(sea_level) - float(bed_elevation), 0.)


def declared_bands(ds):
    """The (name, top, bottom) bands of a column's ``bands`` attribute, m."""
    from tidewater.physics import parse_depth_bands

    return parse_depth_bands(ds.attrs.get('bands', '').split())[0]


def check_time_axis(time, what='the record'):
    """Raise unless a monthly time axis holds whole calendar years."""
    if not isinstance(time, xr.DataArray):
        time = xr.DataArray(np.asarray(time), dims='time')
    months = np.asarray(time.dt.month)
    whole = months.size and not months.size % 12
    if not whole or months[0] != 1 or months[-1] != 12:
        span = (f'from month {months[0]} to month {months[-1]}'
                if months.size else 'empty')
        raise InvalidParamsError(
            f'{what} must be monthly over whole calendar years: it has '
            f'{months.size} steps, {span}')


def check_coordinates(lon, lat, longitude_convention='-180..180'):
    """(lon, lat) of a column in degrees, the longitude on [-180, 180].

    Parameters
    ----------
    lon, lat : float
        degrees east and north
    longitude_convention : str
        '-180..180' or '0..360': the range ``lon`` must be in; 'any' takes
        either
    """
    if longitude_convention not in LONGITUDE_CONVENTIONS:
        raise InvalidParamsError(f'longitude_convention {longitude_convention!r} '
                                 f'is not one of {LONGITUDE_CONVENTIONS}')
    lon, lat = float(lon), float(lat)
    if not -90. <= lat <= 90.:
        raise InvalidParamsError(f'latitude {lat} is outside [-90, 90]')
    low, high = {'-180..180': (-180., 180.), '0..360': (0., 360.),
                 'any': (-180., 360.)}[longitude_convention]
    if not low <= lon <= high:
        raise InvalidParamsError(
            f'longitude {lon} is outside [{low:g}, {high:g}]: pass '
            f'longitude_convention, one of {LONGITUDE_CONVENTIONS}')
    if longitude_convention != '-180..180':
        lon = float(to_180(lon))
    return lon, lat


@dataclass(frozen=True)
class ColumnCheck:
    """Whether a column forms its bands: ``ok``, ``reason``, and per band '' or why not."""

    ok: bool
    reason: str = ''
    bands: dict = field(default_factory=dict)

    def __bool__(self):
        return self.ok


def _band_problems(depth, thetao, so, depth_bands, terminus_depth):
    """Band name -> '' or why the levels at ``depth`` (m) cannot form it."""
    z = np.asarray(depth, dtype=float)
    if not z.size:
        return {name: 'the column has no level' for name, _, _ in depth_bands}
    out = {}
    for name, top, bot in resolve_bands(depth_bands, z, terminus_depth):
        sel = (z >= top) & (z <= bot)
        out[name] = ''
        if not sel.any():
            out[name] = (f'no level in [{top:g}, {bot:g}] m, the column '
                         f'spans {z.min():g}-{z.max():g} m')
        elif thetao is not None and not (np.isfinite(thetao[:, sel]).all()
                                         and np.isfinite(so[:, sel]).all()):
            out[name] = f'non-finite values in [{top:g}, {bot:g}] m'
    return out


def _verdict(problems):
    reason = '; '.join(f'{name}: {why}' for name, why in problems.items() if why)
    return ColumnCheck(not reason, reason, problems)


def check_column(column, depth_bands=None, terminus_depth=None,
                 thetao_var='thetao', so_var='so', depth_var='depth',
                 depth_units='m', depth_positive=None, params=None):
    """Whether a water column can form the depth bands, without raising.

    Parameters
    ----------
    column : path, :py:class:`xarray.Dataset` or array of depths
        as :func:`process_ocean_column` takes it; with depths alone only the
        levels are checked, not the values on them
    depth_bands : list of (name, top, bottom), optional
        m. Default: the column's ``bands`` attribute, else
        ``params.ocean_depth_bands``.
    terminus_depth : float, optional
        water depth at the terminus, m. Default: the column's
        ``terminus_depth_m`` attribute.
    thetao_var, so_var, depth_var : str
        the variable names in the column
    depth_units, depth_positive : see :func:`tidewater.forcing.io.depth_metres`
    params : :py:class:`tidewater.config.OceanParams`, optional

    Returns
    -------
    :class:`ColumnCheck`
        true when every band holds a level with finite values
    """
    from tidewater.physics import parse_depth_bands

    thetao = so = None
    attrs, zattrs = {}, {}
    if isinstance(column, (str, os.PathLike)):
        with xr.open_dataset(column) as ds:
            column = ds.load()
    if isinstance(column, xr.Dataset):
        ds = column.squeeze([d for d, n in column.sizes.items()
                             if n == 1 and d not in ('time', depth_var)],
                            drop=True)
        attrs, zattrs = ds.attrs, ds[depth_var].attrs
        depth = np.atleast_1d(ds[depth_var].values)
        if thetao_var in ds and so_var in ds:
            thetao, so = (
                ds[v].transpose(..., depth_var).values.reshape(-1, depth.size)
                for v in (thetao_var, so_var))
    else:
        depth = np.atleast_1d(np.asarray(column, dtype=float))
    depth = depth_metres(depth, depth_units, depth_positive, zattrs)

    if depth_bands is None and attrs.get('bands'):
        depth_bands = parse_depth_bands(attrs['bands'].split())[0]
    if depth_bands is None:
        depth_bands = parse_depth_bands(
            (DEFAULTS if params is None else params).ocean_depth_bands)[0]
    if terminus_depth is None and 'terminus_depth_m' in attrs:
        terminus_depth = float(attrs['terminus_depth_m'])
    return _verdict(_band_problems(depth, thetao, so, depth_bands, terminus_depth))


def process_ocean_data(thetao=None, so=None, siconc=None,
                       subglacial_discharge=None,
                       depth_bands=None, band_weighting=None,
                       terminus_depth=None,
                       year_range=None,
                       apply_bias_correction=None, ref_ocean=None,
                       teos10=None, source='', ocean_model='',
                       extraction='', n_cells=None, search_radius_km=None,
                       terminus_depth_source='', temperature_units='auto',
                       params=None, *, period=None, salinity_units='psu',
                       siconc_units='1', longitude_convention='-180..180',
                       on_unformable='raise'):
    """The forcing dataset of one glacier from its water column.

    Parameters
    ----------
    thetao, so : :py:class:`xarray.DataArray`
        monthly sea-water potential temperature and salinity, with a
        ``depth`` coordinate in m positive downwards and scalar ``lon`` and
        ``lat`` (degrees): one water column. Whole calendar years only.
    siconc : :py:class:`xarray.DataArray`, optional
        monthly sea-ice concentration on the same time axis
    subglacial_discharge : :py:class:`xarray.DataArray`, optional
        monthly subglacial discharge, m3 s-1
    depth_bands : list of (name, top, bottom), optional
        m. Default: ``params.ocean_depth_bands``.
    band_weighting : dict, optional
        band name -> 'uniform' or 'depth_weighted'
    terminus_depth : float, optional
        water depth at the terminus, m; the bottom of the 'front' band
    year_range : optional
        earlier name of ``period``
    apply_bias_correction : bool, optional
        add the per-band offset onto ``ref_ocean``; when False it is recorded
        only. Default: ``params.ocean_bias_correct``.
    ref_ocean : :py:class:`xarray.Dataset`, optional
        the reference, with ``thermal_forcing(time, band)``
    teos10 : bool, optional
        default: ``params.ocean_use_teos10``
    source, ocean_model, extraction, terminus_depth_source : str
        provenance
    n_cells, search_radius_km : provenance of the column
    temperature_units : 'auto', 'degC' or 'K'
        'auto' reads a mean above 100 as K
    params : :py:class:`tidewater.config.OceanParams`, optional
    period : (first year, last year), optional
        reference period of the bias correction, both years included; see
        :func:`tidewater.periods.as_period`. Default:
        ``params.ocean_tf_ref_period``.
    salinity_units : 'psu' or 'g/kg'
        see :func:`tidewater.forcing.io.salinity_to_psu`
    siconc_units : '1', '%' or 'auto'
        see :func:`tidewater.forcing.seaice.as_fraction`
    longitude_convention : str
        see :func:`check_coordinates`
    on_unformable : 'raise' or 'skip'
        'skip' returns None for a column that cannot form a band, see
        :func:`check_column`

    Returns
    -------
    :py:class:`xarray.Dataset`
        see :mod:`tidewater.forcing.io`; write it with ``write_forcing``
    """
    from tidewater.physics import (open_water_fraction, parse_depth_bands,
                                   thermal_forcing_bands)

    if thetao is None or so is None:
        raise InvalidParamsError('process_ocean_data needs thetao and so')
    if on_unformable not in ON_UNFORMABLE:
        raise InvalidParamsError(f'on_unformable {on_unformable!r} is not '
                                 f'one of {ON_UNFORMABLE}')
    params = DEFAULTS if params is None else params

    check_time_axis(thetao['time'], 'thetao')
    lon, lat = check_coordinates(thetao['lon'], thetao['lat'],
                                 longitude_convention)
    if not np.array_equal(so['time'].values, thetao['time'].values):
        raise InvalidParamsError('thetao and so are not on one time axis')
    if siconc is not None and len(siconc['time']) != len(thetao['time']):
        # the file's time dimension is unlimited: a short siconc would be padded
        raise InvalidParamsError('siconc is not on the time axis of thetao')

    if teos10 is None:
        teos10 = params.ocean_use_teos10
    period = as_period(period or year_range or params.ocean_tf_ref_period)
    if apply_bias_correction is None:
        apply_bias_correction = params.ocean_bias_correct
    def_bands, def_weighting = parse_depth_bands(params.ocean_depth_bands)
    depth_bands = depth_bands or def_bands
    band_weighting = dict(band_weighting or def_weighting)

    t = np.asarray(thetao.values, dtype=float)
    unit = temperature_unit(temperature_units, t)
    if str(temperature_units).strip().lower() == 'auto':
        log.info('temperature units "auto": read as %s', unit)
    if unit == 'K':
        t = t - 273.15
    s = salinity_to_psu(np.asarray(so.values, dtype=float), salinity_units)

    depth = thetao['depth'].values
    if on_unformable == 'skip':
        verdict = _verdict(_band_problems(depth, t, s, depth_bands,
                                          terminus_depth))
        if not verdict:
            log.info('column at %.3f E, %.3f N skipped: %s', lon, lat,
                     verdict.reason)
            return None
    names, tops, bots, weights, tf, th, sa = thermal_forcing_bands(
        t, s, depth, depth_bands, band_weighting,
        terminus_depth=terminus_depth, teos10=teos10, lat=lat)

    offsets = np.zeros(len(names))
    if ref_ocean is not None:
        offsets = band_offsets(tf, thetao['time'].values, names, ref_ocean,
                               period)
        if apply_bias_correction:
            tf = tf + offsets

    sic = owf = None
    if siconc is not None:
        sic = np.clip(as_fraction(siconc.values, siconc_units), 0, 1)
        owf = open_water_fraction(
            sic, threshold=params.ocean_open_water_threshold)
    q_sg = (None if subglacial_discharge is None
            else np.asarray(subglacial_discharge.values, dtype=float))

    attrs = {'ref_pix_lon': lon, 'ref_pix_lat': lat,
             'ocean_source': source, 'ocean_model': ocean_model,
             'extraction': extraction,
             'ref_bathymetry_src': terminus_depth_source,
             'tf_method': ('teos10_gsw' if teos10
                           else 'linear_lambda_jenkins2011'),
             'bias_correction_applied': str(bool(apply_bias_correction))}
    if n_cells is not None:
        attrs['n_cells'] = int(n_cells)
    if search_radius_km is not None:
        attrs['search_radius_km'] = float(search_radius_km)
    if terminus_depth is not None:
        attrs['ref_bathymetry_m'] = float(terminus_depth)
    return forcing_dataset(thetao['time'].values, names, tops, bots, weights,
                           tf, th, sa, siconc=sic, open_water=owf, q_sg=q_sg,
                           offsets=offsets, attrs=attrs)


def _year_slice(y0, y1):
    """Label slice of the years y0 to y1, either of which may be open."""
    if y0 and y1:
        return time_slice((y0, y1))
    return slice(str(y0) if y0 else None, str(y1) if y1 else None)


def process_ocean_column(column, y0=None, y1=None,
                         terminus_depth=None, terminus_depth_source='',
                         thetao_var='thetao', so_var='so',
                         siconc_var='siconc', depth_var='depth',
                         source=None, ocean_model='', *, years=None,
                         depth_units='m', depth_positive=None, **kwargs):
    """The forcing dataset from a water-column file or dataset.

    The column holds ``thetao(time, depth)`` and ``so(time, depth)``,
    optionally ``siconc(time)``, and scalar ``lon`` and ``lat`` in either
    longitude convention, as :func:`tidewater.forcing.footprint.reduce_site`
    returns it. Its attributes ``bands``, ``extraction``, ``n_cells``,
    ``search_radius_km``, ``terminus_depth_m``, ``source`` and ``model`` are
    used when the matching argument is not given.

    Parameters
    ----------
    column : path or :py:class:`xarray.Dataset`
    y0, y1 : int, optional
        clip to these years, both included
    terminus_depth : float, optional
        water depth at the terminus, m
    terminus_depth_source : str
        provenance of that depth
    thetao_var, so_var, siconc_var, depth_var : str
        the variable names in the column
    source, ocean_model : str, optional
        provenance. Default: the column's ``source`` and ``model`` attributes.
    years : (first year, last year), optional
        ``y0`` and ``y1`` as one period, see
        :func:`tidewater.periods.as_period`
    depth_units, depth_positive : see :func:`tidewater.forcing.io.depth_metres`
        default: m, and the direction by the coordinate's ``positive``
        attribute, else by its sign
    **kwargs
        any other argument of :func:`process_ocean_data`
    """
    for key in ('thetao', 'so', 'siconc', 'subglacial_discharge'):
        if key in kwargs:
            raise InvalidParamsError(f'{key} is built here, not passed in')

    name = ''
    if not isinstance(column, xr.Dataset):
        name = os.path.basename(str(column))
        with xr.open_dataset(column) as ds:
            column = ds.load()
    ds = column.squeeze(drop=True)
    if years is not None:
        y0, y1 = as_period(years)
    if y0 is not None or y1 is not None:
        ds = ds.sel(time=_year_slice(y0, y1))
    ds = whole_years(ds)
    if not ds.sizes.get('time'):
        raise InvalidWorkflowError('no complete calendar year in the '
                                   'selection')

    thetao = ds[thetao_var].rename({depth_var: 'depth'})
    thetao = thetao.transpose('time', 'depth')
    so = ds[so_var].rename({depth_var: 'depth'}).transpose('time', 'depth')
    depth = depth_metres(thetao['depth'].values, depth_units, depth_positive,
                         thetao['depth'].attrs)
    if not np.array_equal(depth, thetao['depth'].values):
        thetao = thetao.assign_coords(depth=depth)
        so = so.assign_coords(depth=thetao['depth'])
    siconc = ds[siconc_var] if siconc_var in ds else None

    lon = float(to_180(float(ds['lon'])))
    for da in (thetao, so):
        da.coords['lon'] = lon
        da.coords['lat'] = float(ds['lat'])

    if 'depth_bands' not in kwargs and ds.attrs.get('bands'):
        kwargs['depth_bands'] = declared_bands(ds)
    for key in ('extraction', 'n_cells', 'search_radius_km'):
        if key not in kwargs and key in ds.attrs:
            kwargs[key] = ds.attrs[key]
    if terminus_depth is None and 'terminus_depth_m' in ds.attrs:
        terminus_depth = float(ds.attrs['terminus_depth_m'])

    if kwargs.get('on_unformable') == 'skip':
        verdict = check_column(
            xr.Dataset({'thetao': thetao, 'so': so}),
            depth_bands=kwargs.get('depth_bands'),
            terminus_depth=terminus_depth, depth_positive='down',
            params=kwargs.get('params'))
        if not verdict:
            log.info('column %s skipped: %s', name or 'dataset', verdict.reason)
            return None
    for var, da in (('thetao', thetao), ('so', so)):
        if not np.isfinite(da.values).all():
            raise InvalidWorkflowError(
                f'{var} carries non-finite values; land and '
                'sub-bathymetry levels must be dropped by the extraction')

    source = source or ds.attrs.get('source') or name
    ocean_model = ocean_model or ds.attrs.get('model', '')

    return process_ocean_data(thetao=thetao, so=so, siconc=siconc,
                              terminus_depth=terminus_depth,
                              terminus_depth_source=terminus_depth_source,
                              source=source, ocean_model=ocean_model,
                              **kwargs)
