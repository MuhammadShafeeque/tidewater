"""The per-glacier forcing file, and the neutral layout of an ocean product.

Forcing file: dimensions ``time`` (unlimited), ``band``, ``nchar``;
``thermal_forcing(time, band)`` degC, ``thetao(time, band)`` degC,
``so(time, band)`` psu, ``band_name``, ``band_weighting`` (char),
``band_top``, ``band_bottom`` (m), ``bias_offset(band)`` degC, and optionally
``siconc(time)``, ``open_water_frac(time)`` (0-1) and
``subglacial_discharge(time)`` m3 s-1.

Normalised product: ``thetao`` (degC) and ``so`` (psu) on
``(time, level, cell)`` with ``depth(level)`` in m positive downwards,
``latitude(cell)`` and ``longitude(cell)`` in degrees, optionally
``cell_area_km2(cell)``; ``siconc(time, cell)`` as a 0-1 fraction.
"""
import logging
import os

import numpy as np
import pandas as pd
import xarray as xr
from netCDF4 import Dataset as ncDataset
from netCDF4 import date2num, num2date

from tidewater.config import InvalidParamsError, InvalidWorkflowError
from tidewater.forcing.footprint import R_EARTH_KM, haversine_km, to_180
from tidewater.units import temperature_unit

log = logging.getLogger(__name__)

AUTHOR = 'tidewater'

BAND_VARS = ('thermal_forcing', 'thetao', 'so')
TIME_VARS = ('siconc', 'open_water_frac', 'subglacial_discharge')

# global attributes write_ocean_file sets itself
_OWN_ATTRS = ('ref_pix_lon', 'ref_pix_lat', 'ocean_source',
              'ocean_model', 'extraction', 'n_cells', 'search_radius_km',
              'ref_bathymetry_m', 'ref_bathymetry_src', 'tf_method',
              'tf_lambda1', 'tf_lambda2', 'tf_lambda3',
              'bias_correction_applied', 'yr_0', 'yr_1', 'author',
              'author_info')

# variable names of a product, as normalise() looks them up
PRODUCT_NAMES = {'thetao': 'thetao', 'so': 'so', 'siconc': 'siconc',
                 'depth': 'depth', 'latitude': 'latitude',
                 'longitude': 'longitude', 'time': 'time',
                 'cell_area': 'cell_area'}

# first year an epoch serves -> the time units of the file
_EPOCHS = ((1801, 'days since 1801-01-01 00:00:00'),
           (None, 'days since 0001-01-01 00:00:00'))
_CALENDAR = 'standard'

_BIAS_NOTE = ('offset that would align the reference-period mean '
              'with ref_ocean; added to thermal_forcing only if '
              'bias_correction_applied is True')

# reference-composition salinity over practical salinity, g kg-1
# (TEOS-10; IOC, SCOR and IAPSO, 2010)
SA_PER_SP = 35.16504 / 35.

_PSU = ('psu', 'pss-78', 'pss78', '1e-3', '0.001', '1', 'practical')
_G_PER_KG = ('g/kg', 'g kg-1', 'g kg^-1', 'absolute')
_METRES = ('m', 'metre', 'meter', 'metres', 'meters')
_CENTIMETRES = ('cm', 'centimetre', 'centimeter', 'centimetres', 'centimeters')


def chars(var):
    """The strings of a netCDF4 char variable (rows, nchar)."""
    return [''.join(c.decode() for c in row if c).strip('\x00 ') for row in var[:].data]


def _pad(values, nchar):
    """Fixed-width character array, as netCDF4 wants it for a char variable."""
    out = np.zeros((len(values), nchar), dtype='S1')
    for i, val in enumerate(values):
        out[i, :len(val)] = np.array(list(val), dtype='S1')
    return out


def _as_datetimes(time):
    """Whatever time axis we were handed, as objects date2num accepts."""
    time = np.asarray(time)
    if np.issubdtype(time.dtype, np.datetime64):
        return pd.to_datetime(time).to_pydatetime()
    return list(time)


def _time_units(first_year):
    """The time units of a file whose record starts in ``first_year``."""
    return next(units for since, units in _EPOCHS
                if since is None or first_year >= since)


def _global_attrs(first_year, last_year, lon, lat, glacier_lon, glacier_lat,
                  terminus_depth, terminus_depth_source, source, ocean_model,
                  extraction, n_cells, search_radius_km, teos10, bias_applied,
                  author, author_info, attrs):
    """The global attributes of a forcing file, in file order."""
    from tidewater.physics import LAMBDA1, LAMBDA2, LAMBDA3

    column = {} if lon is None else {'ref_pix_lon': lon, 'ref_pix_lat': lat}
    if None not in (lon, lat, glacier_lon, glacier_lat):
        column['ref_pix_dis'] = 1000 * float(
            haversine_km(lon, lat, glacier_lon, glacier_lat))
    optional = {'n_cells': None if n_cells is None else int(n_cells),
                'search_radius_km': (None if search_radius_km is None
                                     else float(search_radius_km)),
                'ref_bathymetry_m': (None if terminus_depth is None
                                     else float(terminus_depth))}
    out = dict(column, ocean_source=source, ocean_model=ocean_model,
               extraction=extraction, **optional,
               ref_bathymetry_src=terminus_depth_source,
               tf_method='teos10_gsw' if teos10 else 'linear_lambda_jenkins2011',
               tf_lambda1=LAMBDA1, tf_lambda2=LAMBDA2, tf_lambda3=LAMBDA3,
               bias_correction_applied=str(bias_applied),
               yr_0=first_year, yr_1=last_year, author=author,
               author_info=author_info)
    out.update({k: v for k, v in (attrs or {}).items() if k not in _OWN_ATTRS})
    return {k: v for k, v in out.items() if v is not None}


def _file_variables(names, weights, nchar, tops, bots, offsets, tf, th, sa,
                    siconc, open_water, q_sg):
    """(name, type, dimensions, values, attributes) of the data variables, in file order."""
    band, record, text = ('band',), ('time', 'band'), ('band', 'nchar')
    rows = [
        ('band_name', 'S1', text, _pad(names, nchar), {}),
        ('band_weighting', 'S1', text, _pad(weights, nchar), {}),
        ('band_top', 'f4', band, np.asarray(tops, dtype=float), {'units': 'm'}),
        ('band_bottom', 'f4', band, np.asarray(bots, dtype=float), {'units': 'm'}),
        ('bias_offset', 'f4', band,
         None if offsets is None else np.asarray(offsets, dtype=float),
         {'units': 'degC', 'long_name': _BIAS_NOTE}),
        ('thermal_forcing', 'f4', record, tf,
         {'units': 'degC',
          'long_name': 'ocean thermal forcing, in situ minus freezing point'}),
        ('thetao', 'f4', record, th,
         {'units': 'degC', 'long_name': 'band-mean potential temperature'}),
        ('so', 'f4', record, sa,
         {'units': 'psu', 'long_name': 'band-mean practical salinity'}),
        ('siconc', 'f4', ('time',), siconc,
         {'units': '1', 'long_name': 'sea ice area fraction'}),
        ('open_water_frac', 'f4', ('time',), open_water,
         {'units': '1', 'long_name': 'open water fraction'}),
        ('subglacial_discharge', 'f4', ('time',), q_sg,
         {'units': 'm3 s-1', 'long_name': 'subglacial discharge'}),
    ]
    return [row for row in rows if row[3] is not None]


def write_ocean_file(fpath, time, names, tops, bots, weights, tf, th, sa,
                     siconc, open_water, q_sg=None, lon=None, lat=None,
                     terminus_depth=None, terminus_depth_source='',
                     source='', ocean_model='', extraction='', n_cells=None,
                     search_radius_km=None, offsets=None, teos10=False,
                     bias_applied=False, glacier_lon=None, glacier_lat=None,
                     zlib=True, attrs=None, author=AUTHOR, author_info=None):
    """Write one forcing file.

    Parameters
    ----------
    fpath : path of the file, replaced if it exists
    time : monthly time axis (datetime64 or cftime)
    names, tops, bots, weights : band names, tops and bottoms (m), weightings
    tf, th, sa : (time, band) thermal forcing (degC), temperature (degC), salinity (psu)
    siconc, open_water : (time,) fractions 0-1, or None
    q_sg : (time,) subglacial discharge, m3 s-1, or None
    lon, lat : position of the water column, degrees
    terminus_depth : m
    offsets : (band,) bias offsets, degC
    glacier_lon, glacier_lat : the glacier's reference point, degrees; sets
        ``ref_pix_dis`` (m)
    attrs : further global attributes
    author, author_info : the ``author`` and ``author_info`` global attributes
    """
    time = np.asarray(time)
    first_year, last_year = (int(str(stamp)[:4]) for stamp in (time[0], time[-1]))
    units = _time_units(first_year)
    names, weights = [str(n) for n in names], [str(w) for w in weights]
    nchar = max(len(n) for n in names + weights + ['depth_weighted'])
    sizes = {'time': None, 'band': len(names), 'nchar': nchar}
    global_attrs = _global_attrs(
        first_year, last_year, lon, lat, glacier_lon, glacier_lat,
        terminus_depth, terminus_depth_source, source, ocean_model, extraction,
        n_cells, search_radius_km, teos10, bias_applied, author, author_info,
        attrs)
    variables = [('time', 'i4', ('time',),
                  date2num(_as_datetimes(time), units, calendar=_CALENDAR),
                  {'units': units, 'calendar': _CALENDAR})]
    variables += _file_variables(names, weights, nchar, tops, bots, offsets,
                                 tf, th, sa, siconc, open_water, q_sg)

    with ncDataset(fpath, 'w', format='NETCDF4') as nc:
        for dim, size in sizes.items():
            nc.createDimension(dim, size)
        nc.setncatts(global_attrs)
        for name, kind, dims, values, var_attrs in variables:
            packed = zlib and name != 'time' and 'time' in dims
            var = nc.createVariable(name, kind, dims, zlib=packed)
            var.setncatts(var_attrs)
            var[:] = values


def forcing_dataset(time, names, tops, bots, weights, tf, th, sa, siconc=None,
                    open_water=None, q_sg=None, offsets=None, attrs=None):
    """The forcing file's content as an in-memory dataset; units as in the file."""
    ds = xr.Dataset(coords={'time': np.asarray(time)})
    for var, vals in zip(BAND_VARS, (tf, th, sa)):
        ds[var] = (('time', 'band'), np.asarray(vals, dtype=float))
    ds['band_name'] = ('band', np.array([str(n) for n in names], dtype=object))
    ds['band_weighting'] = ('band', np.array([str(w) for w in weights], dtype=object))
    ds['band_top'] = ('band', np.asarray(tops, dtype=float))
    ds['band_bottom'] = ('band', np.asarray(bots, dtype=float))
    ds['bias_offset'] = ('band', np.zeros(len(names)) if offsets is None
                         else np.asarray(offsets, dtype=float))
    for var, vals in zip(TIME_VARS, (siconc, open_water, q_sg)):
        if vals is not None:
            ds[var] = ('time', np.asarray(vals, dtype=float))
    ds.attrs.update(attrs or {})
    return ds


def _strings(ds, var):
    raw = np.asarray(ds[var].values)
    if raw.ndim == 2:
        raw = [b''.join(r) for r in raw]
    return [(r.decode() if isinstance(r, bytes) else str(r)).strip('\x00 ')
            for r in raw]


def write_forcing(fpath, ds, glacier_lon=None, glacier_lat=None, zlib=True,
                  author=None, author_info=None, attrs=None):
    """Write a forcing dataset (of :func:`forcing_dataset` or :func:`read_forcing`).

    ``author`` and ``author_info`` default to those of ``ds``, else to
    ``tidewater`` and none; ``attrs`` are further global attributes.
    """
    a = dict(ds.attrs)
    applied = str(a.get('bias_correction_applied', 'False')) == 'True'

    def opt(var):
        return ds[var].values if var in ds else None

    extra = {k: v for k, v in a.items() if k not in _OWN_ATTRS}
    if glacier_lon is not None:
        extra.pop('ref_pix_dis', None)
    extra.update(attrs or {})
    write_ocean_file(
        fpath, ds['time'].values, _strings(ds, 'band_name'),
        ds['band_top'].values, ds['band_bottom'].values,
        _strings(ds, 'band_weighting'), ds['thermal_forcing'].values,
        ds['thetao'].values, ds['so'].values, opt('siconc'),
        opt('open_water_frac'), q_sg=opt('subglacial_discharge'),
        lon=a.get('ref_pix_lon'), lat=a.get('ref_pix_lat'),
        terminus_depth=a.get('ref_bathymetry_m'),
        terminus_depth_source=a.get('ref_bathymetry_src', ''),
        source=a.get('ocean_source', ''), ocean_model=a.get('ocean_model', ''),
        extraction=a.get('extraction', ''), n_cells=a.get('n_cells'),
        search_radius_km=a.get('search_radius_km'), offsets=opt('bias_offset'),
        teos10=a.get('tf_method') == 'teos10_gsw', bias_applied=applied,
        glacier_lon=glacier_lon, glacier_lat=glacier_lat, zlib=zlib,
        attrs=extra, author=author or a.get('author') or AUTHOR,
        author_info=author_info or a.get('author_info'))


def read_forcing(fpath):
    """A forcing file, loaded, with band names and weightings as strings."""
    with xr.open_dataset(fpath) as ds:
        ds = ds.load()
    for var in ('band_name', 'band_weighting'):
        if var in ds:
            ds[var] = ('band', np.array(_strings(ds, var), dtype=object))
    return ds


def validate(fpath, bands=None, variables=('thermal_forcing',)):
    """Check one file for whole, consecutive, finite months and for its bands.

    Parameters
    ----------
    fpath : path of a forcing file, or of any file with a monthly ``time``
    bands : band names that must be there, optional
    variables : variables that must be finite where present

    Returns
    -------
    dict with ``problem`` ('' when sound), ``y0``, ``y1`` (first and last
    year) and ``filled`` (the bands a fill wrote)
    """
    row = {'problem': '', 'y0': None, 'y1': None, 'filled': ''}
    if not os.path.exists(fpath):
        row['problem'] = 'missing'
        return row
    try:
        with ncDataset(fpath) as d:
            t = d['time']
            dates = num2date(t[:], t.units, getattr(t, 'calendar', 'standard'))
            months = np.array([12 * x.year + x.month - 1 for x in dates])
            row['y0'], row['y1'] = dates[0].year, dates[-1].year
            if dates[0].month != 1 or dates[-1].month != 12 or (np.diff(months) != 1).any():
                row['problem'] = 'months not whole and consecutive'
            if bands:
                have = chars(d['band_name'])
                lost = [b for b in bands if b not in have]
                if lost:
                    row['problem'] = 'band(s) absent: ' + ' '.join(lost)
                row['filled'] = getattr(d, 'filled_bands', '')
            for v in variables:
                if v in d.variables and not np.isfinite(np.ma.filled(d[v][:], np.nan)).all():
                    row['problem'] = row['problem'] or f'{v} not finite'
    except Exception as err:  # an unreadable file is a finding, not a crash
        row['problem'] = f'unreadable: {err}'
    return row


def cell_areas(lat, dlat, dlon):
    """Area of regular lat-lon cells centred at ``lat``, km2; spacings in degrees."""
    lat = np.radians(np.asarray(lat, dtype=float))
    half = np.radians(abs(dlat)) / 2
    top, bot = np.clip(lat + half, -np.pi / 2, np.pi / 2), np.clip(lat - half, -np.pi / 2, np.pi / 2)
    return R_EARTH_KM ** 2 * np.radians(abs(dlon)) * (np.sin(top) - np.sin(bot))


def _declared(units, attrs, what, argument, choices):
    """The units given, else those of the ``units`` attribute; neither is an error."""
    given = attrs.get('units') if units is None else units
    if given is None or not str(given).strip():
        raise InvalidParamsError(f'{what} units missing: pass {argument} {choices}')
    return str(given).strip()


def _to_degc(da, units):
    given = _declared(units, da.attrs, 'temperature', 'temperature_units',
                      '"degC", "K" or "auto"')
    unit = temperature_unit(given, da.values)
    if given.lower() == 'auto':
        log.info('temperature units "auto": read as %s', unit)
    if unit == 'K':
        da = da - 273.15
    da.attrs['units'] = 'degC'
    return da


def salinity_to_psu(salinity, units='psu'):
    """Practical salinity from practical ('psu') or absolute ('g/kg') salinity.

    'g/kg' divides by the reference-composition ratio 35.16504 / 35.
    """
    u = str(units).strip().lower()
    if u in _PSU:
        return salinity
    if u in _G_PER_KG:
        log.info('salinity in g/kg: divided by %.6f to practical salinity', SA_PER_SP)
        return salinity / SA_PER_SP
    raise InvalidParamsError(f'salinity units {units!r} not understood: pass '
                             'salinity_units "psu" or "g/kg"')


def depth_metres(depth, units='m', positive='down', attrs=None):
    """Depths in m, positive downwards.

    Parameters
    ----------
    depth : array
    units : 'm', 'cm' or 'auto'; None reads ``attrs['units']``
        'auto' reads a maximum beyond 20000 as cm
    positive : 'down' or 'up'; None reads ``attrs['positive']``, else the sign
    attrs : dict, optional
        the attributes of the coordinate
    """
    attrs = attrs or {}
    d = np.asarray(depth, dtype=float)
    u = _declared(units, attrs, 'depth', 'depth_units', '"m", "cm" or "auto"').lower()
    if u == 'auto':
        u = 'cm' if np.nanmax(np.abs(d)) > 20000. else 'm'
        log.info('depth units "auto": read as %s', u)
    if u in _CENTIMETRES:
        d = d / 100.
    elif u not in _METRES:
        raise InvalidParamsError(
            f'depth units {u!r} not understood: pass depth_units "m", "cm" or "auto"')
    if positive is None:
        positive = attrs.get('positive')
        if not positive:
            positive = 'up' if (d <= 0).all() and (d < 0).any() else 'down'
            log.info('depth axis without a declared direction: read as positive %s',
                     positive)
    if positive not in ('up', 'down'):
        raise InvalidParamsError(f'depth_positive {positive!r} is not up or down')
    return -d if positive == 'up' else d


def normalise(ds, names=None, temperature_units=None, depth_units=None,
              depth_positive=None, siconc_units=None, cell_area_units='m2',
              around=None, radius_km=None, max_depth=None, drop_dry=True,
              salinity_units='psu'):
    """Any regular, curvilinear or unstructured product in the neutral layout.

    Parameters
    ----------
    ds : :py:class:`xarray.Dataset`
        holds any of thetao, so and siconc with their coordinates; longitudes
        on [-180, 180] or [0, 360], returned on [-180, 180)
    names : dict, optional
        the product's own variable names, overriding :data:`PRODUCT_NAMES`
        (keys thetao, so, siconc, depth, latitude, longitude, time, cell_area)
    temperature_units : 'degC', 'K' or 'auto', optional
        default: the variable's ``units`` attribute; 'auto' reads a mean above
        100 as K
    depth_units : 'm', 'cm' or 'auto', optional
        default: the coordinate's ``units`` attribute; 'auto' reads a maximum
        beyond 20000 as cm
    depth_positive : 'down' or 'up', optional
        default: the coordinate's ``positive`` attribute, else its sign
    siconc_units : '1', '%' or 'auto', optional
        default: the variable's ``units`` attribute; 'auto' reads a maximum
        above 1.5 as a percentage
    salinity_units : 'psu' or 'g/kg'
        see :func:`salinity_to_psu`
    cell_area_units : 'm2' or 'km2'
        units of the product's cell area, if it has one
    around : (lon, lat), optional
        with ``radius_km``, keep only the cells within that great-circle
        distance (degrees, km)
    max_depth : float, optional
        keep only the levels down to this depth, m
    drop_dry : bool
        drop the cells with no finite value at the first time step

    Returns
    -------
    :py:class:`xarray.Dataset`
        in the normalised layout. A regular grid gets ``cell_area_km2`` from
        its spacing when the product has none.
    """
    from tidewater.forcing.seaice import as_fraction

    nm = dict(PRODUCT_NAMES, **(names or {}))
    have = [k for k in ('thetao', 'so', 'siconc') if nm[k] in ds]
    if not have:
        raise InvalidParamsError(f'none of {[nm[k] for k in ("thetao", "so", "siconc")]} '
                                 f'in {list(ds.data_vars)}')
    lat, lon = ds[nm['latitude']], ds[nm['longitude']]
    area = ds[nm['cell_area']] if nm['cell_area'] in ds else None

    if lat.ndim == 1 and lon.ndim == 1 and lat.dims != lon.dims:
        hdims = (lat.dims[0], lon.dims[0])
        la2, lo2 = np.meshgrid(lat.values.astype(float), to_180(lon.values), indexing='ij')
        if area is None and lat.size > 1 and lon.size > 1:
            dlat = np.gradient(lat.values.astype(float))
            dlon = np.gradient(np.unwrap(np.radians(lon.values.astype(float))))
            ar2 = cell_areas(la2, dlat[:, None], np.degrees(dlon)[None, :])
        else:
            ar2 = None
    elif lat.ndim == 2:
        hdims = lat.dims
        la2, lo2, ar2 = lat.values.astype(float), to_180(lon.transpose(*hdims).values), None
    elif lat.ndim == 1:
        hdims = lat.dims
        la2, lo2, ar2 = lat.values.astype(float), to_180(lon.values), None
    else:
        raise InvalidParamsError('latitude must be 1-D or 2-D')
    if area is not None:
        scale = {'m2': 1e-6, 'km2': 1.0}[cell_area_units]
        like = xr.DataArray(np.zeros(la2.shape), dims=hdims)
        ar2 = (area.reset_coords(drop=True) + like).transpose(*hdims).values * scale

    keep = np.ones(la2.shape, dtype=bool)
    if around is not None:
        if radius_km is None:
            raise InvalidParamsError('`around` needs `radius_km`')
        keep = haversine_km(around[0], around[1], lo2, la2) <= radius_km
        if not keep.any():
            raise InvalidWorkflowError(f'no cell within {radius_km} km of {around}')

    # cut to the bounding index window first, so a lazy product is never read whole
    sub = ds[[nm[k] for k in have]]
    if len(hdims) == 2:
        rows, cols = np.where(keep.any(axis=1))[0], np.where(keep.any(axis=0))[0]
        sub = sub.isel({hdims[0]: rows, hdims[1]: cols})
        window = np.ix_(rows, cols)
        keep, la2, lo2 = keep[window], la2[window], lo2[window]
        ar2 = None if ar2 is None else ar2[window]
    flat = np.where(keep.ravel())[0]

    out = xr.Dataset()
    for key in have:
        da = sub[nm[key]]
        if nm['time'] != 'time':
            da = da.rename({nm['time']: 'time'})
        if len(hdims) == 2:
            da = da.stack(cell=hdims).reset_index('cell', drop=True)
        else:
            da = da.rename({hdims[0]: 'cell'})
        da = da.isel(cell=flat)
        if key == 'siconc':
            da = da.transpose('time', 'cell')
            units = da.attrs.get('units') if siconc_units is None else siconc_units
            da = da.copy(data=as_fraction(da.values, units))
            da.attrs['units'] = '1'
        else:
            zdim = ds[nm['depth']].dims[0]
            da = da.rename({zdim: 'level'}).transpose('time', 'level', 'cell')
            if key == 'thetao':
                da = _to_degc(da.load(), temperature_units)
            else:
                da = salinity_to_psu(da, salinity_units)
        out[key] = da.reset_coords(drop=True)

    out = out.drop_vars(['cell', 'level'], errors='ignore')
    out = out.assign_coords(latitude=('cell', la2.ravel()[flat]),
                            longitude=('cell', lo2.ravel()[flat]))
    if ar2 is not None:
        out = out.assign_coords(cell_area_km2=('cell', np.asarray(ar2).ravel()[flat]))
    if 'level' in out.dims:
        zvar = ds[nm['depth']]
        depth = depth_metres(zvar.values, depth_units, depth_positive, zvar.attrs)
        out = out.assign_coords(depth=('level', depth))
        out['depth'].attrs.update(units='m', positive='down')
        order = np.argsort(depth)
        if max_depth is not None:
            order = order[depth[order] <= max_depth]
        out = out.isel(level=order)
    out = out.load()

    if drop_dry:
        first = out[have[0]].isel(time=0)
        wet = np.isfinite(first).any('level') if 'level' in first.dims else np.isfinite(first)
        out = out.isel(cell=np.where(wet.values)[0])
    out.attrs.update({k: v for k, v in ds.attrs.items() if k in ('source', 'model')})
    return out
