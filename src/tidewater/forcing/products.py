"""The naming conventions of ocean products, as data, and one call that opens any of them.

An entry of :data:`PRODUCTS` holds ``names`` (field -> the variable names it
may have, first match wins), the units of temperature, salinity, depth and
sea ice (None: read the file's ``units`` attribute), the depth direction
(None: the ``positive`` attribute, else the sign) and the expected ``grid``
kind. Anything not listed here is added with :func:`register_product`.
"""
import logging
import os

import xarray as xr

from tidewater.config import InvalidParamsError
from tidewater.forcing.io import PRODUCT_NAMES, normalise

log = logging.getLogger(__name__)

FIELDS = tuple(PRODUCT_NAMES)
GRIDS = ('regular', 'curvilinear', 'unstructured')
_UNITS = {'temperature_units': None, 'salinity_units': 'psu',
          'depth_units': None, 'depth_positive': None, 'siconc_units': None,
          'cell_area_units': 'm2'}

# CF standard names, tried when no listed variable name is in the file
STANDARD_NAMES = {
    'thetao': ('sea_water_potential_temperature',),
    'so': ('sea_water_practical_salinity', 'sea_water_salinity'),
    'siconc': ('sea_ice_area_fraction',),
    'depth': ('depth',),
    'latitude': ('latitude',),
    'longitude': ('longitude',),
    'time': ('time',),
    'cell_area': ('cell_area',),
}

PRODUCTS = {
    # CMIP data request names, CF metadata
    'cmip': dict(
        _UNITS,
        names={'thetao': ('thetao',), 'so': ('so',), 'siconc': ('siconc',),
               'depth': ('lev', 'olevel', 'depth'),
               'latitude': ('lat', 'latitude'),
               'longitude': ('lon', 'longitude'),
               'time': ('time',), 'cell_area': ('areacello',)},
        grid=None),
    # NEMO on an ORCA grid, T points
    'nemo': dict(
        _UNITS,
        names={'thetao': ('votemper',), 'so': ('vosaline',),
               'siconc': ('siconc',), 'depth': ('deptht',),
               'latitude': ('nav_lat',), 'longitude': ('nav_lon',),
               'time': ('time_counter', 'time'), 'cell_area': ()},
        depth_positive='down', grid='curvilinear'),
}


def _candidates(names):
    """Field -> tuple of variable names, from names given as str or sequences."""
    unknown = sorted(set(names) - set(FIELDS))
    if unknown:
        raise InvalidParamsError(f'unknown field(s) {unknown}; the fields are {FIELDS}')
    return {key: (val,) if isinstance(val, str) else tuple(val)
            for key, val in names.items()}


def register_product(name, names, temperature_units=None, salinity_units='psu',
                     depth_units=None, depth_positive=None, siconc_units=None,
                     cell_area_units='m2', grid=None, overwrite=False):
    """Add a product's naming convention to :data:`PRODUCTS`.

    Parameters
    ----------
    name : str
    names : dict
        field (thetao, so, siconc, depth, latitude, longitude, time,
        cell_area) -> the variable name, or the names it may have
    temperature_units, salinity_units, depth_units, depth_positive : optional
    siconc_units, cell_area_units : optional
        as in :func:`tidewater.forcing.io.normalise`
    grid : 'regular', 'curvilinear' or 'unstructured', optional
        the grid kind expected; another one is logged
    overwrite : bool
        replace an entry of that name
    """
    if name in PRODUCTS and not overwrite:
        raise InvalidParamsError(f'product {name!r} is registered: pass overwrite=True')
    if grid is not None and grid not in GRIDS:
        raise InvalidParamsError(f'grid {grid!r} is not one of {GRIDS}')
    PRODUCTS[name] = dict(
        names=_candidates(names), temperature_units=temperature_units,
        salinity_units=salinity_units, depth_units=depth_units,
        depth_positive=depth_positive, siconc_units=siconc_units,
        cell_area_units=cell_area_units, grid=grid)
    return PRODUCTS[name]


def _by_standard_name(ds, key):
    wanted = STANDARD_NAMES[key]
    for var in ds.variables:
        if ds[var].attrs.get('standard_name') in wanted:
            return var
    return None


def resolve_names(ds, names=None):
    """Field -> the variable of ``ds`` that holds it, for the fields found.

    ``names``: field -> candidate names; default the normalised names. A field
    none of whose names is there is looked up by CF standard name.
    """
    cand = dict(_candidates(PRODUCT_NAMES), **_candidates(names or {}))
    found = {}
    for key in FIELDS:
        hit = next((n for n in cand[key] if n in ds.variables), None)
        hit = hit or _by_standard_name(ds, key)
        if hit is not None:
            found[key] = hit
    return found


def _usable(found):
    return (any(k in found for k in ('thetao', 'so', 'siconc'))
            and 'latitude' in found and 'longitude' in found)


def detect_product(ds):
    """Name of the first registered product whose names fit ``ds``, else None."""
    for name, entry in PRODUCTS.items():
        if _usable(resolve_names(ds, entry['names'])):
            return name
    return None


def grid_kind(ds, names):
    """'regular', 'curvilinear' or 'unstructured', from the latitude and longitude of ``ds``."""
    lat, lon = ds[names['latitude']], ds[names['longitude']]
    if lat.ndim == 2:
        return 'curvilinear'
    return 'regular' if lat.dims != lon.dims else 'unstructured'


def _open(source):
    """(dataset, whether it was opened here, name of the source)."""
    if isinstance(source, xr.Dataset):
        return source, False, ''
    if isinstance(source, (str, os.PathLike)):
        return xr.open_dataset(source), True, os.path.basename(str(source))
    parts = [_open(part) for part in source]
    merged = xr.merge([p[0] for p in parts], compat='override', join='exact',
                      combine_attrs='override')
    merged.set_close(lambda: [p[0].close() for p in parts if p[1]])
    return merged, True, ' '.join(p[2] for p in parts if p[2])


def open_product(path_or_ds, product=None, **overrides):
    """An ocean product in the normalised layout of :func:`tidewater.forcing.io.normalise`.

    Parameters
    ----------
    path_or_ds : path, :py:class:`xarray.Dataset`, or a sequence of them
        several files of one product on one grid and time axis are merged
    product : str, optional
        a key of :data:`PRODUCTS`. Default: the first entry whose names fit,
        which is logged.
    **overrides
        ``names`` (field -> variable name) and any unit of the entry, or any
        other argument of ``normalise`` (``around``, ``radius_km``,
        ``max_depth``, ``drop_dry``)
    """
    ds, opened, label = _open(path_or_ds)
    try:
        if product is None:
            product = detect_product(ds)
            if product is None and not _usable(resolve_names(ds)):
                raise InvalidParamsError(
                    f'no registered product ({", ".join(PRODUCTS)}) fits the '
                    f'variables {list(ds.variables)}: pass names=, or register '
                    'the product')
            log.info('product read as %s', product or 'the normalised layout')
        elif product not in PRODUCTS:
            raise InvalidParamsError(f'unknown product {product!r}; registered: '
                                     f'{", ".join(PRODUCTS)}')
        entry = dict(PRODUCTS[product]) if product else dict(_UNITS, names={}, grid=None)
        cand = dict(entry.pop('names'), **_candidates(overrides.pop('names', None) or {}))
        expected = overrides.pop('grid', entry.pop('grid'))
        entry.update(overrides)
        names = resolve_names(ds, cand)
        for key in ('latitude', 'longitude'):
            if key not in names:
                raise InvalidParamsError(
                    f'no {key} among {list(ds.variables)}: pass names={{{key!r}: ...}}')
        kind = grid_kind(ds, names)
        if expected is not None and kind != expected:
            log.warning('product %s: a %s grid was expected, the file is %s',
                        product, expected, kind)
        out = normalise(ds, names=names, **entry)
    finally:
        if opened:
            ds.close()
    out.attrs.setdefault('source', label)
    out.attrs.update(product=product or '', grid=kind)
    return out
