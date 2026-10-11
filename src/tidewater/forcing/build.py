"""From one ocean product and a glacier table to one forcing file per glacier."""
import logging
import os
from pathlib import Path

import numpy as np
import pandas as pd

from tidewater.config import DEFAULTS, InvalidParamsError, InvalidWorkflowError
from tidewater.forcing.bands import (ON_UNFORMABLE, check_column, check_coordinates,
                                     front_depth_from_bed, process_ocean_column)
from tidewater.forcing.footprint import radius_schedule, reduce_site
from tidewater.forcing.io import AUTHOR, validate, write_forcing
from tidewater.forcing.products import _open, detect_product, open_product

log = logging.getLogger(__name__)

REPORT_COLUMNS = ('lon', 'lat', 'front_depth', 'status', 'reason', 'path',
                  'bands', 'n_cells')


def read_glaciers(table, id_column='id', lon_column='lon', lat_column='lat',
                  front_depth_column='front_depth', bed_column=None,
                  sea_level=0.):
    """A glacier table indexed by id, with ``lon``, ``lat`` and ``front_depth``.

    Parameters
    ----------
    table : path of a CSV, or :py:class:`pandas.DataFrame`
        one row per glacier: its id, the terminus position in degrees (either
        longitude convention) and optionally the water depth at the front, m
    id_column, lon_column, lat_column, front_depth_column : str
    bed_column : str, optional
        bed elevation at the front, m; gives the front depth where
        ``front_depth_column`` has none
    sea_level : float
        m, on the datum of ``bed_column``

    Returns
    -------
    :py:class:`pandas.DataFrame`
        ``front_depth`` is NaN where unknown
    """
    df = table if isinstance(table, pd.DataFrame) else pd.read_csv(
        table, dtype={id_column: str})
    if id_column not in df.columns and df.index.name == id_column:
        df = df.reset_index()
    missing = [c for c in (id_column, lon_column, lat_column) if c not in df.columns]
    if missing:
        raise InvalidParamsError(f'glacier table without column(s) {missing}; '
                                 f'it has {list(df.columns)}')
    out = pd.DataFrame({'lon': df[lon_column].to_numpy(float),
                        'lat': df[lat_column].to_numpy(float)},
                       index=pd.Index(df[id_column].astype(str), name='id'))
    if out.index.duplicated().any():
        raise InvalidParamsError(f'duplicate glacier ids: '
                                 f'{sorted(set(out.index[out.index.duplicated()]))}')
    depth = (df[front_depth_column].to_numpy(float) if front_depth_column in df.columns
             else np.full(len(df), np.nan))
    if bed_column is not None:
        bed = df[bed_column].to_numpy(float)
        from_bed = [front_depth_from_bed(b, sea_level) if np.isfinite(b) else np.nan
                    for b in bed]
        depth = np.where(np.isfinite(depth), depth, from_bed)
    out['front_depth'] = depth
    for lon, lat in zip(out['lon'], out['lat']):
        check_coordinates(lon, lat, 'any')
    return out


def _reach_km(radii, shelf_radius_km, r0_km, r_max_km, r_grow, max_radius):
    radii = radius_schedule(r0_km, r_max_km, r_grow, max_radius) if radii is None else radii
    return max([float(r) for r in radii] + [float(shelf_radius_km or 0.)])


def build_forcing(product, glaciers, out_dir=None, *, product_name=None,
                  product_options=None, depth_bands=None, scheme='nearest',
                  fallback='shelf', region_km=None, years=None,
                  cell_area_km2=None, on_unformable='skip', pattern='{id}.nc',
                  author=AUTHOR, attrs=None, params=None, source=None,
                  ocean_model='', process_options=None, radii=None,
                  r0_km=15.0, r_max_km=150.0, r_grow=1.5, max_radius=None,
                  shelf_radius_km=None, **footprint):
    """The forcing of every glacier of a table from one ocean product.

    Parameters
    ----------
    product : path, :py:class:`xarray.Dataset`, or a sequence of them
        the ocean product, read through
        :func:`tidewater.forcing.products.open_product`
    glaciers : path of a CSV or :py:class:`pandas.DataFrame`
        as :func:`read_glaciers` returns it, or anything it reads with its
        default column names
    out_dir : path, optional
        write the files here, named by ``pattern``; None keeps them in memory
    product_name : str, optional
        a key of :data:`tidewater.forcing.products.PRODUCTS`; default: detected
    product_options : dict, optional
        ``names`` and units, passed to ``open_product``
    depth_bands : list of str or of (name, top, bottom), optional
        ``name:top:bottom[:weighting]``, m. Default: ``params.ocean_depth_bands``.
    scheme, fallback : str
        see :func:`tidewater.forcing.footprint.reduce_site`
    region_km : float, optional
        only the cells this close to a terminus are read. Default: the
        largest search radius, or ``shelf_radius_km`` if larger.
    years : (first year, last year), optional
        clip the record, both years included
    cell_area_km2 : float, optional
        for a product with no cell area and no regular grid
    on_unformable : 'skip' or 'raise'
        'skip' records why a glacier got no file (no water in reach, a band
        with no level) and goes on
    pattern : str
        file name, formatted with ``id``
    author, attrs : global attributes of the files
    params : :py:class:`tidewater.config.OceanParams`, optional
    source, ocean_model : str, optional
        provenance. Default source: the product's file names.
    process_options : dict, optional
        further arguments of :func:`tidewater.forcing.bands.process_ocean_data`
    radii, r0_km, r_max_km, r_grow, max_radius, shelf_radius_km, **footprint
        further arguments of ``reduce_site``

    Returns
    -------
    (forcing, report) : dict of glacier id -> :py:class:`xarray.Dataset`, and
        a :py:class:`pandas.DataFrame` by glacier id with the columns
        :data:`REPORT_COLUMNS`
    """
    from tidewater.physics import band_names, parse_depth_bands

    if on_unformable not in ON_UNFORMABLE:
        raise InvalidParamsError(f'on_unformable {on_unformable!r} is not '
                                 f'one of {ON_UNFORMABLE}')
    params = DEFAULTS if params is None else params
    table = read_glaciers(glaciers)
    specs = params.ocean_depth_bands if depth_bands is None else depth_bands
    if specs and not isinstance(specs, str) and not isinstance(specs[0], str):
        specs = [':'.join(str(v) for v in band) for band in specs]
    bands, weighting = parse_depth_bands(specs)
    reach = region_km or _reach_km(radii, shelf_radius_km, r0_km, r_max_km,
                                   r_grow, max_radius)
    options = dict(product_options or {})
    kw = dict(radii=radii, r0_km=r0_km, r_max_km=r_max_km, r_grow=r_grow,
              max_radius=max_radius, shelf_radius_km=shelf_radius_km, **footprint)
    if out_dir is not None:
        Path(out_dir).mkdir(parents=True, exist_ok=True)

    ds, opened, label = _open(product)
    forcing, rows = {}, {}
    try:
        product_name = product_name or detect_product(ds)
        for gid, g in table.iterrows():
            anchor = (float(g['lon']), float(g['lat']))
            depth = None if np.isnan(g['front_depth']) else float(g['front_depth'])
            row = dict(g, status='skipped', reason='', path='', bands='', n_cells=0)
            rows[gid] = row
            try:
                local = open_product(ds, product_name, around=anchor,
                                     radius_km=reach, **options)
                if not local.sizes.get('cell'):
                    raise InvalidWorkflowError(f'no wet cell within {reach:g} km')
                sea_ice = local if 'siconc' in local else None
                col = reduce_site(local, sea_ice, anchor, depth, scheme, fallback,
                                  bands=bands, cell_area_km2=cell_area_km2, **kw)
                out = process_ocean_column(
                    col, terminus_depth=depth, years=years,
                    terminus_depth_source='glacier table' if depth is not None else '',
                    source=source or label or local.attrs.get('source', ''),
                    ocean_model=ocean_model or local.attrs.get('model', ''),
                    band_weighting=weighting, temperature_units='degC',
                    on_unformable=on_unformable, params=params,
                    **(process_options or {}))
                if out is None:
                    row['reason'] = check_column(
                        col, (process_options or {}).get('depth_bands'),
                        terminus_depth=depth, params=params).reason
                    continue
            except InvalidWorkflowError as err:
                if on_unformable == 'raise':
                    raise InvalidWorkflowError(f'{gid}: {err}') from err
                row['reason'] = str(err)
                log.info('%s skipped: %s', gid, err)
                continue
            row.update(status='built', bands=' '.join(band_names(out)),
                       n_cells=int(out.attrs.get('n_cells', 0)))
            forcing[gid] = out
            if out_dir is not None:
                path = os.path.join(out_dir, pattern.format(id=gid))
                write_forcing(path, out, glacier_lon=anchor[0], glacier_lat=anchor[1],
                              author=author, attrs=attrs)
                row.update(status='written', path=path)
    finally:
        if opened:
            ds.close()
    report = pd.DataFrame.from_dict(rows, orient='index', columns=list(REPORT_COLUMNS))
    report.index.name = 'id'
    return forcing, report


def validate_directory(directory, pattern='*.nc', bands=None,
                       variables=('thermal_forcing',)):
    """:func:`tidewater.forcing.io.validate` over the files of a directory, by file name."""
    files = sorted(Path(directory).glob(pattern))
    rows = {f.name: validate(str(f), bands=bands, variables=variables) for f in files}
    report = pd.DataFrame.from_dict(rows, orient='index',
                                    columns=['problem', 'y0', 'y1', 'filled'])
    report.index.name = 'file'
    return report
