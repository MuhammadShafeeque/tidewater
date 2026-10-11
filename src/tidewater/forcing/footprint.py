"""Wet-cell footprint around a terminus, reduced to one water column.

Land and sub-bathymetry cells are non-finite in an ocean product, so the cells
that stand for a glacier are searched per depth band among the cells that
carry water there, by growing a radius around the terminus.
"""
import warnings

import numpy as np
import xarray as xr

from tidewater.config import InvalidParamsError, InvalidWorkflowError

# mean Earth radius
R_EARTH_KM = 6371.0

SCHEMES = ('nearest', 'shelf', 'column')
COVERAGE_RULES = ('strict', 'partial')

# Names of the normalised product layout (see tidewater.forcing.io.normalise)
NAMES = {'thetao': 'thetao', 'so': 'so', 'siconc': 'siconc', 'depth': 'depth',
         'latitude': 'latitude', 'longitude': 'longitude',
         'cell_area': 'cell_area_km2'}


def to_180(lon):
    """Longitude on [-180, 180), degrees east."""
    return ((np.asarray(lon, dtype=float) + 180.0) % 360.0) - 180.0


def haversine_km(lon1, lat1, lon2, lat2):
    """Great-circle distance, km; any longitude convention."""
    lo1, la1, lo2, la2 = map(np.radians, (to_180(lon1), lat1, to_180(lon2), lat2))
    d = (np.sin((la2 - la1) / 2) ** 2
         + np.cos(la1) * np.cos(la2) * np.sin((lo2 - lo1) / 2) ** 2)
    return 2 * R_EARTH_KM * np.arcsin(np.sqrt(np.clip(d, 0, 1)))


def centroid(lon, lat):
    """(lon, lat) of the mean unit vector, degrees; safe at the dateline and poles."""
    lo, la = np.radians(to_180(lon)), np.radians(np.asarray(lat, dtype=float))
    x, y, z = (np.cos(la) * np.cos(lo)).mean(), (np.cos(la) * np.sin(lo)).mean(), np.sin(la).mean()
    return float(np.degrees(np.arctan2(y, x))), float(np.degrees(np.arctan2(z, np.hypot(x, y))))


def equal_area_cell_km2(n_global):
    """Cell area of an equal-area global grid of ``n_global`` cells, km2."""
    return 4 * np.pi * R_EARTH_KM ** 2 / n_global


def band_levels(depth, top, bottom):
    """Indices of the levels in [top, bottom], depths in m."""
    return np.where((depth >= top) & (depth <= bottom))[0]


def assign_levels(depth, bands):
    """Level index -> the narrowest band containing it, so overlapping bands stay disjoint."""
    owner = {}
    for name, top, bot in sorted(bands, key=lambda b: b[2] - b[1]):
        for lev in band_levels(depth, top, bot):
            owner.setdefault(int(lev), name)
    return owner


def usable_cells(wet, levels, coverage):
    """Cells with one wet level ('partial') or every level wet ('strict'); wet is (level, cell)."""
    sub = wet[levels, :]
    return sub.any(axis=0) if coverage == 'partial' else sub.all(axis=0)


def radius_schedule(r0_km=15.0, r_max_km=150.0, r_grow=1.5, max_radius=None):
    """The search radii, km: geometric growth from r0 while <= r_max.

    With ``max_radius`` the growth carries on to it and tries that radius last.
    """
    limit = max_radius or r_max_km
    radii, radius = [], float(r0_km)
    while radius <= limit:
        radii.append(radius)
        if max_radius and radius < max_radius < radius * r_grow:
            radius = max_radius
        else:
            radius *= r_grow
    return radii


def select_footprint(lon, lat, wet, levels, coverage, anchor, n_cells,
                     radii=None, r0_km=15.0, r_max_km=150.0, r_grow=1.5,
                     max_radius=None):
    """The nearest usable cells to the anchor at the first radius that finds any.

    Parameters
    ----------
    lon, lat : arrays over cells, degrees
    wet : (level, cell) bool
    levels : indices of the band's levels
    coverage : 'strict' or 'partial'
    anchor : (lon, lat) of the terminus, degrees
    n_cells : the most cells kept
    radii : explicit search radii in km; default :func:`radius_schedule`

    Returns
    -------
    (cell indices sorted by distance, radius in km); empty when nothing is found
    """
    if radii is None:
        radii = radius_schedule(r0_km, r_max_km, r_grow, max_radius)
    ok = usable_cells(wet, levels, coverage)
    dist = haversine_km(anchor[0], anchor[1], lon, lat)
    for radius in radii:
        cand = np.where(ok & (dist <= radius))[0]
        if cand.size:
            keep = cand[np.argsort(dist[cand])][:n_cells]
            return keep, radius
    return np.array([], dtype=int), (radii[-1] if len(radii) else np.nan)


def shelf_footprint(wet, levels, coverage, within=None):
    """Every usable cell of the supplied region (optionally a bool mask over cells)."""
    ok = usable_cells(wet, levels, coverage)
    if within is not None:
        ok = ok & within
    return np.where(ok)[0], np.nan


def whole_years(da):
    """Trim to complete January-December years."""
    months, years = da['time.month'].values, da['time.year'].values
    keep = np.zeros(da.sizes['time'], dtype=bool)
    for y in np.unique(years):
        sel = years == y
        if sel.sum() == 12 and months[sel][0] == 1 and months[sel][-1] == 12:
            keep |= sel
    return da.isel(time=keep)


def cell_dim(da):
    """The horizontal dimension of a (time, [level,] cell) array."""
    return [d for d in da.dims if d not in ('time', 'level')][0]


def _coverage_of(coverage, name):
    rule = coverage if isinstance(coverage, str) else (coverage or {}).get(name, 'strict')
    if rule not in COVERAGE_RULES:
        raise InvalidParamsError(f'coverage of {name} is {rule!r}, not strict or partial')
    return rule


def resolve_bands(bands, depth, terminus_depth=None, front_bands=('front',),
                  unfloored_bands=('terminus',)):
    """Bands with the bottom of the terminus-following ones set to ``terminus_depth``, m.

    As in :func:`tidewater.physics.thermal_forcing_bands`: ``front_bands``
    are floored at the first level, ``unfloored_bands`` are not.
    """
    out = []
    for name, top, bot in bands:
        if terminus_depth is not None and name in front_bands:
            bot = max(float(terminus_depth), float(np.min(depth)))
        elif terminus_depth is not None and name in unfloored_bands:
            bot = float(terminus_depth)
        out.append((name, float(top), float(bot)))
    return out


def reduce_site(ds3d, ds2d, anchor, terminus_depth, scheme, fallback='shelf',
                siconc_max_gap=0, siconc_fill_is_open_water=False, *,
                bands, coverage='strict', target_area_km2=1000.0,
                cell_area_km2=None, max_cells=64, radii=None, r0_km=15.0,
                r_max_km=150.0, r_grow=1.5, max_radius=None,
                shelf_radius_km=None, column_band=None,
                front_bands=('front',), unfloored_bands=('terminus',),
                names=None):
    """One water column, its sea-ice series and the provenance that explains them.

    Parameters
    ----------
    ds3d : :py:class:`xarray.Dataset`
        thetao and so on (time, level, cell) with depth(level) in m positive
        downwards and latitude/longitude per cell in degrees; monthly
    ds2d : :py:class:`xarray.Dataset`, optional
        siconc (0-1) on (time, cell) with latitude/longitude per cell
    anchor : (lon, lat)
        the terminus, degrees
    terminus_depth : float, optional
        water depth at the terminus, m; the bottom of the ``front_bands``
        and ``unfloored_bands``, see :func:`resolve_bands`
    scheme : str
        'nearest' (a footprint per band), 'column' (one footprint, found in
        ``column_band``, for every band) or 'shelf' (every usable cell)
    fallback : str
        'shelf': a band with no water in reach takes every usable cell;
        'drop': it is left out
    siconc_max_gap : int
        interpolate sea-ice gaps of at most this many months
    siconc_fill_is_open_water : bool
        read a missing month of an ocean cell as ice free
    bands : list of (name, top, bottom)
        the depth bands, m
    coverage : str or dict
        'strict' or 'partial', for all bands or per band name (default strict)
    target_area_km2 : float
        footprint area aimed for; the cell count follows from the cell area
    cell_area_km2 : float or array over cells, optional
        default: the ``cell_area`` variable of ``ds3d``
    max_cells : int
        upper bound of the cell count
    radii, r0_km, r_max_km, r_grow, max_radius
        the search radii, see :func:`radius_schedule`
    shelf_radius_km : float, optional
        limit the 'shelf' cells to this distance of the anchor
    column_band : str, optional
        the band that fixes the 'column' footprint. Default: the first.
    names : dict, optional
        variable names in ``ds3d`` and ``ds2d``, overriding :data:`NAMES`

    Returns
    -------
    :py:class:`xarray.Dataset`
        thetao(time, depth), so(time, depth), optionally siconc(time), scalar
        lon and lat, in the units of the input
    """
    if scheme not in SCHEMES:
        raise InvalidParamsError(f'scheme {scheme!r} is not one of {SCHEMES}')
    if fallback not in ('shelf', 'drop'):
        raise InvalidParamsError(f'fallback {fallback!r} is not shelf or drop')
    nm = dict(NAMES, **(names or {}))

    thetao, so = whole_years(ds3d[nm['thetao']]), whole_years(ds3d[nm['so']])
    if not np.array_equal(thetao['time'].values, so['time'].values):
        raise InvalidWorkflowError('thetao and so are on different time axes')
    if not thetao.sizes['time']:
        raise InvalidWorkflowError('no complete calendar year in the product')

    cdim = cell_dim(thetao)
    lon = to_180(ds3d[nm['longitude']].values)
    lat = np.asarray(ds3d[nm['latitude']].values, dtype=float)
    depth = np.asarray(ds3d[nm['depth']].values, dtype=float)

    t = thetao.transpose('time', 'level', cdim).values
    s = so.transpose('time', 'level', cdim).values
    with np.errstate(invalid='ignore'), warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)  # land columns are all-NaN
        wet = np.isfinite(np.nanmean(t, axis=0)) & np.isfinite(np.nanmean(s, axis=0))

    bands = resolve_bands(bands, depth, terminus_depth, front_bands,
                          unfloored_bands)
    dist = haversine_km(anchor[0], anchor[1], lon, lat)
    if radii is None:
        radii = radius_schedule(r0_km, r_max_km, r_grow, max_radius)
    search_radius = float(radii[-1]) if len(radii) else np.nan

    if cell_area_km2 is None:
        if nm['cell_area'] not in ds3d:
            raise InvalidParamsError('the cell area is needed: pass cell_area_km2 '
                                     f'or supply {nm["cell_area"]!r}')
        cell_area_km2 = ds3d[nm['cell_area']].values
    a_cells = np.broadcast_to(np.asarray(cell_area_km2, dtype=float), lon.shape)
    # the cell area that sets the count is the local one: it varies with latitude
    near = np.isfinite(a_cells) & (dist <= search_radius)
    a_pix = float(np.median(a_cells[near] if near.any() else a_cells[np.isfinite(a_cells)]))
    n_cells = int(np.clip(round(target_area_km2 / a_pix), 1, max_cells))

    within = None if shelf_radius_km is None else dist <= shelf_radius_km
    kw = dict(radii=radii)

    prof_t = np.full(t.shape[:2], np.nan)
    prof_s = np.full(s.shape[:2], np.nan)
    owner = assign_levels(depth, bands)
    foot, dropped, notes, kept, origins = {}, [], [], [], {}

    fixed = None
    if scheme == 'column':
        first = bands[0] if column_band is None else \
            [b for b in bands if b[0] == column_band][0]
        lev0 = band_levels(depth, first[1], first[2])
        fixed, _ = select_footprint(lon, lat, wet, lev0, 'partial', anchor, n_cells, **kw)

    for name, top, bot in bands:
        rule = _coverage_of(coverage, name)
        levels = band_levels(depth, top, bot)
        if levels.size == 0:
            dropped.append(f'{name}: no level in [{top:.0f}, {bot:.0f}] m')
            continue
        if scheme == 'shelf':
            keep, radius = shelf_footprint(wet, levels, rule, within)
        elif scheme == 'column':
            keep, radius = fixed, np.nan
            if keep.size and not usable_cells(wet, levels, 'partial')[keep].any():
                keep = np.array([], dtype=int)
        else:
            keep, radius = select_footprint(lon, lat, wet, levels, rule, anchor,
                                            n_cells, **kw)

        origin = scheme
        if keep.size == 0 and fallback == 'shelf':
            keep, radius = shelf_footprint(wet, levels, rule, within)
            origin = 'shelf fallback'
        if keep.size == 0:
            dropped.append(f'{name}: no wet cell in reach')
            continue
        foot[name] = keep
        kept.append((name, top, bot))
        d = dist[keep]
        origins[name] = (f'{origin}: {keep.size} cells, '
                         f'{d.min():.0f}-{d.max():.0f} km from the anchor')
        notes.append(f'{name}: {origins[name]}'
                     + ('' if np.isnan(radius) else f', radius {radius:.0f} km'))
        mine = [lev for lev in levels if owner[int(lev)] == name]
        if mine:
            with warnings.catch_warnings():
                warnings.simplefilter('ignore', RuntimeWarning)  # levels below the sea floor
                prof_t[:, mine] = np.nanmean(t[:, mine, :][:, :, keep], axis=2)
                prof_s[:, mine] = np.nanmean(s[:, mine, :][:, :, keep], axis=2)

    # only levels finite in every month go on, so an empty band is seen as empty
    finite = np.isfinite(prof_t).all(axis=0) & np.isfinite(prof_s).all(axis=0)
    if not finite.any():
        raise InvalidWorkflowError('no level survived the finiteness trim')

    ref = foot[bands[0][0]] if bands[0][0] in foot else next(iter(foot.values()))
    clon, clat = centroid(lon[ref], lat[ref])
    area = float(a_cells[ref].sum())

    out = xr.Dataset(
        {'thetao': (('time', 'depth'), prof_t[:, finite]),
         'so': (('time', 'depth'), prof_s[:, finite])},
        coords={'time': thetao['time'].values, 'depth': depth[finite],
                'lon': clon, 'lat': clat})
    out['thetao'].attrs.update(units=thetao.attrs.get('units', ''),
                               long_name='sea water potential temperature')
    out['so'].attrs.update(units=so.attrs.get('units', 'psu'),
                           long_name='sea water practical salinity')
    out['depth'].attrs.update(units='m', positive='down')

    if ds2d is not None:
        from tidewater.forcing.seaice import fill_short_gaps, mean_siconc

        sic = whole_years(ds2d[nm['siconc']]).sel(time=out['time'])
        r_eq = np.sqrt(area / np.pi)
        vals, n_near, _ = mean_siconc(
            sic.transpose('time', cell_dim(sic)).values,
            ds2d[nm['longitude']].values, ds2d[nm['latitude']].values,
            (clon, clat), r_eq, fill_is_open_water=siconc_fill_is_open_water,
            wet_only=False)
        if n_near:
            vals, n_filled = fill_short_gaps(vals, siconc_max_gap)
            if vals.size == out.sizes['time'] and np.isfinite(vals).all():
                out['siconc'] = ('time', np.clip(vals, 0, 1))
                gap = (f'; {n_filled} month(s) interpolated in time'
                       if n_filled else '')
                out['siconc'].attrs.update(
                    units='1', long_name='sea ice area fraction',
                    months_interpolated=n_filled,
                    fill_is_open_water=int(siconc_fill_is_open_water))
                notes.append(f'siconc: {n_near} cells within {r_eq:.0f} km of '
                             f'the centroid{gap}')
            else:
                notes.append('siconc: dropped, series incomplete over the footprint')

    out.attrs.update(
        extraction=f'{scheme} footprint; ' + '; '.join(notes),
        footprint_scheme=scheme,
        footprint_anchor=f'{anchor[0]:.4f} E, {anchor[1]:.4f} N',
        footprint_area_km2=area,
        footprint_target_area_km2=float(target_area_km2),
        n_cells=int(ref.size),
        search_radius_km=search_radius,
        band_coverage='; '.join(f'{n}={_coverage_of(coverage, n)}' for n, _, _ in bands),
        bands=' '.join(f'{n}:{top:g}:{bot:g}' for n, top, bot in kept),
        dropped_bands='; '.join(dropped) if dropped else 'none',
        levels_kept=f'{int(finite.sum())} of {depth.size}',
    )
    if terminus_depth is not None:
        out.attrs['terminus_depth_m'] = float(terminus_depth)
    for name, origin in origins.items():
        out.attrs[f'band_footprint_{name}'] = origin
    for name, keep in foot.items():
        out.attrs[f'footprint_cells_{name}'] = ' '.join(
            f'{lo:.3f},{la:.3f}' for lo, la in zip(lon[keep], lat[keep]))
    return out


def footprint_cells(ds, band):
    """(lon, lat) arrays of the cells a band of a :func:`reduce_site` column came from."""
    cells = ds.attrs.get(f'footprint_cells_{band}', '').split()
    xy = np.array([[float(v) for v in c.split(',')] for c in cells]).reshape(-1, 2)
    return xy[:, 0], xy[:, 1]
