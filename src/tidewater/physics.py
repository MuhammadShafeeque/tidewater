"""Sea-water thermodynamics, depth-band averaging and front geometry.

Thermal forcing and the linear freezing point follow Jenkins (2011) as used by
Slater et al. (2020); the TEOS-10 form is that of IOC, SCOR and IAPSO (2010).
"""
import numpy as np

from tidewater.config import DEFAULTS, InvalidParamsError, InvalidWorkflowError
from tidewater.units import to_celsius  # noqa: F401

# Jenkins (2011), as used by Slater et al. (2020) Eq. (6):
# degC psu-1, degC, degC m-1
LAMBDA1, LAMBDA2, LAMBDA3 = -5.73e-2, 8.32e-2, 7.61e-4

# Earlier name of the fixed 0-60 m band, still read from older files
BAND_ALIASES = {'terminus': 'shallow_0_60', 'shallow_0_60': 'terminus'}

WEIGHTINGS = ('uniform', 'depth_weighted')


def parse_depth_bands(specs):
    """Depth bands and their weighting from ``name:top:bottom[:weighting]``.

    Parameters
    ----------
    specs : list of str, or one comma-separated str
        depths in m, positive downwards

    Returns
    -------
    (depth_bands, band_weighting) : a list of (name, top, bottom) and a dict
        of name -> 'uniform' or 'depth_weighted'
    """
    if isinstance(specs, str):
        specs = [s for s in specs.split(',') if s.strip()]
    bands, weighting = [], {}
    for spec in specs:
        parts = str(spec).strip().split(':')
        if len(parts) not in (3, 4):
            raise InvalidParamsError(f'depth band {spec!r} is not '
                                     'name:top:bottom[:weighting]')
        bands.append((parts[0], float(parts[1]), float(parts[2])))
        weighting[parts[0]] = parts[3] if len(parts) == 4 else 'uniform'
    return bands, weighting


def band_names(ds):
    """The depth-band names of an ocean dataset, in file order."""
    if 'band_name' not in ds:
        return [str(b) for b in ds['band'].values]
    raw = np.asarray(ds['band_name'].values)
    if raw.ndim == 2:  # netCDF char array, if xarray did not join it
        raw = [b''.join(r) for r in raw]
    return [(r.decode() if isinstance(r, bytes) else str(r)).strip('\x00')
            for r in raw]


def band_index(names, band, aliases=None):
    """Position of ``band`` in ``names``, or of its alias."""
    aliases = BAND_ALIASES if aliases is None else aliases
    for b in (band, aliases.get(band)):
        if b in names:
            return list(names).index(b)
    raise InvalidParamsError(f'band {band!r} not in {list(names)}')


def freezing_point(salinity, depth, teos10=False, lat=None):
    """Freezing point of sea water, degC.

    Parameters
    ----------
    salinity : array
        practical salinity, psu
    depth : array
        m, positive downwards
    teos10 : bool
        use the conservative-temperature freezing point of TEOS-10 (needs
        gsw) instead of the linear form of Slater et al. (2020), Eq. (6)
    lat : float
        degrees north, needed with ``teos10`` to convert depth to pressure
    """
    depth = np.asarray(depth, dtype=float)
    if teos10:
        import gsw
        if lat is None:
            raise InvalidParamsError('the TEOS-10 freezing point needs `lat`')
        p = gsw.p_from_z(-depth, lat)
        return gsw.CT_freezing(salinity, p, 0)
    return LAMBDA1 * np.asarray(salinity) + LAMBDA2 - LAMBDA3 * depth


def thermal_forcing(thetao, so, depth, teos10=False, lat=None):
    """Thermal forcing, temperature minus freezing point, degC (not clipped).

    Parameters
    ----------
    thetao : array
        potential temperature, degC
    so : array
        practical salinity, psu
    depth : array
        m, positive downwards
    teos10, lat :
        see :func:`freezing_point`
    """
    return np.asarray(thetao) - freezing_point(so, depth, teos10=teos10,
                                               lat=lat)


def open_water_fraction(siconc, threshold=None):
    """Open-water fraction (0-1) from sea-ice concentration (0-1).

    Parameters
    ----------
    siconc : array
        sea-ice area fraction, 0-1
    threshold : float, optional
        the concentration at and above which no water is open. Default: 0.15.
    """
    if threshold is None:
        threshold = DEFAULTS.ocean_open_water_threshold
    siconc = np.asarray(siconc, dtype=float)
    return np.clip((threshold - siconc) / threshold, 0, 1)


def thermal_forcing_bands(thetao, so, depth, depth_bands, band_weighting,
                          terminus_depth=None, teos10=False, lat=None,
                          front_bands=('front',),
                          unfloored_bands=('terminus',)):
    """Band-average thermal forcing, temperature and salinity.

    Parameters
    ----------
    thetao, so : ndarray
        (time, depth) potential temperature in degC and salinity in psu
    depth : ndarray
        depth coordinate in m, positive downwards
    depth_bands : list of (name, top, bottom)
        the bands, in m
    band_weighting : dict
        band name -> 'uniform' or 'depth_weighted'
    terminus_depth : float, optional
        m; replaces the bottom of the bands named in ``front_bands`` and
        ``unfloored_bands``
    teos10 : bool
        see :func:`freezing_point`
    lat : float, optional
        degrees north, needed with ``teos10``
    front_bands : sequence of str
        bands whose bottom is ``terminus_depth``, at least the first level
    unfloored_bands : sequence of str
        bands whose bottom is ``terminus_depth`` as given

    Returns
    -------
    names, tops, bottoms, weightings, tf, thetao_b, so_b : the last three with
        shape (time, band), in degC, degC and psu
    """
    z = np.asarray(depth, dtype=float)
    names, tops, bots, weights = [], [], [], []
    tf_b, th_b, so_b = [], [], []

    for name, top, bot in depth_bands:
        if name in front_bands and terminus_depth is not None:
            bot = max(float(terminus_depth), float(z.min()))
        elif name in unfloored_bands and terminus_depth is not None:
            bot = float(terminus_depth)
        sel = (z >= top) & (z <= bot)
        if sel.sum() == 0:
            raise InvalidWorkflowError(
                f'band {name} [{top}, {bot}] m contains no ocean levels; '
                f'available depths are {z.min():.0f}-{z.max():.0f} m')

        if not (np.isfinite(thetao[:, sel]).all()
                and np.isfinite(so[:, sel]).all()):
            # levels below the sea floor pass the check on the coordinate
            raise InvalidWorkflowError(
                f'band {name} [{top}, {bot}] m has non-finite values; land and '
                f'sub-bathymetry levels must be dropped before this call')

        how = band_weighting.get(name, 'uniform')
        if how == 'depth_weighted':
            w = np.gradient(z)[sel]
            w = w / w.sum()
        elif how == 'uniform':
            w = np.full(int(sel.sum()), 1 / sel.sum())
        else:
            raise InvalidParamsError(f'unknown band weighting {how!r}')

        zc = z[sel]
        tf_z = thetao[:, sel] - freezing_point(so[:, sel], zc, teos10=teos10,
                                               lat=lat)
        tf_b.append(tf_z @ w)
        th_b.append(thetao[:, sel] @ w)
        so_b.append(so[:, sel] @ w)
        names.append(name)
        tops.append(float(top))
        bots.append(float(bot))
        weights.append(how)

    return (names, tops, bots, weights,
            np.array(tf_b).T, np.array(th_b).T, np.array(so_b).T)


def flotation_thickness(water_depth, rho_ocean=None, rho_ice=None):
    """Ice thickness at which a front in this water depth floats, m.

    Parameters
    ----------
    water_depth : float
        m
    rho_ocean, rho_ice : float, optional
        kg m-3. Default: 1028 and 900.
    """
    if rho_ocean is None:
        rho_ocean = DEFAULTS.ocean_water_density
    if rho_ice is None:
        rho_ice = DEFAULTS.ice_density
    return float(water_depth) * rho_ocean / rho_ice


def water_depth(bed_elevation, sea_level=0.):
    """Water depth over a bed, m, positive downwards; negative above the water.

    Parameters
    ----------
    bed_elevation : float or array
        m, in any vertical datum
    sea_level : float
        m, in the same datum
    """
    if np.ndim(bed_elevation) == 0:
        return float(sea_level) - float(bed_elevation)
    return float(sea_level) - np.asarray(bed_elevation, dtype=float)


def submerged_depth(thickness, surface_h, sea_level=0.):
    """Water depth at a front from its thickness and surface elevation, m.

    Parameters
    ----------
    thickness : float
        ice thickness, m
    surface_h : float
        surface elevation, m
    sea_level : float
        m, in the datum of ``surface_h``
    """
    return thickness - (surface_h - sea_level)


def front_thickness(surface_h, water_depth, sea_level=0.):
    """Terminus thickness at a prescribed water depth, m: freeboard plus depth.

    Parameters
    ----------
    surface_h : float
        surface elevation of the terminus, m
    water_depth : float
        m
    sea_level : float
        m, in the datum of ``surface_h``
    """
    return max(float(surface_h) - sea_level, 0.) + float(water_depth)


def ocean_cells(bed, mask=None, surface_class=None, ocean_classes=None,
                sea_level=0., bed_mask='bedmachine'):
    """Cells outside the glacier that are ocean.

    Parameters
    ----------
    bed : 2D array
        bed elevation, m
    mask : 2D array of bool, optional
        the glacier mask. Default: no glacier cell.
    surface_class : 2D array, optional
        the surface-type mask of the bed product
    ocean_classes : sequence, optional
        the values of ``surface_class`` that count as water. Default: those
        of ``bed_mask``.
    sea_level : float
        m, in the datum of ``bed``
    bed_mask : str, :class:`tidewater.bed.BedMask` or None
        the mask convention of the bed product, see
        :data:`tidewater.bed.BED_MASKS`; None ignores ``surface_class``

    Returns
    -------
    (cells, source) : a boolean array, and 'mask' when ``surface_class``
        decided or 'bed_below_sea_level' when every bed below sea level
        outside the glacier counts, the bed under neighbouring ice included
    """
    bed = np.asarray(bed, dtype=float)
    wet = np.isfinite(bed) & (bed < sea_level)
    if mask is not None:
        wet = wet & ~np.asarray(mask, dtype=bool)
    if surface_class is not None:
        if ocean_classes is None:
            from tidewater.bed import bed_mask_convention
            ocean_classes = bed_mask_convention(bed_mask).water_classes
        if len(tuple(ocean_classes)):
            cls = np.round(np.asarray(surface_class, dtype=float))
            return wet & np.isin(cls, tuple(ocean_classes)), 'mask'
    return wet, 'bed_below_sea_level'


def terminus_water_depth_from_bed(bed, mask, ocean=None, dilate=2,
                                  min_pixels=5, fallback_trim=0, sea_level=0.,
                                  surface_class=None, ocean_classes=None,
                                  bed_mask='bedmachine'):
    """Water depth at the calving front from a map of the bed.

    The median of the bed over the ocean cells within ``dilate`` cells of the
    glacier mask, or over every ocean cell of the map when fewer than
    ``min_pixels`` such cells exist.

    Parameters
    ----------
    bed : 2D array
        bed elevation, m
    mask : 2D array of bool
        the glacier mask
    ocean : 2D array of bool, optional
        the ocean cells. Default: :func:`ocean_cells`.
    dilate : int
        the width of the ring around the glacier mask, in grid cells
    min_pixels : int
        the smallest number of cells a median is taken over
    fallback_trim : int
        cells cut from each edge of the map before the fallback searches it
    sea_level : float
        m, in the datum of ``bed``
    surface_class, ocean_classes, bed_mask :
        see :func:`ocean_cells`

    Returns
    -------
    (depth, n, source) : the depth in m, positive downwards, the number of
        cells behind it, and 'ring' or 'fallback'
    """
    from scipy import ndimage

    bed = np.asarray(bed)
    mask = np.asarray(mask, dtype=bool)
    if ocean is None:
        ocean, _ = ocean_cells(bed, mask, surface_class=surface_class,
                               ocean_classes=ocean_classes,
                               sea_level=sea_level, bed_mask=bed_mask)

    ring = ndimage.binary_dilation(mask, iterations=int(dilate)) & ~mask
    wet = bed[ring & ocean]
    source = 'ring'
    if wet.size < min_pixels:
        source = 'fallback'
        sea = ocean.copy()
        if fallback_trim:
            t = int(fallback_trim)
            sea[:t, :] = sea[-t:, :] = sea[:, :t] = sea[:, -t:] = False
        wet = bed[sea]
    if wet.size < min_pixels:
        raise InvalidWorkflowError(
            f'only {wet.size} ocean cells in the map; cannot set a water '
            'depth.')
    return water_depth(np.median(wet), sea_level), int(wet.size), source
