"""Gridded bed products: mask conventions, sampling and the depth at the front.

A bed product is any :py:class:`xarray.DataArray` of bed elevation on two
horizontal dimensions, projected or geographic. Points are given in the
coordinates of the grid; no reprojection is done here.
"""
from dataclasses import dataclass

import numpy as np

from tidewater.config import InvalidParamsError, InvalidWorkflowError
from tidewater.physics import water_depth

X_NAMES = ('x', 'lon', 'longitude', 'easting', 'xc')
Y_NAMES = ('y', 'lat', 'latitude', 'northing', 'yc')
GEOGRAPHIC = ('lon', 'longitude')
METHODS = ('nearest', 'bilinear')


@dataclass(frozen=True)
class BedMask:
    """What the classes of a bed product's surface-type mask mean.

    Attributes
    ----------
    name : str
    ocean_classes : tuple
        open ocean
    floating_classes : tuple
        floating ice, with sea water below
    ice_classes : tuple
        grounded ice
    land_classes : tuple
        ice-free land
    note : str
    """

    name: str
    ocean_classes: tuple = ()
    floating_classes: tuple = ()
    ice_classes: tuple = ()
    land_classes: tuple = ()
    note: str = ''

    @property
    def water_classes(self):
        """The classes with sea water over the bed."""
        return self.ocean_classes + self.floating_classes

    @property
    def has_classes(self):
        return bool(self.water_classes + self.ice_classes + self.land_classes)


# name -> BedMask
BED_MASKS = {}


def register_bed_mask(name, ocean_classes=(), ice_classes=(),
                      floating_classes=(), land_classes=(), note=''):
    """Add or replace a mask convention in :data:`BED_MASKS`.

    Parameters
    ----------
    name : str
        stored in lower case
    ocean_classes, ice_classes, floating_classes, land_classes : sequence
        the mask values of open ocean, grounded ice, floating ice and
        ice-free land
    note : str
        where the convention comes from

    Returns
    -------
    :class:`BedMask`
    """
    key = str(name).strip().lower()
    entry = BedMask(key, tuple(ocean_classes), tuple(floating_classes),
                    tuple(ice_classes), tuple(land_classes), str(note))
    BED_MASKS[key] = entry
    return entry


register_bed_mask('none', note='no mask: bed elevation and sea level decide')
register_bed_mask('bedmachine', ocean_classes=(0,), land_classes=(1,),
                  ice_classes=(2,), floating_classes=(3,),
                  note='BedMachine Greenland and Antarctica')


def bed_mask_convention(bed_mask):
    """The :class:`BedMask` of a name, of None ('none') or of itself."""
    if isinstance(bed_mask, BedMask):
        return bed_mask
    key = 'none' if bed_mask is None else str(bed_mask).strip().lower()
    if key not in BED_MASKS:
        raise InvalidParamsError(f'unknown bed mask {bed_mask!r}; available: '
                                 f'{sorted(BED_MASKS)}. Add one with '
                                 'register_bed_mask.')
    return BED_MASKS[key]


def _horizontal_dims(da, x_dim=None, y_dim=None):
    names = {str(d).lower(): d for d in da.dims}
    if x_dim is None:
        x_dim = next((names[n] for n in X_NAMES if n in names), None)
    if y_dim is None:
        y_dim = next((names[n] for n in Y_NAMES if n in names), None)
    if x_dim is None or y_dim is None:
        if da.ndim != 2:
            raise InvalidParamsError(
                f'cannot tell the horizontal dimensions of {da.dims}; pass '
                'x_dim and y_dim')
        y_dim, x_dim = da.dims
    for dim in (x_dim, y_dim):
        if dim not in da.dims:
            raise InvalidParamsError(f'{dim!r} is not a dimension of '
                                     f'{da.dims}')
    return x_dim, y_dim


def _grid(bed, var=None, x_dim=None, y_dim=None):
    """Values as (y, x), the two coordinates, and whether x is a longitude."""
    if hasattr(bed, 'data_vars'):
        if var is None:
            if len(bed.data_vars) != 1:
                raise InvalidParamsError('the dataset holds several '
                                         'variables; pass `var`')
            var = list(bed.data_vars)[0]
        bed = bed[var]
    x_dim, y_dim = _horizontal_dims(bed, x_dim, y_dim)
    extra = [d for d in bed.dims if d not in (x_dim, y_dim)]
    if any(bed.sizes[d] != 1 for d in extra):
        raise InvalidParamsError(f'the bed has more than two dimensions: '
                                 f'{bed.dims}')
    bed = bed.squeeze(extra, drop=True).transpose(y_dim, x_dim)
    return (np.asarray(bed.values, dtype=float),
            np.asarray(bed[x_dim].values, dtype=float),
            np.asarray(bed[y_dim].values, dtype=float),
            str(x_dim).lower() in GEOGRAPHIC)


def _fractional_index(coord, points, name):
    """Position of ``points`` on a coordinate, in cells; nan outside the grid."""
    if coord.size < 2:
        raise InvalidParamsError(f'the {name} coordinate has one value')
    index = np.arange(coord.size, dtype=float)
    if coord[0] > coord[-1]:
        coord, index = coord[::-1], index[::-1]
    step = np.diff(coord)
    if not np.all(step > 0):
        raise InvalidParamsError(f'the {name} coordinate is not monotonic')
    pos = np.interp(points, coord, index)
    outside = ((points < coord[0] - step[0] / 2)
               | (points > coord[-1] + step[-1] / 2))
    return np.where(outside | ~np.isfinite(points), np.nan, pos)


def sample_bed(bed, x, y, method='nearest', var=None, x_dim=None, y_dim=None):
    """A gridded field at points, for example the bed along a flowline.

    Parameters
    ----------
    bed : :py:class:`xarray.DataArray` or :py:class:`xarray.Dataset`
        on two horizontal dimensions with monotonic coordinates
    x, y : array
        the points, in the coordinates of the grid (projected or lon, lat)
    method : str
        'nearest' or 'bilinear'
    var : str, optional
        the variable, when ``bed`` is a dataset with several
    x_dim, y_dim : str, optional
        the horizontal dimensions. Default: found by name (x/y, lon/lat,
        longitude/latitude), else the last two in (y, x) order.

    Returns
    -------
    array
        shaped as ``x``; nan outside the grid, and with 'bilinear' wherever
        one of the four neighbours is nan
    """
    if method not in METHODS:
        raise InvalidParamsError(f'unknown method {method!r}; one of '
                                 f'{METHODS}')
    values, xc, yc, geographic = _grid(bed, var, x_dim, y_dim)
    x, y = np.broadcast_arrays(np.asarray(x, dtype=float),
                               np.asarray(y, dtype=float))
    if geographic:
        # the points on the longitude convention of the grid
        x = x % 360. if np.nanmax(xc) > 180. else (x + 180.) % 360. - 180.
    fx = _fractional_index(xc, x.ravel(), 'x')
    fy = _fractional_index(yc, y.ravel(), 'y')
    inside = np.isfinite(fx) & np.isfinite(fy)
    fx, fy = np.where(inside, fx, 0.), np.where(inside, fy, 0.)

    if method == 'nearest':
        out = values[np.floor(fy + 0.5).astype(int),
                     np.floor(fx + 0.5).astype(int)]
    else:
        j = np.clip(np.floor(fy).astype(int), 0, yc.size - 2)
        i = np.clip(np.floor(fx).astype(int), 0, xc.size - 2)
        ty, tx = fy - j, fx - i
        out = ((1 - ty) * ((1 - tx) * values[j, i] + tx * values[j, i + 1])
               + ty * ((1 - tx) * values[j + 1, i] + tx * values[j + 1, i + 1]))
    return np.where(inside, out, np.nan).reshape(x.shape)


def terminus_water_depth(bed, x, y, sea_level=0., surface_class=None,
                         bed_mask='bedmachine', n_front=1, method='nearest',
                         var=None, x_dim=None, y_dim=None):
    """Water depth at the calving front from a bed product and a flowline.

    The median bed over the last ``n_front`` points of the line, below
    ``sea_level``. A mask convention with classes overrules the bed: a front
    on ice-free land has no water.

    Parameters
    ----------
    bed : :py:class:`xarray.DataArray` or :py:class:`xarray.Dataset`
        bed elevation, m
    x, y : array
        the flowline points, ordered towards the front, in the coordinates
        of the grid
    sea_level : float
        m, in the datum of ``bed``
    surface_class : :py:class:`xarray.DataArray`, optional
        the surface-type mask of the bed product, on any grid that holds
        the points
    bed_mask : str, :class:`BedMask` or None
        the convention of ``surface_class``, see :data:`BED_MASKS`
    n_front : int
        the number of points at the end of the line the median is taken over
    method : str
        'nearest' or 'bilinear', for the bed
    var, x_dim, y_dim :
        see :func:`sample_bed`

    Returns
    -------
    (depth, n, source) : the depth in m, positive downwards and never
        negative, the number of points behind it, and 'bed', 'mask' (the
        mask shows water or ice at the front), 'land' (the mask shows
        ice-free land, depth 0) or 'dry' (the bed is not below sea level,
        depth 0)
    """
    x, y = np.atleast_1d(x)[-int(n_front):], np.atleast_1d(y)[-int(n_front):]
    z = sample_bed(bed, x, y, method=method, var=var, x_dim=x_dim, y_dim=y_dim)
    ok = np.isfinite(z)
    if not ok.any():
        raise InvalidWorkflowError('the bed product holds no value at the '
                                   'front; cannot set a water depth.')
    n, source = int(ok.sum()), 'bed'
    convention = bed_mask_convention(bed_mask)
    if surface_class is not None and convention.has_classes:
        cls = np.round(sample_bed(surface_class, x, y, x_dim=x_dim,
                                  y_dim=y_dim))[ok]
        if cls.size and np.isin(cls, convention.land_classes).all():
            return 0., n, 'land'
        marine = convention.water_classes + convention.ice_classes
        if np.isin(cls, marine).any():
            source = 'mask'
    depth = water_depth(np.median(z[ok]), sea_level)
    if depth <= 0:
        return 0., n, 'dry'
    return depth, n, source
