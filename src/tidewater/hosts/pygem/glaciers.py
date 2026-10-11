"""Glacier ids, model years and the calving front as PyGEM has them.

Nothing here imports PyGEM, so it works where PyGEM is installed but not
configured.
"""
from collections.abc import Mapping

import numpy as np
import pandas as pd

from tidewater import laws as _laws
from tidewater.config import DEFAULTS, InvalidWorkflowError
from tidewater.ids import from_pygem_glacno, parse_id, to_pygem_glacno
from tidewater.physics import submerged_depth
from tidewater.timeconv import yearly_ocean_factor

# PyGEM's dynamics run on calendar years only (``sim_wateryear: calendar``)
START_MONTH = 1
LABEL = 'end'


def rgi_id(glacier):
    """The RGI id of a PyGEM glacier number (``1.03622``, ``05-09345``) or of an RGI id."""
    gid = parse_id(glacier) if isinstance(glacier, str) else None
    if gid is not None and gid.is_rgi:
        return gid.raw
    return from_pygem_glacno(glacier)


def glacno(glacier):
    """The PyGEM glacier number (``'1.03622'``, ``'05-09345'``) of an RGI id or a number."""
    return to_pygem_glacno(rgi_id(glacier))


def glacier_of(mb_model):
    """The RGI id of the glacier a PyGEM mass-balance model runs."""
    table = getattr(mb_model, 'glacier_rgi_table', None)
    if table is None:
        raise InvalidWorkflowError('the mass-balance model carries no '
                                   'glacier_rgi_table')
    return rgi_id(table['RGIId'])


def by_glacier(values):
    """A mapping keyed by RGI id from one keyed by any id form."""
    return {rgi_id(k): v for k, v in dict(values).items()}


def front_cell(flowline, water_level):
    """The cell PyGEM's calving law reads, or None where it does not calve."""
    thick = np.asarray(flowline.thick)
    if not (thick > 0).any():
        return None
    cells = np.flatnonzero((np.asarray(flowline.surface_h) > water_level)
                           & (thick > 0))
    if not cells.size:
        cells = np.flatnonzero((np.asarray(flowline.bed_h) <= water_level)
                               & (thick > 0))
    i = int(cells[-1])
    return i if flowline.bed_h[i] < water_level else None


def cell_geometry(flowline, cell, water_level):
    """Thickness, water depth and width (m) of a front at ``cell``."""
    h = float(flowline.thick[cell])
    return {'thickness': h,
            'water_depth': float(submerged_depth(h, flowline.surface_h[cell],
                                                 water_level)),
            'width': float(flowline.widths_m[cell])}


def front_geometry(flowline, water_level):
    """:func:`cell_geometry` at :func:`front_cell`, or None.

    The keys are those :func:`tidewater.laws.ocean_calving_law` reads for a
    partition (add ``calving_k`` there).
    """
    i = front_cell(flowline, water_level)
    return None if i is None else cell_geometry(flowline, i, water_level)


def yearly_factors(tf_monthly, tf_ref, gamma=None, tf_fraction=None,
                   weights='days', incomplete='raise'):
    """``k(year) / k0`` of ``tf_power`` per PyGEM model year and glacier.

    The mean of the monthly ocean factors over each calendar year
    (:func:`tidewater.timeconv.yearly_ocean_factor`), then
    ``(1 - a) + a * factor``.

    Parameters
    ----------
    tf_monthly : pandas DataFrame or Series indexed by time
        monthly thermal forcing, degC; the columns of a DataFrame are
        glaciers, in any id form
    tf_ref : float or mapping
        degC, > 0, one value or per glacier
    gamma, tf_fraction : float, optional
        default: 1.18 and 1
    weights, incomplete :
        see :func:`tidewater.timeconv.to_model_years`

    Returns
    -------
    the same kind of object, indexed by the integer model year, with RGI
    ids as columns
    """
    if isinstance(tf_monthly, pd.DataFrame):
        tf_monthly = tf_monthly.rename(columns=rgi_id)
        if isinstance(tf_ref, (Mapping, pd.Series)):
            tf_ref = by_glacier(tf_ref)
    factor = yearly_ocean_factor(tf_monthly, tf_ref, gamma,
                                 start_month=START_MONTH, label=LABEL,
                                 weights=weights, incomplete=incomplete)
    a = DEFAULTS.calving_tf_fraction if tf_fraction is None else tf_fraction
    return factor if a == 1 else (1 - a) + a * factor


def laws_from_forcing(forcing, calving_law='tf_power', **law_kwargs):
    """One calving law per glacier from its ocean forcing.

    Parameters
    ----------
    forcing : mapping
        glacier id (any form) -> forcing dataset or the path of a forcing
        file (:mod:`tidewater.forcing.io`)
    calving_law : str
        a name in :data:`tidewater.laws.LAWS`
    **law_kwargs
        passed to :func:`tidewater.laws.ocean_calving_law` for every glacier

    Returns
    -------
    dict
        RGI id -> law
    """
    from tidewater.forcing.io import read_forcing

    out = {}
    for gid, ds in by_glacier(forcing).items():
        if not hasattr(ds, 'data_vars'):
            ds = read_forcing(ds)
        out[gid] = _laws.ocean_calving_law(ds, calving_law, **law_kwargs)
    return out
