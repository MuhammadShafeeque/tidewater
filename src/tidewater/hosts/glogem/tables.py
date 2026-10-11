"""Per-glacier, per-model-year tables in the layout GloGEM reads.

GloGEM computes frontal ablation once per model year. Its year ``Y`` runs from
the start month of calendar year ``Y - 1`` to the month before it in ``Y``
(labelled by the year in which it ends). The start month is set per region,
not per glacier: October, except April for RGI regions 16 to 19 (Low
Latitudes, Southern Andes, New Zealand, Antarctic and Subantarctic). Daily runs
start on day 274 (1 October) or day 121 (1 May) of a 365-day year.
"""

from pathlib import Path

import pandas as pd

from tidewater.export import write_host_bundle, write_region_tables
from tidewater.ids import parse_id
from tidewater.timeconv import model_year_span, yearly_ocean_factor

IDL_DIR = Path(__file__).parent / "idl"
LABEL = "end"
SOUTH_REGIONS = (16, 17, 18, 19)
START_MONTHS = {"monthly": (10, 4), "daily": (10, 5)}
OUTSIDE = ("stop", "one", "nearest")


def start_month(region, time_resolution="monthly"):
    """First calendar month of GloGEM's model year in an RGI region."""
    if time_resolution not in START_MONTHS:
        raise ValueError(f'time_resolution must be "monthly" or "daily", '
                         f"got {time_resolution!r}")
    north, south = START_MONTHS[time_resolution]
    return south if int(region) in SOUTH_REGIONS else north


def _parsed(glacier_ids):
    ids = [parse_id(g) for g in glacier_ids]
    bad = [g.raw for g in ids if not g.is_rgi]
    if bad:
        raise ValueError(f"GloGEM needs RGI ids (region and number): {bad[:5]}")
    versions = {g.version for g in ids}
    if len(versions) > 1:
        raise ValueError(f"ids of several RGI versions: {sorted(versions)}; "
                         "map them to one with tidewater.ids.map_ids")
    return ids


def start_months(glacier_ids, time_resolution="monthly"):
    """GloGEM's start month of each glacier, keyed by id."""
    return {g.raw: start_month(g.region, time_resolution)
            for g in _parsed(glacier_ids)}


def forcing_span(first_year, last_year, region, time_resolution="monthly"):
    """First and last calendar (year, month) of a GloGEM run ``tran``.

    The monthly forcing must cover this span for every model year of the run
    to be complete.
    """
    month = start_month(region, time_resolution)
    return (model_year_span(first_year, month, LABEL)[0],
            model_year_span(last_year, month, LABEL)[1])


def write_tables(directory, yearly, variable="ocean_factor", units="1",
                 time_resolution="monthly", weights="days", outside="stop",
                 params=None, meta=None, fmt="%.6f"):
    """Write a table on GloGEM's model years, one file per RGI region.

    The files are those of :func:`tidewater.export.write_host_bundle`, named
    ``<variable>_<RR>.dat`` after the two-digit RGI region, with the rows keyed
    by the five-digit glacier number within the region (GloGEM's glacier
    id). The header of each file states its start month, label, RGI version
    and time resolution, which the IDL reader checks against the run.

    Parameters
    ----------
    directory : path
    yearly : pandas.DataFrame
        index: integer model years on GloGEM's convention; columns: RGI ids
    variable, units : str
    time_resolution : "monthly" or "daily"
        of the GloGEM run
    weights : str
        the month weighting the yearly values were aggregated with
    outside : "stop", "one" or "nearest"
        what the run does for a year or value the table lacks: stop, use 1,
        or use the nearest year of the glacier
    params, meta : mapping, optional
        see :func:`tidewater.export.write_host_bundle`

    Returns
    -------
    dict : the sidecar
    """
    if outside not in OUTSIDE:
        raise ValueError(f"outside must be one of {OUTSIDE}, got {outside!r}")
    ids = _parsed(yearly.columns)
    months = {g.raw: start_month(g.region, time_resolution) for g in ids}
    host = {"host": "glogem", "time_resolution": time_resolution,
            "rgi_version": ids[0].version, "outside": outside}
    info = write_host_bundle(directory, yearly, params=params,
                             start_month=months, label=LABEL, weights=weights,
                             variable=variable, units=units, fmt=fmt,
                             meta={**host, **dict(meta or {})})
    groups = {}
    for gid in ids:
        groups.setdefault(gid.region, []).append(gid.raw)
    for region, gids in groups.items():
        write_region_tables(
            yearly[gids], directory, variable=variable, units=units, fmt=fmt,
            meta={"start_month": start_month(region, time_resolution),
                  "label": LABEL, "weights": weights, **host})
    return info


def write_ocean_factor_tables(directory, tf_monthly, tf_ref, gamma=None,
                              tf_fraction=1.0, time_resolution="monthly",
                              weights="days", incomplete="raise",
                              outside="stop", params=None, meta=None,
                              fmt="%.6f"):
    """Write the yearly ocean factor of each glacier on GloGEM's model years.

    The factor of a model year is the mean of the monthly factors over it
    (:func:`tidewater.timeconv.yearly_ocean_factor`), with the start month
    GloGEM uses in the glacier's region. With ``tf_fraction`` ``a`` below 1
    the table holds ``(1 - a) + a * F``. GloGEM multiplies its calving
    constant by it, so ``tf_ref`` is best the power mean of the thermal
    forcing over the period that constant was calibrated on.

    Parameters
    ----------
    directory : path
    tf_monthly : pandas.DataFrame
        monthly thermal forcing, degC, indexed by dates; columns: RGI ids
    tf_ref : float or mapping
        reference thermal forcing, degC, one value or per glacier
    gamma : float, optional
    tf_fraction : float
    time_resolution : "monthly" or "daily"
    weights, incomplete :
        see :func:`tidewater.timeconv.yearly_ocean_factor`
    outside, params, meta, fmt :
        see :func:`write_tables`

    Returns
    -------
    (pandas.DataFrame, dict) : the yearly table and the sidecar
    """
    if not 0 <= tf_fraction <= 1:
        raise ValueError(f"tf_fraction must be in [0, 1], got {tf_fraction}")
    factor = yearly_ocean_factor(
        tf_monthly, tf_ref, gamma, start_month=start_months(
            tf_monthly.columns, time_resolution),
        label=LABEL, weights=weights, incomplete=incomplete)
    if tf_fraction != 1:
        factor = (1 - tf_fraction) + tf_fraction * factor
    factor = pd.DataFrame(factor)
    meta = {"tf_fraction": tf_fraction, **dict(meta or {})}
    if gamma is not None:
        meta["gamma"] = gamma
    info = write_tables(directory, factor, time_resolution=time_resolution,
                        weights=weights, outside=outside, params=params,
                        meta=meta, fmt=fmt)
    return factor, info
