"""Writers and readers that hand results to host models."""

import dataclasses
import json
from pathlib import Path

import cftime
import numpy as np
import pandas as pd
import xarray as xr

from . import __version__
from .calibration import column_units
from .ids import parse_id, to_glogem

HEADER_LINES = 4
MISSING = -9999.0
SIDECAR = "tidewater_export.json"
CALIBRATION_FILE = "calibration.csv"
PERIOD_CONVENTION = ("a period is (first year, last year), both included; a "
                     "model run that holds it starts at the first year and "
                     "ends at the last year plus one")

ID_COLUMN = "glacier_id"
K_COLUMN = "calving_k"
K_UNC_COLUMN = "calving_k_unc"
PYGEM_COLUMNS = {ID_COLUMN: "RGIId", K_COLUMN: "calving_k",
                 K_UNC_COLUMN: "calving_k_nmad"}


def _token(value, what):
    text = str(value)
    if not text or any(c.isspace() for c in text) or "=" in text:
        raise ValueError(f"{what} {text!r} must be free of whitespace and '='")
    return text


def _region_label(glacier_id, regions):
    if regions is not None:
        return _token(regions[glacier_id], "region")
    region = parse_id(glacier_id).region
    if region is None:
        raise ValueError(f"no region for {glacier_id!r}: pass `regions`")
    return f"{region:02d}"


def write_region_tables(data, directory, variable="ocean_factor", units="1",
                        regions=None, id_style="within_region", meta=None,
                        filename="{variable}_{region}.dat", fmt="%.6f",
                        missing=MISSING):
    """Write a per-glacier, per-model-year table as text, one file per region.

    Each file has ``HEADER_LINES`` header lines, the last one naming the
    columns, then one whitespace-delimited row per glacier: its id and one
    value per model year.

    Parameters
    ----------
    data : pandas.DataFrame
        index: integer model years; columns: glacier ids
    directory : path
    variable, units : str
        written to the header, for instance "ocean_factor" and "1"
    regions : mapping, optional
        glacier id -> region label; the default is the RGI region of the id
    id_style : "within_region" or "full"
        row ids as the five-digit number within the region, or unchanged
    meta : mapping, optional
        key -> value pairs for the header, for instance the start month and
        the label convention of the model years
    filename : str
        pattern with ``{variable}`` and ``{region}``
    fmt : str
        number format
    missing : float
        written in place of NaN

    Returns
    -------
    dict : region label -> path
    """
    if id_style not in ("within_region", "full"):
        raise ValueError(f"unknown id_style {id_style!r}")
    years = [int(y) for y in data.index]
    if list(data.index) != years:
        raise ValueError("the index must hold integer model years")
    if not data.columns.is_unique:
        raise ValueError("glacier ids must be unique")
    variable = _token(variable, "variable")
    units = _token(units, "units")
    meta_line = " ".join(f"{_token(k, 'meta key')}={_token(v, 'meta value')}"
                         for k, v in (meta or {}).items())
    groups = {}
    for gid in data.columns:
        groups.setdefault(_region_label(gid, regions), []).append(gid)
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    paths = {}
    for region, gids in sorted(groups.items()):
        rows = [_token(to_glogem(g)[1] if id_style == "within_region" else g,
                       "glacier id") for g in gids]
        if len(set(rows)) != len(rows):
            raise ValueError(f"row ids are not unique in region {region}; "
                             'use id_style="full"')
        values = data[gids].to_numpy(dtype=float).T
        values = np.where(np.isfinite(values), values, missing)
        lines = [
            f"# tidewater {__version__} variable={variable} units={units} "
            f"region={region}",
            f"# {meta_line}".rstrip(),
            f"# missing={missing:g} n_glaciers={len(gids)} "
            f"n_years={len(years)} id_style={id_style}",
            " ".join(["id"] + [str(y) for y in years]),
        ]
        for row, vals in zip(rows, values):
            lines.append(" ".join([row] + [fmt % v for v in vals]))
        path = directory / filename.format(variable=variable, region=region)
        path.write_text("\n".join(lines) + "\n")
        paths[region] = path
    return paths


def read_region_table(path):
    """Read a file of :func:`write_region_tables`.

    Returns
    -------
    (pandas.DataFrame, dict) : the table (index: model years, columns: row
        ids as strings, NaN for missing) and the header metadata
    """
    with open(path) as f:
        lines = f.read().splitlines()
    meta = {}
    for line in lines[:HEADER_LINES - 1]:
        for item in line.lstrip("#").split():
            if "=" in item:
                key, value = item.split("=", 1)
                meta[key] = value
    years = [int(y) for y in lines[HEADER_LINES - 1].split()[1:]]
    rows = [line.split() for line in lines[HEADER_LINES:] if line.strip()]
    values = np.array([r[1:] for r in rows], dtype=float).reshape(
        len(rows), len(years))
    values[values == float(meta.get("missing", MISSING))] = np.nan
    frame = pd.DataFrame(values.T, index=pd.Index(years, name="year"),
                         columns=[r[0] for r in rows])
    return frame, meta


def _column_map(columns):
    if columns is None:
        return {}
    if isinstance(columns, str):
        if columns != "pygem":
            raise ValueError(f"unknown column mapping {columns!r}")
        return dict(PYGEM_COLUMNS)
    return dict(columns)


def write_calibration_csv(table, path, columns=None, float_format=None):
    """Write calibrated calving constants keyed by glacier id.

    Parameters
    ----------
    table : pandas.DataFrame
        ``calving_k`` (yr-1) and ``calving_k_unc`` (yr-1, the caller's choice
        of spread), with the glacier ids in the index or in ``glacier_id``;
        further columns are kept
    path : path
    columns : mapping or "pygem", optional
        renames the columns on writing; "pygem" gives ``RGIId, calving_k,
        calving_k_nmad``
    float_format : str, optional
    """
    frame = table.copy()
    if ID_COLUMN not in frame.columns:
        frame = frame.rename_axis(ID_COLUMN).reset_index()
    lacking = {ID_COLUMN, K_COLUMN, K_UNC_COLUMN} - set(frame.columns)
    if lacking:
        raise ValueError(f"missing columns: {sorted(lacking)}")
    if frame[ID_COLUMN].duplicated().any():
        raise ValueError("glacier ids must be unique")
    first = [ID_COLUMN, K_COLUMN, K_UNC_COLUMN]
    frame = frame[first + [c for c in frame.columns if c not in first]]
    frame = frame.rename(columns=_column_map(columns))
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, float_format=float_format)
    return path


def read_calibration_csv(path, columns=None):
    """Read a calibration CSV back into the column names of this package."""
    mapping = _column_map(columns)
    id_column = mapping.get(ID_COLUMN, ID_COLUMN)
    frame = pd.read_csv(path, dtype={id_column: str})
    return frame.rename(columns={v: k for k, v in mapping.items()})


def _time_axis(index, calendar):
    """A time axis and, for integer model years, the years themselves."""
    values = np.asarray(index)
    if np.issubdtype(values.dtype, np.integer):
        cal = calendar or "standard"
        return [cftime.datetime(int(y), 1, 1, calendar=cal)
                for y in values], values.astype("int32")
    if isinstance(index, pd.PeriodIndex):
        index = index.to_timestamp()
    return index, None


def write_netcdf(data, path, variable="ocean_factor", units="1", lon=None,
                 lat=None, calendar=None, long_name=None, attrs=None,
                 time_units="days since 1850-01-01 00:00:00"):
    """Write a ``(time, rgi_id)`` NetCDF file with CF time.

    Parameters
    ----------
    data : pandas.DataFrame
        index: dates (pandas or cftime) or integer model years; columns:
        glacier ids. Integer years are stamped at 1 January and also kept as
        the ``year`` coordinate.
    path : path
    variable, units, long_name : str
    lon, lat : mapping or sequence, optional
        terminus position per glacier id, degrees east and north
    calendar : str, optional
        CF calendar; the default is that of the dates
    attrs : mapping, optional
        global attributes, for instance the model-year convention
    time_units : str
    """
    time, years = _time_axis(data.index, calendar)
    ids = [str(c) for c in data.columns]
    ds = xr.Dataset(
        {variable: (("time", "rgi_id"), data.to_numpy(dtype=float))},
        coords={"time": time, "rgi_id": ids})
    ds[variable].attrs["units"] = units
    if long_name:
        ds[variable].attrs["long_name"] = long_name
    for name, values, unit in (("lon", lon, "degrees_east"),
                               ("lat", lat, "degrees_north")):
        if values is None:
            continue
        if hasattr(values, "keys"):
            values = [values[c] for c in data.columns]
        ds.coords[name] = ("rgi_id", np.asarray(values, dtype=float))
        ds[name].attrs.update(
            units=unit,
            standard_name="longitude" if name == "lon" else "latitude")
    if years is not None:
        ds.coords["year"] = ("time", years)
        ds["year"].attrs["long_name"] = "model year"
    ds["time"].attrs["standard_name"] = "time"
    ds.attrs.update({"title": f"tidewater {variable}",
                     "tidewater_version": __version__, **(attrs or {})})
    encoding = {"time": {"units": time_units, "dtype": "float64"},
                variable: {"_FillValue": np.nan}}
    if calendar is not None:
        encoding["time"]["calendar"] = calendar
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    ds.to_netcdf(path, encoding=encoding)
    return path


def read_netcdf(path, variable=None):
    """Read a file of :func:`write_netcdf` into memory.

    Returns the Dataset, or the DataArray of ``variable`` when it is given.
    """
    with xr.open_dataset(path) as ds:
        ds = ds.load()
    return ds if variable is None else ds[variable]


def _plain(value):
    """A value as JSON can hold it."""
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        value = dataclasses.asdict(value)
    if isinstance(value, dict):
        return {str(k): _plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set, frozenset, np.ndarray)):
        return [_plain(v) for v in value]
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    return str(value)


def write_host_bundle(directory, factor, calibration=None, params=None,
                      flux_unit=None, start_month=1, label="end",
                      weights="days", variable="ocean_factor", units="1",
                      regions=None, id_style="within_region",
                      calibration_columns=None, meta=None, fmt="%.6f"):
    """Write all a host that cannot import Python needs, for any set of glaciers.

    The set is a region or the globe: one text table per region
    (:func:`write_region_tables`), one calibration CSV
    (:func:`write_calibration_csv`) and a JSON sidecar naming the files, the
    units, the conventions, the parameters and the package version.

    Parameters
    ----------
    directory : path
    factor : pandas.DataFrame
        index: integer model years; columns: glacier ids; for instance the
        table of :func:`tidewater.timeconv.yearly_ocean_factor`
    calibration : pandas.DataFrame, optional
        ``calving_k`` and ``calving_k_unc`` by glacier id, for instance a
        table of :mod:`tidewater.calibration`; its list columns
        (``history``, ``rules``) are not written
    params : mapping or dataclass, optional
        the parameters the tables were computed with
    flux_unit : str, optional
        the unit of the flux columns of ``calibration``. Default: its
        ``attrs['flux_unit']``.
    start_month : int or mapping
        first month of the model year, one value or glacier id -> month
    label : "end" or "start"
        which calendar year names a model year
    weights : "days" or "equal"
        the month weighting the yearly values were aggregated with
    variable, units : str
        of ``factor``
    regions, id_style, fmt :
        see :func:`write_region_tables`
    calibration_columns : mapping or "pygem", optional
        see :func:`write_calibration_csv`
    meta : mapping, optional
        further entries of the sidecar

    Returns
    -------
    dict : the sidecar, with its own path under ``sidecar``
    """
    directory = Path(directory)
    per_glacier = hasattr(start_month, "keys")
    tables = write_region_tables(
        factor, directory, variable=variable, units=units, regions=regions,
        id_style=id_style, fmt=fmt,
        meta={"start_month": "per_glacier" if per_glacier else start_month,
              "label": label, "weights": weights})
    glaciers = {}
    for gid in factor.columns:
        glaciers.setdefault(_region_label(gid, regions), []).append(str(gid))
    info = {
        "tidewater_version": __version__,
        "variable": variable,
        "units": {variable: units, "year": "model year"},
        "period_convention": PERIOD_CONVENTION,
        "model_year": {
            "start_month": ({str(k): int(v) for k, v in start_month.items()}
                            if per_glacier else start_month),
            "label": label, "weights": weights,
            "rule": ("a yearly ocean factor is the mean of the monthly "
                     "factors over the model year, not the factor of the "
                     "yearly mean thermal forcing")},
        "years": [int(y) for y in factor.index],
        "tables": {r: path.name for r, path in tables.items()},
        "glaciers": glaciers,
        "header_lines": HEADER_LINES, "missing": MISSING,
        "id_style": id_style,
        "parameters": _plain(params) if params is not None else {},
        **_plain(dict(meta or {})),
    }
    if calibration is not None:
        flux_unit = flux_unit or calibration.attrs.get("flux_unit")
        frame = calibration.drop(columns=[c for c in ("history", "rules")
                                          if c in calibration.columns])
        write_calibration_csv(frame, directory / CALIBRATION_FILE,
                              columns=calibration_columns)
        known = column_units(flux_unit or "unit of the calibration")
        renamed = _column_map(calibration_columns)
        names = [ID_COLUMN] + [c for c in frame.columns if c != ID_COLUMN]
        info["calibration"] = {
            "file": CALIBRATION_FILE, "flux_unit": flux_unit,
            "columns": renamed,
            "units": {renamed.get(c, c): known.get(c, "") for c in names}}
        info["units"].update(info["calibration"]["units"])
    path = directory / SIDECAR
    path.write_text(json.dumps(info, indent=2) + "\n")
    return dict(info, sidecar=path)


def read_host_bundle(directory):
    """Read a directory of :func:`write_host_bundle`.

    Returns
    -------
    dict : ``factor`` (index: model years, columns: full glacier ids),
        ``calibration`` (the names of this package; None when not written)
        and ``meta`` (the sidecar)
    """
    directory = Path(directory)
    meta = json.loads((directory / SIDECAR).read_text())
    parts = []
    for region, name in meta["tables"].items():
        frame, _ = read_region_table(directory / name)
        frame.columns = meta["glaciers"][region]
        parts.append(frame)
    factor = pd.concat(parts, axis=1) if parts else pd.DataFrame()
    calibration = None
    if "calibration" in meta:
        cal = meta["calibration"]
        calibration = read_calibration_csv(directory / cal["file"],
                                           columns=cal["columns"] or None)
        calibration.attrs["flux_unit"] = cal["flux_unit"]
    return {"factor": factor, "calibration": calibration, "meta": meta}
