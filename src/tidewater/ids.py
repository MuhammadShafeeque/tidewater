"""Glacier id helpers: RGI 6 and 7 parsing, host id forms, link-table mapping."""

import re
from typing import NamedTuple

import pandas as pd

_RGI_OLD = re.compile(r"^RGI(\d)(\d)-(\d{2})\.(\d{5})$")
_RGI_V7 = re.compile(r"^RGI2000-v(\d+\.\d+)-([GC])-(\d{2})-(\d{5})$")


class GlacierId(NamedTuple):
    """A parsed glacier id; the fields are None for an unrecognised id."""

    raw: str
    version: str | None = None
    region: int | None = None
    number: int | None = None
    kind: str | None = None

    @property
    def is_rgi(self):
        return self.version is not None


def parse_id(glacier_id):
    """Parse an id; unrecognised formats come back as opaque strings."""
    raw = str(glacier_id).strip()
    m = _RGI_OLD.match(raw)
    if m:
        return GlacierId(raw, f"{m[1]}.{m[2]}", int(m[3]), int(m[4]), "G")
    m = _RGI_V7.match(raw)
    if m:
        return GlacierId(raw, m[1], int(m[3]), int(m[4]), m[2])
    return GlacierId(raw)


def rgi_version(glacier_id):
    """RGI version of an id, such as "6.0" or "7.0", or None."""
    return parse_id(glacier_id).version


def format_rgi(region, number, version="6.0", kind="G"):
    """Build an RGI id from its parts."""
    region, number = int(region), int(number)
    major = int(str(version).split(".")[0])
    if major >= 7:
        if kind not in ("G", "C"):
            raise ValueError(f'kind must be "G" or "C", got {kind!r}')
        return f"RGI2000-v{version}-{kind}-{region:02d}-{number:05d}"
    return f"RGI{str(version).replace('.', '')}-{region:02d}.{number:05d}"


def to_pygem_glacno(glacier_id):
    """PyGEM ``glacno`` of an RGI 6 id ("1.00570") or an RGI 7.0 glacier id ("01-00570").

    Unrecognised ids are returned unchanged. The RGI 7.0 form needs a PyGEM
    run on RGI 7.0; other ids are refused: map them first with :func:`map_ids`.
    """
    gid = parse_id(glacier_id)
    if not gid.is_rgi:
        return gid.raw
    if gid.version.startswith("6"):
        return f"{gid.region}.{gid.number:05d}"
    if gid.version == "7.0" and gid.kind == "G":
        return f"{gid.region:02d}-{gid.number:05d}"
    raise ValueError(f"{gid.raw} is RGI {gid.version} {gid.kind}; glacno is an "
                     "RGI 6 or RGI 7.0 glacier number, map the id first")


def from_pygem_glacno(glacno, version="6.0"):
    """RGI id of a PyGEM ``glacno``: "1.00570" or 1.0057 (``version``), "01-00570" (RGI 7.0)."""
    text = f"{glacno:.5f}" if isinstance(glacno, float) else str(glacno).strip()
    m = re.match(r"^(\d{2})-(\d{5})$", text)
    if m:
        return format_rgi(m[1], m[2], version="7.0")
    m = re.match(r"^(\d{1,2})\.(\d{5})$", text)
    if not m:
        raise ValueError(f"not a glacno: {glacno!r}")
    return format_rgi(m[1], m[2], version=version)


def to_glogem(glacier_id, region_names=None):
    """GloGEM ``(region, within-region id)``, such as ``(1, "01450")``.

    ``region_names`` maps the region number to the host's directory name.
    Unrecognised ids give ``(None, id)``.
    """
    gid = parse_id(glacier_id)
    if not gid.is_rgi:
        return None, gid.raw
    region = gid.region if region_names is None else region_names[gid.region]
    return region, f"{gid.number:05d}"


def from_glogem(region, number, version="6.0", kind="G", region_names=None):
    """RGI id of a GloGEM region and within-region id."""
    if region_names is not None:
        inverse = {name: num for num, name in region_names.items()}
        region = inverse.get(region, region)
    return format_rgi(region, number, version=version, kind=kind)


def map_ids(glacier_ids, link, source, target, weight=None, missing="keep",
            multiple="raise"):
    """Map ids through a caller-supplied link table.

    Parameters
    ----------
    glacier_ids : iterable of str
    link : pandas.DataFrame
        one row per linked pair, for instance an RGI 6 to RGI 7 link table
    source, target : str
        the columns of ``link`` holding the ids to map from and to
    weight : str, optional
        column ranking the candidates of a one-to-many link (the largest
        wins), for instance an overlap area
    missing : "keep", "none" or "raise"
        for ids absent from the table: pass through, give None, or fail
    multiple : "raise" or "first"
        for ids with several candidates when ``weight`` is not given

    Returns
    -------
    list of str, in the order of ``glacier_ids``
    """
    if missing not in ("keep", "none", "raise"):
        raise ValueError(f"unknown missing={missing!r}")
    if multiple not in ("raise", "first"):
        raise ValueError(f"unknown multiple={multiple!r}")
    glacier_ids = [str(g) for g in glacier_ids]
    table = link.dropna(subset=[source, target])
    if weight is not None:
        table = table.sort_values(weight, ascending=False, kind="stable")
    elif multiple == "raise":
        pairs = table.drop_duplicates(subset=[source, target])
        dup = pairs[source][pairs[source].duplicated()].unique()
        dup = sorted(set(dup) & set(glacier_ids))
        if dup:
            raise ValueError(f"several candidates for {dup[:5]}; give a "
                             '`weight` column or multiple="first"')
    lookup = table.drop_duplicates(subset=[source]).set_index(source)[target]
    out = []
    for gid in glacier_ids:
        if gid in lookup.index:
            out.append(str(lookup[gid]))
        elif missing == "keep":
            out.append(gid)
        elif missing == "none":
            out.append(None)
        else:
            raise KeyError(f"{gid} is not in the link table")
    return out


def region_of(glacier_ids):
    """Region number of each id as a Series indexed by id; NA when unknown."""
    ids = [str(g) for g in glacier_ids]
    return pd.Series([parse_id(g).region for g in ids], index=ids,
                     dtype="Int64", name="region")
