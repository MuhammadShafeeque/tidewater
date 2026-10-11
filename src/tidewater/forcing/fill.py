"""Fill a depth band a product's footprint does not reach, from the same
product's nearest glaciers.

A band is absent from a forcing file when no cell in reach of the glacier has
water at its depth. It is filled with the inverse-distance mean of that band
at the ``k`` nearest glaciers where the same product carries it, so the
product keeps its own water mass and variability. A glacier the product has
no file for gets every band that way. Fill the reference product first.
"""
import os
from pathlib import Path

import netCDF4 as nc
import numpy as np
import pandas as pd

from tidewater.config import InvalidParamsError, InvalidWorkflowError
from tidewater.forcing.footprint import haversine_km
from tidewater.forcing.io import BAND_VARS, _pad, chars
from tidewater.periods import as_period, year_mask


def read_band(path, band):
    """The band's series and geometry, or None when the file does not carry it natively."""
    with nc.Dataset(path) as d:
        names = chars(d['band_name'])
        if band not in names or band in getattr(d, 'filled_bands', '').split():
            return None
        i = names.index(band)
        t = d['time']
        out = {'time': nc.num2date(t[:], t.units, t.calendar), 'names': names,
               'top': float(d['band_top'][i]), 'bottom': float(d['band_bottom'][i]),
               'weighting': chars(d['band_weighting'])[i]}
        for v in BAND_VARS:
            out[v] = np.asarray(d[v][:, i], dtype=float)
        return out


def neighbours(gid, have, entities, k, max_km, lon='lon', lat='lat', group=None):
    """The k nearest glaciers in ``have`` within max_km: [(id, km)].

    With ``group`` (a column of ``entities``), glaciers of the same group are
    taken when there are any in reach.
    """
    cand = entities.loc[sorted(have)]
    d = pd.Series(haversine_km(entities.at[gid, lon], entities.at[gid, lat],
                               cand[lon].to_numpy(float), cand[lat].to_numpy(float)),
                  index=cand.index)
    d = d[d <= max_km].sort_values(kind='stable')
    if group is not None:
        same = d[cand.loc[d.index, group] == entities.at[gid, group]]
        d = same if len(same) else d
    return list(d.iloc[:k].items())


def ref_mean(series, time, period=None):
    """Mean of a monthly series over the years of ``period``, both included (all if None)."""
    if period is None:
        return float(np.mean(series))
    return float(series[year_mask([t.year for t in time], period)].mean())


def _position(names, name, order):
    """Where ``name`` goes among ``names`` to keep the band order of ``order``."""
    before = set(order[:order.index(name)]) if name in order else set(names)
    hits = [i for i, n in enumerate(names) if n in before]
    return hits[-1] + 1 if hits else 0


def append_band(path, name, values, top, bottom, weighting, offset, note, order=(),
                attrs=None):
    """Rewrite ``path`` with one more band, placed as in ``order`` (default: last).

    ``attrs`` are further global attributes.
    """
    path = Path(path)
    tmp = path.with_suffix('.fill.nc')
    with nc.Dataset(path) as src, nc.Dataset(tmp, 'w', format='NETCDF4') as dst:
        names = chars(src['band_name'])
        pos = _position(names, name, list(order))
        weights = chars(src['band_weighting'])
        nchar = max(len(s) for s in names + weights + [name, weighting])
        for dim, obj in src.dimensions.items():
            size = {'band': len(names) + 1, 'nchar': nchar}.get(dim, len(obj))
            dst.createDimension(dim, None if obj.isunlimited() else size)
        dst.setncatts({a: src.getncattr(a) for a in src.ncattrs()})
        new = {'band_top': top, 'band_bottom': bottom, 'bias_offset': offset,
               **{v: values[v] for v in BAND_VARS}}
        for vname, var in src.variables.items():
            zlib = var.filters().get('zlib', False)
            out = dst.createVariable(vname, var.dtype, var.dimensions, zlib=zlib)
            out.setncatts({a: var.getncattr(a) for a in var.ncattrs()})
            if vname in ('band_name', 'band_weighting'):
                vals = names if vname == 'band_name' else weights
                vals = vals[:pos] + [name if vname == 'band_name' else weighting] + vals[pos:]
                out[:] = _pad(vals, nchar)
            elif 'band' in var.dimensions:
                ax = var.dimensions.index('band')
                out[:] = np.insert(np.asarray(var[:], dtype=float), pos, new[vname], axis=ax)
            else:
                out[:] = var[:]
        dst.filled_bands = ' '.join(filter(None, [getattr(src, 'filled_bands', ''), name]))
        dst.setncattr(f'fill_{name}', note)
        dst.setncatts(attrs or {})
    os.replace(tmp, path)


def write_full(template, out, bands, offsets, applied, note, attrs=None):
    """A new forcing file on ``template``'s structure, every band variable replaced.

    ``attrs`` are further global attributes.
    """
    with nc.Dataset(template) as src, nc.Dataset(out, 'w', format='NETCDF4') as dst:
        names = chars(src['band_name'])
        for dim, obj in src.dimensions.items():
            dst.createDimension(dim, None if obj.isunlimited() else len(obj))
        dst.setncatts({a: src.getncattr(a) for a in src.ncattrs()})
        for vname, var in src.variables.items():
            out_v = dst.createVariable(vname, var.dtype, var.dimensions,
                                       zlib=var.filters().get('zlib', False))
            out_v.setncatts({a: var.getncattr(a) for a in var.ncattrs()})
            if vname in BAND_VARS:
                cols = [bands[n][vname] + (offsets[n] if applied and vname == 'thermal_forcing'
                                           else 0.0) for n in names]
                out_v[:] = np.stack(cols, axis=1)
            elif vname == 'bias_offset':
                out_v[:] = [offsets[n] for n in names]
            else:
                out_v[:] = var[:]
        dst.bias_correction_applied = str(applied)
        dst.filled_bands = ' '.join(names)
        for n in names:
            dst.setncattr(f'fill_{n}', bands[n]['note'])
        dst.fill_template = f'structure, sea ice and time axis of a neighbour; {note}'
        dst.setncatts(attrs or {})


def _weighted(nb, data):
    """Inverse-distance weights (floored at 1 km) and the weighted band variables."""
    w = np.array([1.0 / max(dist, 1.0) for _, dist in nb])
    w /= w.sum()
    vals = {v: sum(wi * data[n][v] for wi, (n, _) in zip(w, nb)) for v in BAND_VARS}
    src = '; '.join(f'{n} {dist:.0f} km w {wi:.2f}' for (n, dist), wi in zip(nb, w))
    return vals, src


def _row(gid, band, method, nb, src, level, ref_level, offset, corrected, period,
         entities, group):
    row = {'glacier_id': gid, 'band': band, 'method': method, 'n_neighbours': len(nb),
           'neighbours': src, 'nearest_km': round(nb[0][1], 1),
           'tf_fill_mean_degC': round(level, 4), 'tf_ref_mean_degC': round(ref_level, 4),
           'bias_offset_degC': round(offset, 4), 'corrected_file': corrected,
           'period': '' if period is None else '{}-{}'.format(*as_period(period))}
    if group is not None:
        row[group] = entities.at[gid, group]
    return row


def fill_product(files, entities, band, ref_files=None, corrected_files=None,
                 k=3, max_km=300.0, period=None, lon='lon', lat='lat',
                 group=None, target=None, corrected_target=None):
    """Fill ``band`` wherever this product's files lack it; returns the fill table rows.

    Parameters
    ----------
    files : dict
        glacier id -> forcing file of the product (uncorrected)
    entities : :py:class:`pandas.DataFrame`
        indexed by glacier id, with the terminus position in the ``lon`` and
        ``lat`` columns (degrees) and optionally a ``group`` column
    band : str
        the band to fill
    ref_files : dict, optional
        glacier id -> forcing file of the reference product; the filled band
        is offset onto it over ``period`` (degC), and the glaciers it covers
        that ``files`` lacks get a whole file through ``target``
    corrected_files : dict, optional
        glacier id -> bias-corrected file of the product, filled alongside
    k : int
        the most neighbours averaged
    max_km : float
        the farthest a neighbour may be, km
    period : (first year, last year), optional
        the reference period, both years included (see
        :func:`tidewater.periods.as_period`); needed with ``ref_files``
    lon, lat, group : str
        column names of ``entities``
    target, corrected_target : callable, optional
        glacier id -> path of the file to write for a glacier without one
    """
    files, ref = dict(files), dict(ref_files or {})
    bcf = dict(corrected_files or {})
    if ref and period is None:
        raise InvalidParamsError('an offset onto a reference needs `period`')
    period = None if period is None else as_period(period)
    data = {gid: read_band(p, band) for gid, p in files.items()}
    have = {gid for gid, v in data.items() if v is not None}
    rows = []
    for gid in sorted(set(files) - have):
        with nc.Dataset(files[gid]) as d:
            if band in chars(d['band_name']):
                continue  # filled by an earlier pass
            t = d['time']
            time = nc.num2date(t[:], t.units, t.calendar)
        nb = neighbours(gid, have, entities, k, max_km, lon, lat, group)
        if not nb:
            raise InvalidWorkflowError(f'{gid}: no glacier carries {band} within {max_km} km')
        for n, _ in nb:
            if len(data[n]['time']) != len(time) or data[n]['time'][0] != time[0]:
                raise InvalidWorkflowError(f'{n} and {gid} do not share a time axis')
        vals, src = _weighted(nb, data)
        first = data[nb[0][0]]
        level = ref_mean(vals['thermal_forcing'], time, period)
        offset, ref_level = 0.0, np.nan
        if ref:
            if gid not in ref:
                raise InvalidWorkflowError(f'{gid}: no reference file')
            with nc.Dataset(ref[gid]) as d:  # a filled reference band counts
                names = chars(d['band_name'])
                if band not in names:
                    raise InvalidWorkflowError(
                        f'{gid}: the reference lacks {band}; fill it first')
                rt = d['time']
                ref_level = ref_mean(
                    np.asarray(d['thermal_forcing'][:, names.index(band)], float),
                    nc.num2date(rt[:], rt.units, rt.calendar), period)
            offset = ref_level - level
        note = f'inverse-distance mean of {len(nb)} neighbour(s) carrying {band}: {src}'
        append_band(files[gid], band, vals, first['top'], first['bottom'],
                    first['weighting'], offset, note, order=first['names'])
        if gid in bcf:
            with nc.Dataset(bcf[gid]) as d:
                applied = str(getattr(d, 'bias_correction_applied', 'False')) == 'True'
            if not applied:
                raise InvalidWorkflowError(f'{bcf[gid]}: expected a corrected file')
            cor = dict(vals, thermal_forcing=vals['thermal_forcing'] + offset)
            append_band(bcf[gid], band, cor, first['top'], first['bottom'],
                        first['weighting'], offset,
                        note + f'; offset {offset:+.4f} degC onto the reference',
                        order=first['names'])
        rows.append(_row(gid, band, 'same_product_neighbours', nb, src, level, ref_level,
                         offset, gid in bcf, period, entities, group))
    if ref:
        rows += fill_missing_files(files, ref, entities, target, corrected_target,
                                   k=k, max_km=max_km, period=period, lon=lon, lat=lat,
                                   group=group)
    return rows


def fill_missing_files(files, ref_files, entities, target, corrected_target=None,
                       k=3, max_km=300.0, period=None, lon='lon', lat='lat',
                       group=None):
    """Every band of each glacier the reference covers and ``files`` lacks.

    Arguments as in :func:`fill_product`; the bands are those of the
    reference file, and the structure, sea ice and time axis those of the
    nearest glacier carrying them all.
    """
    missing = sorted(set(ref_files) - set(files))
    if not missing:
        return []
    if not files:
        raise InvalidWorkflowError('the product has no forcing file at all; '
                                   'that is a failure, not a gap')
    if target is None:
        raise InvalidParamsError(f'{len(missing)} glacier(s) without a file: pass '
                                 '`target` to say where theirs go')
    with nc.Dataset(ref_files[missing[0]]) as d:
        names = chars(d['band_name'])
    data = {b: {gid: read_band(p, b) for gid, p in files.items()} for b in names}
    have = {b: {g for g, v in data[b].items() if v is not None} for b in names}
    complete = set.intersection(*have.values())
    rows = []
    for gid in missing:
        tmpl = neighbours(gid, complete, entities, 1, max_km, lon, lat, group)
        if not tmpl:
            raise InvalidWorkflowError(f'{gid}: no glacier with every band within {max_km} km')
        template = files[tmpl[0][0]]
        t0 = data[names[0]][tmpl[0][0]]['time']
        bands, offsets = {}, {}
        for b in names:
            nb = neighbours(gid, have[b], entities, k, max_km, lon, lat, group)
            if not nb:
                raise InvalidWorkflowError(f'{gid}: no glacier carries {b} within {max_km} km')
            if any(len(data[b][n]['time']) != len(t0) or data[b][n]['time'][0] != t0[0]
                   for n, _ in nb):
                raise InvalidWorkflowError(
                    f'the neighbours of {gid} do not share a time axis')
            vals, src = _weighted(nb, data[b])
            with nc.Dataset(ref_files[gid]) as d:
                rt = d['time']
                rtf = np.asarray(d['thermal_forcing'][:, chars(d['band_name']).index(b)], float)
                ref_level = ref_mean(rtf, nc.num2date(rt[:], rt.units, rt.calendar), period)
            level = ref_mean(vals['thermal_forcing'], t0, period)
            offsets[b] = ref_level - level
            bands[b] = dict(vals, note=f'inverse-distance mean of {len(nb)} '
                                       f'neighbour(s) carrying {b}: {src}')
            rows.append(_row(gid, b, 'same_product_neighbours_all_bands', nb, src, level,
                             ref_level, offsets[b], corrected_target is not None, period,
                             entities, group))
        note = 'single-level or absent profile: every band from the neighbours'
        write_full(template, target(gid), bands, offsets, False, note)
        if corrected_target is not None:
            write_full(template, corrected_target(gid), bands, offsets, True,
                       note + '; offsets onto the reference')
    return rows
