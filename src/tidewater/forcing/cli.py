"""Command line: ``tidewater-forcing build|validate`` (also ``python -m tidewater.forcing``)."""
import argparse
import logging
import sys

from tidewater.config import InvalidParamsError
from tidewater.forcing.bands import ON_UNFORMABLE
from tidewater.forcing.build import build_forcing, validate_directory
from tidewater.forcing.footprint import SCHEMES
from tidewater.forcing.io import AUTHOR
from tidewater.forcing.products import FIELDS, PRODUCTS
from tidewater.periods import as_period


def _names(pairs):
    out = {}
    for pair in pairs or ():
        key, sep, val = pair.partition('=')
        if not sep or key not in FIELDS:
            raise InvalidParamsError(f'--name {pair!r} is not field=variable, '
                                     f'field one of {FIELDS}')
        out[key] = val
    return out


def _attrs(pairs):
    return dict(p.partition('=')[::2] for p in pairs or ())


def parser():
    """The argument parser."""
    p = argparse.ArgumentParser(prog='tidewater-forcing',
                                description='Per-glacier ocean forcing files.')
    p.add_argument('-v', '--verbose', action='store_true', help='log decisions')
    sub = p.add_subparsers(dest='command', required=True)

    b = sub.add_parser('build', help='forcing files from one ocean product')
    b.add_argument('product', nargs='+', help='product file(s), one grid and time axis')
    b.add_argument('--glaciers', required=True,
                   help='CSV: id, terminus lon and lat (degrees), optional front depth (m)')
    b.add_argument('--out', required=True, help='output directory')
    b.add_argument('--pattern', default='{id}.nc', help='file name, with {id}')
    b.add_argument('--report', help='CSV of what was built and why not')
    g = b.add_argument_group('glacier table')
    g.add_argument('--id-column', default='id')
    g.add_argument('--lon-column', default='lon')
    g.add_argument('--lat-column', default='lat')
    g.add_argument('--front-depth-column', default='front_depth')
    g.add_argument('--bed-column', help='bed elevation at the front, m')
    g.add_argument('--sea-level', type=float, default=0., help='m, datum of --bed-column')
    g = b.add_argument_group('product')
    g.add_argument('--product-name', choices=sorted(PRODUCTS),
                   help='naming convention; default: detected')
    g.add_argument('--name', action='append', metavar='FIELD=VAR',
                   help=f'variable name of a field ({", ".join(FIELDS)})')
    g.add_argument('--temperature-units', choices=('degC', 'K', 'auto'))
    g.add_argument('--salinity-units', choices=('psu', 'g/kg'))
    g.add_argument('--depth-units', choices=('m', 'cm', 'auto'))
    g.add_argument('--depth-positive', choices=('down', 'up'))
    g.add_argument('--siconc-units', choices=('1', '%', 'auto'))
    g.add_argument('--cell-area-units', choices=('m2', 'km2'))
    g.add_argument('--cell-area-km2', type=float,
                   help='cell area of a product that has none')
    g = b.add_argument_group('bands and period')
    g.add_argument('--bands', nargs='+', metavar='NAME:TOP:BOTTOM[:WEIGHTING]',
                   help='depth bands, m; default: the package defaults')
    g.add_argument('--years', help='first-last year kept, both included')
    g = b.add_argument_group('footprint')
    g.add_argument('--scheme', choices=SCHEMES, default='nearest')
    g.add_argument('--fallback', choices=('shelf', 'drop'), default='shelf')
    g.add_argument('--coverage', choices=('strict', 'partial'), default='strict')
    g.add_argument('--target-area-km2', type=float, default=1000.)
    g.add_argument('--max-cells', type=int, default=64)
    g.add_argument('--radii', type=float, nargs='+', help='search radii, km')
    g.add_argument('--r0-km', type=float, default=15.)
    g.add_argument('--r-max-km', type=float, default=150.)
    g.add_argument('--r-grow', type=float, default=1.5)
    g.add_argument('--max-radius', type=float)
    g.add_argument('--shelf-radius-km', type=float)
    g.add_argument('--region-km', type=float, help='read only cells this close')
    g.add_argument('--siconc-max-gap', type=int, default=0)
    g = b.add_argument_group('output')
    g.add_argument('--on-unformable', choices=ON_UNFORMABLE, default='skip')
    g.add_argument('--author', default=AUTHOR)
    g.add_argument('--attr', action='append', metavar='KEY=VALUE',
                   help='further global attribute')
    g.add_argument('--source', help='provenance; default: the product file names')
    g.add_argument('--ocean-model', default='')

    v = sub.add_parser('validate', help='check the forcing files of a directory')
    v.add_argument('directory')
    v.add_argument('--pattern', default='*.nc')
    v.add_argument('--bands', nargs='+', help='band names every file must carry')
    v.add_argument('--report', help='CSV of the result')
    return p


def _build(a):
    from tidewater.forcing.build import read_glaciers

    glaciers = read_glaciers(a.glaciers, a.id_column, a.lon_column, a.lat_column,
                             a.front_depth_column, a.bed_column, a.sea_level)
    units = {k: getattr(a, k) for k in ('temperature_units', 'salinity_units',
                                        'depth_units', 'depth_positive',
                                        'siconc_units', 'cell_area_units')
             if getattr(a, k) is not None}
    _, report = build_forcing(
        a.product if len(a.product) > 1 else a.product[0], glaciers, a.out,
        product_name=a.product_name, product_options=dict(units, names=_names(a.name)),
        depth_bands=a.bands, scheme=a.scheme, fallback=a.fallback,
        region_km=a.region_km, years=as_period(a.years) if a.years else None,
        cell_area_km2=a.cell_area_km2, on_unformable=a.on_unformable,
        pattern=a.pattern, author=a.author, attrs=_attrs(a.attr), source=a.source,
        ocean_model=a.ocean_model, radii=a.radii, r0_km=a.r0_km, r_max_km=a.r_max_km,
        r_grow=a.r_grow, max_radius=a.max_radius, shelf_radius_km=a.shelf_radius_km,
        coverage=a.coverage, target_area_km2=a.target_area_km2, max_cells=a.max_cells,
        siconc_max_gap=a.siconc_max_gap)
    if a.report:
        report.to_csv(a.report)
    n = int((report['status'] == 'written').sum())
    print(f'{n} of {len(report)} glacier(s) written to {a.out}')
    for gid, row in report[report['status'] != 'written'].iterrows():
        print(f'{gid}: {row["reason"]}')
    return 0


def _validate(a):
    report = validate_directory(a.directory, a.pattern, bands=a.bands)
    if a.report:
        report.to_csv(a.report)
    bad = report[report['problem'] != '']
    print(f'{len(report) - len(bad)} of {len(report)} file(s) sound')
    for name, row in bad.iterrows():
        print(f'{name}: {row["problem"]}')
    return 1 if len(bad) or not len(report) else 0


def main(argv=None):
    """Run the command line; returns the exit status."""
    a = parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO if a.verbose else logging.WARNING,
                        format='%(levelname)s %(name)s: %(message)s')
    try:
        return _build(a) if a.command == 'build' else _validate(a)
    except InvalidParamsError as err:
        print(f'error: {err}', file=sys.stderr)
        return 2
