"""Write reference_vectors.csv, the test vectors of SPEC.md, from the laws.

Run as ``python -m tidewater.spec.make_vectors [path]``.
"""
import csv
import itertools
import os
import sys

import numpy as np

from tidewater import laws, physics

PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                    'reference_vectors.csv')

# One column per input; a cell is empty when the function does not read it
INPUTS = ['k', 'thick', 'water_depth', 'width', 'surface_h', 'tf', 'tf_ref',
          'gamma', 'tf_fraction', 'q_sg', 'q_sg_per_area', 'open_water', 'A',
          'B', 'alpha', 'beta', 'lam', 'delta', 'melt_rate', 'thetao', 'so',
          'depth', 'siconc', 'threshold', 'rho_ocean', 'rho_ice', 'band_top',
          'band_bottom', 'weighting', 'series', 'series2', 'series3',
          'sea_level', 'bed_elevation']
COLUMNS = ['id', 'function'] + INPUTS + ['out1', 'out2']
TEXT = ('weighting',)
SERIES = ('series', 'series2', 'series3')

MELT = dict(A=3e-4, B=0.15, alpha=0.39, beta=1.18)


def _law(name, c):
    """Total and melt flux of one law at constant forcing, m3 s-1."""
    yrs = np.array([0., 1.])
    kw = dict(open_water=(None if c.get('open_water') is None
                          else np.full(2, c['open_water'])),
              q_sg=None if c.get('q_sg') is None else np.full(2, c['q_sg']))
    tf = np.full(2, c.get('tf', 0.))
    if name == 'constant':
        law = laws.ConstantK(k=c['k'])
    elif name == 'tf_power':
        law = laws.TFPower(yrs, tf, tf_ref=c['tf_ref'], k0=c['k'],
                           gamma=c['gamma'], tf_fraction=c['tf_fraction'])
    else:
        par = {p: c[p] for p in ('A', 'B', 'alpha', 'beta', 'lam')}
        if name == 'sea_ice':
            par['delta'] = c['delta']
        law = laws.LAWS[name](yrs, tf, k_c=c['k'], **par, **kw)
    front = c['thick'], c['water_depth'], c['width'], 0.5
    return law.flux(*front), law.flux_components(*front)[1]


def evaluate(function, c):
    """``(out1, out2)`` of one case; out2 is nan for a single-valued function."""
    nan = float('nan')
    if function == 'freezing_point':
        return float(physics.freezing_point(c['so'], c['depth'])), nan
    if function == 'thermal_forcing':
        return float(physics.thermal_forcing(c['thetao'], c['so'],
                                             c['depth'])), nan
    if function == 'open_water_fraction':
        return float(physics.open_water_fraction(c['siconc'],
                                                 c['threshold'])), nan
    if function == 'flotation_thickness':
        return physics.flotation_thickness(c['water_depth'], c['rho_ocean'],
                                           c['rho_ice']), nan
    if function == 'thermal_forcing_band':
        z = np.asarray(c['series3'])
        out = physics.thermal_forcing_bands(
            np.asarray(c['series'])[None, :], np.asarray(c['series2'])[None, :],
            z, [('b', c['band_top'], c['band_bottom'])], {'b': c['weighting']})
        return float(out[4][0, 0]), nan
    if function == 'ocean_factor':
        return laws.ocean_factor(c['tf'], c['tf_ref'], c['gamma']), nan
    if function == 'tf_scaling':
        return laws.tf_scaling(c['tf'], c['tf_ref'], c['gamma'],
                               c['tf_fraction']), nan
    if function == 'tf_power_mean':
        return laws.tf_power_mean(c['series'], c['gamma']), nan
    if function == 'inversion_k_from_ocean':
        return laws.inversion_k_from_ocean(
            c['k'], c['tf'], tf_ref=c['tf_ref'], gamma=c['gamma'],
            tf_fraction=c['tf_fraction']), nan
    if function == 'discharge_per_front_area':
        return laws.discharge_per_front_area(c['q_sg'], c['water_depth'],
                                             c['width']), nan
    if function == 'submarine_melt_rate':
        return laws.submarine_melt_rate(
            c['water_depth'], c['q_sg_per_area'], c['tf'],
            **{p: c[p] for p in MELT}), nan
    if function == 'calving_law_flux':
        out = laws.calving_law_flux(c['water_depth'], c['k'],
                                    width=c['width'], surface_h=c['surface_h'],
                                    sea_level=c.get('sea_level', 0.))
        return out.flux, out.thickness
    if function == 'water_depth':
        return physics.water_depth(c['bed_elevation'], c['sea_level']), nan
    if function == 'submerged_depth':
        return physics.submerged_depth(c['thick'], c['surface_h'],
                                       c['sea_level']), nan
    if function == 'front_thickness':
        return physics.front_thickness(c['surface_h'], c['water_depth'],
                                       c['sea_level']), nan
    if function == 'partition_calving_constant':
        return laws.partition_calving_constant(c['k'], c['melt_rate'],
                                               c['thick'], lam=c['lam'])
    if function == 'larger_of_calving_constant':
        return laws.larger_of_calving_constant(c['k'], c['series'], c['thick'])
    if function.startswith('law_'):
        return _law(function[4:], c)
    raise ValueError(function)


def cases():
    """The inputs of every vector, as (function, dict)."""
    out = []

    def add(function, **c):
        out.append((function, c))

    for so, z in itertools.product((0., 30., 34.8, 36.), (0., 60., 350., 700.)):
        add('freezing_point', so=so, depth=z)
    for th, so, z in itertools.product((-1.9, 0., 2.5, 6.), (30., 34.8),
                                       (30., 350.)):
        add('thermal_forcing', thetao=th, so=so, depth=z)
    for sic, thr in itertools.product((0., 0.05, 0.15, 0.4, 1.), (0.15, 0.5)):
        add('open_water_fraction', siconc=sic, threshold=thr)
    for d, ro, ri in ((50., 1028., 900.), (300., 1028., 917.),
                      (120., 1000., 900.)):
        add('flotation_thickness', water_depth=d, rho_ocean=ro, rho_ice=ri)

    z = [5., 15., 30., 50., 80., 150., 250., 400., 600.]
    th = [-1.2, -0.8, -0.1, 0.6, 1.4, 2.2, 2.9, 3.1, 2.4]
    so = [31.0, 32.5, 33.4, 34.0, 34.4, 34.7, 34.8, 34.9, 34.9]
    for top, bot, how in ((0., 60., 'uniform'), (200., 500., 'uniform'),
                          (0., 700., 'depth_weighted'),
                          (0., 100., 'depth_weighted')):
        add('thermal_forcing_band', series=th, series2=so, series3=z,
            band_top=top, band_bottom=bot, weighting=how)

    tfs = (-1.5, 0., 0.3, 1.5, 4.2)
    for tf, ref, g in itertools.product(tfs, (0.8, 1.5), (0., 1., 1.18, 2.)):
        add('ocean_factor', tf=tf, tf_ref=ref, gamma=g)
    for tf, g, a in itertools.product(tfs, (1., 1.18), (0., 0.4, 1.)):
        add('tf_scaling', tf=tf, tf_ref=1.5, gamma=g, tf_fraction=a)
    for g in (0., 0.5, 1., 1.18, 2.):
        add('tf_power_mean', gamma=g, series=[-0.5, 0.5, 1., 2., 4.])
    for tf, ref, a in itertools.product((-0.2, 0.9, 2.4), (1.2, 2.4),
                                        (0.5, 1.)):
        add('inversion_k_from_ocean', k=0.6, tf=tf, tf_ref=ref, gamma=1.18,
            tf_fraction=a)

    for q, d, w in ((0., 100., 1000.), (25., 100., 1000.), (-3., 80., 500.),
                    (300., 400., 4000.), (5., 0., 1000.)):
        add('discharge_per_front_area', q_sg=q, water_depth=d, width=w)
    for d, q, tf in itertools.product((0., 80., 400.), (0., 0.5, 30.),
                                      (0.3, 1.8, 5.)):
        add('submarine_melt_rate', water_depth=d, q_sg_per_area=q, tf=tf,
            **MELT)
    add('submarine_melt_rate', water_depth=150., q_sg_per_area=2., tf=1.8,
        **dict(MELT, alpha=0.54, beta=1.61))

    for d, k, w, s in ((150., 0.6, 2000., 40.), (60., 2.4, 800., 0.),
                       (300., 0.1, 5000., -5.)):
        add('calving_law_flux', water_depth=d, k=k, width=w, surface_h=s)

    fronts = ((200., 100., 1000.), (350., 300., 4000.), (50., 60., 800.),
              (80., -10., 800.))
    for (h, d, w), k in itertools.product(fronts, (0.6, 2.4)):
        add('law_constant', k=k, thick=h, water_depth=d, width=w)
    for (h, d, w), tf, a in itertools.product(fronts, tfs, (0.4, 1.)):
        add('law_tf_power', k=0.6, thick=h, water_depth=d, width=w, tf=tf,
            tf_ref=1.5, gamma=1.18, tf_fraction=a)
    for (h, d, w), tf, q, lam in itertools.product(
            fronts, (-0.5, 0.6, 3.), (None, 40.), (0., 1., 400.)):
        c = dict(k=0.6, thick=h, water_depth=d, width=w, tf=tf, lam=lam,
                 **MELT)
        if q is not None:
            c['q_sg'] = q
        add('law_melt_calving', **c)
        add('law_larger_of', **c)
        for ow, delta in ((0., 1.), (0.35, 1.), (0.35, 2.), (1., 0.)):
            add('law_sea_ice', open_water=ow, delta=delta, **c)

    for k, m, h, lam in ((0.6, 0.1 / 86400., 200., 1.),
                         (0.6, 0.1 / 86400., 200., 0.25),
                         (0.01, 1e-5, 200., 1.), (0., 1e-7, 150., 1.),
                         (2.4, 3e-6, 420., 1.)):
        add('partition_calving_constant', k=k, melt_rate=m, thick=h, lam=lam)
    u = 0.6 / 31536000. * 200.
    for rel in ([0.1, 0.5, 1.5, 0.2, 0.8, 0.3], [0., 0.5], [1., 4.],
                [0.9, 0.9, 0.9, 1.2], [0.2, -0.1, 2.5]):
        add('larger_of_calving_constant', k=0.6, thick=200.,
            series=[r * u for r in rel])

    for bed, sl in ((-150., 0.), (-150., 2.5), (30., 0.), (-80., -1.2),
                    (415., 420.)):
        add('water_depth', bed_elevation=bed, sea_level=sl)
    for h, s, sl in ((200., 100., 0.), (200., 100., 2.5), (350., 20., -1.2),
                     (60., 470., 420.)):
        add('submerged_depth', thick=h, surface_h=s, sea_level=sl)
    for s, d, sl in ((40., 150., 0.), (40., 150., 2.5), (1., 150., 2.5),
                     (460., 35., 420.)):
        add('front_thickness', surface_h=s, water_depth=d, sea_level=sl)
    for d, k, w, s, sl in ((150., 0.6, 2000., 40., 2.5),
                           (60., 2.4, 800., 0., -1.2),
                           (35., 1.1, 600., 460., 420.)):
        add('calving_law_flux', water_depth=d, k=k, width=w, surface_h=s,
            sea_level=sl)
    return out


def _cell(key, value):
    if value is None:
        return ''
    if key in TEXT:
        return value
    if key in SERIES:
        return ';'.join(repr(float(v)) for v in value)
    return repr(float(value))


def rows():
    """Every vector as a dict of strings, in file order."""
    out = []
    for i, (function, c) in enumerate(cases(), start=1):
        o1, o2 = evaluate(function, c)
        row = {'id': str(i), 'function': function}
        row.update({key: _cell(key, c.get(key)) for key in INPUTS})
        row['out1'], row['out2'] = repr(float(o1)), repr(float(o2))
        out.append(row)
    return out


def parse(row):
    """The inputs of one CSV row, as :func:`evaluate` takes them."""
    c = {}
    for key in INPUTS:
        cell = row[key]
        if cell == '':
            continue
        if key in TEXT:
            c[key] = cell
        elif key in SERIES:
            c[key] = [float(v) for v in cell.split(';')]
        else:
            c[key] = float(cell)
    return c


def read(path=PATH):
    with open(path, newline='') as f:
        return list(csv.DictReader(f))


def write(path=PATH):
    with open(path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS, lineterminator='\n')
        writer.writeheader()
        writer.writerows(rows())
    return path


if __name__ == '__main__':
    print(write(sys.argv[1] if len(sys.argv) > 1 else PATH))
