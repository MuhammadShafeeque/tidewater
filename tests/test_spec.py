"""reference_vectors.csv against the code, and against SPEC.md written out
again below without the package."""
import math

import pytest

from tidewater.spec import make_vectors

RTOL = 1e-12
NAN = float('nan')
DAY, YEAR = 86400., 31536000.
L1, L2, L3 = -5.73e-2, 8.32e-2, 7.61e-4


def _same(got, want):
    if math.isnan(want):
        return math.isnan(got)
    return got == pytest.approx(want, rel=RTOL, abs=0.)


@pytest.fixture(scope='module')
def table():
    return make_vectors.read()


def test_the_file_is_what_the_script_writes(table, tmp_path):
    fresh = make_vectors.read(make_vectors.write(tmp_path / 'vectors.csv'))
    assert len(table) == len(fresh) > 600
    for old, new in zip(table, fresh):
        assert list(old) == make_vectors.COLUMNS
        assert {k: v for k, v in old.items() if k not in ('out1', 'out2')} == \
            {k: v for k, v in new.items() if k not in ('out1', 'out2')}
        for out in ('out1', 'out2'):
            assert _same(float(new[out]), float(old[out])), old['id']


def test_the_code_reproduces_every_vector(table):
    for row in table:
        got = make_vectors.evaluate(row['function'], make_vectors.parse(row))
        for value, out in zip(got, ('out1', 'out2')):
            assert _same(float(value), float(row[out])), (row['id'], out)


def test_every_law_and_kernel_has_vectors(table):
    functions = {row['function'] for row in table}
    assert functions == {
        'freezing_point', 'thermal_forcing', 'open_water_fraction',
        'flotation_thickness', 'thermal_forcing_band', 'ocean_factor',
        'tf_scaling', 'tf_power_mean', 'inversion_k_from_ocean',
        'discharge_per_front_area', 'submarine_melt_rate', 'calving_law_flux',
        'law_constant', 'law_tf_power', 'law_melt_calving', 'law_sea_ice',
        'law_larger_of', 'partition_calving_constant',
        'larger_of_calving_constant', 'water_depth', 'submerged_depth',
        'front_thickness'}
    assert any(r['function'] == 'calving_law_flux' and r['sea_level']
               for r in table)
    # the clipping rules are exercised
    assert any(r['function'] == 'law_tf_power' and float(r['tf']) < 0
               and float(r['out1']) == 0. for r in table)
    assert any(r['function'] == 'law_melt_calving' and float(r['water_depth']) < 0
               for r in table)
    assert any(r['function'] == 'law_constant' and float(r['out1']) < 0
               for r in table)


# --- SPEC.md, sections 2 to 8, in plain Python ----------------------------------

def _t_f(s, z):
    return L1 * s + L2 - L3 * z


def _factor(tf, ref, gamma):
    return (max(tf, 0.) / ref) ** gamma


def _scaling(tf, ref, gamma, a):
    f = _factor(tf, ref, gamma)
    return f if a == 1 else (1 - a) + a * f


def _discharge(q_sg, d, w):
    return max(q_sg, 0.) * DAY / max(d * w, 1e-3)


def _mdot(d, q, tf, c):
    return (c['A'] * max(d, 0.) * q ** c['alpha'] + c['B']) * tf ** c['beta']


def _band(c):
    z, th, so = c['series3'], c['series'], c['series2']
    n = len(z)
    g = [z[1] - z[0]] + [(z[i + 1] - z[i - 1]) / 2 for i in range(1, n - 1)]
    g.append(z[-1] - z[-2])
    sel = [i for i in range(n) if c['band_top'] <= z[i] <= c['band_bottom']]
    if c['weighting'] == 'uniform':
        wt = [1 / len(sel)] * len(sel)
    else:
        wt = [g[i] / sum(g[j] for j in sel) for i in sel]
    return sum(w * (th[i] - _t_f(so[i], z[i])) for w, i in zip(wt, sel))


def _law(name, c):
    k_s = c['k'] / YEAR
    h, d, w = c['thick'], c['water_depth'], c['width']
    if name == 'constant':
        return k_s * d * h * w, 0.
    if d <= 0 or h <= 0:
        return 0., 0.
    if name == 'tf_power':
        k_t = k_s * _scaling(c['tf'], c['tf_ref'], c['gamma'], c['tf_fraction'])
        return max(k_t * d * h * w, 0.), 0.
    q = _discharge(c['q_sg'], d, w) if 'q_sg' in c else 0.
    u_c = k_s * h
    u_m = c['lam'] * (_mdot(d, q, max(c['tf'], 0.), c) / DAY)
    if name == 'sea_ice':
        u_m = max(c['open_water'], 0.) ** c['delta'] * u_m
    if name == 'larger_of':
        u_c, u_m = (u_c, 0.) if u_c >= u_m else (0., u_m)
    total = u_c + u_m
    share = min(max(u_m / total, 0.), 1.) if total > 0 else 0.
    q_f = max(total * d * w, 0.)
    return q_f, q_f * share


def _partition(c):
    u_tot, u_m = c['k'] / YEAR * c['thick'], c['lam'] * c['melt_rate']
    if u_tot <= 0:
        return 0., 0.
    if u_m >= u_tot:
        return 0., 1.
    return (u_tot - u_m) / c['thick'] * YEAR, u_m / u_tot


def _larger_of(c):
    u_tot = c['k'] / YEAR * c['thick']
    u = sorted(max(v, 0.) for v in c['series'])
    n = len(u)
    if u_tot <= 0 or n == 0:
        return max(c['k'], 0.), 0.
    if sum(u) / n >= u_tot:
        return 0., 1.
    for j in range(1, n + 1):
        cc = (n * u_tot - sum(u[j:])) / j
        if j == n or cc <= u[j]:
            break
    return cc / c['thick'] * YEAR, sum(v for v in u if v > cc) / n / u_tot


def spec(function, c):
    if function == 'freezing_point':
        return _t_f(c['so'], c['depth']), NAN
    if function == 'thermal_forcing':
        return c['thetao'] - _t_f(c['so'], c['depth']), NAN
    if function == 'open_water_fraction':
        t = c['threshold']
        return min(max((t - c['siconc']) / t, 0.), 1.), NAN
    if function == 'flotation_thickness':
        return c['water_depth'] * c['rho_ocean'] / c['rho_ice'], NAN
    if function == 'thermal_forcing_band':
        return _band(c), NAN
    if function == 'ocean_factor':
        return _factor(c['tf'], c['tf_ref'], c['gamma']), NAN
    if function == 'tf_scaling':
        return _scaling(c['tf'], c['tf_ref'], c['gamma'], c['tf_fraction']), NAN
    if function == 'tf_power_mean':
        tf, g = [max(v, 0.) for v in c['series']], c['gamma']
        if g == 0:
            return sum(tf) / len(tf), NAN
        return (sum(v ** g for v in tf) / len(tf)) ** (1 / g), NAN
    if function == 'inversion_k_from_ocean':
        return c['k'] * _scaling(c['tf'], c['tf_ref'], c['gamma'],
                                 c['tf_fraction']), NAN
    if function == 'discharge_per_front_area':
        return _discharge(c['q_sg'], c['water_depth'], c['width']), NAN
    if function == 'submarine_melt_rate':
        if c['tf'] <= 0:
            return NAN, NAN
        return _mdot(c['water_depth'], c['q_sg_per_area'], c['tf'], c), NAN
    sl = c.get('sea_level', 0.)
    if function == 'calving_law_flux':
        h = max(c['surface_h'] - sl, 0.) + c['water_depth']
        return max(c['k'] * h * c['water_depth'] * c['width'] / 1e9, 0.), h
    if function == 'water_depth':
        return sl - c['bed_elevation'], NAN
    if function == 'submerged_depth':
        return c['thick'] - (c['surface_h'] - sl), NAN
    if function == 'front_thickness':
        return max(c['surface_h'] - sl, 0.) + c['water_depth'], NAN
    if function == 'partition_calving_constant':
        return _partition(c)
    if function == 'larger_of_calving_constant':
        return _larger_of(c)
    return _law(function[4:], c)


def test_the_written_specification_reproduces_every_vector(table):
    for row in table:
        got = spec(row['function'], make_vectors.parse(row))
        for value, out in zip(got, ('out1', 'out2')):
            assert _same(value, float(row[out])), (row['id'], row['function'], out)
