"""A band a footprint misses is filled from the same product's neighbours, and offset like the rest."""
import numpy as np
import pandas as pd
import pytest

from tidewater.config import InvalidParamsError, InvalidWorkflowError
from tidewater.forcing.fill import fill_product, neighbours
from tidewater.forcing.io import read_forcing, validate, write_ocean_file
from tidewater.physics import band_names

BANDS = {'near': (0, 60, 'uniform'), 'mid': (200, 500, 'uniform'),
         'wide': (0, 700, 'depth_weighted')}
FULL = ['near', 'mid', 'wide']
TIME = pd.date_range('1995-01-01', '2014-12-01', freq='MS')
PERIOD = (1995, 2014)


def gid(n):
    return f'G-{n:05d}'


def write(wd, n, product, bands, tf, bc=False, offsets=None):
    f = wd / gid(n) / f'ocean_{product}{"_bc" if bc else ""}.nc'
    f.parent.mkdir(parents=True, exist_ok=True)
    vals = np.outer(np.ones(TIME.size), [tf[b] for b in bands])
    write_ocean_file(f, TIME.values, bands, [BANDS[b][0] for b in bands],
                     [BANDS[b][1] for b in bands], [BANDS[b][2] for b in bands],
                     vals, 0.5 * vals, 10.0 * vals, np.full(TIME.size, 0.5),
                     np.zeros(TIME.size), offsets=offsets or [0.0] * len(bands),
                     bias_applied=bc)
    return f


def files(wd, product, bc=False):
    name = f'ocean_{product}{"_bc" if bc else ""}.nc'
    return {p.parent.name: p for p in wd.glob(f'*/{name}')}


def fill(wd, ent, product='P', ref='REF', **kw):
    kw.setdefault('group', 'sector')
    return fill_product(
        files(wd, product), ent, 'mid', ref_files=files(wd, ref) if ref else None,
        corrected_files=files(wd, product, bc=True), period=PERIOD,
        target=lambda g: wd / g / f'ocean_{product}.nc',
        corrected_target=lambda g: wd / g / f'ocean_{product}_bc.nc', **kw)


@pytest.fixture
def setup(tmp_path):
    # southern hemisphere, astride the dateline
    ent = pd.DataFrame({'glacier_id': [gid(i) for i in (1, 2, 3, 4)],
                        'sector': ['east', 'east', 'east', 'north'],
                        'lon': [179.9, 179.9, 179.9, -179.9],
                        'lat': [-70.0, -70.09, -70.27, -70.0]}).set_index('glacier_id')
    wd = tmp_path
    # glacier 1 lacks mid; 2 (10 km) and 3 (30 km) carry it; 4 is nearer but in another sector
    write(wd, 1, 'P', ['near', 'wide'], {'near': 0.5, 'wide': 1.0})
    write(wd, 1, 'P', ['near', 'wide'], {'near': 0.7, 'wide': 1.2}, bc=True)
    write(wd, 2, 'P', FULL, {'near': 0.5, 'mid': 2.0, 'wide': 1.0})
    write(wd, 3, 'P', FULL, {'near': 0.5, 'mid': 4.0, 'wide': 1.0})
    write(wd, 4, 'P', FULL, {'near': 0.5, 'mid': 9.0, 'wide': 1.0})
    for n in (1, 2, 3, 4):
        write(wd, n, 'REF', FULL, {'near': 0.6, 'mid': 3.5, 'wide': 1.1})
    return wd, ent


def test_neighbours_across_the_dateline(setup):
    _, ent = setup
    nb = neighbours(gid(1), {gid(2), gid(3), gid(4)}, ent, 3, 300.)
    assert [n for n, _ in nb] == [gid(4), gid(2), gid(3)]
    assert nb[0][1] == pytest.approx(7.6, abs=0.1)  # 0.2 degrees of longitude at 70 S
    nb = neighbours(gid(1), {gid(2), gid(3), gid(4)}, ent, 3, 300., group='sector')
    assert [n for n, _ in nb] == [gid(2), gid(3)]


def test_filled_from_same_sector_neighbours_and_offset_onto_the_reference(setup):
    wd, ent = setup
    rows = fill(wd, ent, k=3, max_km=300)
    assert [r['glacier_id'] for r in rows] == [gid(1)]
    w2, w3 = 1 / 10.0, 1 / 30.0
    expect = (2.0 * w2 + 4.0 * w3) / (w2 + w3)
    assert rows[0]['n_neighbours'] == 2 and rows[0]['sector'] == 'east'
    assert rows[0]['tf_fill_mean_degC'] == pytest.approx(expect, abs=2e-3)
    assert rows[0]['bias_offset_degC'] == pytest.approx(3.5 - expect, abs=2e-3)
    for bc, tf in ((False, expect), (True, 3.5)):
        f = wd / gid(1) / f'ocean_P{"_bc" if bc else ""}.nc'
        ds = read_forcing(f)
        assert band_names(ds) == FULL  # placed as in the neighbour's file
        np.testing.assert_allclose(ds.thermal_forcing.isel(band=1), tf, atol=2e-3)
        # the bands already there are untouched
        np.testing.assert_allclose(ds.thermal_forcing.isel(band=0), 0.7 if bc else 0.5)
        assert ds.band_top.values[1] == 200 and ds.attrs['filled_bands'] == 'mid'
        assert ds.sizes['time'] == 240 and ds.siconc.size == 240
        assert validate(f, bands=FULL) == {'problem': '', 'y0': 1995, 'y1': 2014,
                                           'filled': 'mid'}


def test_without_a_group_the_nearest_wins(setup):
    wd, ent = setup
    rows = fill(wd, ent, group=None, k=1)
    assert rows[0]['neighbours'].startswith(gid(4)) and 'sector' not in rows[0]
    assert rows[0]['tf_fill_mean_degC'] == pytest.approx(9.0)


def test_without_a_reference_no_offset(setup):
    wd, ent = setup
    rows = fill_product(files(wd, 'P'), ent, 'mid', group='sector')
    assert rows[0]['bias_offset_degC'] == 0.0 and rows[0]['period'] == ''
    with pytest.raises(InvalidParamsError, match='period'):
        fill_product(files(wd, 'P'), ent, 'mid', ref_files=files(wd, 'REF'))


def test_a_second_pass_changes_nothing(setup):
    wd, ent = setup
    fill(wd, ent)
    assert fill(wd, ent) == []


def test_the_reference_must_be_filled_first(setup):
    wd, ent = setup
    write(wd, 1, 'REF', ['near', 'wide'], {'near': 0.6, 'wide': 1.1})
    with pytest.raises(InvalidWorkflowError, match='fill it first'):
        fill(wd, ent)
    assert len(fill(wd, ent, product='REF', ref=None)) == 1
    assert len(fill(wd, ent)) == 1


def test_nothing_within_reach_refuses(setup):
    wd, ent = setup
    with pytest.raises(InvalidWorkflowError, match='within 5 km'):
        fill(wd, ent, max_km=5)


def test_a_glacier_without_a_file_gets_every_band(setup):
    wd, ent = setup
    ent.loc[gid(5)] = ['east', 179.9, -70.18]
    write(wd, 5, 'REF', FULL, {'near': 0.6, 'mid': 3.5, 'wide': 1.1})
    rows = fill(wd, ent)
    got = [r for r in rows if r['glacier_id'] == gid(5)]
    assert [r['band'] for r in got] == FULL
    assert {r['method'] for r in got} == {'same_product_neighbours_all_bands'}
    ds = read_forcing(wd / gid(5) / 'ocean_P.nc')
    bc = read_forcing(wd / gid(5) / 'ocean_P_bc.nc')
    assert band_names(ds) == FULL
    assert ds.attrs['filled_bands'] == 'near mid wide'
    np.testing.assert_allclose(ds.thermal_forcing.isel(band=0), 0.5, atol=1e-4)
    np.testing.assert_allclose(bc.thermal_forcing.mean('time'), [0.6, 3.5, 1.1], atol=2e-3)
    assert bc.attrs['bias_correction_applied'] == 'True'
    assert ds.attrs['bias_correction_applied'] == 'False'


def test_a_missing_file_needs_a_target(setup):
    wd, ent = setup
    ent.loc[gid(5)] = ['east', 179.9, -70.18]
    write(wd, 5, 'REF', FULL, {'near': 0.6, 'mid': 3.5, 'wide': 1.1})
    with pytest.raises(InvalidParamsError, match='target'):
        fill_product(files(wd, 'P'), ent, 'mid', ref_files=files(wd, 'REF'), period=PERIOD)


def test_a_product_with_no_file_at_all_is_a_failure(setup):
    wd, ent = setup
    for f in wd.glob('*/ocean_P*.nc'):
        f.unlink()
    with pytest.raises(InvalidWorkflowError, match='failure, not a gap'):
        fill(wd, ent)
