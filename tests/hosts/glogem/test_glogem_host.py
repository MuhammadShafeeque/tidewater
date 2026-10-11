"""GloGEM host: the table writer, and the IDL side when an IDL or GDL interpreter is found.

The interpreter is ``$TIDEWATER_IDL`` or the first of ``gdl``, ``idl`` on PATH.
``$GLOGEM_DIR``, a GloGEM checkout with the calving factor hook, adds the test
of GloGEM's own calving routine.
"""
import os
import re
import shutil
import subprocess

import numpy as np
import pandas as pd
import pytest

from tidewater.export import read_host_bundle, read_region_table
from tidewater.hosts import glogem
from tidewater.spec.make_vectors import PATH as VECTORS
from tidewater.timeconv import yearly_ocean_factor

NORTH, SOUTH, NORTH2 = "RGI60-01.00570", "RGI60-17.01234", "RGI60-01.00001"


def monthly_tf(start="1999-01-01", end="2003-12-01", ids=(NORTH, SOUTH, NORTH2)):
    time = pd.date_range(start, end, freq="MS")
    n = len(time)
    return pd.DataFrame({gid: 2.0 + (i + 1) * np.sin(np.arange(n) / (i + 2))
                         for i, gid in enumerate(ids)}, index=time)


@pytest.mark.parametrize("region, resolution, month", [
    (1, "monthly", 10), (11, "monthly", 10), (16, "monthly", 4),
    (17, "monthly", 4), (18, "monthly", 4), (19, "monthly", 4),
    (5, "daily", 10), (19, "daily", 5)])
def test_start_month_per_region(region, resolution, month):
    assert glogem.start_month(region, resolution) == month


def test_start_months_and_bad_ids():
    assert glogem.start_months([NORTH, SOUTH]) == {NORTH: 10, SOUTH: 4}
    with pytest.raises(ValueError, match="RGI ids"):
        glogem.start_months(["glacier_a"])
    with pytest.raises(ValueError, match="several RGI versions"):
        glogem.start_months([NORTH, "RGI2000-v7.0-G-01-00570"])
    with pytest.raises(ValueError, match="time_resolution"):
        glogem.start_month(1, "yearly")


def test_forcing_span():
    assert glogem.forcing_span(2000, 2005, 1) == ((1999, 10), (2005, 9))
    assert glogem.forcing_span(2000, 2005, 17) == ((1999, 4), (2005, 3))
    assert glogem.forcing_span(2000, 2005, 17, "daily") == ((1999, 5), (2005, 4))


def test_round_trip_ids_and_headers(tmp_path):
    tf = monthly_tf()
    factor, info = glogem.write_ocean_factor_tables(tmp_path, tf, 2.5, gamma=1.18,
                                                    incomplete="drop")
    back = read_host_bundle(tmp_path)
    pd.testing.assert_frame_equal(back["factor"][factor.columns], factor,
                                  check_names=False, atol=5e-7, rtol=0)
    assert info["model_year"]["start_month"] == {NORTH: 10, SOUTH: 4, NORTH2: 10}
    assert info["host"] == "glogem" and info["rgi_version"] == "6.0"
    north, meta = read_region_table(tmp_path / "ocean_factor_01.dat")
    assert list(north.columns) == ["00570", "00001"]
    assert meta["start_month"] == "10" and meta["label"] == "end"
    assert meta["rgi_version"] == "6.0" and meta["time_resolution"] == "monthly"
    assert meta["outside"] == "stop"
    south, meta = read_region_table(tmp_path / "ocean_factor_17.dat")
    assert list(south.columns) == ["01234"] and meta["start_month"] == "4"


def test_years_follow_glogem_convention(tmp_path):
    tf = monthly_tf(ids=(NORTH, SOUTH))
    factor, _ = glogem.write_ocean_factor_tables(tmp_path, tf, 2.5, incomplete="drop")
    for gid, month in ((NORTH, 10), (SOUTH, 4)):
        want = yearly_ocean_factor(tf[gid], 2.5, start_month=month, label="end",
                                   incomplete="drop")
        got = factor[gid].dropna()
        pd.testing.assert_series_equal(got, want, check_names=False)
    # October to December 2001 belong to the northern model year 2002 only
    bumped = tf.copy()
    bumped.loc["2001-10-01":"2001-12-01", NORTH] += 1.0
    other, _ = glogem.write_ocean_factor_tables(tmp_path / "b", bumped, 2.5,
                                                incomplete="drop")
    changed = (other[NORTH] != factor[NORTH])
    assert changed[changed].index.tolist() == [2002]
    # the first northern year starts in October of the year before
    assert factor[NORTH].first_valid_index() == 2000
    assert factor[SOUTH].first_valid_index() == 2000


def test_tf_fraction(tmp_path):
    tf = monthly_tf()
    full, _ = glogem.write_ocean_factor_tables(tmp_path / "a", tf, 2.5, incomplete="drop")
    half, info = glogem.write_ocean_factor_tables(tmp_path / "b", tf, 2.5, tf_fraction=0.5,
                                                  incomplete="drop")
    np.testing.assert_allclose(half, 0.5 + 0.5 * full)
    assert info["tf_fraction"] == 0.5
    with pytest.raises(ValueError, match="tf_fraction"):
        glogem.write_ocean_factor_tables(tmp_path / "c", tf, 2.5, tf_fraction=2)


def test_incomplete_years_are_refused_by_default(tmp_path):
    with pytest.raises(ValueError, match="incomplete model years"):
        glogem.write_ocean_factor_tables(tmp_path, monthly_tf(), 2.5)


def test_outside_is_written_and_checked(tmp_path):
    glogem.write_ocean_factor_tables(tmp_path, monthly_tf(), 2.5, incomplete="drop",
                                     outside="nearest")
    assert read_region_table(tmp_path / "ocean_factor_01.dat")[1]["outside"] == "nearest"
    with pytest.raises(ValueError, match="outside"):
        glogem.write_ocean_factor_tables(tmp_path, monthly_tf(), 2.5, incomplete="drop",
                                         outside="clip")


def test_generic_table(tmp_path):
    yearly = pd.DataFrame({NORTH: [1.5, 2.5]}, index=[2000, 2001])
    glogem.write_tables(tmp_path, yearly, variable="thermal_forcing", units="degC",
                        time_resolution="daily")
    table, meta = read_region_table(tmp_path / "thermal_forcing_01.dat")
    assert meta["units"] == "degC" and meta["time_resolution"] == "daily"
    assert table["00570"].tolist() == [1.5, 2.5]


def test_idl_files_are_package_data():
    names = sorted(p.name for p in glogem.IDL_DIR.glob("*.pro"))
    assert names == ["tidewater_laws.pro", "tw_spec_check.pro", "tw_table_factor.pro"]


# --- IDL side ------------------------------------------------------------------

def _interpreter():
    return os.environ.get("TIDEWATER_IDL") or shutil.which("gdl") or shutil.which("idl")


needs_idl = pytest.mark.skipif(_interpreter() is None, reason="no IDL or GDL interpreter")

PROBE = """
pro tw_probe, dir, region, ids, years, bal_month, res, rgi, outside
  compile_opt idl2
  if outside ne '' then how = outside
  for i = 0, n_elements(ids) - 1 do begin
    v = call_function('tw_table_factor', dir, region, ids[i], years[i], $
      bal_month = bal_month, time_resolution = res, rgi_version = rgi, outside = how)
    print, 'TWOUT ', ids[i], ' ', strtrim(years[i], 2), ' ', string(v, format = '(e25.17)')
  endfor
end
"""


def _run_idl(command, extra_path=(), cwd=None):
    path = ":".join([str(glogem.IDL_DIR), *map(str, extra_path)])
    full = f"!path = '{path}:' + !path & {command}"
    out = subprocess.run([_interpreter(), "-quiet", "-e", full], capture_output=True,
                         text=True, timeout=600, cwd=cwd)
    return out.stdout + out.stderr


def _idl_list(values):
    return "[" + ", ".join(values) + "]"


def _probe(tmp_path, table_dir, region, cases, bal_month=10, res="monthly", rgi="6",
           outside=""):
    """Values of the cases; a run halted by tw_table_factor ends the list with None."""
    (tmp_path / "tw_probe.pro").write_text(PROBE)
    ids = _idl_list(f"'{g}'" for g, _ in cases)
    years = _idl_list(f"{y}L" for _, y in cases)
    text = _run_idl(f"tw_probe, '{table_dir}', '{region}', {ids}, {years}, {bal_month}, "
                    f"'{res}', '{rgi}', '{outside}'", extra_path=[tmp_path])
    found = [float(v) for v in re.findall(r"^TWOUT \S+ \S+\s+(\S+)\s*$", text,
                                                re.MULTILINE)]
    if len(found) < len(cases):
        assert "TW_TABLE_FACTOR:" in text, text
        found.append(None)
    return found


@needs_idl
def test_idl_laws_reproduce_the_reference_vectors():
    text = _run_idl(f"tw_spec_check, '{VECTORS}'")
    m = re.search(r"tw_spec_check rows=(\d+) failures=(\d+) max_rel_diff=(\S+)", text)
    assert m, text
    assert int(m[1]) > 600 and int(m[2]) == 0, text
    assert float(m[3]) <= 1e-12


@needs_idl
def test_idl_lookup_matches_the_table(tmp_path):
    factor, _ = glogem.write_ocean_factor_tables(tmp_path / "t", monthly_tf(), 2.5,
                                                 incomplete="drop")
    years = factor.index.tolist()
    cases = [("00570", y) for y in years] + [("00001", years[1]), ("99999", years[0])]
    got = _probe(tmp_path, tmp_path / "t", "01", cases)
    want = factor[NORTH].tolist() + [factor[NORTH2].iloc[1], 1.0]
    np.testing.assert_allclose(got, want, rtol=0, atol=5e-7)
    south = _probe(tmp_path, tmp_path / "t", "17", [("01234", years[1])], bal_month=4)
    np.testing.assert_allclose(south, [factor[SOUTH].iloc[1]], atol=5e-7)


@needs_idl
def test_idl_lookup_outside_and_checks(tmp_path):
    factor, _ = glogem.write_ocean_factor_tables(tmp_path / "t", monthly_tf(), 2.5,
                                                 incomplete="drop")
    last = int(factor.index.max())
    late = [("00570", last + 3)]
    assert _probe(tmp_path, tmp_path / "t", "01", late) == [None]
    assert _probe(tmp_path, tmp_path / "t", "01", late, outside="one") == [1.0]
    nearest = _probe(tmp_path, tmp_path / "t", "01", late, outside="nearest")
    np.testing.assert_allclose(nearest, [factor[NORTH].loc[last]], atol=5e-7)
    # without the keyword the table's own setting applies
    glogem.write_ocean_factor_tables(tmp_path / "n", monthly_tf(), 2.5,
                                     incomplete="drop", outside="nearest")
    assert _probe(tmp_path, tmp_path / "n", "01", late) == nearest
    assert _probe(tmp_path, tmp_path / "n", "01", late, outside="stop") == [None]
    # a run whose year starts in another month, or on other RGI ids, is refused
    assert _probe(tmp_path, tmp_path / "t", "17", [("01234", 2001)], bal_month=10) == [None]
    assert _probe(tmp_path, tmp_path / "t", "01", [("00570", 2001)], rgi="7") == [None]
    # daily runs give the start as a day of the year: 274 is 1 October
    assert _probe(tmp_path, tmp_path / "t", "01", [("00570", 2001)], bal_month=274,
                  res="daily")[0] is not None
    # a region without a table is left unchanged
    assert _probe(tmp_path, tmp_path / "t", "05", [("00001", 2001)]) == [1.0]


# --- GloGEM's calving routine ---------------------------------------------------

GLOGEM_DIR = os.environ.get("GLOGEM_DIR")

DRIVER = """
pro tw_drive, fn, dir, rid, region, year
  compile_opt idl2
  nb = 20
  thick = dblarr(nb) + 200d
  bed_elev = [-150d, -100d, -20d, dindgen(nb - 3) * 50d + 10d]
  bed_elev_p = bed_elev
  length = dblarr(nb) + 10d
  slope = dblarr(nb) + 5d
  width = dblarr(nb) + 2000d
  area = dblarr(nb) + 0.5d
  acc = dblarr(nb) + 2d
  dvol = -1d6 & dens = 0.9 & ar_gl = total(area)
  tran = [year - 5, year + 5] & ye = 5 & flux_calv = dblarr(11)
  frontal_ablation = 'y' & alpha_f = 0.7 & length_corrfact = 1.4 & c_calving = 0.01
  crit_ccorrdist = 3000. & ccorr_expon = 2.5 & calv_sep = 1.25
  id = [rid] & gg = [0] & g = 0
  rgiregion = region & RGIversion = '6' & bal_month = 10 & time_resolution = 'monthly'
  calving_factor_function = fn & calving_factor_dir = dir
  @procedures/processing/calving_model.pro
  print, 'TWOUT ', string(q_front, format = '(e25.17)'), string(dvol, format = '(e25.17)')
end
"""


def _drive(tmp_path, function, table_dir, glacier, year):
    (tmp_path / "tw_drive.pro").write_text(DRIVER)
    text = _run_idl(f"tw_drive, '{function}', '{table_dir}', '{glacier}', '01', {year}",
                    extra_path=[tmp_path], cwd=GLOGEM_DIR)
    m = re.search(r"^TWOUT\s+(\S+)\s+(\S+)\s*$", text, re.MULTILINE)
    assert m, text
    return float(m[1]), float(m[2])


@needs_idl
@pytest.mark.skipif(not GLOGEM_DIR, reason="no GloGEM checkout in $GLOGEM_DIR")
def test_glogem_calving_routine_scales_with_the_factor(tmp_path):
    factor, _ = glogem.write_ocean_factor_tables(tmp_path / "t", monthly_tf(), 2.5,
                                                 incomplete="drop")
    year = int(factor.index[1])
    off = _drive(tmp_path, "", "", "00001", year)
    on = _drive(tmp_path, "tw_table_factor", tmp_path / "t", "00001", year)
    assert off[0] > 0
    np.testing.assert_allclose(on[0] / off[0], factor[NORTH2].loc[year], rtol=1e-6)
    # a glacier that is not in the table keeps GloGEM's own flux
    assert _drive(tmp_path, "tw_table_factor", tmp_path / "t", "99999", year) == off
