"""Smoke tests of the examples and the figure scripts, on the synthetic case."""
import importlib.util
import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
EXAMPLES = ROOT / "examples"
FIGURES = ROOT / "docs" / "figures"
SYNTHETIC = EXAMPLES / "results" / "synthetic"


def load(path, name):
    """Import a script by its path, with its own directory on the path."""
    for d in (str(path.parent), str(EXAMPLES)):
        if d not in sys.path:
            sys.path.insert(0, d)
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def test_synthetic_case_calibrates_and_runs(tmp_path):
    case = load(EXAMPLES / "example_case.py", "example_case")
    run = load(EXAMPLES / "synthetic" / "run.py", "tw_example_synthetic_run")
    summary = run.main(tmp_path, quick=True)
    assert len(summary) == 6 and summary["converged"].all()
    assert (summary["ratio"] - 1).abs().max() < 0.05
    for route in ("law_each_step", "yearly_constant", "factor_table"):
        rows = pd.read_csv(tmp_path / f"toy_{route}.csv")
        assert list(rows.columns) == case.COLUMNS
        assert set(rows["run"]) == {"control", "ocean"}
        assert set(rows["scenario"]) == {"historical", "ssp126", "ssp585"}
        assert (rows.loc[rows["run"] == "control", "ocean_factor"] == 1).all()
    for name in ("outline", "flowline", "front", "ocean_cells"):
        assert len(pd.read_csv(tmp_path / "map" / f"{name}.csv")) > 0


def test_shipped_synthetic_results_are_complete():
    for name in ("forcing.csv", "forcing.json", "calibration.csv", "map/water_depth.nc"):
        assert (SYNTHETIC / name).exists(), name
    assert len(list(SYNTHETIC.glob("toy_*.csv"))) == 3


def test_fetch_data_list_matches_the_shipped_results():
    fetch = load(EXAMPLES / "fetch_data.py", "tw_fetch_data")
    rows = [r for r in fetch.read_files() if r["set"] == "results"]
    assert len(rows) == 9
    for r in rows:
        assert fetch.sha256(EXAMPLES / "results" / "05-09345" / r["target"]) == r["sha256"], r["target"]


def test_fetch_data_checks_the_checksum(tmp_path):
    fetch = load(EXAMPLES / "fetch_data.py", "tw_fetch_data")
    (tmp_path / "home").mkdir()
    (tmp_path / "home" / "a.txt").write_text("a")
    row = {"path": "a.txt", "target": "sub/a.txt", "sha256": fetch.sha256(tmp_path / "home" / "a.txt")}
    url = (tmp_path / "home").as_uri()
    assert fetch.fetch([row], tmp_path / "out", url) == 1
    assert fetch.fetch([row], tmp_path / "out", url) == 0
    with pytest.raises(ValueError):
        fetch.fetch([dict(row, target="b.txt", sha256="0" * 64)], tmp_path / "out", url)
    assert not (tmp_path / "out" / "b.txt.part").exists()


def test_glogem_example_writes_tables(tmp_path):
    ex = load(EXAMPLES / "glogem" / "run_glogem.py", "tw_example_glogem")
    factors = ex.main(SYNTHETIC, tmp_path)
    assert (tmp_path / "ssp126" / "ocean_factor_05.dat").exists()
    assert abs(factors["ssp126"].loc[2000:2009].mean() - 1) < 0.05


def test_oggm_example_writes_a_valid_forcing_file(tmp_path):
    from tidewater.forcing.io import validate

    ex = load(EXAMPLES / "oggm" / "run_oggm.py", "tw_example_oggm")
    out = tmp_path / "ocean_data.nc"
    law = ex.main([str(SYNTHETIC), "--out", str(out)])
    assert validate(out, bands=["shallow_0_60"])["problem"] == ""
    assert law.name == "tf_power"


def test_oggm_example_host_part():
    pytest.importorskip("oggm.core")  # not the examples/oggm directory
    import tidewater.hosts.oggm
    from tidewater.hosts.oggm import tasks

    if not tidewater.hosts.oggm.supported():
        pytest.skip("this OGGM has no glacier settings (1.6.3 or earlier)")
    assert callable(tasks.run_dynamic_calving_k_calibration)
    assert callable(tasks.run_with_ocean_forcing)


def test_pygem_example_builds_the_law():
    ex = load(EXAMPLES / "pygem" / "run_pygem.py", "tw_example_pygem")
    laws, factors = ex.main([str(SYNTHETIC)])
    assert list(laws) == ["RGI2000-v7.0-G-05-09345"]
    assert factors.index.min() <= 2000 and factors.index.max() == 2100
    laws6, factors6 = ex.main([str(SYNTHETIC), "--rgi6"])
    assert list(laws6) == ["RGI60-05.13043"]
    assert factors6.equals(factors.set_axis(["RGI60-05.13043"], axis=1))


def test_pygem_example_host_part():
    pytest.importorskip("pygem.bin.run")  # not the examples/pygem directory
    from tidewater.hosts.pygem.dynamics import ocean_dynamics

    assert callable(ocean_dynamics)


@pytest.fixture
def figure_out(tmp_path, monkeypatch):
    pytest.importorskip("matplotlib")
    style = load(FIGURES / "style.py", "style")
    monkeypatch.setattr(style, "OUT", tmp_path)
    return tmp_path


def test_schematics_draw(figure_out):
    schematics = load(FIGURES / "schematics.py", "tw_schematics")
    schematics.main()
    assert len(list(figure_out.glob("*.svg"))) == len(schematics.FIGURES)


NAMES = ["example-map", "example-forcing", "example-calibration", "example-scatter",
         "example-historical", "example-ocean", "example-projection"]


@pytest.mark.parametrize("case", ["05-09345", "synthetic"])
def test_result_figures_draw(figure_out, case):
    results = load(FIGURES / "results.py", "tw_results")
    paths = results.main(EXAMPLES / "results" / case)
    assert [p.stem for p in paths] == NAMES
