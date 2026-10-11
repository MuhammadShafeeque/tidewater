import cftime
import numpy as np
import pandas as pd
import pytest

from tidewater import export as ex


def factor_table():
    cols = ["RGI60-01.00570", "RGI60-01.01450", "RGI60-19.00003"]
    data = np.array([[1.0, 0.5, 2.0], [1.25, np.nan, 0.0], [0.75, 1.5, 3.0]])
    return pd.DataFrame(data, index=[2001, 2002, 2003], columns=cols)


def test_region_tables_round_trip(tmp_path):
    table = factor_table()
    meta = {"start_month": 10, "label": "end"}
    paths = ex.write_region_tables(table, tmp_path / "out", meta=meta)
    assert sorted(paths) == ["01", "19"]
    assert paths["01"].name == "ocean_factor_01.dat"
    frame, header = ex.read_region_table(paths["01"])
    assert list(frame.columns) == ["00570", "01450"]
    assert list(frame.index) == [2001, 2002, 2003]
    np.testing.assert_allclose(frame.to_numpy(), table.iloc[:, :2].to_numpy())
    assert header["variable"] == "ocean_factor" and header["units"] == "1"
    assert header["region"] == "01" and header["start_month"] == "10"
    assert header["label"] == "end" and header["n_glaciers"] == "2"
    south, _ = ex.read_region_table(paths["19"])
    assert list(south.columns) == ["00003"]


def test_region_table_layout(tmp_path):
    path = ex.write_region_tables(factor_table(), tmp_path)["01"]
    lines = path.read_text().splitlines()
    assert len(lines) == ex.HEADER_LINES + 2
    assert all(line.startswith("#") for line in lines[:ex.HEADER_LINES - 1])
    assert lines[ex.HEADER_LINES - 1].split() == ["id", "2001", "2002", "2003"]
    assert all(len(line.split()) == 4 for line in lines[ex.HEADER_LINES:])
    assert lines[ex.HEADER_LINES + 1].split()[2] == "-9999.000000"
    bare = np.loadtxt(path, skiprows=ex.HEADER_LINES)
    assert bare.shape == (2, 4) and bare[1, 0] == 1450


def test_region_tables_custom_regions_and_ids(tmp_path):
    table = pd.DataFrame({"glacier_a": [1.0, 2.0], "glacier_b": [3.0, 4.0]},
                         index=[1990, 1991])
    with pytest.raises(ValueError, match="regions"):
        ex.write_region_tables(table, tmp_path)
    regions = {"glacier_a": "Svalbard", "glacier_b": "Patagonia"}
    paths = ex.write_region_tables(
        table, tmp_path, variable="thermal_forcing", units="degC",
        regions=regions, filename="{region}/{variable}.dat".replace("/", "_"))
    assert sorted(paths) == ["Patagonia", "Svalbard"]
    frame, header = ex.read_region_table(paths["Svalbard"])
    assert list(frame.columns) == ["glacier_a"]
    assert header["units"] == "degC" and list(frame["glacier_a"]) == [1.0, 2.0]


def test_region_tables_refuse_bad_input(tmp_path):
    table = factor_table()
    with pytest.raises(ValueError, match="whitespace"):
        ex.write_region_tables(table.rename(columns={"RGI60-01.00570": "a b"}),
                               tmp_path, regions={"a b": "x",
                                                  "RGI60-01.01450": "x",
                                                  "RGI60-19.00003": "x"})
    clash = pd.DataFrame({"RGI60-01.00001": [1.0],
                          "RGI2000-v7.0-G-01-00001": [2.0]}, index=[2000])
    with pytest.raises(ValueError, match="not unique"):
        ex.write_region_tables(clash, tmp_path)
    full = ex.write_region_tables(clash, tmp_path, id_style="full")
    frame, _ = ex.read_region_table(full["01"])
    assert list(frame.columns) == list(clash.columns)
    dated = table.set_axis(pd.date_range("2001-01-01", periods=3, freq="YS"))
    with pytest.raises((ValueError, TypeError)):
        ex.write_region_tables(dated, tmp_path)


def calibration():
    return pd.DataFrame(
        {"calving_k": [0.6, 1.25], "calving_k_unc": [0.1, 0.3],
         "n_iter": [4, 7]},
        index=["RGI60-01.00570", "RGI60-17.00005"])


def test_calibration_csv_round_trip(tmp_path):
    path = ex.write_calibration_csv(calibration(), tmp_path / "a" / "k.csv")
    assert path.read_text().splitlines()[0] == (
        "glacier_id,calving_k,calving_k_unc,n_iter")
    back = ex.read_calibration_csv(path).set_index("glacier_id")
    pd.testing.assert_frame_equal(back, calibration(), check_names=False)


def test_calibration_csv_pygem_columns(tmp_path):
    path = ex.write_calibration_csv(calibration().reset_index(names="glacier_id"),
                                    tmp_path / "k.csv", columns="pygem")
    raw = pd.read_csv(path)
    assert list(raw.columns[:3]) == ["RGIId", "calving_k", "calving_k_nmad"]
    assert raw["RGIId"][0] == "RGI60-01.00570"
    back = ex.read_calibration_csv(path, columns="pygem")
    assert list(back.columns[:3]) == ["glacier_id", "calving_k", "calving_k_unc"]
    custom = ex.write_calibration_csv(
        calibration(), tmp_path / "c.csv",
        columns={"glacier_id": "id", "calving_k_unc": "k_sd"})
    assert list(pd.read_csv(custom).columns[:3]) == ["id", "calving_k", "k_sd"]


def test_calibration_csv_numeric_ids_stay_strings(tmp_path):
    table = pd.DataFrame({"calving_k": [1.0], "calving_k_unc": [0.2]},
                         index=["01450"])
    back = ex.read_calibration_csv(ex.write_calibration_csv(table,
                                                            tmp_path / "k.csv"))
    assert back["glacier_id"][0] == "01450"


def test_calibration_csv_refuses_bad_tables(tmp_path):
    with pytest.raises(ValueError, match="missing columns"):
        ex.write_calibration_csv(calibration().drop(columns="calving_k_unc"),
                                 tmp_path / "k.csv")
    twice = pd.concat([calibration(), calibration()])
    with pytest.raises(ValueError, match="unique"):
        ex.write_calibration_csv(twice, tmp_path / "k.csv")
    with pytest.raises(ValueError):
        ex.write_calibration_csv(calibration(), tmp_path / "k.csv",
                                 columns="glogem")


def test_netcdf_round_trip_dates(tmp_path):
    idx = pd.date_range("2000-01-01", periods=6, freq="MS")
    cols = ["RGI60-01.00570", "RGI2000-v7.0-G-19-00003"]
    data = pd.DataFrame(np.arange(12.0).reshape(6, 2), index=idx, columns=cols)
    data.iloc[2, 1] = np.nan
    lon = {cols[0]: -140.5, cols[1]: 170.25}
    path = ex.write_netcdf(data, tmp_path / "tf.nc", variable="thermal_forcing",
                           units="degC", lon=lon, lat=[60.0, -72.5],
                           attrs={"ocean_product": "synthetic"})
    ds = ex.read_netcdf(path)
    da = ds["thermal_forcing"]
    assert da.dims == ("time", "rgi_id") and da.attrs["units"] == "degC"
    assert list(ds["rgi_id"].values) == cols
    np.testing.assert_allclose(da.values, data.to_numpy())
    np.testing.assert_allclose(ds["lon"].values, [-140.5, 170.25])
    np.testing.assert_allclose(ds["lat"].values, [60.0, -72.5])
    assert ds["lat"].attrs["units"] == "degrees_north"
    assert ds.attrs["ocean_product"] == "synthetic"
    assert list(pd.DatetimeIndex(ds["time"].values)) == list(idx)
    only = ex.read_netcdf(path, "thermal_forcing")
    assert only.name == "thermal_forcing"


def test_netcdf_noleap_and_model_years(tmp_path):
    idx = [cftime.datetime(2000, m, 1, calendar="noleap") for m in (1, 2, 3)]
    data = pd.DataFrame({"a": [1.0, 2.0, 3.0]}, index=idx)
    ds = ex.read_netcdf(ex.write_netcdf(data, tmp_path / "noleap.nc"))
    assert ds["time"].values[1].calendar == "noleap"
    assert (ds["time"].values[2].year, ds["time"].values[2].month) == (2000, 3)

    years = pd.DataFrame({"a": [1.0, 2.0], "b": [3.0, 4.0]}, index=[2001, 2002])
    path = ex.write_netcdf(years, tmp_path / "years.nc", calendar="360_day",
                           attrs={"start_month": 10, "year_label": "end"})
    ds = ex.read_netcdf(path)
    assert list(ds["year"].values) == [2001, 2002]
    assert ds["time"].values[0].calendar == "360_day"
    assert ds.attrs["start_month"] == 10 and ds.attrs["year_label"] == "end"
    np.testing.assert_allclose(ds["ocean_factor"].values, years.to_numpy())


def test_host_bundle_round_trip(tmp_path):
    years = [2001, 2002, 2003]
    factor = pd.DataFrame({"RGI60-05.00001": [1.0, 1.1, np.nan],
                           "RGI60-05.00002": [0.9, 1.0, 1.2],
                           "RGI60-19.00010": [1.0, 0.8, 0.7]},
                          index=pd.Index(years, name="year"))
    cal = pd.DataFrame({"calving_k": [0.5, 1.0, 2.0], "calving_k_unc": [0.1, 0.2, 0.3],
                        "flux": [0.1, 0.2, 0.3], "history": [[], [], []]},
                       index=pd.Index(list(factor.columns), name="glacier_id"))
    cal.attrs["flux_unit"] = "Gt yr-1"
    info = ex.write_host_bundle(tmp_path, factor, cal, params={"gamma": 1.18},
                                start_month={"RGI60-05.00001": 10, "RGI60-05.00002": 10,
                                             "RGI60-19.00010": 4})
    assert sorted(info["tables"]) == ["05", "19"]
    assert info["sidecar"].name == ex.SIDECAR
    back = ex.read_host_bundle(tmp_path)
    pd.testing.assert_frame_equal(back["factor"], factor, check_names=False,
                                  check_index_type=False)
    meta = back["meta"]
    assert meta["tidewater_version"] == ex.__version__
    assert meta["parameters"] == {"gamma": 1.18}
    assert meta["model_year"]["start_month"]["RGI60-19.00010"] == 4
    assert meta["units"]["flux"] == "Gt yr-1" and meta["units"]["calving_k"] == "yr-1"
    assert "both included" in meta["period_convention"]
    c = back["calibration"]
    assert c.attrs["flux_unit"] == "Gt yr-1" and "history" not in c.columns
    np.testing.assert_allclose(c["calving_k"], cal["calving_k"])
    _, header = ex.read_region_table(tmp_path / info["tables"]["19"])
    assert header["start_month"] == "per_glacier" and header["label"] == "end"

    other = tmp_path / "pygem"
    ex.write_host_bundle(other, factor, cal, start_month=1, calibration_columns="pygem")
    c = ex.read_host_bundle(other)["calibration"]
    assert "glacier_id" in c.columns and c["glacier_id"].tolist() == list(factor.columns)
    assert ex.read_host_bundle(other)["meta"]["units"]["calving_k_nmad"] == "yr-1"
    alone = ex.write_host_bundle(tmp_path / "f", factor)
    assert "calibration" not in alone
    assert ex.read_host_bundle(tmp_path / "f")["calibration"] is None
