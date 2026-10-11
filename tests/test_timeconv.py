import cftime
import numpy as np
import pandas as pd
import pytest
import xarray as xr

from tidewater import timeconv as tc


def monthly(start, periods):
    return pd.date_range(start, periods=periods, freq="MS")


def test_resolve_start_month():
    assert tc.resolve_start_month() == 1
    assert tc.resolve_start_month(7) == 7
    assert tc.resolve_start_month("calendar") == 1
    assert tc.resolve_start_month("hydro_north") == 10
    assert tc.resolve_start_month("hydro_south") == 4
    assert tc.resolve_start_month("auto", latitude=70.0) == 10
    assert tc.resolve_start_month("auto", latitude=-54.0) == 4
    with pytest.raises(ValueError):
        tc.resolve_start_month("auto")
    with pytest.raises(ValueError):
        tc.resolve_start_month(13)
    with pytest.raises(ValueError):
        tc.resolve_start_month("water")


def test_latitude_is_ignored_without_auto():
    s = pd.Series(1.0, index=monthly("2001-01-01", 12))
    out = tc.to_model_years(s, latitude=-60.0)
    assert list(out.index) == [2001]


def test_model_year_labels():
    y = np.array([2000, 2000, 2001, 2001])
    m = np.array([9, 10, 3, 4])
    assert list(tc.model_year(y, m)) == [2000, 2000, 2001, 2001]
    assert list(tc.model_year(y, m, 10, "end")) == [2000, 2001, 2001, 2001]
    assert list(tc.model_year(y, m, 10, "start")) == [1999, 2000, 2000, 2000]
    assert list(tc.model_year(y, m, 4, "end")) == [2001, 2001, 2001, 2002]
    with pytest.raises(ValueError):
        tc.model_year(y, m, 10, "middle")


def test_model_year_span():
    assert tc.model_year_span(2001) == ((2001, 1), (2001, 12))
    assert tc.model_year_span(2001, 10, "end") == ((2000, 10), (2001, 9))
    assert tc.model_year_span(2001, 10, "start") == ((2001, 10), (2002, 9))
    assert tc.model_year_span(2001, 4, "end") == ((2000, 4), (2001, 3))


def test_days_in_month():
    assert list(tc.days_in_month([2000, 2001, 1900], [2, 2, 2])) == [29, 28, 28]
    assert list(tc.days_in_month([2000], [2], "noleap")) == [28]
    assert list(tc.days_in_month([2001], [2], "all_leap")) == [29]
    assert list(tc.days_in_month([2001, 2001], [2, 7], "360_day")) == [30, 30]
    assert list(tc.days_in_month([1900], [2], "julian")) == [29]
    with pytest.raises(ValueError):
        tc.days_in_month([2000], [1], "lunar")


def test_calendar_year_weighted_mean():
    idx = monthly("2001-01-01", 24)
    s = pd.Series(np.tile(np.arange(1.0, 13.0), 2), index=idx, name="tf")
    out = tc.to_model_years(s)
    days = idx[:12].days_in_month.to_numpy()
    assert list(out.index) == [2001, 2002]
    assert out.index.name == "year" and out.name == "tf"
    assert out[2001] == pytest.approx(np.average(np.arange(1, 13), weights=days))
    equal = tc.to_model_years(s, weights="equal")
    assert equal[2001] == pytest.approx(6.5)


def test_hydro_year_and_labels():
    idx = monthly("2000-10-01", 24)
    s = pd.Series(np.repeat([1.0, 3.0], 12), index=idx)
    end = tc.to_model_years(s, start_month=10, label="end")
    start = tc.to_model_years(s, start_month="hydro_north", label="start")
    assert list(end.index) == [2001, 2002]
    assert list(start.index) == [2000, 2001]
    assert list(end) == pytest.approx([1.0, 3.0])
    south = tc.to_model_years(pd.Series(2.0, index=monthly("2000-04-01", 12)),
                              start_month="auto", latitude=-45.0)
    assert list(south.index) == [2001]


def test_incomplete_years():
    s = pd.Series(1.0, index=monthly("2000-01-01", 24))
    with pytest.raises(ValueError, match="incomplete"):
        tc.to_model_years(s, start_month=10)
    drop = tc.to_model_years(s, start_month=10, incomplete="drop")
    assert list(drop.index) == [2001]
    keep = tc.to_model_years(s, start_month=10, incomplete="keep")
    assert list(keep.index) == [2000, 2001, 2002]
    with pytest.raises(ValueError):
        tc.to_model_years(s, incomplete="maybe")


def test_gap_and_duplicates_are_refused():
    idx = monthly("2001-01-01", 12)
    with pytest.raises(ValueError, match="incomplete"):
        tc.to_model_years(pd.Series(1.0, index=idx.delete(5)))
    daily = pd.Series(1.0, index=pd.date_range("2001-01-01", periods=365))
    with pytest.raises(ValueError, match="not monthly"):
        tc.to_model_years(daily)


def test_nan_propagates_and_sum():
    idx = monthly("2001-01-01", 24)
    df = pd.DataFrame({"a": 2.0, "b": 1.0}, index=idx)
    df.iloc[3, 1] = np.nan
    out = tc.to_model_years(df, how="sum")
    assert out.loc[2001, "a"] == pytest.approx(24.0)
    assert np.isnan(out.loc[2001, "b"])
    assert out.loc[2002, "b"] == pytest.approx(12.0)
    assert list(out.columns) == ["a", "b"]


@pytest.mark.parametrize("calendar, feb", [("noleap", 28), ("360_day", 30)])
def test_cftime_calendars(calendar, feb):
    idx = [cftime.datetime(2000 + (i // 12), i % 12 + 1, 15, calendar=calendar)
           for i in range(24)]
    vals = np.tile(np.arange(1.0, 13.0), 2)
    out = tc.to_model_years(pd.Series(vals, index=idx))
    days = tc.days_in_month(np.full(12, 2000), np.arange(1, 13), calendar)
    assert days[1] == feb
    assert out[2000] == pytest.approx(np.average(np.arange(1, 13), weights=days))
    cfidx = xr.CFTimeIndex(idx)
    hyd = tc.to_model_years(pd.Series(vals, index=cfidx), start_month=10,
                            incomplete="drop")
    assert list(hyd.index) == [2001]


def test_calendar_override():
    s = pd.Series(np.arange(1.0, 13.0), index=monthly("2000-01-01", 12))
    out = tc.to_model_years(s, calendar="360_day")
    assert out[2000] == pytest.approx(6.5)


def test_period_index():
    idx = pd.period_range("2001-01", periods=12, freq="M")
    out = tc.to_model_years(pd.Series(1.0, index=idx))
    assert list(out.index) == [2001] and out[2001] == pytest.approx(1.0)


def test_dataarray():
    idx = monthly("2000-10-01", 24)
    da = xr.DataArray(
        np.repeat([[1.0, 2.0], [3.0, 4.0]], 12, axis=0).T,
        dims=("rgi_id", "time"),
        coords={"time": idx, "rgi_id": ["a", "b"], "lat": ("rgi_id", [60., 61.])},
        name="tf", attrs={"units": "degC"})
    out = tc.to_model_years(da, start_month=10)
    assert out.dims == ("rgi_id", "year")
    assert list(out["year"].values) == [2001, 2002]
    np.testing.assert_allclose(out.sel(rgi_id="b").values, [2.0, 4.0])
    assert out.attrs["units"] == "degC" and "lat" in out.coords


def test_bad_input():
    with pytest.raises(TypeError):
        tc.to_model_years([1.0, 2.0])
    with pytest.raises(TypeError):
        tc.to_model_years(pd.Series([1.0, 2.0]))


def test_start_months_per_glacier_in_one_call():
    idx = monthly("2000-01-01", 36)
    df = pd.DataFrame({"north": np.arange(36.0), "south": np.arange(36.0),
                       "cal": np.arange(36.0)}, index=idx)
    months = {"north": 10, "south": 4, "cal": 1}
    out = tc.to_model_years(df, start_month=months, incomplete="drop")
    assert list(out.columns) == ["north", "south", "cal"] and out.index.name == "year"
    for gid, m in months.items():
        one = tc.to_model_years(df[gid], start_month=m, incomplete="drop")
        pd.testing.assert_series_equal(out[gid].dropna(), one, check_names=False,
                                       check_index_type=False)
    # a model year only some glaciers have is NaN for the others
    assert list(out.index) == [2000, 2001, 2002]
    assert np.isnan(out.loc[2000, "north"]) and np.isnan(out.loc[2000, "south"])
    assert out.loc[2001, "north"] == pytest.approx(
        tc.to_model_years(df["north"], 10, incomplete="drop")[2001])
    # from latitude, as a list in the order of the columns
    lat = tc.to_model_years(df, start_month="auto", latitude=[70.0, -60.0, 70.0],
                            incomplete="drop")
    pd.testing.assert_series_equal(lat["south"], out["south"].dropna())
    np.testing.assert_array_equal(tc.resolve_start_months(["a", "b"], "auto",
                                                          {"a": 1.0, "b": -1.0}), [10, 4])
    with pytest.raises(ValueError, match="no start_month"):
        tc.to_model_years(df, start_month={"north": 10})
    with pytest.raises(ValueError, match="values of start_month"):
        tc.to_model_years(df, start_month=[10, 4])
    with pytest.raises(TypeError):
        tc.to_model_years(df["north"], start_month=[10])

    da = xr.DataArray(df.to_numpy(), dims=("time", "rgi_id"),
                      coords={"time": idx, "rgi_id": list(df.columns)})
    got = tc.to_model_years(da, start_month=months, incomplete="drop")
    assert got.dims == ("year", "rgi_id") and list(got.rgi_id.values) == list(df.columns)
    np.testing.assert_allclose(got.transpose("year", "rgi_id").values,
                               out.loc[got.year.values].to_numpy())


def test_a_yearly_factor_is_the_mean_of_the_monthly_factors():
    idx = monthly("2000-01-01", 24)
    tf = pd.Series(np.tile([0.0, -1.0, 4.0] * 4, 2), index=idx)
    got = tc.yearly_ocean_factor(tf, tf_ref=2.0, gamma=1.18)
    w = tc.days_in_month(idx.year, idx.month)[:12].astype(float)
    want = np.sum(w * (np.clip(tf[:12], 0, None) / 2.0) ** 1.18) / w.sum()
    assert got[2000] == pytest.approx(want)
    # not the factor of the yearly mean forcing
    mean_tf = tc.to_model_years(tf)[2000]
    assert got[2000] != pytest.approx((mean_tf / 2.0) ** 1.18)
    equal = tc.yearly_ocean_factor(tf, 2.0, gamma=1.18, weights="equal")
    assert equal[2000] == pytest.approx(np.mean((np.clip(tf[:12], 0, None) / 2.0) ** 1.18))

    df = pd.DataFrame({"a": tf.to_numpy(), "b": 2 * tf.to_numpy()}, index=idx)
    two = tc.yearly_ocean_factor(df, {"a": 2.0, "b": 4.0}, gamma=1.18,
                                 start_month={"a": 1, "b": 1})
    np.testing.assert_allclose(two["a"], got)
    np.testing.assert_allclose(two["b"], got)
    da = xr.DataArray(df.to_numpy(), dims=("time", "rgi_id"),
                      coords={"time": idx, "rgi_id": ["a", "b"]})
    ref = xr.DataArray([2.0, 4.0], dims="rgi_id", coords={"rgi_id": ["a", "b"]})
    fx = tc.yearly_ocean_factor(da, ref, gamma=1.18)
    np.testing.assert_allclose(fx.sel(rgi_id="b").values, got.to_numpy())
    assert fx.attrs["units"] == "1"
    with pytest.raises(TypeError):
        tc.yearly_ocean_factor([1.0], 1.0)
