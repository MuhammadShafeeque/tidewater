import numpy as np
import pandas as pd
import pytest
import xarray as xr

from tidewater.config import InvalidParamsError
from tidewater.periods import (as_period, covers, n_years, run_bounds,
                               time_slice, year_mask)


def test_a_period_is_two_included_years():
    for period in ((2000, 2010), [2000., 2010.], '2000-2010', '2000,2010',
                   ' 2000 , 2010 ', np.array([2000, 2010]), ('2000', '2010')):
        assert as_period(period) == (2000, 2010)
    y0, y1 = as_period((2000., 2010.))
    assert isinstance(y0, int) and isinstance(y1, int)
    assert as_period((2005, 2005)) == (2005, 2005)


def test_an_exclusive_end_is_converted():
    # a run to 2020 holds the years up to 2019
    assert as_period((2000, 2020), end='exclusive') == (2000, 2019)
    assert as_period('2000-2001', end='exclusive') == (2000, 2000)
    with pytest.raises(InvalidParamsError, match='empty'):
        as_period((2000, 2000), end='exclusive')
    with pytest.raises(InvalidParamsError, match='inclusive or exclusive'):
        as_period((2000, 2010), end='open')


def test_what_is_not_a_period_raises():
    for bad in ((2000,), (2000, 2005, 2010), 'the noughties', None, 2000,
                ('a', 'b')):
        with pytest.raises(InvalidParamsError, match='two years'):
            as_period(bad)
    with pytest.raises(InvalidParamsError, match='empty'):
        as_period((2010, 2000))


def test_run_bounds_and_length():
    assert run_bounds((2000, 2019)) == (2000, 2020)
    assert as_period(run_bounds((2000, 2019)), end='exclusive') == (2000, 2019)
    assert n_years((2000, 2019)) == 20
    assert n_years('2005-2005') == 1


def test_year_mask_on_calendar_and_decimal_years():
    np.testing.assert_array_equal(
        year_mask([1999, 2000, 2005, 2010, 2011], (2000, 2010)),
        [False, True, True, True, False])
    monthly = np.arange(1999, 2012, 1 / 12) + 0.5 / 12
    sel = year_mask(monthly, (2000, 2010))
    assert sel.sum() == 11 * 12
    assert monthly[sel][0] == pytest.approx(2000 + 0.5 / 12)
    assert monthly[sel][-1] == pytest.approx(2010 + 11.5 / 12)
    # the same years as a test on the calendar year
    cal = np.floor(monthly)
    np.testing.assert_array_equal(sel, (cal >= 2000) & (cal <= 2010))
    assert not year_mask([np.nan], (2000, 2010))[0]
    assert year_mask(2010.999, '2000-2010') and not year_mask(2011., '2000-2010')


def test_time_slice_selects_whole_years():
    time = pd.date_range('1998-01-01', '2012-12-01', freq='MS')
    da = xr.DataArray(np.arange(len(time)), coords={'time': time}, dims='time')
    sel = da.sel(time=time_slice((2000, 2010)))
    assert len(sel) == 11 * 12
    assert (int(sel['time.year'][0]), int(sel['time.month'][0])) == (2000, 1)
    assert (int(sel['time.year'][-1]), int(sel['time.month'][-1])) == (2010, 12)
    assert time_slice('2000-2010') == slice('2000', '2010')
    assert time_slice((850, 900)) == slice('0850', '0900')
    # the label slice and the mask agree
    years = da['time.year'].values + (da['time.month'].values - 0.5) / 12
    np.testing.assert_array_equal(da.values[year_mask(years, (2000, 2010))],
                                  sel.values)


def test_covers():
    assert covers(range(1990, 2021), (2000, 2010))
    assert covers([2000.04, 2000.9, 2001.5], (2000, 2001))
    assert not covers(range(2001, 2021), (2000, 2010))
    assert not covers([2000, 2001, 2003], (2000, 2003))
    assert not covers([], (2000, 2000))
