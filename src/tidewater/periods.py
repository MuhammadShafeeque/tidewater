"""One period convention: ``(first year, last year)``, both included.

A host whose runs end before their last year (a run to 2020 holds the years
up to 2019) converts with :func:`as_period` and :func:`run_bounds`.
"""
import numpy as np

from tidewater.config import InvalidParamsError


def as_period(period, end='inclusive'):
    """``(y0, y1)`` with both years included, as integers.

    Parameters
    ----------
    period : sequence of two years, or 'y0-y1' / 'y0,y1'
    end : str
        'inclusive': ``period`` already includes its last year.
        'exclusive': ``period`` ends at the start of its last year.
    """
    if isinstance(period, str):
        period = period.replace(',', '-').split('-')
    try:
        y0, y1 = (int(float(y)) for y in period)
    except (TypeError, ValueError):
        raise InvalidParamsError(f'a period is two years, not {period!r}')
    if end not in ('inclusive', 'exclusive'):
        raise InvalidParamsError(f'end {end!r} is not inclusive or exclusive')
    if end == 'exclusive':
        y1 -= 1
    if y1 < y0:
        raise InvalidParamsError(f'the period {y0}-{y1} is empty')
    return y0, y1


def run_bounds(period):
    """Start and end of a model run that holds every year of ``period``."""
    y0, y1 = as_period(period)
    return y0, y1 + 1


def n_years(period):
    """Number of years in ``period``."""
    y0, y1 = as_period(period)
    return y1 - y0 + 1


def year_mask(years, period):
    """True where calendar or decimal ``years`` fall in ``period``."""
    y0, y1 = as_period(period)
    years = np.asarray(years, dtype=float)
    return (years >= y0) & (years < y1 + 1)


def time_slice(period):
    """Label slice that selects ``period`` on a datetime axis."""
    y0, y1 = as_period(period)
    return slice(f'{y0:04d}', f'{y1:04d}')


def covers(years, period):
    """Whether ``years`` hold every year of ``period``."""
    y0, y1 = as_period(period)
    have = set(np.floor(np.asarray(years, dtype=float)).astype(int).tolist())
    return all(y in have for y in range(y0, y1 + 1))
