"""Unit handling shared by the laws, the forcing and the calibration."""
import numpy as np

from tidewater.config import SEC_IN_YEAR, InvalidParamsError

GT_PER_YR, KM3_PER_YR, M3_PER_S = 'Gt yr-1', 'km3 yr-1', 'm3 s-1'
VOLUME_UNITS = ('m3 s-1', 'm3 yr-1', 'km3 yr-1')
MASS_UNITS = ('kg s-1', 'kg yr-1', 'Gt yr-1')
FLUX_UNITS = VOLUME_UNITS + MASS_UNITS

_CELSIUS = ('degc', 'c', 'celsius', 'degree_celsius', 'degrees_celsius',
            'deg_c', 'degree_c')
_KELVIN = ('k', 'kelvin', 'degk', 'deg_k')


def convert_flux(values, from_unit, to_unit, ice_density=None,
                 sec_in_year=SEC_IN_YEAR):
    """A frontal ablation flux on another rate, volume or mass basis.

    Parameters
    ----------
    values : float or array
        the flux in ``from_unit``; volumes are volumes of ice
    from_unit, to_unit : str
        one of :data:`FLUX_UNITS`
    ice_density : float
        kg m-3, required between a volume and a mass
    sec_in_year : float
        s
    """
    for unit in (from_unit, to_unit):
        if unit not in FLUX_UNITS:
            raise InvalidParamsError(f'unknown flux unit {unit!r}; '
                                     f'one of {FLUX_UNITS}')
    if from_unit == to_unit:
        return values
    crosses = (from_unit in MASS_UNITS) != (to_unit in MASS_UNITS)
    if crosses and (ice_density is None or not ice_density > 0):
        raise InvalidParamsError(f'{from_unit} to {to_unit} needs the ice '
                                 'density, ice_density in kg m-3')
    if isinstance(values, (list, tuple)):
        values = np.asarray(values, dtype=float)
    trio = (GT_PER_YR, KM3_PER_YR, M3_PER_S)
    if from_unit in trio and to_unit in trio:
        # through km3 yr-1 of ice
        if from_unit == GT_PER_YR:
            values = values / (ice_density / 1000.)
        elif from_unit == M3_PER_S:
            values = values * sec_in_year / 1e9
        if to_unit == GT_PER_YR:
            values = values * (ice_density / 1000.)
        elif to_unit == M3_PER_S:
            values = values * 1e9 / sec_in_year
        return values
    rho = ice_density if crosses or ice_density else 1.
    # each unit in m3 s-1 of ice
    to_si = {'m3 s-1': 1., 'm3 yr-1': 1. / sec_in_year,
             'km3 yr-1': 1e9 / sec_in_year, 'kg s-1': 1. / rho,
             'kg yr-1': 1. / rho / sec_in_year,
             'Gt yr-1': 1e12 / rho / sec_in_year}
    return values * to_si[from_unit] / to_si[to_unit]


def temperature_unit(units, values=None):
    """'degC' or 'K' from a unit string; 'auto' or None tells them apart by magnitude."""
    u = str('auto' if units is None else units).strip().lower()
    if u in _CELSIUS:
        return 'degC'
    if u in _KELVIN:
        return 'K'
    if u != 'auto':
        raise InvalidParamsError(f'temperature units {units!r} not understood: '
                                 'pass "degC", "K" or "auto"')
    values = None if values is None else np.asarray(values, dtype=float)
    if values is None or not np.isfinite(values).any():
        raise InvalidParamsError('units "auto" need finite values: pass '
                                 '"degC" or "K"')
    # sea water is never 100 degC
    return 'K' if np.nanmean(values) > 100 else 'degC'


def to_celsius(temperature, units='auto'):
    """Temperature in degC from degC or K.

    ``units``: 'degC', 'K' or 'auto' (told apart by magnitude).
    """
    t = np.asarray(temperature, dtype=float)
    if temperature_unit(units, t) == 'K':
        t = t - 273.15
    return t
