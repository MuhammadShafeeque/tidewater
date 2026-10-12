"""Ocean-forced frontal ablation laws and the algebra of their constants.

Five laws, as a nested family rooted at the constant-k law
``Q = k * d * h * w``:

===================  =========================================================
``constant``         :class:`ConstantK`, after Oerlemans and Nick (2005). The
                     control.
``tf_power``         :class:`TFPower`,
                     ``k(t) = k0 ((1 - a) + a (TF/TF_ref)**gamma)``.
                     Reduces to the control when ``TF == TF_ref`` or ``a = 0``.
``melt_calving``     :class:`MeltPlusCalving`, an explicit submarine-melt term
                     after Rignot et al. (2016) plus a calving term.
``sea_ice``          :class:`SeaIceModulated`, the melt term gated by the
                     open-water fraction. Reduces to ``melt_calving`` at
                     ``delta = 0``.
``larger_of``        :class:`LargerOf`, the larger of the calving and the
                     submarine-melt speed, after Malles et al. (2025).
                     Reduces to the control at ``lam = 0``.
===================  =========================================================

Calving constants are given in yr-1 and applied in s-1; a law returns a volume
flux of ice in m3 s-1 (:func:`convert_flux` for other bases). ``h`` is the ice
thickness at the front, ``d`` the water depth and ``w`` the width, all in m.
The thermal forcing is clipped at zero before any power.

A law takes plain numbers (:meth:`_OceanCalvingLaw.flux`, or
:meth:`_OceanCalvingLaw.fluxes` for arrays of fronts). A host model may also
call it as ``law(host, line, front)``: the law then reads the front from the
attributes named in :data:`HOST_ATTRS`.

The source of each equation and constant is given in ``docs/references.md``
and in section 10 of ``spec/SPEC.md``.
"""
import logging
from dataclasses import asdict, dataclass

import numpy as np

from tidewater import units as _units
from tidewater.config import (DEFAULTS, SEC_IN_DAY, SEC_IN_YEAR,
                              InvalidParamsError, InvalidWorkflowError)
from tidewater.periods import as_period, time_slice, year_mask
from tidewater.physics import (band_index, band_names, flotation_thickness,
                               front_thickness, submerged_depth)
from tidewater.units import FLUX_UNITS  # noqa: F401

log = logging.getLogger(__name__)

# What a law reads from the host objects it is called with: the cell arrays
# of the line (m), and on the host its sea level (m), its calving constant
# (s-1), its time (float years), its clock (s), its cumulative frontal
# ablation (m3) and the law it runs.
HOST_ATTRS = {'thickness': 'thick', 'surface': 'surface_h',
              'width': 'widths_m', 'sea_level': 'water_level',
              'calving_k': 'calving_k', 'year': 'yr', 'clock': 't',
              'total': 'calving_m3_since_y0', 'law': 'calving_law'}


def _default(value, key):
    return getattr(DEFAULTS, key) if value is None else value


def convert_flux(flux, from_unit='m3 s-1', to_unit='m3 yr-1', ice_density=None,
                 sec_in_year=SEC_IN_YEAR):
    """:func:`tidewater.units.convert_flux` with the defaults of the laws.

    Parameters
    ----------
    flux : float or array
        in ``from_unit``; volumes are volumes of ice
    from_unit, to_unit : str
        one of :data:`tidewater.units.FLUX_UNITS`
    ice_density : float, optional
        kg m-3, used between a volume and a mass. Default: 900.
    sec_in_year : float
        s
    """
    return _units.convert_flux(flux, from_unit, to_unit,
                               ice_density=_default(ice_density,
                                                    'ice_density'),
                               sec_in_year=sec_in_year)


def melt_share(melt, total=None, calving=None):
    """The share of frontal ablation that is submarine melt.

    Parameters
    ----------
    melt : float or array
        the submarine-melt part, in any unit
    total : float or array, optional
        the frontal ablation, in the same unit
    calving : float or array, optional
        the calving part, in place of ``total``

    Returns
    -------
    float or array
        ``melt / total``, and 0 where the total is not positive
    """
    if (total is None) == (calving is None):
        raise InvalidParamsError('melt_share needs either the total or the '
                                 'calving part')
    scalar = np.ndim(melt) == 0 and np.ndim(calving if total is None
                                            else total) == 0
    if not scalar:
        melt = np.asarray(melt, dtype=float)
    if total is None:
        total = (calving if scalar else np.asarray(calving, dtype=float)) + melt
    if scalar:
        return float(melt / total) if total > 0 else 0.
    total = np.asarray(total, dtype=float)
    return np.where(total > 0, melt / np.where(total > 0, total, 1.), 0.)


def frontal_ablation_specific_mb(flux, area, ice_density=None):
    """Frontal ablation as a specific mass balance, kg m-2 yr-1.

    Parameters
    ----------
    flux : float
        km3 yr-1 of ice
    area : float
        the glacier area, m2
    ice_density : float, optional
        kg m-3. Default: 900.
    """
    return flux * 1e9 * _default(ice_density, 'ice_density') / area


def ocean_factor(tf, tf_ref, gamma=None):
    """The dimensionless ocean factor ``(max(TF, 0) / TF_ref) ** gamma``.

    Parameters
    ----------
    tf : float or array
        thermal forcing, degC
    tf_ref : float
        reference thermal forcing, degC, > 0
    gamma : float, optional
        default: 1.18
    """
    gamma = _default(gamma, 'calving_tf_exponent')
    if np.ndim(tf) == 0:
        return (max(tf, 0.) / tf_ref) ** gamma
    return (np.clip(np.asarray(tf, dtype=float), 0., None) / tf_ref) ** gamma


def tf_scaling(tf, tf_ref, gamma=None, tf_fraction=None):
    """``k(t) / k0 = (1 - a) + a * ocean_factor``, dimensionless.

    Parameters
    ----------
    tf : float or array
        thermal forcing, degC
    tf_ref : float
        reference thermal forcing, degC, > 0
    gamma : float, optional
        default: 1.18
    tf_fraction : float, optional
        ``a``, in [0, 1]. Default: 1.
    """
    tf_fraction = _default(tf_fraction, 'calving_tf_fraction')
    s = ocean_factor(tf, tf_ref, gamma)
    if tf_fraction == 1:
        return s
    return (1 - tf_fraction) + tf_fraction * s


def tf_power_mean(tf, gamma):
    """The reference thermal forcing at which ``tf_power`` leaves k unchanged.

    The power mean ``mean(max(TF, 0)**gamma)**(1/gamma)``, i.e. the ``TF_ref``
    for which ``(TF/TF_ref)**gamma`` averages to one over ``tf``.

    Parameters
    ----------
    tf : array
        thermal forcing, degC
    gamma : float
        the exponent of the law

    Returns
    -------
    float
        the reference in degC, or nan when ``tf`` holds no finite value
    """
    tf = np.asarray(tf, dtype=float)
    tf = np.clip(tf[np.isfinite(tf)], 0., None)
    if not tf.size:
        return np.nan
    if gamma == 0:
        return float(np.mean(tf))
    return float(np.mean(tf ** gamma) ** (1. / gamma))


def inversion_k_from_ocean(k_ref, tf_mean, tf_ref=None, gamma=None,
                           tf_fraction=None):
    """The calving constant scaled by a period-mean thermal forcing, yr-1.

    ``k_ref * ((1 - a) + a * (max(tf_mean, 0) / TF_ref) ** gamma)``

    Parameters
    ----------
    k_ref : float
        the calving constant at the reference forcing, yr-1
    tf_mean : float
        the period-mean thermal forcing, degC
    tf_ref : float, optional
        degC. Default: ``tf_mean``, which leaves ``k_ref`` unchanged.
    gamma, tf_fraction : float, optional
        default: 1.18 and 1
    """
    gamma = _default(gamma, 'calving_tf_exponent')
    tf_fraction = _default(tf_fraction, 'calving_tf_fraction')
    if tf_ref is None:
        tf_ref = tf_mean
    if not np.isfinite(tf_ref) or tf_ref <= 0:
        raise InvalidParamsError(
            f'TF_ref = {tf_ref} is not usable; give a positive reference or '
            'pick a different band.')
    scale = (max(tf_mean, 0.) / tf_ref) ** gamma
    if tf_fraction != 1:
        scale = (1 - tf_fraction) + tf_fraction * scale
    return float(k_ref * scale)


def discharge_per_front_area(q_sg, water_depth, width):
    """Subglacial discharge over the submerged front, m d-1.

    Parameters
    ----------
    q_sg : float
        subglacial discharge, m3 s-1 of water
    water_depth, width : float
        of the front, m
    """
    area = max(water_depth * width, 1e-3)
    return max(q_sg, 0.) * SEC_IN_DAY / area


def _rignot(water_depth, q, tf, A, B, alpha, beta):
    return (A * max(water_depth, 0.) * q ** alpha + B) * tf ** beta


def submarine_melt_rate(water_depth, q_sg, tf, A=None, B=None, alpha=None,
                        beta=None):
    """Submarine melt rate after Rignot et al. (2016), Eq. 1, m d-1.

    ``(A * h * q**alpha + B) * TF**beta``

    Parameters
    ----------
    water_depth : float
        the water depth at the front, m
    q_sg : float
        the subglacial discharge per unit of calving-front area, m d-1
    tf : float
        the thermal forcing, degC
    A, B, alpha, beta : float, optional
        default: 3e-4, 0.15, 0.39 and 1.18

    Returns
    -------
    float
        the melt rate in m d-1, nan where the thermal forcing is not positive
    """
    A = _default(A, 'calving_melt_A')
    B = _default(B, 'calving_melt_B')
    alpha = _default(alpha, 'calving_melt_alpha')
    beta = _default(beta, 'calving_melt_beta')
    if not np.isfinite(water_depth) or not np.isfinite(tf) or tf <= 0:
        return np.nan
    q = 0. if not np.isfinite(q_sg) else max(q_sg, 0.)
    return _rignot(water_depth, q, tf, A, B, alpha, beta)


def _melt_fraction(u_calving, u_melt):
    """The melt share of a front-normal speed, bounded to [0, 1]."""
    total = u_calving + u_melt
    if total <= 0:
        return 0.
    return float(min(max(u_melt / total, 0.), 1.))


class _OceanCalvingLaw:
    """Base for time-varying calving laws driven by an ocean record.

    Sub-classes implement :meth:`frontal_speed_components`, returning
    front-normal speeds in m s-1. This class multiplies by the submerged area:
    ``k*d*h*w == (k*h) * (d*w)``. :class:`TFPower` multiplies in the order
    of the control instead, so that it reduces to it to the bit.

    Parameters
    ----------
    years : array
        the time axis of the forcing, in float years
    tf : array
        thermal forcing on that axis, degC
    open_water : array, optional
        open-water fraction on that axis, 0-1
    q_sg : array, optional
        subglacial discharge on that axis, m3 s-1
    sec_in_year : float
        s, the year of the yr-1 constants
    sea_level : float or callable, optional
        m, or a function of the time in float years, for the call by a host.
        Default: the host's own.
    """

    name = 'ocean'
    # law kwarg -> the OceanParams field it defaults to
    settings_keys = {}
    host_attrs = HOST_ATTRS
    # the attributes describe() reports
    _parameters = ()

    def __init__(self, years, tf, open_water=None, q_sg=None,
                 sec_in_year=SEC_IN_YEAR, sea_level=None):
        self._init_accounting()
        self.sea_level = sea_level
        self.years = np.asarray(years, dtype=float)
        self.tf = np.asarray(tf, dtype=float)
        self.open_water = (None if open_water is None
                           else np.asarray(open_water, dtype=float))
        self.q_sg = None if q_sg is None else np.asarray(q_sg, dtype=float)
        self.sec_in_year = sec_in_year
        self._model_k = None

    def _at(self, arr, yr):
        """Clamped linear interpolation, so a run past the forcing freezes it."""
        if arr is None:
            return None
        return float(np.interp(yr, self.years, arr, left=arr[0], right=arr[-1]))

    def _tf_at(self, yr):
        return max(self._at(self.tf, yr), 0.)  # clip before any power

    def frontal_speed(self, h, d, w, yr):
        """The total front-normal speed, m s-1."""
        return sum(self.frontal_speed_components(h, d, w, yr))

    def frontal_speed_components(self, h, d, w, yr):
        """``(calving, submarine melt)`` front-normal speeds, m s-1.

        They sum to :meth:`frontal_speed`. A law with no explicit melt term
        puts everything in the first slot.
        """
        raise NotImplementedError

    def _flux(self, h, d, w, yr):
        """The flux in m3 s-1 and its melt fraction."""
        if d <= 0 or h <= 0:
            return 0., 0.
        u_c, u_m = self.frontal_speed_components(h, d, w, yr)
        return max((u_c + u_m) * d * w, 0.), _melt_fraction(u_c, u_m)

    def flux(self, h, d, w, yr=None, k=None):
        """Frontal ablation of one front, m3 s-1 of ice.

        Parameters
        ----------
        h, d, w : float
            ice thickness, water depth and width at the front, m
        yr : float
            the time, in float years
        k : float, optional
            the calving constant in yr-1 of a law built without its own
        """
        if k is not None:
            self._model_k = k / self.sec_in_year
        return self._flux(h, d, w, yr)[0]

    def flux_per_year(self, h, d, w, yr=None, k=None):
        """:meth:`flux` in m3 yr-1 of ice."""
        return self.flux(h, d, w, yr, k=k) * self.sec_in_year

    def fluxes(self, h, d, w, yr=None, k=None):
        """:meth:`flux` over arrays of fronts or times, broadcast together."""
        args = [h, d, w, np.nan if yr is None else yr]
        if k is not None:
            args.append(k)
        cases = np.broadcast(*args)
        out = np.empty(cases.shape)
        out.flat = [self.flux(*c[:4], k=c[4] if k is not None else None)
                    for c in cases]
        return out if out.ndim else float(out)

    def describe(self):
        """The law and its parameters, as a plain dict."""
        out = {'law': self.name, 'sec_in_year': self.sec_in_year,
               'sea_level': self.sea_level}
        out.update({key: getattr(self, key) for key in self._parameters})
        yrs = getattr(self, 'years', None)
        out['record'] = (None if yrs is None or not len(yrs)
                         else (float(yrs[0]), float(yrs[-1])))
        return out

    def flux_components(self, h, d, w, yr=None, k=None):
        """``(calving, submarine melt)`` parts of :meth:`flux`, m3 s-1."""
        if k is not None:
            self._model_k = k / self.sec_in_year
        q, f = self._flux(h, d, w, yr)
        return q * (1 - f), q * f

    def _host_sea_level(self, host, yr):
        level = self.sea_level
        if level is None:
            return getattr(host, self.host_attrs['sea_level'])
        return level(yr) if callable(level) else level

    def __call__(self, host, line, front):
        """Frontal ablation at cell ``front`` of ``line``, m3 s-1.

        Parameters
        ----------
        host : the model that steps the glacier, see :data:`HOST_ATTRS`
        line : the object holding the cell arrays of the glacier
        front : int
            the cell the host takes as the calving front
        """
        names = self.host_attrs
        yr = getattr(host, names['year'])
        thick = getattr(line, names['thickness'])[front]
        depth = submerged_depth(thick, getattr(line, names['surface'])[front],
                                self._host_sea_level(host, yr))
        self._model_k = getattr(host, names['calving_k'])
        q, f = self._flux(thick, depth, getattr(line, names['width'])[front],
                          yr)
        self._account(host, line, q, f)
        return q

    def _k(self, value):
        """A calving constant in s-1, from `value` in yr-1 or from the model."""
        if value is not None:
            return value / self.sec_in_year
        if self._model_k is None:
            raise InvalidWorkflowError('no calving constant: pass one to the '
                                       'law or let the model provide '
                                       'calving_k')
        return self._model_k

    # --- the melt/calving split ------------------------------------------------
    #
    # A host integrates the one number a law returns, so the law keeps its
    # own books of the two parts. The length of a step is not known when the
    # law is called: each step is closed at the next call, from the host's
    # clock, and the last one against the host's total in components_m3.

    def _init_accounting(self):
        self.frontal_ablation_m3 = 0.
        self.submarine_melt_m3 = 0.
        self._t_prev = None
        self._pending = {}
        self._series = []

    def reset_accounting(self):
        """Start the books again, e.g. before reusing a law for a second run."""
        self._init_accounting()

    def _account(self, host, line, q, f_melt):
        t = getattr(host, self.host_attrs['clock'], None)
        if t is None:
            return  # no clock: a bare call, not a run
        if self._t_prev is None:
            self._t_prev = t
        elif t > self._t_prev:
            self._close(t - self._t_prev,
                        getattr(host, self.host_attrs['year'], None))
            self._t_prev = t
        self._pending[id(line)] = (q, f_melt)

    def _close(self, dt, yr=None):
        """Integrate the previous step, whose length is only now known."""
        for q, f in self._pending.values():
            self.frontal_ablation_m3 += q * dt
            self.submarine_melt_m3 += q * f * dt
        self._pending = {}
        if yr is None:
            return
        # one record per month bounds the memory
        if not self._series or int(yr * 12) > int(self._series[-1][0] * 12):
            self._series.append((float(yr), self.frontal_ablation_m3,
                                 self.submarine_melt_m3))

    def _pending_melt_fraction(self):
        tot = sum(q for q, _ in self._pending.values())
        if tot <= 0:
            return 0.
        return sum(q * f for q, f in self._pending.values()) / tot

    def components_m3(self, model=None):
        """``(calving, submarine melt)`` since the run started, m3.

        With a host model they sum to its cumulative frontal ablation: the
        step still open when the run stopped is closed here at its own melt
        fraction.
        """
        total, melt = self.frontal_ablation_m3, self.submarine_melt_m3
        ref = getattr(model, self.host_attrs['total'], None)
        if ref is not None and ref > total:
            melt += (ref - total) * self._pending_melt_fraction()
            total = ref
        return total - melt, melt

    def component_series(self, model=None):
        """Monthly cumulative (years, frontal ablation, submarine melt), m3."""
        rec = list(self._series)
        if model is not None:
            yr = getattr(model, self.host_attrs['year'], None)
            calving, melt = self.components_m3(model)
            if yr is not None and (not rec or yr > rec[-1][0]):
                rec.append((float(yr), calving + melt, melt))
        if not rec:
            return (np.zeros(0), np.zeros(0), np.zeros(0))
        a = np.asarray(rec, dtype=float)
        return a[:, 0], a[:, 1], a[:, 2]

    def melt_share(self, model=None):
        """:func:`melt_share` of the run so far."""
        calving, melt = self.components_m3(model)
        return melt_share(melt, calving=calving)


CalvingLaw = _OceanCalvingLaw


class ConstantK(_OceanCalvingLaw):
    """The control: a front-normal speed ``k * h`` over the submerged area.

    ``Q = k * d * h * w``, the water-depth law of Oerlemans and Nick (2005).
    Unlike its siblings it applies no test on ``d`` or ``h`` and does not
    clip at zero.

    Parameters
    ----------
    k : float, optional
        the calving constant in yr-1. Default: the host's.
    """

    name = 'constant'
    _parameters = ('k',)

    def __init__(self, years=None, tf=None, k=None, sec_in_year=SEC_IN_YEAR,
                 sea_level=None, **kwargs):
        self._init_accounting()
        self.sea_level = sea_level
        self.k = k
        self.years = None if years is None else np.asarray(years, dtype=float)
        self.tf = None if tf is None else np.asarray(tf, dtype=float)
        self.sec_in_year = sec_in_year
        self._model_k = None

    def frontal_speed_components(self, h, d, w, yr):
        return self._k(self.k) * h, 0.

    def _flux(self, h, d, w, yr):
        # constant, depth, thickness, width, multiplied left to right
        return self._k(self.k) * d * h * w, 0.


class TFPower(_OceanCalvingLaw):
    """The constant-k law with a constant scaled by the thermal forcing.

    ``k(t) = k0 * ((1 - a) + a * (max(TF(t), 0) / TF_ref) ** gamma)``

    The default ``gamma`` is the thermal-forcing exponent of the submarine
    melt rate of Rignot et al. (2016), which assumes that frontal ablation
    inherits the thermal dependence of the melt. ``a`` is the share of the
    calving constant that follows the ocean; the rest is a background term
    that calves at the freezing point too.

    Parameters
    ----------
    tf_ref : float
        the reference thermal forcing, degC. Must be positive:
        :func:`tf_power_mean` over a reference period makes the law return
        ``k0`` on average over that period.
    k0 : float, optional
        the calving constant at ``TF_ref``, yr-1. Default: the host's.
    gamma : float, optional
        default: 1.18
    tf_fraction : float, optional
        ``a``, in [0, 1]. Default: 1.
    """

    name = 'tf_power'
    settings_keys = {'gamma': 'calving_tf_exponent',
                     'tf_fraction': 'calving_tf_fraction'}
    _parameters = ('k0', 'tf_ref', 'gamma', 'tf_fraction')

    def __init__(self, *args, tf_ref=None, k0=None, gamma=None,
                 tf_fraction=None, **kwargs):
        super().__init__(*args, **kwargs)
        if tf_ref is None or not np.isfinite(tf_ref) or tf_ref <= 0:
            raise InvalidParamsError(
                f'TF_ref = {tf_ref} is not usable: the tf_power law needs a '
                'positive reference thermal forcing.')
        self.tf_ref = float(tf_ref)
        self.k0 = k0
        self.gamma = _default(gamma, 'calving_tf_exponent')
        self.tf_fraction = _default(tf_fraction, 'calving_tf_fraction')
        if not 0 <= self.tf_fraction <= 1:
            raise InvalidParamsError(
                f'tf_fraction = {self.tf_fraction} is not in [0, 1]')

    def scaling(self, yr):
        """``k(t) / k0`` at ``yr``."""
        return tf_scaling(self._tf_at(yr), self.tf_ref, self.gamma,
                          self.tf_fraction)

    def frontal_speed_components(self, h, d, w, yr):
        k = self._k(self.k0)
        return k * self.scaling(yr) * h, 0.

    def _flux(self, h, d, w, yr):
        if type(self).frontal_speed_components is not \
                TFPower.frontal_speed_components:
            return super()._flux(h, d, w, yr)
        if d <= 0 or h <= 0:
            return 0., 0.
        # the control's product in its order, so gamma = 0 or a = 0 is the
        # control to the bit
        k = self._k(self.k0) * self.scaling(yr)
        return max(k * d * h * w, 0.), 0.


class MeltPlusCalving(_OceanCalvingLaw):
    """Frontal ablation as a sum of front-normal speeds.

    ``Q_f = w * d * (k_c * h + lam * mdot)``, with ``mdot`` from Rignot et
    al. (2016) Eq. (1): ``mdot = (A * h_w * q_sg**alpha + B) * TF**beta`` in
    m d-1, ``h_w`` being the water depth at the front.

    ``mdot`` is the horizontally averaged maximum melt rate, not the
    area-averaged one, so ``lam`` scales its action on the whole submerged
    front. Without a subglacial discharge the law is ``B * TF**beta``.

    Parameters
    ----------
    k_c : float, optional
        the calving constant, yr-1. Default: the host's, in
        which case the melt term adds to a constant that may already contain
        it (see :func:`partition_law_k`).
    lam : float, optional
        default: 1
    water_depth : float, optional
        the water depth of the melt rate, m. Default: the front's own.
    A, B, alpha, beta : float, optional
        default: 3e-4, 0.15, 0.39 and 1.18
    """

    name = 'melt_calving'
    settings_keys = {'A': 'calving_melt_A', 'B': 'calving_melt_B',
                     'alpha': 'calving_melt_alpha',
                     'beta': 'calving_melt_beta',
                     'lam': 'calving_undercut_efficiency'}
    _parameters = ('k_c', 'lam', 'water_depth', 'A', 'B', 'alpha', 'beta')

    def __init__(self, *args, k_c=None, lam=None, water_depth=None, A=None,
                 B=None, alpha=None, beta=None, **kwargs):
        super().__init__(*args, **kwargs)
        for kwarg, value in (('A', A), ('B', B), ('alpha', alpha),
                             ('beta', beta), ('lam', lam)):
            setattr(self, kwarg, _default(value, self.settings_keys[kwarg]))
        self.k_c = k_c
        self.water_depth = water_depth

    def melt_rate(self, d, w, yr):
        """Rignot et al. (2016) Eq. (1), in m s-1."""
        tf = self._tf_at(yr)
        hw = d if self.water_depth is None else self.water_depth
        if self.q_sg is None:
            q = 0.
        else:
            q = discharge_per_front_area(self._at(self.q_sg, yr), hw, w)
        mdot = _rignot(hw, q, tf, self.A, self.B, self.alpha, self.beta)
        return mdot / SEC_IN_DAY

    def melt_speed(self, d, w, yr):
        """The front-normal melt speed the law applies, m s-1."""
        return self.lam * self.melt_rate(d, w, yr)

    def frontal_speed_components(self, h, d, w, yr):
        return self._k(self.k_c) * h, self.melt_speed(d, w, yr)


class SeaIceModulated(MeltPlusCalving):
    """:class:`MeltPlusCalving` with the melt term gated by open water.

    ``Q_f = w * d * (k_ice * h + f_ow(t)**delta * lam * mdot(TF))``

    The open-water fraction ``f_ow`` acts on the melt term only; the calving
    term is left as it is. ``delta = 0`` is :class:`MeltPlusCalving`.

    Parameters
    ----------
    k_ice : float, optional
        the calving constant, yr-1 (``k_c`` of :class:`MeltPlusCalving`)
    delta : float, optional
        default: 1
    """

    name = 'sea_ice'
    settings_keys = dict(MeltPlusCalving.settings_keys,
                         delta='calving_openwater_exponent')

    _parameters = MeltPlusCalving._parameters + ('delta',)

    def __init__(self, *args, k_ice=None, delta=None, **kwargs):
        if kwargs.get('k_c') is None:
            kwargs['k_c'] = k_ice
        super().__init__(*args, **kwargs)
        self.delta = _default(delta, 'calving_openwater_exponent')

    @property
    def k_ice(self):
        """The calving constant, yr-1: ``k_c`` under the name of this law."""
        return self.k_c

    def melt_speed(self, d, w, yr):
        if self.open_water is None:
            raise InvalidWorkflowError('SeaIceModulated needs an open_water '
                                       'record')
        f = max(self._at(self.open_water, yr), 0.)
        melt = self.lam * self.melt_rate(d, w, yr)
        return (f ** self.delta) * melt


class LargerOf(MeltPlusCalving):
    """The larger of the calving and the submarine-melt speed.

    ``Q_f = w * d * max(k_c * h, lam * mdot)``, after Malles et al. (2025).
    The whole flux is reported as the larger term. ``lam = 0`` is the control.
    With :func:`partition_law_k`, ``k_c`` is lowered until the period mean of
    the larger term equals the calibrated total.
    """

    name = 'larger_of'

    def frontal_speed_components(self, h, d, w, yr):
        u_c, u_m = self._k(self.k_c) * h, self.melt_speed(d, w, yr)
        return (u_c, 0.) if u_c >= u_m else (0., u_m)


LAWS = {'constant': ConstantK, 'tf_power': TFPower,
        'melt_calving': MeltPlusCalving, 'sea_ice': SeaIceModulated,
        'larger_of': LargerOf}


def register_law(name, cls, laws=None):
    """Add or replace a law class in a registry.

    Parameters
    ----------
    name : str
        the name the factories and the parameters select it by
    cls : type
        a subclass of :class:`CalvingLaw`
    laws : dict, optional
        the registry. Default: :data:`LAWS`.
    """
    if not (isinstance(cls, type) and issubclass(cls, _OceanCalvingLaw)):
        raise InvalidParamsError(f'{cls!r} is not a subclass of CalvingLaw')
    (LAWS if laws is None else laws)[str(name)] = cls
    return cls


def _law_class(name, laws=None):
    laws = LAWS if laws is None else laws
    if name not in laws:
        raise InvalidParamsError(f'unknown calving law {name!r}; '
                                 f'available: {sorted(laws)}')
    return laws[name]


def partition_calving_constant(k_total, melt_rate, thick, lam=None,
                               sec_in_year=SEC_IN_YEAR):
    """The residual calving constant that keeps the total frontal ablation.

    Solves ``k_c*h + lam*mdot == k_total*h`` for ``k_c``: both terms act on
    the same submerged area.

    Parameters
    ----------
    k_total : float
        the calibrated calving constant, yr-1
    melt_rate : float
        the reference submarine melt rate, m s-1
    thick : float
        the terminus ice thickness the constant acts on, m
    lam : float, optional
        default: 1
    sec_in_year : float
        s

    Returns
    -------
    (k_c, melt_fraction) : the residual constant in yr-1, and the melt share
        of the unchanged total
    """
    lam = _default(lam, 'calving_undercut_efficiency')
    if thick <= 0:
        raise InvalidParamsError(f'thick = {thick} is not a terminus thickness')
    u_total = k_total / sec_in_year * thick
    u_melt = lam * melt_rate
    if u_total <= 0:
        return 0., 0.
    if u_melt >= u_total:
        log.warning('submarine melt alone (%.3e m s-1) matches or exceeds the '
                    'calibrated frontal ablation speed (%.3e): the residual '
                    'calving constant is zero', u_melt, u_total)
        return 0., 1.
    k_c = (u_total - u_melt) / thick * sec_in_year
    return float(k_c), float(u_melt / u_total)


def larger_of_calving_constant(k_total, melt_speeds, thick,
                               sec_in_year=SEC_IN_YEAR):
    """The calving constant at which the larger-of law keeps the total.

    Solves ``mean(max(k_c*h, u_melt(t))) == k_total*h`` for ``k_c`` over the
    melt speeds of a reference period.

    Parameters
    ----------
    k_total : float
        the calibrated calving constant, yr-1
    melt_speeds : array
        the front-normal melt speeds of the period, m s-1
    thick : float
        the terminus ice thickness the constant acts on, m
    sec_in_year : float
        s

    Returns
    -------
    (k_c, melt_fraction) : the constant in yr-1, and the share of the total
        that the melt term carries
    """
    if thick <= 0:
        raise InvalidParamsError(f'thick = {thick} is not a terminus thickness')
    u_total = k_total / sec_in_year * thick
    u = np.sort(np.clip(np.asarray(melt_speeds, dtype=float), 0., None))
    if u_total <= 0 or u.size == 0:
        return float(max(k_total, 0.)), 0.
    if u.mean() >= u_total:
        log.warning('submarine melt alone (%.3e m s-1) matches or exceeds the '
                    'calibrated frontal ablation speed (%.3e): the calving '
                    'constant is zero', u.mean(), u_total)
        return 0., 1.
    # mean(max(c, u)) is piecewise linear in c; with j speeds below c it is
    # (j*c + sum of the others) / n, so take the first segment that holds.
    n = u.size
    above = np.concatenate([np.cumsum(u[::-1])[::-1], [0.]])
    for j in range(1, n + 1):
        c = (n * u_total - above[j]) / j
        if c <= (u[j] if j < n else np.inf):
            break
    frac = float(np.where(u > c, u, 0.).sum() / n / u_total)
    return float(c / thick * sec_in_year), frac


def partition_law_k(law, k_total, thick, depth, width, period=None):
    """Give a melt-bearing law the residual calving constant, keeping the total.

    A calving constant fitted to an observed frontal ablation already contains
    the calving that submarine melt drives, so a melt term added on top of it
    counts that melt twice. :func:`partition_calving_constant` solves
    ``k_c h + u_melt = k h`` for ``k_c`` at the given front, with the melt
    speed the law applies (its open-water gate included) averaged over
    ``period``. For :class:`LargerOf`, :func:`larger_of_calving_constant`
    solves ``mean(max(k_c h, u_melt)) = k h`` instead.

    Parameters
    ----------
    law : :class:`MeltPlusCalving`
        the law, whose ``k_c`` is set in place
    k_total : float
        the calibrated calving constant, yr-1
    thick, depth, width : float
        ice thickness, water depth and width of the front the constant was
        calibrated on, m
    period : optional
        first and last year, both included, as
        :func:`tidewater.periods.as_period` reads it. Default: (1995, 2014).

    Returns
    -------
    (k_c, melt_fraction) : the residual constant in yr-1 and the melt share of
        the total, or ``(None, 0.)`` for a law without a melt term
    """
    if not hasattr(law, 'melt_rate'):
        return None, 0.
    if None in (k_total, thick, depth, width):
        raise InvalidWorkflowError(
            'partitioning needs the calibrated constant and the thickness, '
            'water depth and width of the front it was calibrated on.')

    y0, y1 = as_period(period or DEFAULTS.ocean_tf_ref_period)
    yrs = np.asarray(law.years, dtype=float)
    sel = year_mask(yrs, (y0, y1))
    if not sel.any():
        raise InvalidWorkflowError(
            f'the ocean record does not cover {y0}-{y1}, so the '
            'reference melt rate cannot be formed. Give the law an explicit '
            'k_c, or pick a period the record covers.')
    u_melt = [law.melt_speed(depth, width, y) for y in yrs[sel]]
    if isinstance(law, LargerOf):
        k_c, frac = larger_of_calving_constant(k_total, u_melt, thick,
                                               sec_in_year=law.sec_in_year)
    else:
        k_c, frac = partition_calving_constant(k_total, float(np.mean(u_melt)),
                                               thick, lam=1.,
                                               sec_in_year=law.sec_in_year)
    law.k_c = k_c
    return k_c, frac


def calving_law_from_series(calving_law, years=None, tf=None, open_water=None,
                            q_sg=None, tf_ref=None, tf_ref_period=None,
                            laws=None, **law_kwargs):
    """Build a calving law from an ocean record.

    Parameters
    ----------
    calving_law : str
        a name in ``laws``
    years : array
        the time axis, in float years (mid-month: ``year + (month - 0.5)/12``)
    tf : array
        thermal forcing on that axis, degC
    open_water : array, optional
        open-water fraction on that axis, 0-1
    q_sg : array, optional
        subglacial discharge on that axis, m3 s-1
    tf_ref : float, optional
        the reference thermal forcing of 'tf_power', degC. Default:
        :func:`tf_power_mean` of the record over ``tf_ref_period``.
    tf_ref_period : optional
        first and last calendar year, both included, as
        :func:`tidewater.periods.as_period` reads it. Default: (1995, 2014).
    laws : dict, optional
        the registry of law classes. Default: :data:`LAWS`.
    **law_kwargs
        passed to the law (``k0``, ``gamma``, ``lam``, ``delta`` ...)
    """
    cls = _law_class(calving_law, laws)
    if issubclass(cls, ConstantK):
        return cls(**law_kwargs)
    if years is None or tf is None:
        raise InvalidWorkflowError(f'the {calving_law} law needs an ocean '
                                   'record')
    for kwarg, key in cls.settings_keys.items():
        if law_kwargs.get(kwarg) is None:
            law_kwargs[kwarg] = getattr(DEFAULTS, key)

    if issubclass(cls, TFPower):
        if tf_ref is None:
            y0, y1 = as_period(tf_ref_period or DEFAULTS.ocean_tf_ref_period)
            sel = year_mask(years, (y0, y1))
            if not sel.any():
                raise InvalidWorkflowError(
                    f'the ocean record does not cover {y0}-{y1}, so there is '
                    'no reference thermal forcing. Pass tf_ref.')
            tf_ref = tf_power_mean(np.asarray(tf)[sel], law_kwargs['gamma'])
        law_kwargs['tf_ref'] = tf_ref

    return cls(years, tf, open_water=open_water, q_sg=q_sg, **law_kwargs)


def _front_value(front, *keys):
    for key in keys:
        value = (front.get(key) if hasattr(front, 'get')
                 else getattr(front, key, None))
        if value is not None:
            return value
    return None


def ocean_calving_law(ds, calving_law=None, band=None, tf_ref=None,
                      tf_ref_period=None, partition=False,
                      partition_period=None, front=None,
                      tf_var='thermal_forcing', open_water_var='open_water_frac',
                      q_sg_var='subglacial_discharge', laws=None,
                      **law_kwargs):
    """Build a calving law for one glacier from its ocean dataset.

    Parameters
    ----------
    ds : :py:class:`xarray.Dataset`
        monthly, with ``tf_var(time, band)`` in degC and optionally
        ``open_water_var(time)`` (0-1) and ``q_sg_var(time)`` (m3 s-1)
    calving_law : str, optional
        a name in ``laws``. Default: 'constant'.
    band : str, optional
        the depth band the law reads. Default: 'shallow_0_60'.
    tf_ref, tf_ref_period :
        see :func:`calving_law_from_series`
    partition : bool
        for a melt-bearing law, set ``k_c`` from :func:`partition_law_k`
    partition_period : optional
        see :func:`partition_law_k`. Default: (1995, 2014).
    front : :class:`FrontFlux` or mapping, optional
        needed with ``partition``: ``calving_k`` (yr-1) and ``thickness``,
        ``water_depth``, ``width`` (m). A mapping may also hold the last
        three as ``calving_front_thick``, ``terminus_water_depth`` and
        ``calving_front_width``.
    tf_var, open_water_var, q_sg_var : str
        the variable names in ``ds``
    laws : dict, optional
        the registry of law classes. Default: :data:`LAWS`.
    **law_kwargs
        passed to the law

    Returns
    -------
    the law; after a partition it carries ``k_c`` and
    ``melt_fraction_reference``
    """
    calving_law = calving_law or DEFAULTS.calving_law
    cls = _law_class(calving_law, laws)
    if issubclass(cls, ConstantK):
        return cls(**law_kwargs)
    if ds is None:
        raise InvalidWorkflowError('no ocean dataset')
    if 'band' not in ds.dims:
        raise InvalidWorkflowError('the ocean dataset has no band dimension')

    band = band or DEFAULTS.ocean_tf_band
    tf = ds[tf_var].values[:, band_index(band_names(ds), band)]
    yrs = ds['time.year'].values + (ds['time.month'].values - 0.5) / 12
    law = calving_law_from_series(
        calving_law, yrs, tf,
        open_water=ds[open_water_var].values if open_water_var in ds else None,
        q_sg=ds[q_sg_var].values if q_sg_var in ds else None,
        tf_ref=tf_ref, tf_ref_period=tf_ref_period, laws=laws, **law_kwargs)
    if partition:
        front = {} if front is None else front
        k_c, frac = partition_law_k(
            law, _front_value(front, 'calving_k'),
            _front_value(front, 'thickness', 'calving_front_thick'),
            _front_value(front, 'water_depth', 'terminus_water_depth'),
            _front_value(front, 'width', 'calving_front_width'),
            period=partition_period)
        if k_c is not None:
            law.melt_fraction_reference = float(frac)
    return law


def ocean_tf_mean(ds, band=None, period=None, gamma=None,
                  tf_var='thermal_forcing'):
    """Period-mean thermal forcing of one glacier in one depth band, degC.

    Parameters
    ----------
    ds : :py:class:`xarray.Dataset`
        with ``tf_var(time, band)`` in degC
    band : str, optional
        default: 'shallow_0_60'
    period : optional
        first and last year, both included, as
        :func:`tidewater.periods.as_period` reads it. Default: (1995, 2014).
    gamma : float, optional
        return :func:`tf_power_mean` at this exponent instead of the
        arithmetic mean
    tf_var : str
        the variable name in ``ds``
    """
    band = band or DEFAULTS.ocean_tf_band
    period = period or DEFAULTS.ocean_tf_ref_period
    tf = ds[tf_var].isel(band=band_index(band_names(ds), band))
    tf = tf.sel(time=time_slice(period))
    if gamma is not None:
        return tf_power_mean(tf.values, gamma)
    return float(tf.mean('time'))


@dataclass(frozen=True)
class FrontFlux:
    """The flux of the constant-k law at one front, with its geometry.

    Attributes
    ----------
    flux : float
        km3 yr-1 of ice, never negative
    width, thickness, water_depth, freeboard : float
        of the front, m
    flotation_thickness : float
        the thickness at which a front in this water depth floats, m
    calving_k : float
        yr-1
    sea_level : float
        m
    """

    flux: float
    width: float
    thickness: float
    water_depth: float
    freeboard: float
    flotation_thickness: float
    calving_k: float
    sea_level: float = 0.

    def __getitem__(self, field):
        return asdict(self)[field]

    def as_dict(self, names=None):
        """The fields as a dict, renamed by ``names`` (field -> host key)."""
        names = names or {}
        return {names.get(key, key): value
                for key, value in asdict(self).items()}


def calving_law_flux(water_depth, k, thick=None, width=None, surface_h=None,
                     rho_ocean=None, rho_ice=None, sea_level=0.):
    """The flux of the constant-k law at a prescribed water depth.

    Parameters
    ----------
    water_depth : float
        m
    k : float
        yr-1
    thick : float, optional
        the terminus thickness, m. Default: freeboard plus depth, from
        ``surface_h`` and ``sea_level``.
    width : float
        the width of the front, m
    surface_h : float, optional
        surface elevation of the terminus, m
    rho_ocean, rho_ice : float, optional
        kg m-3, for the flotation thickness. Default: 1028 and 900.
    sea_level : float
        m, in the datum of ``surface_h``

    Returns
    -------
    :class:`FrontFlux`
    """
    if width is None:
        raise InvalidParamsError('the flux at a front needs its width')
    if thick is None:
        if surface_h is None:
            raise InvalidParamsError('the flux at a front needs its thickness '
                                     'or its surface elevation')
        thick = front_thickness(surface_h, water_depth, sea_level)
    m3_per_year = k * thick * water_depth * width
    return FrontFlux(
        flux=max(m3_per_year / 1e9, 0.), width=width, thickness=thick,
        water_depth=water_depth, freeboard=thick - water_depth,
        flotation_thickness=flotation_thickness(water_depth,
                                                rho_ocean=rho_ocean,
                                                rho_ice=rho_ice),
        calving_k=k, sea_level=sea_level)


def frontal_ablation_components(model):
    """``(calving, submarine melt)`` of a finished run, m3, from its model."""
    law = getattr(model, HOST_ATTRS['law'], None)
    if not hasattr(law, 'components_m3'):
        # a law without books: nothing attributable to melt
        return float(getattr(model, HOST_ATTRS['total'], 0.)), 0.
    return law.components_m3(model)


def frontal_component_series(law, time, calving_m3, model=None):
    """Split a run's cumulative frontal ablation into melt and calving.

    Parameters
    ----------
    law : the calving law the run used
    time : array
        the time axis of ``calving_m3``, in float years
    calving_m3 : array
        cumulative frontal ablation of the run, m3
    model : the model of the run, optional
        closes the last open step of the split

    Returns
    -------
    (submarine_melt_m3, calving_only_m3, melt_fraction) : two cumulative
        series in m3 that sum to ``calving_m3``, and the melt share of the
        run's total
    """
    calving, melt = law.components_m3(model)
    frac = melt_share(melt, calving=calving)

    cum = np.asarray(calving_m3)
    yrs, cum_total, cum_melt = law.component_series(model)
    if len(yrs) == 0:
        f_t = np.full(cum.shape, frac)
    else:
        f = melt_share(cum_melt, cum_total)
        f_t = np.interp(np.asarray(time).astype(float), yrs, f,
                        left=f[0], right=f[-1])
    return cum * f_t, cum * (1 - f_t), frac
