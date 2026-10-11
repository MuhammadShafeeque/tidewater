"""The calving constant between tidewater's calibration and PyGEM's.

PyGEM calibrates one ``calving_k`` (yr-1) per glacier so that its dynamical
model returns the observed frontal ablation (``fa_gta_obs``, Gt yr-1) as a
mean over ``ref_startyear`` to ``ref_endyear``, and writes ``RGIId,
calving_k, calving_k_nmad`` for its simulations. The constant and its unit
are those of tidewater, so the two calibrations exchange tables:
:func:`read_targets` reads PyGEM's observations,
:func:`run_flux` drives
:func:`tidewater.calibration.dynamic.calibrate_dynamic_k` through PyGEM's
models, :func:`write_calibration` writes the file
``run_simulation`` reads, and :func:`law_from_calibration` turns a
calibrated constant into an ocean-forced law.
"""
import os

import numpy as np

from tidewater.calibration import targets as _targets
from tidewater.config import DEFAULTS, InvalidParamsError, InvalidWorkflowError
from tidewater.export import read_calibration_csv, write_calibration_csv
from tidewater.laws import LAWS, ConstantK, TFPower, ocean_calving_law
from tidewater.periods import as_period, run_bounds
from tidewater.units import GT_PER_YR, convert_flux

# PyGEM's default ref_startyear and ref_endyear, both included
REF_PERIOD = (2000, 2019)

WATER_DENSITY = 1000.

# tidewater target column -> the column of PyGEM's merged observations
TARGET_COLUMNS = {'glacier_id': 'RGIId', 'flux': 'fa_gta_obs',
                  'flux_err': 'fa_gta_obs_unc'}


def read_targets(source, **kwargs):
    """PyGEM's observed frontal ablation as a tidewater target table, Gt yr-1.

    Parameters
    ----------
    source : path or :py:class:`pandas.DataFrame`
        PyGEM's merged frontal-ablation data or its calibration output, with
        ``RGIId``, ``fa_gta_obs`` and optionally ``fa_gta_obs_unc``,
        ``start_date``, ``end_date``, ``source``
    **kwargs
        passed to :func:`tidewater.calibration.targets.read_targets`
    """
    kwargs.setdefault('flux_unit', GT_PER_YR)
    return _targets.read_targets(source, columns=dict(TARGET_COLUMNS),
                                 **kwargs)


def calibration_file(prms):
    """The calibration file ``run_simulation`` reads, from PyGEM's settings."""
    fa = prms['calib']['data']['frontalablation']
    return os.path.join(f"{prms['root']}/{fa['frontalablation_relpath']}",
                        'analysis', fa['frontalablation_cal_fn'])


def read_calibration(path):
    """A PyGEM calibration file with tidewater's column names."""
    return read_calibration_csv(path, columns='pygem')


def write_calibration(table, path):
    """Write calibrated constants as ``RGIId, calving_k, calving_k_nmad``.

    ``table`` is a tidewater calibration table (``calving_k`` and
    ``calving_k_unc`` in yr-1, by glacier id); further columns are kept.
    PyGEM samples ``calving_k_nmad`` as the standard deviation of ``k``.
    """
    return write_calibration_csv(table, path, columns='pygem')


def _redistributes(model):
    from pygem.glacierdynamics import MassRedistributionCurveModel
    return isinstance(model, MassRedistributionCurveModel)


def run_flux(build_model, period=REF_PERIOD, flux_unit=GT_PER_YR,
             ice_density=None, as_pygem=False):
    """The ``run_flux`` of ``calibrate_dynamic_k``, through PyGEM's models.

    Parameters
    ----------
    build_model : callable
        ``build_model(rgi_id, calving_k)`` returns the dynamical model of one
        glacier at the start of ``period``, with ``calving_k`` in yr-1, or
        None where the glacier cannot run
    period : (int, int)
        first and last year, both included
    flux_unit : str
        the unit of the targets
    ice_density : float, optional
        kg m-3. Default: 900, PyGEM's ``density_ice``.
    as_pygem : bool
        multiply the flux of a ``FluxBasedModel`` by ``ice_density / 1000``,
        as PyGEM's frontal-ablation calibration reports it, to compare with
        the constants that calibration finds

    Returns
    -------
    callable
        dict glacier id -> ``calving_k`` to dict glacier id -> the mean
        frontal ablation over ``period`` in ``flux_unit``
    """
    t0, t1 = run_bounds(as_period(period))
    rho = DEFAULTS.ice_density if ice_density is None else ice_density

    def run(ks):
        out = {}
        for gid, k in ks.items():
            model = build_model(gid, k)
            if model is None:
                out[gid] = np.nan
                continue
            if model.yr != t0:
                raise InvalidWorkflowError(
                    f'{gid}: the model starts at {model.yr}, not {t0}')
            model.run_until(t1)
            q = convert_flux(model.calving_m3_since_y0 / (t1 - t0), 'm3 yr-1',
                             flux_unit, ice_density=rho)
            if as_pygem and not _redistributes(model):
                q *= rho / WATER_DENSITY
            out[gid] = float(q)
        return out
    return run


def law_from_calibration(ds, calving_k, calving_law='tf_power',
                         period=REF_PERIOD, front=None, **law_kwargs):
    """An ocean-forced law that keeps PyGEM's calibrated constant.

    ``tf_power`` takes ``k0 = calving_k`` at the power mean of the record
    over ``period`` as ``TF_ref``, so its constant averages to ``calving_k``
    there. A melt-bearing law takes the residual calving constant that keeps
    the frontal ablation of ``front`` over ``period``
    (:func:`tidewater.laws.partition_law_k`).

    Parameters
    ----------
    ds : :py:class:`xarray.Dataset`
        the glacier's ocean forcing (:mod:`tidewater.forcing.io`)
    calving_k : float
        yr-1
    calving_law : str
        a name in :data:`tidewater.laws.LAWS`
    period : (int, int)
        the calibration period, both years included
    front : mapping, optional
        ``thickness``, ``water_depth`` and ``width`` (m) of the front the
        constant was calibrated on
        (:func:`tidewater.hosts.pygem.front_geometry`); needed by the
        melt-bearing laws
    **law_kwargs
        passed to :func:`tidewater.laws.ocean_calving_law`
    """
    cls = LAWS.get(calving_law)
    if cls is None:
        raise InvalidParamsError(f'unknown calving law {calving_law!r}; '
                                 f'available: {sorted(LAWS)}')
    if issubclass(cls, ConstantK):
        return cls(k=calving_k, **law_kwargs)
    if issubclass(cls, TFPower):
        return ocean_calving_law(ds, calving_law, tf_ref_period=period,
                                 k0=calving_k, **law_kwargs)
    if front is None:
        raise InvalidWorkflowError(
            f'the {calving_law} law needs the front the constant was '
            'calibrated on, to keep its frontal ablation')
    return ocean_calving_law(ds, calving_law, partition=True,
                             partition_period=period,
                             front=dict(front, calving_k=calving_k),
                             **law_kwargs)
