"""The inversion side of ocean-forced frontal ablation.

The inversion takes one calving constant per glacier, which this module can
scale by the period-mean thermal forcing
(:func:`set_inversion_k_from_ocean`), fit to observed frontal ablation
(:func:`fit_calving_k`, :func:`fit_calving_k_model`) and use at a water depth
taken from measured bathymetry instead of solved for
(:func:`terminus_water_depth_from_bed`,
:func:`find_inversion_calving_from_bathymetry`).

On the bathymetry route the shape of the calving law (free board, water depth,
width) does not depend on the climate, so ``k = observed / shape`` is exact
before the mass balance is calibrated, and the frontal flux can be passed to
the mass-balance calibration through ``gdir.inversion_calving_rate``.
"""
import logging

import numpy as np
import xarray as xr
from oggm import cfg
from oggm import entity_task
from oggm.exceptions import InvalidParamsError, InvalidWorkflowError
from oggm.utils import DisableLogger

from tidewater import laws as _laws
from tidewater import physics as _physics
from tidewater.calibration import covariates as _covariates
from tidewater.calibration import static as _static
from tidewater.calibration.covariates import _design  # noqa: F401
from tidewater.calibration.covariates import calving_k_from_model  # noqa: F401
from tidewater.hosts.oggm.errors import oggm_errors, translated
from tidewater.hosts.oggm.forcing import (  # noqa: F401
    subglacial_discharge_from_mb, write_subglacial_discharge)
from tidewater.hosts.oggm.params import param, setting as _setting

log = logging.getLogger(__name__)

# The mask of BedMachine: 0 ocean, 1 ice-free land, 2 grounded ice, 3 floating
BEDMACHINE_OCEAN, BEDMACHINE_FLOATING = 0, 3


def ocean_tf_mean(gdir, band=None, period=None, ocean_filesuffix='',
                  gamma=None):
    """Period-mean thermal forcing of one glacier in one depth band.

    Parameters
    ----------
    gdir : :py:class:`oggm.GlacierDirectory`
        the glacier directory to process
    band : str, optional
        default: ``gdir.settings['ocean_tf_band']``
    period : tuple of two years, optional
        default: ``gdir.settings['ocean_tf_ref_period']``
    ocean_filesuffix : str
        the filesuffix of the ``ocean_data`` file
    gamma : float, optional
        return :func:`tidewater.hosts.oggm.laws.tf_power_mean` at this exponent
        instead of the arithmetic mean

    Returns
    -------
    float
        the mean, degC
    """
    if not gdir.has_file('ocean_data', filesuffix=ocean_filesuffix):
        raise InvalidWorkflowError(f'({gdir.rgi_id}) no ocean_data file; run '
                                   'process_ocean_data first.')
    band = band or gdir.settings['ocean_tf_band']
    period = period or gdir.settings['ocean_tf_ref_period']

    fp = gdir.get_filepath('ocean_data', filesuffix=ocean_filesuffix)
    with xr.open_dataset(fp) as ds:
        ds = ds.load()
    with oggm_errors(gdir):
        return _laws.ocean_tf_mean(ds, band=band, period=period, gamma=gamma)


@entity_task(log)
def set_inversion_k_from_ocean(gdir, k_ref=None, gamma=None, band=None,
                               period=None, tf_ref=None, ocean_filesuffix='',
                               tf_fraction=None):
    """Scale the inversion's calving constant by the period-mean thermal forcing.

    ``k_inv = k_ref * ((1 - a) + a * (mean(TF, period) / TF_ref) ** gamma)``
    is written to the glacier's settings as ``inversion_calving_k``.

    Parameters
    ----------
    gdir : :py:class:`oggm.GlacierDirectory`
        the glacier directory to process
    k_ref : float, optional
        the calving constant at the reference forcing, yr-1. Default: the
        glacier's current ``inversion_calving_k``.
    gamma : float, optional
        default: ``gdir.settings['calving_tf_exponent']``
    band : str, optional
        default: ``gdir.settings['ocean_tf_band']``
    period : tuple of two years, optional
        the period whose mean forcing scales k. Default:
        ``gdir.settings['ocean_tf_ref_period']``.
    tf_ref : float, optional
        the reference forcing, degC. Default:
        ``gdir.settings['ocean_tf_ref']``, else the period mean itself, which
        leaves ``k_ref`` unchanged.
    ocean_filesuffix : str
        the filesuffix of the ``ocean_data`` file
    tf_fraction : float, optional
        ``a``, the share of the constant that follows the ocean. Default:
        ``gdir.settings['calving_tf_fraction']``.

    Returns
    -------
    float
        the scaled constant, or None for a glacier that is not tidewater
    """
    if not gdir.is_tidewater:
        log.warning('(%s) not tidewater, inversion k untouched', gdir.rgi_id)
        return None

    band = band or gdir.settings['ocean_tf_band']
    if gamma is None:
        gamma = gdir.settings['calving_tf_exponent']
    k_ref = gdir.settings['inversion_calving_k'] if k_ref is None else k_ref

    tf_mean = ocean_tf_mean(gdir, band=band, period=period,
                            ocean_filesuffix=ocean_filesuffix)
    if tf_ref is None:
        tf_ref = gdir.settings['ocean_tf_ref']
    if tf_ref is None:
        tf_ref = tf_mean
    if tf_fraction is None:
        tf_fraction = gdir.settings['calving_tf_fraction']
    with oggm_errors(gdir):
        k_inv = _laws.inversion_k_from_ocean(k_ref, tf_mean, tf_ref=tf_ref,
                                             gamma=gamma,
                                             tf_fraction=tf_fraction)
    gdir.settings['inversion_calving_k'] = k_inv
    gdir.add_to_diagnostics('ocean_inversion_k', k_inv)
    gdir.add_to_diagnostics('ocean_inversion_tf_mean', tf_mean)
    gdir.add_to_diagnostics('ocean_inversion_tf_ref', float(tf_ref))
    gdir.add_to_diagnostics('ocean_inversion_band', band)
    return k_inv


@entity_task(log)
def calving_vs_geodetic_residual(gdir, ref_mb=None, ref_mb_err=None):
    """How much of the geodetic mass change the inverted frontal ablation is.

    Parameters
    ----------
    gdir : :py:class:`oggm.GlacierDirectory`
        the glacier directory to process
    ref_mb : float, optional
        the geodetic mass balance, kg m-2 yr-1
    ref_mb_err : float, optional
        its uncertainty, same units

    Returns
    -------
    dict
        the frontal ablation as a specific mass balance and its ratio to
        ``ref_mb`` and ``ref_mb_err``, also written to the diagnostics
    """
    rate = gdir.inversion_calving_rate  # km3 yr-1
    cmb = _laws.frontal_ablation_specific_mb(
        rate, gdir.rgi_area_m2, ice_density=gdir.settings['ice_density'])

    out = {'calving_specific_mb': cmb, 'calving_rate_km3_per_a': rate}
    if ref_mb is not None:
        out['calving_over_geodetic'] = (cmb / abs(ref_mb) if ref_mb else np.inf)
        if ref_mb_err:
            out['calving_over_geodetic_err'] = cmb / abs(ref_mb_err)
        if cmb > abs(ref_mb):
            log.warning('(%s) frontal ablation %.1f kg m-2 yr-1 exceeds the '
                        'geodetic mass balance %.1f', gdir.rgi_id, cmb, ref_mb)
            out['calving_exceeds_geodetic'] = True
    for k, v in out.items():
        gdir.add_to_diagnostics(k, v if isinstance(v, bool) else float(v))
    return out


@translated
def partition_calving_constant(k_total, melt_rate, thick, lam=None):
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
        default: ``cfg.PARAMS['calving_undercut_efficiency']``

    Returns
    -------
    (k_c, melt_fraction) : the residual constant in yr-1, and the melt share
        of the unchanged total
    """
    return _laws.partition_calving_constant(
        k_total, melt_rate, thick,
        lam=param('calving_undercut_efficiency', lam),
        sec_in_year=cfg.SEC_IN_YEAR)


@translated
def larger_of_calving_constant(k_total, melt_speeds, thick):
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

    Returns
    -------
    (k_c, melt_fraction) : the constant in yr-1, and the share of the total
        that the melt term carries
    """
    return _laws.larger_of_calving_constant(k_total, melt_speeds, thick,
                                            sec_in_year=cfg.SEC_IN_YEAR)


def ocean_cells(ds, mask, bed_var='bedmachine_bed'):
    """Cells outside the glacier that are ocean.

    Parameters
    ----------
    ds : :py:class:`xarray.Dataset`
        the ``gridded_data`` of the glacier
    mask : array of bool
        the glacier mask
    bed_var : str
        the variable holding the bed

    Returns
    -------
    (cells, source) : a boolean array, and 'mask' when ``bedmachine_mask``
        decided (ocean and floating ice) or 'bed_below_sea_level' when the
        file has no such mask and every bed below sea level outside the
        glacier counts, the bed under neighbouring ice included
    """
    surface_class = (ds['bedmachine_mask'].values if 'bedmachine_mask' in ds
                     else None)
    return _physics.ocean_cells(
        ds[bed_var].values, mask, surface_class=surface_class,
        ocean_classes=(BEDMACHINE_OCEAN, BEDMACHINE_FLOATING))


@entity_task(log)
def terminus_water_depth_from_bed(gdir, bed_var='bedmachine_bed',
                                  dilate=2, min_pixels=5, fallback_trim=None):
    """Water depth at the calving front from a measured bed.

    The median of the bed over the ocean cells within ``dilate`` cells of the
    glacier mask. When fewer than ``min_pixels`` such cells exist, the median
    is taken over every ocean cell of the map instead. Writes
    ``terminus_water_depth``, ``terminus_water_depth_n``,
    ``terminus_water_depth_source`` ('ring' or 'fallback') and
    ``terminus_water_depth_ocean_source`` to the settings.

    Parameters
    ----------
    gdir : :py:class:`oggm.GlacierDirectory`
        the glacier directory to process
    bed_var : str
        the ``gridded_data`` variable holding the bed, e.g. from
        :func:`tidewater.hosts.oggm.shop_bed.bedmachine_bed_to_gdir`
    dilate : int
        the width of the ring around the glacier mask, in grid cells
    min_pixels : int
        the smallest number of cells a median is taken over
    fallback_trim : int, optional
        cells cut from each edge of the map before the fallback searches it.
        Default: ``gdir.settings['terminus_depth_fallback_trim']``, else 0.

    Returns
    -------
    float
        the depth in m, positive downwards, or None for a glacier that is
        not tidewater
    """
    if not gdir.is_tidewater:
        return None
    if fallback_trim is None:
        fallback_trim = int(_setting(gdir, 'terminus_depth_fallback_trim', 0))

    with xr.open_dataset(gdir.get_filepath('gridded_data')) as ds:
        if bed_var not in ds:
            raise InvalidWorkflowError(
                f'({gdir.rgi_id}) no {bed_var} in gridded_data; run '
                'bedmachine_bed_to_gdir first.')
        bed = ds[bed_var].values
        mask = ds['glacier_mask'].values.astype(bool)
        ocean, ocean_source = ocean_cells(ds, mask, bed_var=bed_var)

    with oggm_errors(gdir):
        depth, n, source = _physics.terminus_water_depth_from_bed(
            bed, mask, ocean=ocean, dilate=dilate, min_pixels=min_pixels,
            fallback_trim=fallback_trim)
    gdir.settings['terminus_water_depth'] = depth
    gdir.settings['terminus_water_depth_n'] = n
    gdir.settings['terminus_water_depth_source'] = source
    gdir.settings['terminus_water_depth_ocean_source'] = ocean_source
    return depth


def flotation_thickness(water_depth, rho_ocean=None, rho_ice=None):
    """Ice thickness at which a front in this water depth floats, m."""
    return _physics.flotation_thickness(
        water_depth, rho_ocean=param('ocean_water_density', rho_ocean),
        rho_ice=param('ice_density', rho_ice))


def front_thickness(gdir, water_depth, input_filesuffix=''):
    """Terminus thickness at a prescribed water depth: free board plus depth.

    The free board is the surface elevation of the last cell of the inversion
    flowline, so this is the thickness
    :func:`tidewater.hosts.oggm.bed.bedmachine_terminus_bed` gives the
    front of the run.
    """
    fl = gdir.read_store('inversion_flowlines',
                          filesuffix=input_filesuffix)[-1]
    return _physics.front_thickness(fl.surface_h[-1], water_depth)


def calving_law_flux(gdir, water_depth=None, k=None, thick=None,
                     input_filesuffix=''):
    """The flux of the calving law at a prescribed water depth.

    ``k * thick * water_depth * width``. Unlike
    :func:`oggm.core.inversion.calving_flux_from_depth`, the depth is given
    and the thickness follows from it (:func:`front_thickness`).

    Parameters
    ----------
    gdir : :py:class:`oggm.GlacierDirectory`
        the glacier directory to process
    water_depth : float, optional
        m. Default: ``gdir.settings['terminus_water_depth']``.
    k : float, optional
        yr-1. Default: ``gdir.settings['inversion_calving_k']``.
    thick : float, optional
        the terminus thickness, m. Default: :func:`front_thickness`.
    input_filesuffix : str
        the filesuffix of the inversion flowlines

    Returns
    -------
    dict
        the flux in km3 yr-1 and the geometry it was computed with
    """
    if water_depth is None:
        water_depth = gdir.settings['terminus_water_depth']
    if k is None:
        k = gdir.settings['inversion_calving_k']
    fl = gdir.read_store('inversion_flowlines',
                          filesuffix=input_filesuffix)[-1]
    with oggm_errors(gdir):
        return _laws.calving_law_flux(
            water_depth, k, thick=thick, width=fl.widths[-1] * gdir.grid.dx,
            surface_h=fl.surface_h[-1],
            rho_ocean=cfg.PARAMS['ocean_water_density'],
            rho_ice=cfg.PARAMS['ice_density']).as_dict(_FRONT_KEYS)


_FRONT_KEYS = {'thickness': 'thick', 'freeboard': 'free_board',
               'flotation_thickness': 'thick_flotation',
               'calving_k': 'inversion_calving_k'}


@entity_task(log, writes=['diagnostics'])
def find_inversion_calving_from_bathymetry(gdir, settings_filesuffix='',
                                           input_filesuffix='',
                                           output_filesuffix='',
                                           water_depth=None, k=None,
                                           mb_model=None, mb_years=None,
                                           glen_a=None, fs=None):
    """Calving inversion with the water depth prescribed instead of solved for.

    The counterpart of
    :func:`oggm.core.inversion.find_inversion_calving_from_any_mb` when the
    water depth at the front is known: the flux follows from the depth and the
    calving constant (:func:`calving_law_flux`) and the thickness inversion is
    run against it. Selected in :func:`oggm.workflow.inversion_tasks` by
    ``PARAMS['inversion_calving_from_bathymetry']``.

    Parameters
    ----------
    gdir : :py:class:`oggm.GlacierDirectory`
        the glacier directory to process
    settings_filesuffix : str
        You can use a different set of settings by providing a filesuffix.
        This is useful for sensitivity experiments. Code-wise the
        settings_filesuffix is set in the @entity-task decorater.
    input_filesuffix : str
        the filesuffix used for the inversion flowlines to start
    output_filesuffix : str
        the filesuffix used for all outputs
    water_depth : float, optional
        m. Default: ``gdir.settings['terminus_water_depth']``, as written by
        :func:`terminus_water_depth_from_bed`.
    k : float, optional
        yr-1. Default: ``gdir.settings['inversion_calving_k']``.
    mb_model : :py:class:`oggm.core.massbalance.MassBalanceModel`, optional
        the mass balance model to use
    mb_years : array, optional
        the years of the apparent mass balance
    glen_a : float, optional
        default: ``gdir.settings['inversion_glen_a']``
    fs : float, optional
        default: ``gdir.settings['inversion_fs']``

    Returns
    -------
    dict
        the calving flux and the front geometry, as the stock task returns
        them, or None for a glacier that is not tidewater
    """
    from oggm.core import massbalance
    from oggm.core.inversion import (prepare_for_inversion,
                                     mass_conservation_inversion)

    if not gdir.is_tidewater:
        return None

    if water_depth is None:
        water_depth = _setting(gdir, 'terminus_water_depth')
    if water_depth is None:
        raise InvalidWorkflowError(
            f'({gdir.rgi_id}) no terminus_water_depth; run '
            'terminus_water_depth_from_bed first, or pass one.')

    # Volume without calving, for the statistic the stock task records
    gdir.inversion_calving_rate = 0
    with DisableLogger():
        massbalance.apparent_mb_from_any_mb(
            gdir, settings_filesuffix=settings_filesuffix,
            input_filesuffix=input_filesuffix,
            output_filesuffix=output_filesuffix,
            mb_model=mb_model, mb_years=mb_years)
        prepare_for_inversion(gdir, settings_filesuffix=settings_filesuffix,
                              input_filesuffix=input_filesuffix,
                              output_filesuffix=output_filesuffix)
        v_ref = mass_conservation_inversion(
            gdir, settings_filesuffix=settings_filesuffix,
            input_filesuffix=output_filesuffix,
            output_filesuffix=output_filesuffix, glen_a=glen_a, fs=fs)

    out = calving_law_flux(gdir, water_depth=water_depth, k=k,
                           input_filesuffix=output_filesuffix)
    gdir.inversion_calving_rate = out['flux']

    with DisableLogger():
        massbalance.apparent_mb_from_any_mb(
            gdir, settings_filesuffix=settings_filesuffix,
            input_filesuffix=input_filesuffix,
            output_filesuffix=output_filesuffix,
            mb_model=mb_model, mb_years=mb_years)
        prepare_for_inversion(gdir, settings_filesuffix=settings_filesuffix,
                              input_filesuffix=output_filesuffix,
                              output_filesuffix=output_filesuffix)
        mass_conservation_inversion(
            gdir, settings_filesuffix=settings_filesuffix,
            input_filesuffix=output_filesuffix,
            output_filesuffix=output_filesuffix,
            water_level=0., glen_a=glen_a, fs=fs)

    fl = gdir.read_store('inversion_flowlines',
                          filesuffix=output_filesuffix)[-1]
    f_calving = (fl.flux[-1] * (gdir.grid.dx ** 2) * 1e-9
                 / gdir.settings['ice_density'])

    odf = {'volume_before_calving': v_ref,
           'calving_flux': f_calving,
           'calving_rate_myr': f_calving * 1e9 / (out['thick'] * out['width']),
           'calving_law_flux': out['flux'],
           'calving_water_level': 0.,
           'calving_inversion_k': out['inversion_calving_k'],
           'calving_front_water_depth': out['water_depth'],
           'calving_front_free_board': out['free_board'],
           'calving_front_thick': out['thick'],
           'calving_front_thick_flotation': out['thick_flotation'],
           'calving_front_width': out['width'],
           'calving_depth_source': 'bathymetry'}
    for key, val in odf.items():
        gdir.settings[key] = val
        # as the stock task does
        if settings_filesuffix == '':
            gdir.add_to_diagnostics(key, val)
    return odf


def find_inversion_calving(gdir, settings_filesuffix='', **kwargs):
    """The calving inversion ``inversion_calving_from_bathymetry`` selects.

    :func:`find_inversion_calving_from_bathymetry` when it is set, else
    :func:`oggm.core.inversion.find_inversion_calving_from_any_mb`; the task
    :func:`tidewater.hosts.oggm.register` gives ``oggm.workflow.inversion_tasks``.
    """
    from oggm.core.inversion import find_inversion_calving_from_any_mb

    gdir.settings_filesuffix = settings_filesuffix
    task = find_inversion_calving_from_any_mb
    if gdir.settings['inversion_calving_from_bathymetry']:
        task = find_inversion_calving_from_bathymetry
    return task(gdir, settings_filesuffix=settings_filesuffix, **kwargs)


def fit_calving_k(gdirs, observed_flux, water_depth=None, input_filesuffix=''):
    """Pooled calving constant matching the summed observed frontal ablation.

    Parameters
    ----------
    gdirs : list of :py:class:`oggm.GlacierDirectory` objects
        the glacier directories to process
    observed_flux : dict
        rgi_id -> observed frontal ablation in km3 yr-1 of ice. Glaciers
        absent from it take no part in the fit.
    water_depth : float, optional
        m, for every glacier. Default: each glacier's
        ``terminus_water_depth``.
    input_filesuffix : str
        the filesuffix of the inversion flowlines

    Returns
    -------
    (k, k_per_glacier) : the pooled constant in yr-1, and a dict of the
        constants that reproduce each glacier's own observation
    """
    shape = {}
    for gdir in gdirs:
        if observed_flux.get(gdir.rgi_id) is None or not gdir.is_tidewater:
            continue
        d = water_depth
        if d is None:
            d = _setting(gdir, 'terminus_water_depth')
        shape[gdir.rgi_id] = calving_law_flux(
            gdir, water_depth=d, k=1.,
            input_filesuffix=input_filesuffix)['flux']
    with oggm_errors():
        return _static.fit_calving_k(shape, observed_flux)


def submarine_melt_rate(water_depth, q_sg, tf, A=None, B=None, alpha=None,
                        beta=None):
    """Submarine melt rate after Rignot et al. (2016), Eq. 1.

    ``(A * h * q**alpha + B) * TF**beta``, the scalar twin of
    :meth:`~tidewater.hosts.oggm.laws.MeltPlusCalving.melt_rate`.

    Parameters
    ----------
    water_depth : float
        the water depth at the front, m
    q_sg : float
        the subglacial discharge per unit of calving-front area, m d-1
    tf : float
        the thermal forcing, degC
    A, B, alpha, beta : float, optional
        default: ``cfg.PARAMS['calving_melt_A']`` and so on

    Returns
    -------
    float
        the melt rate in m d-1, nan where the thermal forcing is not positive
    """
    return _laws.submarine_melt_rate(
        water_depth, q_sg, tf, A=param('calving_melt_A', A),
        B=param('calving_melt_B', B), alpha=param('calving_melt_alpha', alpha),
        beta=param('calving_melt_beta', beta))


def calving_covariates(gdir, observed_flux=None, ocean_filesuffix='',
                       band=None, period=None, input_filesuffix=''):
    """Candidate predictors of one glacier's calving constant.

    The geometry of the calving law at the prescribed front, the subglacial
    discharge, and the thermal forcing and melt rate when the glacier has an
    ocean file.

    Parameters
    ----------
    gdir : :py:class:`oggm.GlacierDirectory`
        the glacier directory to process
    observed_flux : dict, optional
        rgi_id -> observed frontal ablation in km3 yr-1 of ice; adds the
        constant ``k`` that reproduces it
    ocean_filesuffix : str
        the filesuffix of the ``ocean_data`` file
    band, period : optional
        passed to :func:`ocean_tf_mean` and
        :func:`subglacial_discharge_from_mb`
    input_filesuffix : str
        the filesuffix of the inversion flowlines

    Returns
    -------
    dict
        one row of covariates
    """
    out = {'rgi_id': gdir.rgi_id, 'is_tidewater': bool(gdir.is_tidewater)}
    if not gdir.is_tidewater:
        return out

    cls = gdir.read_store('inversion_input', filesuffix=input_filesuffix)[-1]
    fl = gdir.read_store('inversion_flowlines',
                          filesuffix=input_filesuffix)[-1]
    yrs, q = subglacial_discharge_from_mb(gdir, period=period,
                                          input_filesuffix=input_filesuffix)
    tf = None
    if gdir.has_file('ocean_data', filesuffix=ocean_filesuffix):
        tf = ocean_tf_mean(gdir, band=band, period=period,
                           ocean_filesuffix=ocean_filesuffix)
    with oggm_errors(gdir):
        row = _covariates.calving_covariates(
            gdir.rgi_id, gdir.settings['terminus_water_depth'],
            width=fl.widths[-1] * gdir.grid.dx, surface_h=fl.surface_h[-1],
            observed_flux=(observed_flux or {}).get(gdir.rgi_id),
            q_sg=None if yrs is None else float(np.nanmean(q)), tf=tf,
            rho_ocean=cfg.PARAMS['ocean_water_density'],
            rho_ice=cfg.PARAMS['ice_density'],
            melt_kwargs={k: cfg.PARAMS[f'calving_melt_{k}']
                         for k in ('A', 'B', 'alpha', 'beta')},
            area_km2=gdir.rgi_area_km2,
            terminus_slope=float(cls['slope_angle'][-1]),
            terminus_elev=float(fl.surface_h[-1]),
            flowline_length=float(fl.dx_meter * len(fl.surface_h)))
    del row['glacier_id']
    out.update(row)
    return out


# Covariates that enter a model of log k as logs
_LOG_TERMS = _covariates.LOG_TERMS


def fit_calving_k_model(gdirs, observed_flux, terms=(), covariates=None,
                        **kwargs):
    """Regress ``log k`` on covariates across a set of glaciers.

    With no ``terms`` the model is the intercept alone, i.e. the geometric
    mean of the constants that reproduce each observation.

    Parameters
    ----------
    gdirs : list of :py:class:`oggm.GlacierDirectory` objects
        the glacier directories to process
    observed_flux : dict
        rgi_id -> observed frontal ablation in km3 yr-1 of ice
    terms : sequence of str
        the keys of :func:`calving_covariates` to regress on
    covariates : list of dict, optional
        the rows to fit, instead of computing them from ``gdirs``
    **kwargs
        passed to :func:`calving_covariates`

    Returns
    -------
    dict
        the coefficients, the terms, the in-sample and leave-one-out log
        residuals, and the constant the model implies for each glacier
    """
    rows = covariates if covariates is not None else [
        calving_covariates(g, observed_flux=observed_flux, **kwargs)
        for g in gdirs]
    with oggm_errors():
        model = _covariates.fit_calving_k_model(
            [dict(r, glacier_id=r.get('rgi_id')) for r in rows], terms,
            _LOG_TERMS)
    return {('rgi_ids' if key == 'glacier_ids' else key): val
            for key, val in model.items()}


@entity_task(log)
def set_inversion_k_from_model(gdir, model=None, k_fallback=None, **kwargs):
    """Write the calving constant a fitted covariate model implies.

    Sets both ``inversion_calving_k`` and ``calving_k`` in the settings.

    Parameters
    ----------
    gdir : :py:class:`oggm.GlacierDirectory`
        the glacier directory to process
    model : dict
        as returned by :func:`fit_calving_k_model`
    k_fallback : float, optional
        the constant of a glacier that lacks one of the model's covariates.
        Default: raise.
    **kwargs
        passed to :func:`calving_covariates`

    Returns
    -------
    float
        the constant in yr-1, or None for a glacier that is not tidewater
    """
    if not gdir.is_tidewater:
        return None
    if model is None:
        raise InvalidParamsError('set_inversion_k_from_model needs a model '
                                 'from fit_calving_k_model')
    k = model['k_model'].get(gdir.rgi_id)
    if k is None:
        row = calving_covariates(gdir, **kwargs)
        terms = model['terms']
        if all(np.isfinite(row.get(t, np.nan)) for t in terms):
            k = calving_k_from_model(model, row)
        elif k_fallback is not None:
            k = float(k_fallback)
        else:
            raise InvalidWorkflowError(
                f'({gdir.rgi_id}) is missing {terms} and no k_fallback was '
                'given.')
    gdir.settings['inversion_calving_k'] = k
    gdir.settings['calving_k'] = k
    gdir.add_to_diagnostics('covariate_inversion_k', float(k))
    return float(k)
