"""Ocean-forced frontal ablation laws for the flowline models.

Five laws, as a nested family rooted at the stock model:

===================  =========================================================
``constant``         :class:`ConstantK`, identical to
                     :func:`oggm.core.flowline.k_calving_law`. The control.
``tf_power``         :class:`TFPower`,
                     ``k(t) = k0 ((1 - a) + a (TF/TF_ref)**gamma)``.
                     Reduces to the control when ``TF == TF_ref`` or ``a = 0``.
``melt_calving``     :class:`MeltPlusCalving`, an explicit submarine-melt term
                     after Rignot et al. (2016) plus a calving term.
``sea_ice``          :class:`SeaIceModulated`, the melt term gated by the
                     open-water fraction. Reduces to ``melt_calving`` at
                     ``delta = 0``.
``larger_of``        :class:`LargerOf`, the larger of the calving and the
                     submarine-melt speed. Reduces to the control at
                     ``lam = 0``.
===================  =========================================================

Each is passed as the ``calving_law`` of ``FluxBasedModel`` or
``SemiImplicitModel``, and reads the ``ocean_data`` file written by
:mod:`tidewater.hosts.oggm.forcing`.

Each law also reports the split of what it returns into calving and submarine
melt, which :func:`write_frontal_components` writes beside the run output. The
split is a model diagnostic: melt undercutting drives calving, so a calving
constant calibrated against an observed frontal ablation already contains the
melt-driven part (see ``partition`` in :func:`ocean_calving_law`).

``model.calving_k`` is in s-1 while ``cfg.PARAMS['calving_k']`` is in yr-1, and
a law returns m3 s-1. The thermal forcing is clipped at zero before any power.
"""
import logging
import os

import xarray as xr
from oggm import cfg
from oggm import entity_task, global_task
from oggm.exceptions import InvalidParamsError, InvalidWorkflowError

from tidewater import laws as _laws
from tidewater import physics as _physics
from tidewater.forcing.discharge import mid_month_years
from tidewater.hosts.oggm.errors import oggm_errors, translated
from tidewater.hosts.oggm.params import setting as _setting
from tidewater.laws import (_melt_fraction,  # noqa: F401
                            frontal_ablation_components, tf_power_mean)
from tidewater.physics import BAND_ALIASES, band_names  # noqa: F401

log = logging.getLogger(__name__)


@translated
def band_index(names, band):
    """Position of ``band`` in ``names``, or of its earlier name."""
    return _physics.band_index(names, band, aliases=BAND_ALIASES)


class _OceanCalvingLaw(_laws._OceanCalvingLaw):
    """Base for time-varying calving laws driven by an ocean_data file.

    Sub-classes implement :meth:`frontal_speed_components`, returning
    front-normal speeds in m s-1. This class multiplies by the submerged area,
    which is the algebra of the stock law: ``k*d*h*w == (k*h) * (d*w)``.

    Parameters
    ----------
    years : array
        the time axis of the forcing, in float years
    tf : array
        thermal forcing on that axis, degC
    open_water : array, optional
        open-water fraction on that axis
    q_sg : array, optional
        subglacial discharge on that axis, m3 s-1
    """
    @translated
    def __init__(self, *args, **kwargs):
        for kwarg, key in self.settings_keys.items():
            if kwargs.get(kwarg) is None:
                kwargs[kwarg] = cfg.PARAMS[key]
        kwargs.setdefault('sec_in_year', cfg.SEC_IN_YEAR)
        super().__init__(*args, **kwargs)

    _k = translated(_laws._OceanCalvingLaw._k)


class ConstantK(_OceanCalvingLaw, _laws.ConstantK):
    """The control: :func:`oggm.core.flowline.k_calving_law` with accounting.

    Like the stock law, and unlike its siblings, it does not clip at zero.

    Parameters
    ----------
    k : float, optional
        the calving constant in yr-1. Default: the model's ``calving_k``.
    """


class TFPower(_OceanCalvingLaw, _laws.TFPower):
    """The stock law with a calving constant scaled by the thermal forcing.

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
        the calving constant at ``TF_ref``, yr-1. Default: the model's
        ``calving_k``.
    gamma : float, optional
        default: ``cfg.PARAMS['calving_tf_exponent']``
    tf_fraction : float, optional
        ``a``, in [0, 1]. Default: ``cfg.PARAMS['calving_tf_fraction']``
    """


class MeltPlusCalving(_OceanCalvingLaw, _laws.MeltPlusCalving):
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
        the calving constant, yr-1. Default: the model's ``calving_k``, in
        which case the melt term adds to a constant that may already contain
        it (see ``partition`` in :func:`ocean_calving_law`).
    lam : float, optional
        default: ``cfg.PARAMS['calving_undercut_efficiency']``
    water_depth : float, optional
        the water depth of the melt rate, m. Default: the model's own.
    A, B, alpha, beta : float, optional
        default: ``cfg.PARAMS['calving_melt_A']`` and so on
    """


class SeaIceModulated(MeltPlusCalving, _laws.SeaIceModulated):
    """:class:`MeltPlusCalving` with the melt term gated by open water.

    ``Q_f = w * d * (k_ice * h + f_ow(t)**delta * lam * mdot(TF))``

    The open-water fraction ``f_ow`` acts on the melt term only; the calving
    term is left as it is. ``delta = 0`` is :class:`MeltPlusCalving`.

    Parameters
    ----------
    k_ice : float, optional
        the calving constant, yr-1 (``k_c`` of :class:`MeltPlusCalving`)
    delta : float, optional
        default: ``cfg.PARAMS['calving_openwater_exponent']``
    """
    melt_speed = translated(_laws.SeaIceModulated.melt_speed)


class LargerOf(MeltPlusCalving, _laws.LargerOf):
    """The larger of the calving and the submarine-melt speed.

    ``Q_f = w * d * max(k_c * h, lam * mdot)``, after Malles et al. (2025).
    The whole flux is reported as the larger term. ``lam = 0`` is the control.
    With ``partition``, ``k_c`` is lowered until the period mean of the larger
    term equals the calibrated total.
    """


LAWS = {'constant': ConstantK, 'tf_power': TFPower,
        'melt_calving': MeltPlusCalving, 'sea_ice': SeaIceModulated,
        'larger_of': LargerOf}


def partition_law_k(gdir, law, period=None):
    """Give a melt-bearing law the residual calving constant, keeping the total.

    A calving constant fitted to an observed frontal ablation already contains
    the calving that submarine melt drives, so a melt term added on top of it
    counts that melt twice.
    :func:`tidewater.hosts.oggm.inversion.partition_calving_constant` solves
    ``k_c h + u_melt = k h`` for ``k_c`` at the front the inversion
    prescribed, with the melt speed the law applies (its open-water gate
    included) averaged over ``period``. The total is then unchanged over that
    period and only its split moves. For :class:`LargerOf`,
    :func:`tidewater.hosts.oggm.inversion.larger_of_calving_constant` solves
    ``mean(max(k_c h, u_melt)) = k h`` instead.

    Parameters
    ----------
    gdir : :py:class:`oggm.GlacierDirectory`
        the glacier directory, inverted with
        :func:`tidewater.hosts.oggm.inversion.find_inversion_calving_from_bathymetry`
    law : :class:`MeltPlusCalving`
        the law, whose ``k_c`` is set in place
    period : tuple of two years, optional
        default: ``gdir.settings['ocean_tf_ref_period']``

    Returns
    -------
    (k_c, melt_fraction) : the residual constant in yr-1 and the melt share of
        the total, or ``(None, 0.)`` for a law without a melt term
    """
    if not hasattr(law, 'melt_rate'):
        return None, 0.
    k_total = _setting(gdir, 'calving_k', _setting(gdir, 'inversion_calving_k'))
    thick = _setting(gdir, 'calving_front_thick')
    depth = _setting(gdir, 'terminus_water_depth')
    width = _setting(gdir, 'calving_front_width')
    if None in (k_total, thick, depth, width):
        raise InvalidWorkflowError(
            f'({gdir.rgi_id}) partitioning needs calving_k, '
            'calving_front_thick, terminus_water_depth and calving_front_width '
            'in the settings; run find_inversion_calving_from_bathymetry '
            'first.')

    period = period or gdir.settings['ocean_tf_ref_period']
    with oggm_errors(gdir):
        return _laws.partition_law_k(law, k_total, thick, depth, width,
                                     period=period)


def ocean_calving_law(gdir, calving_law=None, band=None, ocean_filesuffix='',
                      tf_ref=None, partition=False, partition_period=None,
                      **law_kwargs):
    """Build a calving law for one glacier from its ocean_data file.

    Parameters
    ----------
    gdir : :py:class:`oggm.GlacierDirectory`
        the glacier directory to process
    calving_law : str, optional
        'constant', 'tf_power', 'melt_calving', 'sea_ice' or 'larger_of'.
        Default: ``gdir.settings['calving_law']``.
    band : str, optional
        the depth band of the ocean file the law reads. Default:
        ``gdir.settings['ocean_tf_band']``.
    ocean_filesuffix : str
        the filesuffix of the ``ocean_data`` file
    tf_ref : float, optional
        the reference thermal forcing of 'tf_power', ignored by the other
        laws. Default: ``gdir.settings['ocean_tf_ref']``, and when that is not
        set either, :func:`tf_power_mean` of the glacier's own record over
        ``gdir.settings['ocean_tf_ref_period']``, so that the law returns the
        calving constant on average over that period.
    partition : bool
        for a melt-bearing law, set ``k_c`` from :func:`partition_law_k` so
        that the law keeps the calibrated total over ``partition_period``
        instead of adding melt to it.
    partition_period : tuple of two years, optional
        the period over which the partition keeps the total. Default:
        ``gdir.settings['ocean_tf_ref_period']``.
    **law_kwargs
        passed to the law (``k0``, ``gamma``, ``lam``, ``delta`` ...). The
        ones not given are read from the glacier's settings.

    Returns
    -------
    the law, to be passed as ``calving_law`` to a flowline model
    """
    calving_law = calving_law or gdir.settings['calving_law']
    if calving_law not in LAWS:
        raise InvalidParamsError(f'unknown calving law {calving_law!r}; '
                                 f'available: {sorted(LAWS)}')
    if calving_law == 'constant':
        return ConstantK(**law_kwargs)

    if not gdir.has_file('ocean_data', filesuffix=ocean_filesuffix):
        raise InvalidWorkflowError(f'({gdir.rgi_id}) no ocean_data file; run '
                                   'process_ocean_data first.')

    band = band or gdir.settings['ocean_tf_band']
    fp = gdir.get_filepath('ocean_data', filesuffix=ocean_filesuffix)
    with xr.open_dataset(fp) as ds:
        ds = ds.load()
    if 'band' not in ds.dims:
        raise InvalidWorkflowError('ocean_data has no band dimension')
    tf = ds['thermal_forcing'].values[:, band_index(band_names(ds), band)]
    yrs = mid_month_years(ds['time.year'].values, ds['time.month'].values)

    cls = LAWS[calving_law]
    for kwarg, key in cls.settings_keys.items():
        if law_kwargs.get(kwarg) is None:
            law_kwargs[kwarg] = gdir.settings[key]

    if cls is TFPower:
        if tf_ref is None:
            tf_ref = gdir.settings['ocean_tf_ref']
        if tf_ref is None:
            y0, y1 = (int(y) for y in gdir.settings['ocean_tf_ref_period'])
            sel = (ds['time.year'].values >= y0) & (ds['time.year'].values <= y1)
            if not sel.any():
                raise InvalidWorkflowError(
                    f'({gdir.rgi_id}) the ocean record does not cover '
                    f'{y0}-{y1}, so there is no reference thermal forcing. '
                    'Pass tf_ref or set ocean_tf_ref.')
            tf_ref = tf_power_mean(tf[sel], law_kwargs['gamma'])
        law_kwargs['tf_ref'] = tf_ref

    law = cls(yrs, tf,
              open_water=(ds['open_water_frac'].values
                          if 'open_water_frac' in ds else None),
              q_sg=(ds['subglacial_discharge'].values
                    if 'subglacial_discharge' in ds else None),
              **law_kwargs)
    if partition:
        k_c, frac = partition_law_k(gdir, law, period=partition_period)
        if k_c is not None:
            gdir.add_to_diagnostics('calving_k_residual', float(k_c))
            gdir.add_to_diagnostics('calving_melt_fraction_reference',
                                    float(frac))
    return law


def write_frontal_components(gdir, law, model=None, output_filesuffix=''):
    """Write the melt/calving split of a finished run to its own file.

    The two series are a share of the ``calving_m3`` of the run's
    ``model_diagnostics`` file, which is left untouched, so they sum to it at
    every step.

    Parameters
    ----------
    gdir : :py:class:`oggm.GlacierDirectory`
        the glacier directory to process
    law : the calving law the run used
    model : the flowline model of the run, optional
        closes the last open step of the split
    output_filesuffix : str
        the filesuffix of the run

    Returns
    -------
    (calving, submarine melt) : the totals of the run, m3
    """
    calving, melt = law.components_m3(model)
    fp = gdir.get_filepath('model_diagnostics', filesuffix=output_filesuffix)
    ds = None
    if os.path.exists(fp):
        with xr.open_dataset(fp) as ds:
            ds = ds.load()
    has_series = ds is not None and 'calving_m3' in ds
    cum_melt, cum_calving, frac = _laws.frontal_component_series(
        law, ds['time'].values if has_series else [],
        ds['calving_m3'].values if has_series else [], model)
    gdir.add_to_diagnostics('submarine_melt_m3', float(melt))
    gdir.add_to_diagnostics('calving_only_m3', float(calving))
    gdir.add_to_diagnostics('submarine_melt_fraction', frac)
    if ds is None:
        return calving, melt
    if not has_series:
        log.warning("(%s) no calving_m3 in the diagnostics: add 'calving' to "
                    "store_diagnostic_variables to get the split", gdir.rgi_id)
        return calving, melt

    out = xr.Dataset(coords={c: ds[c] for c in ds.coords})
    out['calving_m3'] = ds['calving_m3']
    out['submarine_melt_m3'] = ('time', cum_melt)
    out['submarine_melt_m3'].attrs = {
        'description': 'Accumulated submarine melt part of the frontal '
                       'ablation',
        'unit': 'm 3'}
    out['calving_only_m3'] = ('time', cum_calving)
    out['calving_only_m3'].attrs = {
        'description': 'Accumulated iceberg calving part of the frontal '
                       'ablation',
        'unit': 'm 3'}
    out.attrs.update({'rgi_id': gdir.rgi_id,
                      'cenlon': float(gdir.cenlon),
                      'cenlat': float(gdir.cenlat),
                      'calving_law': law.name,
                      'submarine_melt_fraction': frac,
                      'partition': 'modelled, not observed'})
    out.to_netcdf(gdir.get_filepath('frontal_ablation_diagnostics',
                                    filesuffix=output_filesuffix, delete=True))
    return calving, melt


@global_task(log)
def compile_frontal_components(gdirs, input_filesuffix='', path=True):
    """Compile the frontal ablation split of several glaciers into one file.

    The counterpart of :func:`oggm.utils.compile_run_output` for the
    ``frontal_ablation_diagnostics`` files.

    Parameters
    ----------
    gdirs : list of :py:class:`oggm.GlacierDirectory` objects
        the glacier directories to process
    input_filesuffix : str
        the filesuffix of the runs
    path : str or bool
        where to store the file (default is on the working dir). Set to
        `False` to disable disk storage.

    Returns
    -------
    ds : :py:class:`xarray.Dataset`
        with dimensions (time, rgi_id), or None when no glacier has the file
    """
    dss = []
    for gdir in gdirs:
        fp = gdir.get_filepath('frontal_ablation_diagnostics',
                               filesuffix=input_filesuffix)
        if not os.path.exists(fp):
            log.warning('(%s) no frontal_ablation_diagnostics%s, skipped',
                        gdir.rgi_id, input_filesuffix)
            continue
        with xr.open_dataset(fp) as ds:
            ds = ds.load()
        ds = ds[['calving_m3', 'submarine_melt_m3', 'calving_only_m3']]
        ds = ds.expand_dims(rgi_id=[gdir.rgi_id])
        ds.coords['lon'] = ('rgi_id', [float(gdir.cenlon)])
        ds.coords['lat'] = ('rgi_id', [float(gdir.cenlat)])
        dss.append(ds)
    if not dss:
        return None
    out = xr.concat(dss, dim='rgi_id')
    out.attrs['partition'] = 'modelled, not observed'
    if path:
        if path is True:
            path = os.path.join(
                cfg.PATHS['working_dir'],
                f'frontal_ablation_compiled{input_filesuffix}.nc')
        out.to_netcdf(path)
    return out


@entity_task(log, writes=['frontal_ablation_diagnostics'])
def run_with_ocean_forcing(gdir, calving_law=None, band=None,
                           ocean_filesuffix='', tf_ref=None, law_kwargs=None,
                           climate_filename='gcm_data',
                           climate_input_filesuffix='',
                           output_filesuffix='', **kwargs):
    """Run the flowline model with an ocean-forced calving law.

    A wrapper around :func:`oggm.core.flowline.run_from_climate_data`, which
    also writes the calving and submarine melt parts of the frontal ablation
    (see :func:`write_frontal_components`).

    Parameters
    ----------
    gdir : :py:class:`oggm.GlacierDirectory`
        the glacier directory to process
    calving_law : str, optional
        'constant', 'tf_power', 'melt_calving', 'sea_ice' or 'larger_of'.
        Default: ``gdir.settings['calving_law']``.
    band : str, optional
        the depth band of the ocean file the law reads. Default:
        ``gdir.settings['ocean_tf_band']``.
    ocean_filesuffix : str
        the filesuffix of the ``ocean_data`` file
    tf_ref : float, optional
        the reference thermal forcing of 'tf_power' (see
        :func:`ocean_calving_law`)
    law_kwargs : dict, optional
        passed to :func:`ocean_calving_law` (``partition``, ``k0``, ``gamma``,
        ``lam``, ``delta`` ...)
    climate_filename : str
        name of the climate file, e.g. 'climate_historical' or 'gcm_data'
    climate_input_filesuffix : str
        filesuffix of the climate file
    output_filesuffix : str
        filesuffix of the output files
    **kwargs
        passed to :func:`oggm.core.flowline.run_from_climate_data`

    Returns
    -------
    the flowline model, as returned by ``run_from_climate_data``
    """
    from oggm.core.flowline import run_from_climate_data

    # A law that cannot be built must not leave an earlier run's files behind
    for name in ('model_diagnostics', 'frontal_ablation_diagnostics'):
        gdir.get_filepath(name, filesuffix=output_filesuffix, delete=True)

    law = ocean_calving_law(gdir, calving_law=calving_law, band=band,
                            ocean_filesuffix=ocean_filesuffix, tf_ref=tf_ref,
                            **(law_kwargs or {}))
    gdir.add_to_diagnostics('ocean_calving_law', law.name)
    if band:
        gdir.add_to_diagnostics('ocean_calving_band', band)

    model = run_from_climate_data(
        gdir, calving_law=law, climate_filename=climate_filename,
        climate_input_filesuffix=climate_input_filesuffix,
        output_filesuffix=output_filesuffix, **kwargs)
    write_frontal_components(gdir, law, model=model,
                             output_filesuffix=output_filesuffix)
    return model
