"""The files and parameters the scheme adds to OGGM, and how they are read."""
import functools
import os
import tempfile
from dataclasses import fields

from configobj import ConfigObj
from oggm import cfg

from tidewater import config as _config

BASENAMES = {
    'ocean_data': (
        'ocean_data.nc',
        'The monthly ocean boundary conditions at the glacier terminus: '
        'thermal forcing in one or more depth bands, sea-ice concentration '
        'and, optionally, subglacial discharge.'),
    'frontal_ablation_diagnostics': (
        'frontal_ablation_diagnostics.nc',
        'The submarine melt and calving parts of the frontal ablation of a '
        'run, on the time axis of its model_diagnostics file.'),
}

# The defaults, as OGGM parses them; see docs/hosts.md
DEFAULTS = {
    'inversion_calving_from_bathymetry': False,
    'ocean_depth_bands': ['shallow_0_60:0:60:uniform', 'ismip6:200:500:uniform',
                          'front:0:60:uniform'],
    'ocean_tf_band': 'shallow_0_60',
    'ocean_tf_ref_period': [1995, 2014],
    'ocean_tf_ref': None,
    'ocean_use_teos10': False,
    'ocean_bias_correct': False,
    'ocean_open_water_threshold': 0.15,
    'ocean_water_density': 1028.,
    'calving_law': 'constant',
    'calving_tf_exponent': 1.18,
    'calving_tf_fraction': 1.0,
    'calving_melt_A': 3e-4,
    'calving_melt_B': 0.15,
    'calving_melt_alpha': 0.39,
    'calving_melt_beta': 1.18,
    'calving_undercut_efficiency': 1.0,
    'calving_openwater_exponent': 1.0,
    'bedmachine_version': None,
    'bed_extension_width_method': 'inversion',
    'bed_extension_match_terminus': False,
    'calving_front_width_rtol': 0.05,
}

def add_basenames():
    """Add the files of the scheme to ``cfg.BASENAMES``."""
    for name, (fname, doc) in BASENAMES.items():
        if name not in cfg.BASENAMES:
            cfg.add_to_basenames(name, fname, docstr=doc)


_LISTS = {'ocean_depth_bands': str, 'ocean_tf_ref_period': int}
_OPTIONAL = {'ocean_tf_ref': float, 'bedmachine_version': str}


def parse(key, value):
    """A value of a parameter file (a string or a list of them), typed as in OGGM."""
    if key in _LISTS:
        if isinstance(value, str):
            value = [v for v in value.split(',') if v.strip()]
        return [_LISTS[key](str(v).strip()) for v in value]
    if not isinstance(value, str):
        return value
    if key in _OPTIONAL:
        return _OPTIONAL[key](value) if value.strip() else None
    default = DEFAULTS[key]
    if isinstance(default, bool):
        return cfg.strtobool(value)
    return type(default)(value)


def set_params(values=None, overwrite=False):
    """Write the defaults missing from ``cfg.PARAMS``, then ``values``."""
    prev, cfg.PARAMS.do_log = cfg.PARAMS.do_log, False
    try:
        for key, val in DEFAULTS.items():
            if overwrite or key not in cfg.PARAMS:
                cfg.PARAMS[key] = list(val) if isinstance(val, list) else val
        for key, val in (values or {}).items():
            cfg.PARAMS[key] = parse(key, val)
    finally:
        cfg.PARAMS.do_log = prev


def _split(mapping):
    ours = {k: v for k, v in mapping.items() if k in DEFAULTS}
    return ours, {k: v for k, v in mapping.items() if k not in DEFAULTS}


def wrap_initialize():
    """Make ``cfg.initialize_minimal`` (and so ``cfg.initialize``) set the parameters."""
    stock = cfg.initialize_minimal
    if getattr(stock, 'tidewater', False):
        return

    @functools.wraps(stock)
    def initialize_minimal(file=None, logging_level='INFO', params=None):
        ours, tmp = {}, None
        if file:
            cp = ConfigObj(file, file_error=True)
            ours = {k: cp.pop(k) for k in list(cp) if k in DEFAULTS}
            if ours:
                fd, tmp = tempfile.mkstemp(suffix='.cfg')
                os.close(fd)
                cp.filename = tmp
                cp.write()
                file = tmp
        if params:
            more, params = _split(params)
            ours.update(more)
        try:
            stock(file=file, logging_level=logging_level, params=params)
        finally:
            if tmp:
                os.remove(tmp)
        set_params(ours, overwrite=True)

    initialize_minimal.tidewater = True
    cfg.initialize_minimal = initialize_minimal


def setting(gdir, key, default=None):
    """One setting, or `default` when neither the glacier nor PARAMS has it."""
    try:
        return gdir.settings[key]
    except KeyError:
        return default


def param(key, value=None, settings=None):
    """``value``, or the setting ``key`` when it is None."""
    if value is not None:
        return value
    return (cfg.PARAMS if settings is None else settings)[key]


def ocean_params(settings=None, keys=None):
    """The parameters of tidewater from ``cfg.PARAMS`` or a glacier's settings.

    Parameters
    ----------
    settings : mapping, optional
        default: ``cfg.PARAMS``
    keys : sequence of str, optional
        read these only; the others keep the defaults of tidewater
    """
    settings = cfg.PARAMS if settings is None else settings
    if keys is None:
        keys = [f.name for f in fields(_config.OceanParams)
                if f.name != 'sec_in_year']
    values = {}
    for key in keys:
        try:
            values[key] = settings[key]
        except KeyError:
            pass
    return _config.OceanParams.from_mapping(values,
                                            sec_in_year=cfg.SEC_IN_YEAR)
