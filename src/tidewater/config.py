"""Parameters, constants and exceptions shared by the package."""
from dataclasses import dataclass, fields, replace

# 365-day year, as in OGGM (Maussion et al., 2019)
SEC_IN_YEAR = 365 * 24 * 3600
SEC_IN_DAY = 24 * 3600


class InvalidParamsError(ValueError):
    """A parameter value the scheme cannot use."""


class InvalidWorkflowError(ValueError):
    """An input is missing or does not cover what was asked of it."""


def _float_or_none(value):
    if value is None or (isinstance(value, str) and not value.strip()):
        return None
    return float(value)


def _bool(value):
    if isinstance(value, str):
        return value.strip().lower() in ('true', '1', 'yes')
    return bool(value)


def _specs(value):
    if isinstance(value, str):
        value = value.split(',')
    return tuple(str(v).strip() for v in value if str(v).strip())


def _period(value):
    from tidewater.periods import as_period
    return as_period(value)


@dataclass(frozen=True)
class OceanParams:
    """The parameters of the scheme, with their default values.

    The depth bands, ``ocean_tf_band``, ``ocean_tf_ref_period`` (and an
    explicit ``ocean_tf_ref``), ``calving_tf_exponent`` and the four
    ``calving_melt_*`` coefficients are Greenland values: revisit them for
    another region, with :meth:`for_region` or by hand.
    ``ocean_open_water_threshold`` and ``ocean_sea_level`` depend on the data
    products in use. The source of every value is given in
    ``docs/references.md``.

    Attributes
    ----------
    ocean_depth_bands : tuple of str
        each ``name:top:bottom[:weighting]``, depths in m, positive downwards
    ocean_tf_band : str
        the band the laws and the inversion read
    ocean_tf_ref_period : (int, int)
        first and last year of the reference period, both included
    ocean_tf_ref : float or None
        reference thermal forcing of ``tf_power``, degC; None: the power mean
        of the record over ``ocean_tf_ref_period``
    ocean_use_teos10 : bool
        TEOS-10 freezing point (needs gsw) instead of the linear form
    ocean_bias_correct : bool
        add the offset to a reference ocean to the thermal forcing
    ocean_open_water_threshold : float
        sea-ice concentration (0-1) at and above which no water is open
    ocean_sea_level : float
        m, in the vertical datum of the bed and the surface
    ocean_water_density : float
        kg m-3
    ice_density : float
        kg m-3
    calving_law : str
        constant, tf_power, melt_calving, sea_ice or larger_of
    calving_tf_exponent : float
        exponent of the thermal forcing in ``tf_power``
    calving_tf_fraction : float
        share (0-1) of the calving constant that follows the thermal forcing
    calving_melt_A, calving_melt_B, calving_melt_alpha, calving_melt_beta : float
        ``(A * depth * q_sg**alpha + B) * TF**beta``, m d-1
    calving_undercut_efficiency : float
        share of that melt rate acting on the whole submerged front
    calving_openwater_exponent : float
        exponent of the open-water fraction in ``sea_ice``
    sec_in_year : float
        s, the year the yr-1 constants refer to
    """

    # Greenland values; set per region. 200-500 m: Slater et al. (2019, 2020);
    # 0-60 m: Bendtsen et al. (2017)
    ocean_depth_bands: tuple = ('shallow_0_60:0:60:uniform',
                                'ismip6:200:500:uniform',
                                'front:0:60:uniform')
    # Greenland value; set per region
    ocean_tf_band: str = 'shallow_0_60'
    # Greenland value; set per region
    ocean_tf_ref_period: tuple = (1995, 2014)
    ocean_tf_ref: float = None
    ocean_use_teos10: bool = False
    ocean_bias_correct: bool = False
    ocean_open_water_threshold: float = 0.15
    ocean_sea_level: float = 0.0
    ocean_water_density: float = 1028.
    # as in OGGM (Maussion et al., 2019)
    ice_density: float = 900.
    calving_law: str = 'constant'
    # Greenland value; set per region. The exponent of Rignot et al. (2016)
    calving_tf_exponent: float = 1.18
    calving_tf_fraction: float = 1.0
    # Greenland values; set per region. Rignot et al. (2016), Eq. 1
    calving_melt_A: float = 3e-4
    # Greenland value; set per region
    calving_melt_B: float = 0.15
    # Greenland value; set per region
    calving_melt_alpha: float = 0.39
    # Greenland value; set per region
    calving_melt_beta: float = 1.18
    calving_undercut_efficiency: float = 1.0
    calving_openwater_exponent: float = 1.0
    sec_in_year: float = SEC_IN_YEAR

    _CASTS = {'ocean_depth_bands': _specs, 'ocean_tf_band': str,
              'ocean_tf_ref_period': _period, 'ocean_tf_ref': _float_or_none,
              'ocean_use_teos10': _bool, 'ocean_bias_correct': _bool,
              'calving_law': str}

    @classmethod
    def from_mapping(cls, mapping, **overrides):
        """Build from a host's configuration; keys it does not hold keep the
        defaults, keys this class does not know are ignored."""
        values = {}
        for f in fields(cls):
            if f.name in overrides:
                raw = overrides[f.name]
            else:
                try:
                    raw = mapping[f.name]
                except (KeyError, TypeError):
                    continue
            values[f.name] = cls._CASTS.get(f.name, float)(raw)
        return cls(**values)

    @classmethod
    def for_region(cls, name, **overrides):
        """The parameters of a region registered in :data:`REGIONS`."""
        key = str(name).strip().lower()
        if key not in REGIONS:
            raise InvalidParamsError(f'no parameters for region {name!r}; '
                                     f'available: {sorted(REGIONS)}. Add one '
                                     'with register_region.')
        return cls.from_mapping(REGIONS[key], **overrides)

    def replace(self, **changes):
        """A copy with some values changed."""
        return replace(self, **changes)

    def depth_bands(self):
        """``(depth_bands, band_weighting)`` of ``ocean_depth_bands``."""
        from tidewater.physics import parse_depth_bands
        return parse_depth_bands(self.ocean_depth_bands)

    def law_kwargs(self, calving_law=None, laws=None):
        """The keyword arguments of one law, by its ``settings_keys``.

        ``laws``: the registry the name is looked up in. Default:
        :data:`tidewater.laws.LAWS`.
        """
        if laws is None:
            from tidewater.laws import LAWS as laws
        name = calving_law or self.calving_law
        if name not in laws:
            raise InvalidParamsError(f'unknown calving law {name!r}; '
                                     f'available: {sorted(laws)}')
        out = {kwarg: getattr(self, key)
               for kwarg, key in laws[name].settings_keys.items()}
        out['sec_in_year'] = self.sec_in_year
        return out


DEFAULTS = OceanParams()

# The fields whose defaults were derived for one region
REGIONAL_FIELDS = ('ocean_depth_bands', 'ocean_tf_band',
                   'ocean_tf_ref_period', 'ocean_tf_ref',
                   'calving_tf_exponent', 'calving_melt_A', 'calving_melt_B',
                   'calving_melt_alpha', 'calving_melt_beta')

# region name -> the OceanParams fields it sets
REGIONS = {}


def register_region(name, **values):
    """Add or replace the parameter set of a region in :data:`REGIONS`.

    Parameters
    ----------
    name : str
        stored in lower case
    **values
        :class:`OceanParams` fields; the others keep their defaults
    """
    known = {f.name for f in fields(OceanParams)}
    unknown = sorted(set(values) - known)
    if unknown:
        raise InvalidParamsError(f'not parameters of the scheme: {unknown}')
    OceanParams.from_mapping(values)  # the values must be readable
    REGIONS[str(name).strip().lower()] = dict(values)


register_region('greenland',
                **{key: getattr(DEFAULTS, key) for key in REGIONAL_FIELDS})
