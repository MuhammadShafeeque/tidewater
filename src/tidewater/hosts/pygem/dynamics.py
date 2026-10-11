"""PyGEM's dynamical models with ocean-forced frontal ablation.

OGGM's ``FluxBasedModel``, which PyGEM runs for tidewater glaciers, takes a
calving law as its ``calving_law``: a :mod:`tidewater.laws` law goes there as
it is. PyGEM's mass-redistribution model removes frontal ablation once per
model year; :class:`YearlyCalving` turns a law or a table of yearly factors
into that yearly amount.

:func:`ocean_dynamics` gives PyGEM's run scripts both for the duration of a
``with`` block, by one of two routes. A PyGEM whose ``run_simulation`` has
the ``calving_law_for`` hook is given the hook. A PyGEM without it has the
two model names of the module replaced, the mass-redistribution one by
:class:`OceanMassRedistributionCurveModel`. Both routes build the calving
law of a glacier in the same way and give the same numbers.

Importing this module imports PyGEM, which reads its configuration file.
"""
import copy
import importlib
from collections.abc import Mapping
from contextlib import contextmanager

import numpy as np
import pandas as pd
from pygem.glacierdynamics import MassRedistributionCurveModel

from tidewater.config import InvalidParamsError, InvalidWorkflowError
from tidewater.hosts.pygem.glaciers import (by_glacier, cell_geometry,
                                            front_cell, glacier_of, rgi_id)
from tidewater.laws import frontal_ablation_components as _law_components

# the mid-month times of a model year, as fractions of the year
MONTHS = (np.arange(12) + 0.5) / 12

# the PyGEM modules ocean_dynamics acts on by default
RUN_MODULES = ('pygem.bin.run.run_simulation',)

# the hook of a PyGEM run module, and PyGEM's names of its two dynamics
HOOK = 'calving_law_for'
FLUX, REDISTRIBUTION = 'OGGM', 'MassRedistributionCurves'
ROUTES = ('hook', 'swap')


class StepCalving:
    """A law as the ``calving_law`` of ``FluxBasedModel``; keeps its model."""

    def __init__(self, calving_law):
        self.calving_law = calving_law
        self.model = None

    def __call__(self, model, flowline, front):
        self.model = model
        return self.calving_law(model, flowline, front)

    def components_m3(self, model=None):
        """``(calving, submarine melt)`` since the run started, m3 ice."""
        return self.calving_law.components_m3(self.model if model is None
                                              else model)


class YearlyCalving:
    """A law or a yearly ocean factor as PyGEM's once-a-year frontal ablation.

    PyGEM's mass-redistribution model removes ``k * d * h * w`` (m3 of ice)
    at its front once per model year. Here each year has its own ``k``: the
    model's ``calving_k`` times the year's ocean factor, or the constant at
    which that product equals the mean flux of a law over the twelve
    mid-month times of the year, on the front at the start of the year.

    Called as ``calving(model, flowline, front, year)`` it returns the
    frontal ablation of the year in m3 of ice, which is the ``calving_law``
    of ``MassRedistributionCurveModel`` where PyGEM has that argument.

    Parameters
    ----------
    calving_law : a :mod:`tidewater.laws` law, optional
    ocean_factor : mapping or Series, optional
        model year -> factor (:func:`tidewater.hosts.pygem.yearly_factors`)
    """

    def __init__(self, calving_law=None, ocean_factor=None):
        if (calving_law is None) == (ocean_factor is None):
            raise InvalidParamsError('give a calving law or a yearly ocean '
                                     'factor')
        self.calving_law = calving_law
        self.ocean_factor = ocean_factor
        self.model = None
        self._years = []

    def constant(self, flowline, front, water_level, year, k):
        """The calving constant of ``year`` in yr-1, and its melt share."""
        if self.ocean_factor is not None:
            try:
                return k * float(self.ocean_factor[int(year)]), 0.
            except KeyError:
                raise InvalidWorkflowError(
                    f'no ocean factor for the model year {int(year)}'
                ) from None
        geo = cell_geometry(flowline, front, water_level)
        h, d, w = geo['thickness'], geo['water_depth'], geo['width']
        if d * h * w <= 0:
            return k, 0.
        parts = np.array([self.calving_law.flux_components(h, d, w,
                                                           int(year) + m, k=k)
                          for m in MONTHS])
        calving, melt = parts.mean(axis=0)
        total = calving + melt
        share = melt / total if total > 0 else 0.
        return total * self.calving_law.sec_in_year / (d * h * w), share

    def open(self, model, year, share):
        """Start the books of ``year``; the model's total closes them."""
        self.model = model
        self._years.append((float(year), float(share),
                            float(model.calving_m3_since_y0)))

    def __call__(self, model, flowline, front, year):
        k, share = self.constant(flowline, front, model.water_level, year,
                                 model.calving_k)
        self.open(model, year, share)
        geo = cell_geometry(flowline, front, model.water_level)
        # PyGEM's own product, in its order
        return k * geo['water_depth'] * geo['thickness'] * geo['width']

    def components(self):
        """Frontal ablation and its melt part per calving year, m3 ice."""
        rec = np.asarray(self._years, dtype=float).reshape(-1, 3)
        end = [] if self.model is None else [self.model.calving_m3_since_y0]
        total = np.diff(np.append(rec[:, 2], end))
        out = pd.DataFrame({'frontal_ablation_m3': total,
                            'submarine_melt_m3': total * rec[:, 1]},
                           index=pd.Index(rec[:, 0].astype(int), name='year'))
        out['calving_m3'] = out.frontal_ablation_m3 - out.submarine_melt_m3
        return out

    def components_m3(self):
        """``(calving, submarine melt)`` since the run started, m3 ice."""
        out = self.components()
        return float(out.calving_m3.sum()), float(out.submarine_melt_m3.sum())


class OceanMassRedistributionCurveModel(MassRedistributionCurveModel):
    """PyGEM's mass-redistribution model with a :class:`YearlyCalving`.

    For a PyGEM whose model has no ``calving_law`` argument: each year the
    model's ``calving_k`` is set to the constant of the year for PyGEM's own
    frontal ablation, and restored. PyGEM's front, cap and accounting are
    unchanged.

    Parameters
    ----------
    calving_law : a :mod:`tidewater.laws` law or a :class:`YearlyCalving`
    ocean_factor : mapping or Series, optional
        model year -> factor, in place of a law
    *args, **kwargs
        those of ``MassRedistributionCurveModel``
    """

    def __init__(self, *args, calving_law=None, ocean_factor=None, **kwargs):
        if not isinstance(calving_law, YearlyCalving):
            calving_law = YearlyCalving(calving_law, ocean_factor)
        super().__init__(*args, **kwargs)
        self.yearly_calving = calving_law

    def _get_annual_frontalablation(self, heights, year=None, fls=None,
                                    fl_id=None, calving_k=None, debug=False):
        run = super()._get_annual_frontalablation
        front = front_cell(fls[fl_id], self.water_level)
        if front is None:
            return run(heights, year=year, fls=fls, fl_id=fl_id, debug=debug)
        k = self.calving_k
        self.calving_k, share = self.yearly_calving.constant(
            fls[fl_id], front, self.water_level, year, k)
        self.yearly_calving.open(self, year, share)
        try:
            return run(heights, year=year, fls=fls, fl_id=fl_id, debug=debug)
        finally:
            self.calving_k = k

    def frontal_components(self):
        """Frontal ablation and its melt part per calving year, m3 ice."""
        return self.yearly_calving.components()

    def components_m3(self):
        """``(calving, submarine melt)`` since the run started, m3 ice."""
        return self.yearly_calving.components_m3()


def frontal_ablation_components(run):
    """``(calving, submarine melt)`` of a finished run, m3 of ice.

    ``run`` is a model of either kind, or what :func:`ocean_dynamics` lists.
    """
    for obj in (run, getattr(run, 'yearly_calving', None),
                getattr(run, 'calving_law', None)):
        if isinstance(obj, (YearlyCalving, StepCalving)):
            return obj.components_m3()
    return _law_components(run)


def _is_factor(value):
    return isinstance(value, (pd.Series, Mapping))


def _fresh(law):
    law = copy.deepcopy(law)
    law.reset_accounting()
    return law


def glacier_calving(value, gid, dynamics):
    """The calving law PyGEM's model of ``dynamics`` takes for one glacier.

    Parameters
    ----------
    value : a :mod:`tidewater.laws` law, or a table of yearly factors
    gid : str
        the glacier, for messages
    dynamics : str
        PyGEM's ``option_dynamics``: 'OGGM' or 'MassRedistributionCurves'

    Returns
    -------
    :class:`StepCalving` or :class:`YearlyCalving`, on a fresh copy of the law
    """
    if dynamics == REDISTRIBUTION:
        if _is_factor(value):
            return YearlyCalving(ocean_factor=value)
        return YearlyCalving(calving_law=_fresh(value))
    if dynamics != FLUX:
        raise InvalidParamsError(f'no calving in the dynamics {dynamics!r}')
    if _is_factor(value):
        raise InvalidWorkflowError(
            f'{gid}: FluxBasedModel needs a calving law; a yearly factor is '
            'for the mass-redistribution model')
    return StepCalving(_fresh(value))


def _module(module):
    return (importlib.import_module(module) if isinstance(module, str)
            else module)


def coupling_route(module=RUN_MODULES[0]):
    """'hook' where the PyGEM module has the calving-law hook, else 'swap'."""
    return 'hook' if hasattr(_module(module), HOOK) else 'swap'


def _swapped(cls, dynamics, calving_for):
    """``cls`` as the run scripts call it, with the glacier's calving law."""
    def build(*args, **kwargs):
        calving = None
        if kwargs.get('is_tidewater'):
            calving = calving_for(glacier_of(kwargs.get('mb_model')),
                                  dynamics)
        if calving is None:
            return cls(*args, **kwargs)
        if dynamics == REDISTRIBUTION:
            cls_ = OceanMassRedistributionCurveModel
            return cls_(*args, calving_law=calving, **kwargs)
        return cls(*args, calving_law=calving, **kwargs)
    return build


@contextmanager
def ocean_dynamics(laws, modules=RUN_MODULES, route=None):
    """Run PyGEM's scripts with ocean-forced frontal ablation.

    Inside the block each tidewater glacier with an entry in ``laws`` runs
    with its law (a fresh copy per model) or, for the mass-redistribution
    model, its yearly factor; the others run as before. On exit the modules
    are as they were. Worker processes started with ``spawn`` do not see the
    change: run with one core, or with ``fork``.

    Parameters
    ----------
    laws : mapping or callable
        glacier id (any form) -> law, or a yearly factor (Series) for the
        mass-redistribution model; or a function of the RGI id returning
        one or None
    modules : sequence of str or module
        default: ``pygem.bin.run.run_simulation``
    route : str, optional
        'hook': set the ``calving_law_for`` hook of the module. 'swap':
        replace its ``FluxBasedModel`` and ``MassRedistributionCurveModel``.
        Default: the hook where the module has it (:func:`coupling_route`).

    Yields
    ------
    list
        ``(rgi_id, calving)`` per model run with a law or a factor;
        ``calving`` is the :class:`StepCalving` or :class:`YearlyCalving`
        the model ran with (:func:`frontal_ablation_components`)
    """
    if route not in (None,) + ROUTES:
        raise InvalidParamsError(f'unknown route {route!r}; one of {ROUTES}')
    lookup = (laws if callable(laws) and not isinstance(laws, Mapping)
              else by_glacier(laws).get)
    built, saved = [], []

    def calving_for(gid, dynamics):
        value = lookup(gid)
        if value is None:
            return None
        built.append((gid, glacier_calving(value, gid, dynamics)))
        return built[-1][1]

    def hook(gdir, option_dynamics):
        return calving_for(rgi_id(gdir.rgi_id), option_dynamics)

    def put(mod, name, value):
        saved.append((mod, name, getattr(mod, name)))
        setattr(mod, name, value)

    try:
        for mod in modules:
            mod = _module(mod)
            if (route or coupling_route(mod)) == 'hook':
                if not hasattr(mod, HOOK):
                    raise InvalidWorkflowError(
                        f'{getattr(mod, "__name__", mod)} has no {HOOK} hook')
                put(mod, HOOK, hook)
                continue
            for name, dynamics in (('FluxBasedModel', FLUX),
                                   ('MassRedistributionCurveModel',
                                    REDISTRIBUTION)):
                if hasattr(mod, name):
                    put(mod, name, _swapped(getattr(mod, name), dynamics,
                                            calving_for))
        yield built
    finally:
        for mod, name, value in reversed(saved):
            setattr(mod, name, value)
