"""The PyGEM host of tidewater: ocean-forced frontal ablation in PyGEM.

PyGEM runs a tidewater glacier with OGGM's ``FluxBasedModel`` or with its
own mass-redistribution model, both on calendar model years. The laws of
:mod:`tidewater.laws` are passed as the ``calving_law`` of the first;
:mod:`tidewater.hosts.pygem.dynamics` adds them to the second and to PyGEM's
run scripts, and imports PyGEM. The helpers here and in
:mod:`tidewater.hosts.pygem.calibration` do not.
"""
from tidewater.hosts.pygem.glaciers import (LABEL, START_MONTH,  # noqa: F401
                                            by_glacier, cell_geometry,
                                            front_cell, front_geometry,
                                            glacier_of, glacno,
                                            laws_from_forcing, rgi_id,
                                            yearly_factors)
