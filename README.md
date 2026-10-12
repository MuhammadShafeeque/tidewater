# tidewater

**Ocean-forced frontal ablation for glacier models.**

[![PyPI](https://img.shields.io/pypi/v/tidewater.svg)](https://pypi.org/project/tidewater/)
[![Documentation](https://readthedocs.org/projects/tidewater/badge/?version=latest)](https://tidewater.readthedocs.io)
[![Tests](https://github.com/MuhammadShafeeque/tidewater/actions/workflows/tests.yml/badge.svg)](https://github.com/MuhammadShafeeque/tidewater/actions/workflows/tests.yml)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.23296214.svg)](https://doi.org/10.5281/zenodo.23296214)
[![License: BSD-3-Clause](https://img.shields.io/badge/license-BSD--3--Clause-blue.svg)](LICENSE)
[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue.svg)](pyproject.toml)

<!-- docs: intro -->
`tidewater` makes the frontal ablation of a marine-terminating glacier depend
on the ocean in front of it. It turns an ocean product into a thermal forcing
at each calving front, turns that forcing into a frontal ablation flux or a
dimensionless ocean factor, and calibrates the calving constant against
observed frontal ablation. It is a library for glacier models, not a glacier
model: the host model keeps its geometry, its dynamics and its mass budget.

- **Host-neutral.** The core imports no glacier model. Adapters for
  [OGGM](https://github.com/OGGM/oggm),
  [PyGEM](https://github.com/PyGEM-Community/PyGEM) and
  [GloGEM](https://github.com/eth-vaw-glaciology/GloGEM) are included, and any
  other model can couple through a plain-text table or the written spec.
- **Global.** No region, hemisphere, ocean product, bed product, observation
  dataset, glacier inventory version or year convention is built in. They are
  arguments.
- **Reproducible across languages.** The laws come with a specification and
  678 reference vectors, so an implementation in IDL, MATLAB or Fortran can be
  checked against the Python one.

## Contents

- [What it does](#what-it-does)
- [Supported glacier models](#supported-glacier-models)
- [Installation](#installation)
- [Quick start](#quick-start)
- [The laws](#the-laws)
- [Host integration contract](#host-integration-contract)
- [Bringing your own data](#bringing-your-own-data)
- [Documentation](#documentation)
- [Package layout](#package-layout)
- [Testing](#testing)
- [Project status](#project-status)
- [Contributing](#contributing)
- [Citing](#citing)
- [References](#references)
- [Licence and acknowledgements](#licence-and-acknowledgements)

<!-- docs: overview -->
## What it does

![From an ocean product to the host model: footprint, depth range, thermal
forcing, bias correction, ocean factor, law and host](docs/img/overview.svg)

| Step | What you get | Module |
|---|---|---|
| Ocean forcing | The thermal forcing of an ocean product in depth bands in front of each terminus, with sea-ice cover, optional subglacial discharge and an optional bias offset onto a reference product | `tidewater.forcing` |
| Frontal ablation | The flux of one of five nested laws, or the ocean factor `(max(TF, 0) / TF_ref)^gamma` that scales a host's own calving constant | `tidewater.laws` |
| Calibration | The calving constant `k` from observed frontal ablation, on the inverted front and through the host's dynamic model, and a constant for glaciers without an observation | `tidewater.calibration` |
| Hand-over to the host | Model years in the host's convention, glacier ids in the host's form, and files the host can read | `tidewater.timeconv`, `tidewater.ids`, `tidewater.export` |

## Supported glacier models

| Model | Language | How it couples | What the host needs |
|---|---|---|---|
| [OGGM](https://github.com/OGGM/oggm) | Python | `tidewater.hosts.oggm`: the laws are called at every time step of the flowline model; ocean-aware inversion and dynamic calibration of `k` are OGGM tasks | the branch [`oggm-tidewater`](https://github.com/MuhammadShafeeque/oggmHydro/tree/oggm-tidewater), which adds four small hooks to OGGM |
| [PyGEM](https://github.com/PyGEM-Community/PyGEM) | Python | `tidewater.hosts.pygem`: a law on PyGEM's OGGM dynamics, or a yearly constant on its mass-redistribution model | released PyGEM 1.1.0, or the branch [`pygem-tidewater`](https://github.com/MuhammadShafeeque/PyGEM-tidewater/tree/pygem-tidewater) with a calving-law hook; both give the same results. The branch [`pygem-tidewater-rgi7`](https://github.com/MuhammadShafeeque/PyGEM-tidewater/tree/pygem-tidewater-rgi7) adds RGI 7.0 to it |
| [GloGEM](https://github.com/eth-vaw-glaciology/GloGEM), GloGEMflow | IDL | `tidewater.hosts.glogem`: yearly ocean factor tables that scale GloGEM's own frontal ablation | branch [`glogem-tidewater`](https://github.com/MuhammadShafeeque/GloGEM-tidewater/tree/glogem-tidewater) (a calving factor hook, off by default) and the IDL files shipped with this package |
| Any other model | any | a factor table, a reimplementation from the spec, or an offline diagnostic | see [docs/hosts.md](docs/hosts.md) |

The host branches carry generic hooks only. They contain no ocean code and do
not import `tidewater`, so a user who does not install this package gets the
host model's own results.

<!-- docs: getting-started -->
## Installation

`tidewater` needs Python 3.11 or later:

```bash
pip install tidewater            # core
pip install "tidewater[teos10]"  # adds the TEOS-10 freezing point (gsw)
pip install "tidewater[plot]"    # adds matplotlib, for the examples' figures
```

Core dependencies: numpy, pandas, xarray, netCDF4, cftime, scipy.

The examples and the tests are in the repository, not in the installed
package. To run them or to change the code, install from a clone:

```bash
git clone https://github.com/MuhammadShafeeque/tidewater.git
pip install -e "tidewater[test,plot]"
```

A host model is installed separately, into the same environment. For OGGM:

```bash
git clone -b oggm-tidewater https://github.com/MuhammadShafeeque/oggmHydro.git
pip install -e "oggmHydro[full]"
pip install tidewater
```

Released OGGM 1.6.3 and earlier are not supported by the OGGM adapter. PyGEM
pins its own OGGM and numpy versions, so give PyGEM an environment of its own.

## Quick start

### On its own

No glacier model is needed to compute a forcing, a factor or a flux:

```python
import numpy as np
import pandas as pd
from tidewater.laws import TFPower, tf_power_mean
from tidewater.timeconv import yearly_ocean_factor

time = pd.date_range("1995-01-01", "2014-12-01", freq="MS")
tf = pd.Series(3.0 + np.sin(2 * np.pi * time.month / 12), index=time)  # degC
tf_ref = tf_power_mean(tf.to_numpy(), gamma=1.18)                        # degC

# a host with its own calving law: one factor per glacier and model year
factor = yearly_ocean_factor(tf.to_frame("RGI60-05.00001"), tf_ref, start_month=10,
                             incomplete="drop")

# a host that calls the law at each time step
years = time.year + (time.month - 0.5) / 12
law = TFPower(years, tf.to_numpy(), tf_ref=tf_ref, k0=0.6)                # k0 in yr-1
q = law.flux(h=250.0, d=180.0, w=1500.0, yr=2001.5)                       # m3 ice s-1
```

### With OGGM

```python
from oggm import cfg
import tidewater.hosts.oggm
from tidewater.hosts.oggm import tasks as tw_tasks

cfg.initialize()
tidewater.hosts.oggm.register()   # files, parameters, calving inversion
# ... glacier directories, then for example
# tw_tasks.process_ocean_data(gdir, ...)
# tw_tasks.run_with_ocean_forcing(gdir, calving_law='tf_power', ...)
# tw_tasks.run_dynamic_calving_k_calibration(gdir, ref_fa=..., ref_fa_err=...)
```

Call `register()` once, before the glacier directories are opened. The tasks
and the settings they read are listed in [docs/hosts.md](docs/hosts.md#oggm).

### With PyGEM

```python
import sys
from pygem.bin.run import run_simulation
from tidewater.hosts.pygem import laws_from_forcing
from tidewater.hosts.pygem.dynamics import ocean_dynamics

laws = laws_from_forcing({"1.03622": "ocean/RGI60-01.03622.nc"}, "tf_power")
sys.argv = ["run_simulation", "-rgi_glac_number", "1.03622",
            "-option_dynamics", "OGGM"]
with ocean_dynamics(laws):
    run_simulation.main()
```

Glaciers without an entry, and land-terminating glaciers, run as before.
Details, the two routes and the calibration tables are in
[docs/hosts-pygem.md](docs/hosts-pygem.md).

### With GloGEM

```python
from tidewater.hosts import glogem

factor, info = glogem.write_ocean_factor_tables(
    "ocean_factor", tf_monthly, tf_ref, gamma=1.18,
    time_resolution="monthly", incomplete="drop")
```

This writes one `ocean_factor_<region>.dat` per RGI region in GloGEM's model
years. GloGEM, on the branch `glogem-tidewater`, is then run as usual with
three more lines in its `config.pro`:

```
!PATH = !PATH + ':<tidewater>/hosts/glogem/idl'
calving_factor_function = 'tw_table_factor'
calving_factor_dir = 'ocean_factor'
```

GloGEM keeps its own calving law and calibrated constant and multiplies its
frontal ablation by the factor of each glacier and year. Without these lines
it gives its stock results. The model-year rules and a note for MATLAB hosts
are in [docs/hosts-glogem.md](docs/hosts-glogem.md).

## The laws

Five laws form a nested family around the constant-k law `q = k d h w`
(`d` water depth, `h` ice thickness, `w` width at the front):

| Name | Law |
|---|---|
| `constant` | `q = k d h w` |
| `tf_power` | `k(t) = k0 ((1 - a) + a (TF(t) / TF_ref)^gamma)` |
| `melt_calving` | a submarine melt rate after [Rignot et al. (2016)](docs/references.md#references) added to the calving term, `q = w d (k_c h + lambda m)` |
| `sea_ice` | `melt_calving` with the melt term multiplied by the open-water fraction to the power `delta` |
| `larger_of` | the larger of the two speeds, `q = w d max(k_c h, lambda m)`, after [Malles et al. (2025)](docs/references.md#references) |

Each law reduces to the one before it for a value of its parameters
(`TF = TF_ref`, `gamma = 0` or `a = 0`; `lambda = 0`; `delta = 0`), and
`larger_of` reduces to the constant-k law at `lambda = 0`. A host can therefore
switch the ocean on gradually and compare each step with its present results.
At `gamma = 0`, `a = 0` or `TF = TF_ref`, `tf_power` gives the same
floating-point number as the constant-k law, so a host that calls the law at
each time step runs exactly as it does with its own law; the other reductions
hold to rounding.
The thermal forcing is clipped at zero before any power. The freezing point is
linear in salinity and depth by default; the TEOS-10 form is available with
the `teos10` extra.

The equations, their limits and the reference vectors are in
[`src/tidewater/spec/SPEC.md`](src/tidewater/spec/SPEC.md).

## Host integration contract

A host model supplies, at the calving front and at each of its time steps:

| The host supplies | Unit |
|---|---|
| ice thickness at the terminus | m |
| water depth at the terminus | m |
| width of the front | m |
| time, on the host's own model-year convention | float years |
| glacier id | string, any inventory |
| terminus longitude and latitude | degrees east, degrees north |

It gets back one of:

| The host receives | Unit |
|---|---|
| frontal ablation flux | m3 ice s-1, or m3 ice yr-1 |
| the dimensionless ocean factor, to multiply its own calving constant | 1 |

Three points complete the contract:

- **One front per glacier.** A host whose glacier has several marine fronts
  calls the law once per front and adds the fluxes.
- **The yearly factor is a mean of monthly factors** over the model year
  (`tidewater.timeconv.yearly_ocean_factor`), not the factor of the yearly
  mean thermal forcing.
- **The host keeps the glacier.** Capping the flux by the available ice,
  removing the mass, moving the front and the accounting of the removed volume
  stay with the host. `tidewater` never changes the state of a host glacier
  outside a host adapter.

The coupling routes for Python hosts and for hosts that cannot import Python
are compared in [docs/hosts.md](docs/hosts.md).

## Bringing your own data

Ocean products on regular, curvilinear and unstructured grids are read into
one neutral layout; variable names, units and depth direction are declared or
taken from CF metadata. Per-glacier forcing files are built in Python or from
the command line:

```bash
tidewater-forcing build ocean_*.nc --glaciers glaciers.csv --out forcing/
tidewater-forcing validate forcing/
```

`glaciers.csv` holds a glacier id and the terminus longitude and latitude,
with an optional front depth. Bed products and frontal-ablation datasets are
brought in the same way, through a column mapping and a stated unit. See
[docs/data.md](docs/data.md), and [docs/regions.md](docs/regions.md) for the
parameters to set per region: the defaults are the values of the Greenland
application the package was developed on, and each is marked.

## Documentation

| Document | Content |
|---|---|
| [docs/how-it-works.md](docs/how-it-works.md) | the method in six diagrams: forcing, the front, the laws, the contract, the coupling routes, time |
| [docs/examples.md](docs/examples.md) and [examples/](examples/) | one glacier through the whole chain, an example per host model, and its figures |
| [docs/hosts.md](docs/hosts.md) | coupling a host model: the routes, how to choose one, and the OGGM adapter |
| [docs/hosts-pygem.md](docs/hosts-pygem.md) | the PyGEM adapter: two routes, the two dynamics models, calibration |
| [docs/hosts-glogem.md](docs/hosts-glogem.md) | GloGEM and GloGEMflow: factor tables, model years, the IDL files |
| [docs/data.md](docs/data.md) | bringing an ocean product, a bed product or a frontal-ablation dataset |
| [docs/regions.md](docs/regions.md) | parameters with a regional origin and how to set them elsewhere |
| [docs/conventions.md](docs/conventions.md) | units, periods, model years, glacier ids and file formats |
| [docs/references.md](docs/references.md) | the source of every equation and constant in the code, with full references |
| [src/tidewater/spec/SPEC.md](src/tidewater/spec/SPEC.md) | the laws as equations, with reference vectors for other languages |
| [CHANGELOG.md](CHANGELOG.md) | what changed in each release |

The same pages, with an API reference generated from the docstrings, are
online at [tidewater.readthedocs.io](https://tidewater.readthedocs.io). To
build the site locally:

```bash
pip install -e ".[docs]"
sphinx-build -W docs docs/_build/html
```

## Package layout

| Layer | Module | What it does |
|---|---|---|
| Physics | `tidewater.physics` | freezing point, thermal forcing, depth-band averages, open-water fraction, flotation and front geometry |
| Laws | `tidewater.laws` | the frontal ablation laws and the dimensionless ocean factor |
| Forcing | `tidewater.forcing` | from an ocean product to a per-glacier forcing record |
| Calibration | `tidewater.calibration` | the calving constant against observed frontal ablation |
| Time, ids, export | `tidewater.timeconv`, `tidewater.ids`, `tidewater.export` | host model years, glacier ids, files for hosts |
| Parameters, units, periods | `tidewater.config`, `tidewater.units`, `tidewater.periods` | every parameter as an argument, one unit converter, one period convention |
| Host adapters | `tidewater.hosts` | the only place that imports a glacier model |
| Spec | `tidewater.spec` | reference vectors for hosts that reimplement the law in another language |

Outside the package: `examples/` (the example glacier, a script per host and
the result files the figures read) and `docs/figures/` (the scripts that draw
every figure of the documentation into `docs/img/`).

<!-- docs: project -->
## Testing

```bash
pip install -e ".[test]"
pytest
```

The core tests need no glacier model and no download. The host tests under
`tests/hosts/` run when their host is installed and skip otherwise, each with
the reason:

- **OGGM**: installed from the branch above. The tests take OGGM's options
  (`pytest --run-slow`). Run them with `HOME` set to a scratch directory,
  because OGGM's test helpers delete other directories under their test
  directory.
- **PyGEM**: released PyGEM or the branch; the test ids state which route ran.
- **GloGEM**: the IDL files run when `idl` or `gdl` is on the path, or when
  the environment variable `TIDEWATER_IDL` names an interpreter. `GLOGEM_DIR`,
  a GloGEM checkout with the hook, adds a test of GloGEM's calving routine.

## Project status

From version 1.0 on, names and signatures change only in a major release.

| Part | State |
|---|---|
| Core (laws, forcing, calibration, export) | tested on synthetic data; the laws are checked against the reference vectors |
| OGGM adapter | reproduces, number for number, the scheme's original implementation on OGGM for two glaciers in the Greenland periphery, including the dynamic calibration of `k` |
| PyGEM adapter | reproduces PyGEM's own frontal ablation for one sample glacier when the ocean factor is 1, on both of PyGEM's dynamics options |
| GloGEM, GloGEMflow | the IDL code matches the reference vectors and scales GloGEM's calving routine by exactly the tabled factor; GloGEM runs in full, in IDL; GloGEMflow has not been run |

All three hosts run the example glacier, RGI2000-v7.0-G-05-09345, with and
without the ocean term: calibration on 2000-2009, evaluation on 2010-2019 and
two scenarios to 2100. The results and their figures are in
[docs/examples.md](docs/examples.md).

Known limits of this version: one calving front per glacier; runs on several
processes with PyGEM need forked workers; in GloGEMflow the frontal ablation
does not yet change the flow model's geometry. The scheme is experimental,
like the calving parameterizations it extends.

## Contributing

Bug reports, questions and pull requests are welcome through the issue
tracker of this repository. Three rules keep the package portable:

- nothing outside `src/tidewater/hosts/` imports a glacier model (a test
  enforces it);
- every parameter is an argument with a documented default, never a global;
- a change to a law comes with updated reference vectors
  (`python -m tidewater.spec.make_vectors`) and a test.

A change that a user can notice gets a line under "Unreleased" in
`CHANGELOG.md`.

A release is two steps: the "Unreleased" heading of `CHANGELOG.md` becomes the
version and its date, and a GitHub release is published with the tag
`vX.Y.Z`. The package version, the upload to PyPI, the documentation and the
Zenodo archive follow from the tag.

A new host adapter goes in `src/tidewater/hosts/<host>/` with its tests in
`tests/hosts/<host>/` and a page in `docs/`.

## Citing

Please cite the software using [CITATION.cff](CITATION.cff). The releases are
archived on Zenodo: the DOI
[10.5281/zenodo.23296214](https://doi.org/10.5281/zenodo.23296214) stands for
all versions and resolves to the latest one, and each release has a DOI of its
own on that page.

## References

The works this package builds on are in one list, the
[bibliography of the documentation](docs/references.md#references); the same
page gives the source of every equation and constant in the code.

## Licence and acknowledgements

BSD-3-Clause, see [LICENSE](LICENSE).

`tidewater` builds on the k-calving parameterization that OGGM, PyGEM and
GloGEM each provide. It is an independent project and is not developed or
endorsed by the teams of those models.
