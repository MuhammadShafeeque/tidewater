# Changelog

All notable changes to `tidewater` are listed here, newest first. The format
follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the
version numbers follow [Semantic Versioning](https://semver.org/spec/v2.0.0.html).
While the version is below 1.0, a minor release may change names and
signatures; such a change is listed under "Changed" or "Removed".

A pull request that changes what a user sees adds a line under "Unreleased".
At a release that heading becomes the version and its date.

## [Unreleased]

## [1.0.1] - 2026-10-12

Version 1.0.0 was tagged and archived on Zenodo but not published on PyPI;
this release takes its place, with the same code.

### Fixed

- `tf_power` with `gamma = 0` or `a = 0` now returns the same floating-point
  number as the constant-k law, as a host's own `k * d * h * w` does. It
  formed `(k * h) * d * w`, which differs in the last bit in about a third of
  the calls; a flowline model can carry such a difference into its geometry,
  so a `gamma = 0` run was close to the host's own run but not equal to it.
  The law now forms its constant `k(t)` first and multiplies in the order of
  the constant-k law, in Python and in the IDL code for GloGEM. Other values
  of `gamma` and `a` change at the level of rounding. A subclass of `TFPower`
  that overrides `frontal_speed_components` keeps the previous product.
- `spec/SPEC.md` gives this order for `tf_power`, and
  `spec/reference_vectors.csv` is regenerated from it: 4 of its 40 `tf_power`
  vectors change in the last digit.

## [0.3.0] - 2026-10-11

Version 0.2.0 was tagged and archived on Zenodo but not published on PyPI;
this release takes its place.

### Added

- `examples/fetch_data.py` fetches the results and the inputs of the example
  glacier from the data home and verifies each file against the checksums in
  `examples/data_files.csv`.
- The Zenodo DOI [10.5281/zenodo.23296214](https://doi.org/10.5281/zenodo.23296214),
  which stands for all versions, in `CITATION.cff`, the README and the
  documentation.
- Issue forms for bug reports, feature requests and questions, and a pull
  request template.
- This changelog, also a page of the documentation.

### Changed

- The version number comes from the release tag: `pyproject.toml` and
  `tidewater.__version__` no longer carry it. An install from a commit between
  releases reports a development version that names the commit.
- `CITATION.cff` and the README no longer state a version number; the version
  of an archived release is on its Zenodo page.
- The README points to the bibliography of the documentation instead of
  carrying a reference list of its own.

## [0.1.0] - 2026-10-11

The first release.

### Added

- Ocean forcing (`tidewater.forcing`): the thermal forcing of an ocean product
  in depth bands in front of each terminus, with sea-ice cover, optional
  subglacial discharge and an optional bias offset onto a reference product;
  the command `tidewater-forcing`.
- Five nested frontal ablation laws (`tidewater.laws`): `constant`,
  `tf_power`, `melt_calving`, `sea_ice` and `larger_of`, and the dimensionless
  ocean factor that scales a host's own calving constant.
- Calibration of the calving constant against observed frontal ablation
  (`tidewater.calibration`), on the inverted front and through the host's
  dynamic model, with a constant for glaciers without an observation.
- Hand-over to a host (`tidewater.timeconv`, `tidewater.ids`,
  `tidewater.export`): model years, glacier ids and files in the host's form.
- Host adapters for OGGM, PyGEM and GloGEM (`tidewater.hosts`), the last with
  the IDL files that read the factor tables.
- The specification of the laws with 678 reference vectors
  (`tidewater.spec`), for implementations in other languages.
- The documentation at [tidewater.readthedocs.io](https://tidewater.readthedocs.io)
  and the examples: one glacier through the whole chain on synthetic inputs
  and in each of the three hosts.

[Unreleased]: https://github.com/MuhammadShafeeque/tidewater/compare/v1.0.1...HEAD
[1.0.1]: https://github.com/MuhammadShafeeque/tidewater/releases/tag/v1.0.1
[0.3.0]: https://github.com/MuhammadShafeeque/tidewater/releases/tag/v0.3.0
[0.1.0]: https://github.com/MuhammadShafeeque/tidewater/releases/tag/v0.1.0
