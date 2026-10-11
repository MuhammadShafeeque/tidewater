"""Ocean-forced frontal ablation for marine-terminating glaciers.

Submodules are imported on demand, for example ``import tidewater.timeconv``.
"""

from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _version

try:
    __version__ = _version("tidewater")
except PackageNotFoundError:  # run from a source tree that is not installed
    __version__ = "unknown"
