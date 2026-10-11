"""OGGM's test options (``--run-slow`` ...) and fixtures, when OGGM is installed."""
from importlib.util import find_spec

if find_spec('oggm') is not None:
    pytest_plugins = ['oggm.tests.conftest']
