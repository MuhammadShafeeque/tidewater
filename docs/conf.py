"""Sphinx configuration of the tidewater documentation."""
from importlib.metadata import version as _version

project = "tidewater"
author = "Muhammad Shafeeque and tidewater contributors"
copyright = f"2026, {author}"
release = _version("tidewater")
version = release

extensions = [
    "myst_parser",
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
]
exclude_patterns = ["_build", "figures"]

myst_heading_anchors = 3

# the host models are not installed for the documentation build
autodoc_mock_imports = ["oggm", "pygem", "configobj", "gsw"]
autodoc_member_order = "bysource"
autodoc_default_options = {"members": True, "show-inheritance": True}
napoleon_google_docstring = False

html_theme = "furo"
html_title = f"tidewater {release}"
