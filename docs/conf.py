"""
Sphinx documentation configuration for llm_std_lib.

See https://www.sphinx-doc.org/en/master/usage/configuration.html for the
full list of available configuration values.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

# -- Project information -------------------------------------------------------

project = "llm-std-lib"
author = "llm-std-lib contributors"
release = "1.0.0"

# -- General configuration -----------------------------------------------------

extensions: list[str] = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.intersphinx",
    "sphinx.ext.autosummary",
]

templates_path: list[str] = ["_templates"]
exclude_patterns: list[str] = ["_build", "Thumbs.db", ".DS_Store"]

autosummary_generate = True

napoleon_google_docstring = True
napoleon_numpy_docstring = False
napoleon_include_init_with_doc = True

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "pydantic": ("https://docs.pydantic.dev/latest", None),
}

# -- Options for HTML output ---------------------------------------------------

html_theme = "furo"
html_static_path: list[str] = ["_static"]
html_title = "llm-std-lib"
