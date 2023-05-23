# This code is part of Qiskit.
#
# (C) Copyright IBM 2022.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

# pylint: disable=invalid-name

"""
Sphinx documentation builder
"""

# General options:
from pathlib import Path
import sys

from importlib_metadata import version as metadata_version

project = "Circuit Knitting Toolbox"
copyright = "2023"  # pylint: disable=redefined-builtin
author = "IBM Quantum"

_rootdir = Path(__file__).parent.parent
sys.path.insert(0, str(_rootdir))

# The full version, including alpha/beta/rc tags
release = metadata_version("circuit_knitting_toolbox")
# The short X.Y version
version = ".".join(release.split(".")[:2])

extensions = [
    "sphinx.ext.napoleon",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.mathjax",
    "sphinx.ext.viewcode",
    "sphinx.ext.extlinks",
    # "sphinx.ext.autosectionlabel",
    "jupyter_sphinx",
    "sphinx_autodoc_typehints",
    "reno.sphinxext",
    "nbsphinx",
    "sphinx_copybutton",
]
templates_path = ["_templates"]
numfig = True
numfig_format = {"table": "Table %s"}
language = "en"
pygments_style = "colorful"
add_module_names = False
modindex_common_prefix = ["circuit_knitting_toolbox."]
html_css_files = ["style.css"]

# html theme options
html_static_path = ["_static"]
# html_logo = "_static/images/logo.png"

# autodoc/autosummary options
autosummary_generate = True
autosummary_generate_overwrite = False
autoclass_content = "both"

# nbsphinx options (for tutorials)
nbsphinx_timeout = 180
nbsphinx_execute = "auto"
nbsphinx_widgets_path = ""
exclude_patterns = [
    "_build",
    "**.ipynb_checkpoints",
    "test_notebooks",
    "**/README.rst",
]

# Enable nitpicky mode, and read from nitpick-exceptions file
nitpicky = True
nitpick_ignore = []
with open(Path(__file__).parent / "nitpick-exceptions") as f:
    for line in f:
        line = line.strip()
        if line and not line.startswith("#"):
            nitpick_ignore.append(tuple(line.split(None, 1)))
