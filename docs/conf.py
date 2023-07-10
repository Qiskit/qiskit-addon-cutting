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
    "sphinx_reredirects",
    "qiskit_sphinx_theme",
]
templates_path = ["_templates"]
numfig = True
numfig_format = {"table": "Table %s"}
language = "en"
pygments_style = "colorful"
add_module_names = False
modindex_common_prefix = ["circuit_knitting_toolbox."]

html_theme = "qiskit"

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

# Redirects for pages that have moved
redirects = {
    "circuit_cutting/tutorials/gate_cutting_to_reduce_circuit_width.html": "01_gate_cutting_to_reduce_circuit_width.html",
    "circuit_cutting/tutorials/gate_cutting_to_reduce_circuit_depth.html": "02_gate_cutting_to_reduce_circuit_depth.html",
}
