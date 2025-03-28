# This code is a Qiskit project.
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

"""Sphinx documentation builder."""

# General options:
import inspect
from pathlib import Path
import os
import re
import sys

import qiskit_addon_cutting

project = "Qiskit addon: circuit cutting"
copyright = "2024"  # pylint: disable=redefined-builtin
author = "Qiskit addons team"

_rootdir = Path(__file__).parent.parent
sys.path.insert(0, str(_rootdir))

# The full version, including alpha/beta/rc tags
release = qiskit_addon_cutting.__version__
# The short X.Y version
version = ".".join(release.split(".")[:2])

extensions = [
    "sphinx.ext.napoleon",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.mathjax",
    "sphinx.ext.viewcode",
    "sphinx.ext.extlinks",
    "matplotlib.sphinxext.plot_directive",
    # "sphinx.ext.autosectionlabel",
    "jupyter_sphinx",
    "sphinx_autodoc_typehints",
    "reno.sphinxext",
    "nbsphinx",
    "sphinx_copybutton",
    "sphinx_reredirects",
    "sphinx.ext.intersphinx",
    "qiskit_sphinx_theme",
]
templates_path = ["_templates"]
numfig = False
numfig_format = {"table": "Table %s"}
language = "en"
pygments_style = "colorful"
add_module_names = False
modindex_common_prefix = ["qiskit_addon_cutting."]

html_theme = "qiskit-ecosystem"
html_title = f"{project} {release}"
html_theme_options = {
    "footer_icons": [
        # https://pradyunsg.me/furo/customisation/footer/#using-embedded-svgs
        {
            "name": "GitHub",
            "url": "https://github.com/Qiskit/qiskit-addon-cutting",
            "html": """
                <svg stroke="currentColor" fill="currentColor" stroke-width="0" viewBox="0 0 16 16">
                    <path fill-rule="evenodd" d="M8 0C3.58 0 0 3.58 0 8c0 3.54 2.29 6.53 5.47 7.59.4.07.55-.17.55-.38 0-.19-.01-.82-.01-1.49-2.01.37-2.53-.49-2.69-.94-.09-.23-.48-.94-.82-1.13-.28-.15-.68-.52-.01-.53.63-.01 1.08.58 1.23.82.72 1.21 1.87.87 2.33.66.07-.52.28-.87.51-1.07-1.78-.2-3.64-.89-3.64-3.95 0-.87.31-1.59.82-2.15-.08-.2-.36-1.02.08-2.12 0 0 .67-.21 2.2.82.64-.18 1.32-.27 2-.27.68 0 1.36.09 2 .27 1.53-1.04 2.2-.82 2.2-.82.44 1.1.16 1.92.08 2.12.51.56.82 1.27.82 2.15 0 3.07-1.87 3.75-3.65 3.95.29.25.54.73.54 1.48 0 1.07-.01 1.93-.01 2.2 0 .21.15.46.55.38A8.013 8.013 0 0 0 16 8c0-4.42-3.58-8-8-8z"></path>
                </svg>
            """,
            "class": "",
        },
    ],
    "dark_logo": "images/qiskit-dark-logo.svg",
    "light_logo": "images/qiskit-light-logo.svg",
}
html_static_path = ["_static"]
html_css_files = ["css/custom.css"]

# Options for autodoc. These reflect the values from Qiskit SDK and Runtime.
autosummary_generate = True
autosummary_generate_overwrite = False
autoclass_content = "both"
autodoc_typehints = "description"
autodoc_default_options = {
    "inherited-members": None,
    "show-inheritance": True,
}
napoleon_google_docstring = True
napoleon_numpy_docstring = False

# nbsphinx options (for tutorials)
nbsphinx_timeout = 180
nbsphinx_execute = "always" if os.environ.get("CI") == "true" else "auto"
nbsphinx_widgets_path = ""
exclude_patterns = [
    "_build",
    "**.ipynb_checkpoints",
    "test_notebooks",
    "**/README.rst",
]

# matplotlib.sphinxext.plot_directive options
plot_html_show_formats = False
plot_formats = ["svg"]

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "scipy": ("https://docs.scipy.org/doc/scipy/", None),
    "qiskit": ("https://docs.quantum.ibm.com/api/qiskit/", None),
    "qiskit-ibm-runtime": (
        "https://docs.quantum.ibm.com/api/qiskit-ibm-runtime/",
        None,
    ),
    "qiskit-aer": ("https://qiskit.github.io/qiskit-aer/", None),
    "rustworkx": ("https://www.rustworkx.org/", None),
}

# ----------------------------------------------------------------------------------
# Redirects
# ----------------------------------------------------------------------------------

_inlined_apis = [
    ("qiskit_addon_cutting", "cut_wires"),
    ("qiskit_addon_cutting", "expand_observables"),
    ("qiskit_addon_cutting", "partition_circuit_qubits"),
    ("qiskit_addon_cutting", "partition_problem"),
    ("qiskit_addon_cutting", "cut_gates"),
    ("qiskit_addon_cutting", "generate_cutting_experiments"),
    ("qiskit_addon_cutting", "reconstruct_expectation_values"),
    ("qiskit_addon_cutting", "PartitionedCuttingProblem"),
    ("qiskit_addon_cutting", "find_cuts"),
    ("qiskit_addon_cutting", "OptimizationParameters"),
    ("qiskit_addon_cutting", "DeviceConstraints"),
    ("qiskit_addon_cutting.qpd", "WeightType"),
    ("qiskit_addon_cutting.qpd", "generate_qpd_weights"),
    ("qiskit_addon_cutting.qpd", "decompose_qpd_instructions"),
    ("qiskit_addon_cutting.qpd", "qpdbasis_from_instruction"),
    ("qiskit_addon_cutting.utils.bitwise", "bit_count"),
    ("qiskit_addon_cutting.utils.iteration", "unique_by_id"),
    ("qiskit_addon_cutting.utils.iteration", "unique_by_eq"),
    (
        "qiskit_addon_cutting.utils.observable_grouping",
        "observables_restricted_to_subsystem",
    ),
    ("qiskit_addon_cutting.utils.observable_grouping", "CommutingObservableGroup"),
    ("qiskit_addon_cutting.utils.observable_grouping", "ObservableCollection"),
    ("qiskit_addon_cutting.utils.simulation", "simulate_statevector_outcomes"),
    ("qiskit_addon_cutting.utils.simulation", "ExactSampler"),
    ("qiskit_addon_cutting.utils.transforms", "separate_circuit"),
    ("qiskit_addon_cutting.utils.transforms", "SeparatedCircuits"),
]

redirects = {
    f"stubs/{module}.{name}": f"../apidocs/{module}.html#{module}.{name}"
    for module, name in _inlined_apis
}

# ----------------------------------------------------------------------------------
# Source code links
# ----------------------------------------------------------------------------------


def determine_github_branch() -> str:
    """Determine the GitHub branch name to use for source code links.

    We need to decide whether to use `stable/<version>` vs. `main` for dev builds.
    Refer to https://docs.github.com/en/actions/learn-github-actions/variables
    for how we determine this with GitHub Actions.
    """
    # If CI env vars not set, default to `main`. This is relevant for local builds.
    if "GITHUB_REF_NAME" not in os.environ:
        return "main"

    # PR workflows set the branch they're merging into.
    if base_ref := os.environ.get("GITHUB_BASE_REF"):
        return base_ref

    ref_name = os.environ["GITHUB_REF_NAME"]

    # Check if the ref_name is a tag like `1.0.0` or `1.0.0rc1`. If so, we need
    # to transform it to a Git branch like `stable/1.0`.
    version_without_patch = re.match(r"(\d+\.\d+)", ref_name)
    return (
        f"stable/{version_without_patch.group()}" if version_without_patch else ref_name
    )


GITHUB_BRANCH = determine_github_branch()


def linkcode_resolve(domain, info):
    """Resolve link."""
    if domain != "py":
        return None

    module_name = info["module"]
    module = sys.modules.get(module_name)
    if module is None or "qiskit_addon_cutting" not in module_name:
        return None

    def is_valid_code_object(obj):
        return inspect.isclass(obj) or inspect.ismethod(obj) or inspect.isfunction(obj)

    obj = module
    for part in info["fullname"].split("."):
        try:
            obj = getattr(obj, part)
        except AttributeError:
            return None
        if not is_valid_code_object(obj):
            return None

    # Unwrap decorators. This requires they used `functools.wrap()`.
    while hasattr(obj, "__wrapped__"):
        obj = getattr(obj, "__wrapped__")
        if not is_valid_code_object(obj):
            return None

    try:
        full_file_name = inspect.getsourcefile(obj)
    except TypeError:
        return None
    if full_file_name is None or "/qiskit_addon_cutting/" not in full_file_name:
        return None
    file_name = full_file_name.split("/qiskit_addon_cutting/")[-1]

    try:
        source, lineno = inspect.getsourcelines(obj)
    except (OSError, TypeError):
        linespec = ""
    else:
        ending_lineno = lineno + len(source) - 1
        linespec = f"#L{lineno}-L{ending_lineno}"
    return f"https://github.com/Qiskit/qiskit-addon-cutting/tree/{GITHUB_BRANCH}/qiskit_addon_cutting/{file_name}{linespec}"
