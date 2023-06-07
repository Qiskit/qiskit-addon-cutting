# This code is a Qiskit project.

# (C) Copyright IBM 2023.

# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Deprecated import location ``circuit_knitting_toolbox``."""

import sys
from warnings import warn

from circuit_knitting import utils

sys.modules["circuit_knitting_toolbox.utils"] = utils

warn(
    f"The package namespace {__name__} is deprecated and will be removed "
    "no sooner than Circuit Knitting Toolbox 0.4.0. Use namespace "
    "circuit_knitting instead.",
    DeprecationWarning,
    stacklevel=2,
)
