# This code is a Qiskit project.

# (C) Copyright IBM 2022.

# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Main Circuit Knitting Toolbox public functionality."""

import sys
import warnings


if sys.version_info < (3, 8):
    warnings.warn(
        "Using the Circuit Knitting Toolbox with Python 3.7 is deprecated."
        "Support for Python 3.7 will be removed in the near future, as soon as "
        "https://github.com/Qiskit-Extensions/circuit-knitting-toolbox/pull/80 "
        "is merged.",
        DeprecationWarning,
    )
