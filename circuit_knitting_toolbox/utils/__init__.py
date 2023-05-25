# This code is a Qiskit project.

# (C) Copyright IBM 2022.

# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""
Utility functions.

.. automodule:: circuit_knitting_toolbox.utils.bitwise
.. automodule:: circuit_knitting_toolbox.utils.conversion
.. automodule:: circuit_knitting_toolbox.utils.iteration
.. automodule:: circuit_knitting_toolbox.utils.metrics
.. automodule:: circuit_knitting_toolbox.utils.observable_grouping
.. automodule:: circuit_knitting_toolbox.utils.orbital_reduction
.. automodule:: circuit_knitting_toolbox.utils.simulation
.. automodule:: circuit_knitting_toolbox.utils.transforms
"""

from .orbital_reduction import reduce_bitstrings

__all__ = [
    "reduce_bitstrings",
]
