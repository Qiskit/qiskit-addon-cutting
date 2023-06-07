# This code is a Qiskit project.

# (C) Copyright IBM 2022.

# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""
Helper functions for reducing orbitals.

.. currentmodule:: circuit_knitting.utils.orbital_reduction

.. autosummary::
   :toctree: ../stubs

   reduce_bitstrings
"""

import numpy as np


def reduce_bitstrings(bitstrings, orbitals_to_reduce):
    """
    Eliminates the specified orbitals in the bitstrings.

    This is achieved by simply deleting those orbitals from the
    elements of the bitstrings.

    Example:
    >>> reduce_bitstrings([[1, 0, 0, 1, 0], [1, 0, 0, 0, 1], [1, 1, 1, 1, 0]], [0, 1])
    [(0, 1, 0), (0, 0, 1), (1, 1, 0)]

    Args:
        bitstrings: The list of bitstrings to be reduced
        orbitals_to_reduce: The positions/orbitals to remove from the bitstrings

    Returns:
        The list of reduced bitstrings

    """
    reduced_bitstrings_list = np.delete(
        bitstrings, orbitals_to_reduce, axis=-1
    ).tolist()
    return [tuple(bs) for bs in reduced_bitstrings_list]
