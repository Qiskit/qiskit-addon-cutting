# This code is a Qiskit project.

# (C) Copyright IBM 2022.

# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""File containing the entanglement forging operator."""

from dataclasses import dataclass
from typing import Sequence

import numpy as np
from qiskit.quantum_info import Pauli


@dataclass
class EntanglementForgingOperator:  # noqa: D301
    r"""Operator class for Entanglement Forging.

    A class that contains the :math:`2N` qubit Pauli operator :math:`\hat{O} = \sum_{i, j} w_{i, j} \hat{T}_{i, j} \otimes \sum_{a, b} \hat{S}_{a, b}`
    and associated weights. These operators are knitted by the :class:`EntanglementForgingKnitter` to provide esimates of the
    energy for the :class:`EntanglementForgingVQE`.
    """

    def __init__(
        self,
        tensor_paulis: Sequence[Pauli],
        superposition_paulis: Sequence[Pauli],
        w_ij: np.ndarray,
        w_ab: np.ndarray,
    ):
        r"""
        Assign the necessary member variables.

        Args:
            - tensor_paulis: The operators acting on the subsystems that have the
                same Schmidt coefficients
            - superposition_paulis: The operators acting on subsystems that have
                different Schmidt coefficients
            - w_ij: The weight matrix associated with the tensor paulis
            - w_ab: The weight matrix associated with the superposition paulis

        Returns:
            None
        """
        self.tensor_paulis = tensor_paulis
        self.superposition_paulis = superposition_paulis
        self.w_ij = w_ij
        self.w_ab = w_ab

    def __repr__(self) -> str:
        """
        Representation function for EntanglementForgingOperator.

        Returns:
            Printable repesentation of class
        """
        repr = "EntanglementForgingOperator\nTensor Paulis:\n"
        repr += str(self.tensor_paulis)
        repr += "\nSuperposition Paulis:\n"
        repr += str(self.superposition_paulis)
        repr += "\nTensor Weight Matrix:\n"
        repr += np.array_str(self.w_ij, precision=4, suppress_small=True)
        repr += "\nSuperposition Weight Matrix:\n"
        repr += np.array_str(self.w_ab, precision=4, suppress_small=True)
        return repr
