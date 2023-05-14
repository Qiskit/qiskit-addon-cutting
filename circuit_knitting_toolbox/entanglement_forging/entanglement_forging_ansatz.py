# This code is a Qiskit project.

# (C) Copyright IBM 2022.

# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""File containing the entanglement forging ansatz class."""

from __future__ import annotations

from qiskit import QuantumCircuit


class EntanglementForgingAnsatz:
    """Class to hold features of the ansatz for Entanglement Forging.

    Entanglement Forging is based on a Schmidt decomposition of a 2N-qubit
    wavefunction into two N-qubit subsystems. These subsystems are described by a
    unitary operator U acting on a set of computational basis states. The unitary
    can be represented by a parametrized quantum circuit whose parameters are
    optimized using a VQE. The computational basis states of each subsystem that
    contribute to the Schmidt decomposition can be specified by a list of
    bitstrings. This list can be chosen differently for each subsystem.

    For a 2N-qubit operator, the circuit should act on N qubits and the bitstrings
    should be of length N. This class functions as a container for information about
    the circuit and bitstrings, and is required for the EntanglementForgingGroundStateSolver.
    """

    def __init__(
        self,
        circuit_u: QuantumCircuit,
        bitstrings_u: list[tuple[int, ...]],
        bitstrings_v: list[tuple[int, ...]] | None = None,
    ):
        """
        Assign the necessary member variables and check for shaping errors.

        Args:
            - circuit_u: the parameterized circuit that is optimized to
                find the minimum energy of the original problem. It represents the
                unitary U for both N-qubit subsystems in the Schmidt decomposition.
            - bitstrings_u: the input bitstrings for each N-qubit
                subsystem. The bitstrings represent the computational basis states
                contributing to the Schmidt decomposition. List must contain less than
                or equal to 2^N elements and each bitstring must have length N. These
                bitstrings are used for each subsystem unless bitstrings_v is provided.
            - bitstrings_v: specifies the bitstrings to be
                used for the second subsystem in the Schmidt decomposition. Must be the
                same shape as bitstrings_u. If not provided, then bitstrings_u is used
                for both subsystems.

        Returns:
            None

        Raises:
            - ValueError: The input bitstrings are of incorrect shapes.
        """
        if any(len(bitstring) != circuit_u.num_qubits for bitstring in bitstrings_u):
            raise ValueError(
                "Length of every U bitstring must be the same as the number of qubits in the ansatz."
            )

        if bitstrings_v and bitstrings_v != bitstrings_u:
            if any(
                len(bitstring) != circuit_u.num_qubits for bitstring in bitstrings_v
            ):
                raise ValueError(
                    "Length of every V bitstring must be the same as the number of qubits in the ansatz."
                )

            if len(bitstrings_u) != len(bitstrings_v):
                raise ValueError(
                    "There must be the same number of V bitstrings as U bitstrings."
                )

        self._circuit_u: QuantumCircuit = circuit_u
        self._bitstrings_u: list[tuple[int, ...]] = bitstrings_u
        self._bitstrings_v: list[tuple[int, ...]] = bitstrings_v or bitstrings_u

    @property
    def circuit_u(self) -> QuantumCircuit:
        """Property function for the circuit."""
        return self._circuit_u

    @property
    def bitstrings_u(self) -> list[tuple[int, ...]]:
        """Property function for the first bitstrings."""
        return self._bitstrings_u

    @property
    def bitstrings_v(self) -> list[tuple[int, ...]]:
        """Property function for the second bitstrings."""
        return self._bitstrings_v

    @property
    def bitstrings_are_symmetric(self) -> bool:
        """Property function for the symmetry of bitstrings."""
        return self._bitstrings_v == self._bitstrings_u

    @property
    def subspace_dimension(self) -> int:
        """Property function for the length of bitstrings."""
        return len(self._bitstrings_u)

    def __repr__(self) -> str:
        """Representation function for EntanglementForgingAnsatz."""
        repr = "EntanglementForgingAnsatz\nCircuit:\n"
        repr += str(self._circuit_u.draw())
        repr += "\nBitstrings U:\n"
        repr += str(self.bitstrings_u)
        repr += "\nBitstrings V:\n"
        repr += str(self._bitstrings_v)
        repr += f"\nBitstring are symmetric: {self.bitstrings_are_symmetric}\n"
        repr += f"Subspace dimension: {self.subspace_dimension}"
        return repr
