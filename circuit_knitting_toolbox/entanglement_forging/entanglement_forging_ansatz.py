# This code is part of Qiskit.
#
# (C) Copyright IBM 2022
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""File containing the entanglement forging ansatz class."""

from typing import List, Tuple, Optional

from qiskit import QuantumCircuit


Bitstring = Tuple[int]


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

    Attributes:
        - circuit_u (QuantumCircuit): the parameterized circuit that is optimized to
            find the minimum energy of the original problem. It represents the
            unitary U for both N-qubit subsystems in the Schmidt decomposition.
        - bitstrings_u (List[Tuple[Int]]): the input bitstrings for each N-qubit
            subsystem. The bitstrings represent the computational basis states
            contributing to the Schmidt decomposition. List must contain less than
            or equal to 2^N elements and each bitstring must have length N. These
            bitstrings are used for each subsystem unless bitstrings_v is provided.
        - bitstrings_v (List[Tuple[Int]], optional): specifies the bitstrings to be
            used for the second subsystem in the Schmidt decomposition. Must be the
            same shape as bitstrings_u. If not provided, then bitstrings_u is used
            for both subsystems.
    """

    def __init__(
        self,
        circuit_u: QuantumCircuit,
        bitstrings_u: List[Bitstring],
        bitstrings_v: Optional[List[Bitstring]] = None,
    ):
        """
        Assign the necessary member variables and check for shaping errors.

        Args:
            - circuit_u (QuantumCircuit): the parameterized circuit that is optimized to
                find the minimum energy of the original problem. It represents the
                unitary U for both N-qubit subsystems in the Schmidt decomposition.
            - bitstrings_u (List[Tuple[Int]]): the input bitstrings for each N-qubit
                subsystem. The bitstrings represent the computational basis states
                contributing to the Schmidt decomposition. List must contain less than
                or equal to 2^N elements and each bitstring must have length N. These
                bitstrings are used for each subsystem unless bitstrings_v is provided.
            - bitstrings_v (List[Tuple[Int]], optional): specifies the bitstrings to be
                used for the second subsystem in the Schmidt decomposition. Must be the
                same shape as bitstrings_u. If not provided, then bitstrings_u is used
                for both subsystems.

        Returns:
            - None

        Raises:
            - ValueError: If the input bitstrings are of incorrect shapes (as defined above).
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
        self._bitstrings_u: List[Bitstring] = bitstrings_u
        self._bitstrings_v: List[Bitstring] = bitstrings_v or bitstrings_u

    @property
    def circuit_u(self) -> QuantumCircuit:
        """
        Property function for the circuit.

        Args:
            - self

        Returns:
            - (QuantumCircuit): the _circuit_u member variable
        """
        return self._circuit_u

    @property
    def bitstrings_u(self) -> List[Bitstring]:
        """
        Property function for the first bitstrings.

        Args:
            - self

        Returns:
            - (List[Bitstring]): the _bitstrings_u member variable
        """
        return self._bitstrings_u

    @property
    def bitstrings_v(self) -> List[Bitstring]:
        """
        Property function for the second bitstrings.

        Args:
            - self

        Returns:
            - (List[Bitstring]): the _bitstrings_v member variable
        """
        return self._bitstrings_v

    @property
    def bitstrings_are_symmetric(self) -> bool:
        """
        Property function for the symmetry of bitstrings.

        Args:
            - self

        Returns:
            - (bool): whether the first and second set of bitstrings are the same
        """
        return self._bitstrings_v == self._bitstrings_u

    @property
    def subspace_dimension(self) -> int:
        """
        Property function for the length of bitstrings.

        Args:
            - self

        Returns:
            - (int): the number of bitstrings
        """
        return len(self._bitstrings_u)

    def __repr__(self) -> str:
        """
        Representation function for EntanglementForgingAnsatz.

        Args:
            - self

        Returns:
            - (str): printable repesentation of class
        """
        repr = "EntanglementForgingAnsatz\nCircuit:\n"
        repr += str(self._circuit_u.draw())
        repr += "\nBitstrings U:\n"
        repr += str(self.bitstrings_u)
        repr += "\nBitstrings V:\n"
        repr += str(self._bitstrings_v)
        repr += f"\nBitstring are symmetric: {self.bitstrings_are_symmetric}\n"
        repr += f"Subspace dimension: {self.subspace_dimension}"
        return repr
