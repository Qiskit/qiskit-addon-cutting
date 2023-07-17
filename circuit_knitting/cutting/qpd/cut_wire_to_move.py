# This code is part of Qiskit.
#
# (C) Copyright IBM 2023.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Function to transform a CutWire instruction to a Move instruction."""
from __future__ import annotations

from qiskit.circuit import QuantumCircuit


def transform_to_move(circuit: QuantumCircuit) -> QuantumCircuit:
    """Transform a ``cut_wire`` instruction to a ``move`` instruction

    Args:
        circuit (QuantumCircuit): original circuit with cut_wire instructions.

    Returns:
        circuit (QuantumCircuit): new circuit with move instructions.

    """
    cut_wire_ins = circuit.get_instructions("cut_wire")

    subcircuit: QuantumCircuit = QuantumCircuit(3)
    for index, instructions in enumerate(circuit.data):
        print(index, instructions)
        subcircuit = subcircuit.compose(other=instructions[0], qubits=instructions[1])
        if instructions in cut_wire_ins:
            print([circuit.find_bit(qubit) for qubit in instructions.qubits])

