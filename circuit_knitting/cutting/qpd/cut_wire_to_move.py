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

from qiskit.circuit import Qubit, QuantumCircuit, QuantumRegister
from circuit_knitting.cutting.qpd.instructions.move import Move


def transform_to_move(circuit: QuantumCircuit) -> QuantumCircuit:
    """Transform a ``cut_wire`` instruction to a ``move`` instruction

    Args:
        circuit (QuantumCircuit): original circuit with cut_wire instructions.

    Returns:
        circuit (QuantumCircuit): new circuit with move instructions.

    """

    cut_wire_ins = circuit.get_instructions("cut_wire")

    new_circ = QuantumCircuit(len(circuit.qubits))
    qubit_sequence = list(range(len(new_circ.qubits) + len(cut_wire_ins)))

    count_cut_wire = 0
    for index, instructions in enumerate(circuit.data):
        if instructions in cut_wire_ins:
            count_cut_wire += 1
            new_circ.add_bits([Qubit(QuantumRegister(1), 0)])

            # Make changes to qubit sequence
            cut_wire_index = [circuit.find_bit(qubit).index for qubit in instructions.qubits]
            qubit_sequence[cut_wire_index[0]] = len(new_circ.qubits) - count_cut_wire
            qubit_sequence[len(new_circ.qubits) - count_cut_wire] = cut_wire_index[0]

            # Replace cut_wire with move instruction
            new_circ = new_circ.compose(other=Move(), qubits=[cut_wire_index[0], len(new_circ.qubits) - count_cut_wire])
        else:
            new_circ = new_circ.compose(other=instructions[0],
                                        qubits=[qubit_sequence[circuit.find_bit(qubit).index] for qubit in
                                                instructions.qubits])

    return new_circ
