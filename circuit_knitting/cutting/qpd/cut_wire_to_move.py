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
    new_circ = QuantumCircuit(len(circuit.qubits))
    qubit_sequence = new_circ.qubits

    count_cut_wire = 0
    for _, instructions in enumerate(circuit.data):
        gate_index = [circuit.find_bit(qubit).index for qubit in instructions.qubits]

        if instructions in circuit.get_instructions("cut_wire"):
            count_cut_wire += 1
            new_circ.add_bits([Qubit(QuantumRegister(1), 0)])
            qubit_sequence.append(new_circ.qubits[-1])

            # Make changes to qubit sequence
            temp = qubit_sequence[gate_index[0]]
            qubit_sequence[gate_index[0]] = qubit_sequence[len(new_circ.qubits) - 1]
            qubit_sequence[len(new_circ.qubits) - 1] = temp

            # Replace cut_wire with move instruction
            new_circ = new_circ.compose(
                other=Move(),
                qubits=[
                    qubit_sequence[gate_index[0]],
                    qubit_sequence[len(new_circ.qubits) - 1],
                ],
            )
        else:
            new_circ = new_circ.compose(
                other=instructions[0],
                qubits=[qubit_sequence[index] for index in gate_index],
            )

    return new_circ
