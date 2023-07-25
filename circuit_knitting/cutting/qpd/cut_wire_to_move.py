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

from itertools import groupby
from qiskit.circuit import Qubit, QuantumCircuit, QuantumRegister
from circuit_knitting.cutting.qpd.instructions.move import Move


def transform_to_move(circuit: QuantumCircuit) -> QuantumCircuit:
    """Transform a :class:`.cut_wire` instruction to a :class:`.move` instruction

    Args:
        circuit (QuantumCircuit): original circuit with :class:`.cut_wire` instructions.

    Returns:
        circuit (QuantumCircuit): new circuit with :class`.move` instructions.
    """

    new_circuit, mapping = _circuit_structure_mapping(circuit)

    for _, instructions in enumerate(circuit.data):
        gate_index = [circuit.find_bit(qubit).index for qubit in instructions.qubits]

        if instructions in circuit.get_instructions("cut_wire"):
            # Replace cut_wire with move instruction
            new_circuit = new_circuit.compose(
                other=Move(),
                qubits=[mapping[gate_index[0]], mapping[gate_index[0]] + 1],
            )
            mapping[gate_index[0]] += 1
        else:
            new_circuit = new_circuit.compose(
                other=instructions[0],
                qubits=[mapping[index] for index in gate_index],
            )

    return new_circuit


def _circuit_structure_mapping(circuit: QuantumCircuit):
    new_circuit = QuantumCircuit()
    mapping = [qubit for qubit in range(len(circuit.qubits))]

    cut_wire_index = [
        circuit.find_bit(instruction.qubits[0]).index
        for instruction in circuit.get_instructions("cut_wire")
    ]

    # TODO: Optimize this.
    bits = []
    for index in range(len(circuit.qubits)):
        for key, group in groupby(cut_wire_index):
            if index == key:
                for cuts in range(len(list(group))):
                    mapping[index + 1 :] = map(
                        lambda item: item + 1, iter(mapping[index + 1 :])
                    )
                    bits.append(Qubit())
        bits.append(circuit.qregs[0][index])

    new_qregs = QuantumRegister(name=circuit.qregs[0].name, bits=bits)
    new_circuit.add_register(new_qregs)

    return new_circuit, mapping
