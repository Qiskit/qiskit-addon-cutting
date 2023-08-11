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

"""Function to transform a :class:`.CutWire` instruction to a :class:`.Move` instruction."""
from __future__ import annotations

from itertools import groupby
from qiskit.circuit import Qubit, QuantumCircuit
from circuit_knitting.cutting.instructions.move import Move


def transform_to_move(circuit: QuantumCircuit) -> QuantumCircuit:
    """Transform a :class:`.CutWire` instruction to a :class:`.Move` instruction.

    Args:
        circuit (QuantumCircuit): original circuit with :class:`.cut_wire` instructions.

    Returns:
        circuit (QuantumCircuit): new circuit with :class`.move` instructions.
    """
    new_circuit, mapping = _circuit_structure_mapping(circuit)

    for instructions in circuit.data:
        gate_index = [circuit.find_bit(qubit).index for qubit in instructions.qubits]
        
        if instructions in circuit.get_instructions("cut_wire"):
            # Replace cut_wire with move instruction
            new_circuit = new_circuit.compose(
                other=Move(),
                qubits=[mapping[gate_index[0]], mapping[gate_index[0]] + 1],
                inplace=False,
            )
            mapping[gate_index[0]] += 1
        else:
            new_circuit = new_circuit.compose(
                other=instructions[0],
                qubits=[mapping[index] for index in gate_index],
                inplace=False,
            )

    return new_circuit


def _circuit_structure_mapping(
    circuit: QuantumCircuit,
) -> tuple[QuantumCircuit, list[int]]:
    new_circuit = QuantumCircuit()
    mapping = list(range(len(circuit.qubits)))

    cut_wire_index = [
        circuit.find_bit(instruction.qubits[0]).index
        for instruction in circuit.get_instructions("cut_wire")
    ]
    cut_wire_freq = {key: len(list(group)) for key, group in groupby(cut_wire_index)}

    # Get intermediate mapping and add quantum bits to new_circuit
    for qubit in circuit.qubits:
        index = circuit.find_bit(qubit).index
        if index in cut_wire_freq.keys():
            for _ in range(cut_wire_freq[index]):
                mapping[index + 1 :] = map(
                    lambda item: item + 1, iter(mapping[index + 1 :])
                )
                new_circuit.add_bits([Qubit()])
        new_circuit.add_bits([qubit])

    # Add quantum and classical registers
    for qreg in circuit.qregs:
        new_circuit.add_register(qreg)

    new_circuit.add_bits(circuit.clbits)
    for creg in circuit.cregs:
        new_circuit.add_register(creg)

    return new_circuit, mapping
