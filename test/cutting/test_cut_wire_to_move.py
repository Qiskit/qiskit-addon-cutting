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

"""Test for the transform_to_move function."""
from __future__ import annotations

from pytest import fixture, mark
from qiskit.circuit import QuantumCircuit, QuantumRegister, Qubit, ClassicalRegister
from circuit_knitting.cutting.instructions import Move, CutWire
from circuit_knitting.cutting import transform_cuts_to_moves


@fixture
def circuit1() -> QuantumCircuit:
    circuit = QuantumCircuit(3, 3)
    circuit.cx(1, 2)
    circuit.append(CutWire(), [1])
    circuit.cx(0, 1)
    circuit.append(CutWire(), [1])
    circuit.cx(1, 2)

    return circuit


@fixture
def resulting_circuit1() -> tuple[QuantumCircuit, list[int]]:
    circuit = QuantumCircuit()
    reg = QuantumRegister(3, "q")
    creg = ClassicalRegister(3, "c")
    circuit.add_register(creg)
    circuit.add_bits(
        [
            reg[0],
            Qubit(),
            Qubit(),
            reg[1],
            reg[2],
        ]
    )
    circuit.add_register(reg)
    circuit.cx(1, 4)
    circuit.append(Move(), (1, 2))
    circuit.cx(0, 2)
    circuit.append(Move(), (2, 3))
    circuit.cx(3, 4)

    mapping = [0, 3, 4]

    return circuit, mapping


@fixture
def circuit2() -> QuantumCircuit:
    circuit = QuantumCircuit(4, 4)
    circuit.cx(0, 1)
    circuit.append(CutWire(), [1])
    circuit.cx(1, 2)
    circuit.append(CutWire(), [2])
    circuit.cx(2, 3)

    return circuit


@fixture
def resulting_circuit2() -> tuple[QuantumCircuit, list[int]]:
    circuit = QuantumCircuit()
    reg = QuantumRegister(4, "q")
    creg = ClassicalRegister(4, "c")
    circuit.add_register(creg)
    circuit.add_bits(
        [
            reg[0],
            Qubit(),
            reg[1],
            Qubit(),
            reg[2],
            reg[3],
        ]
    )
    circuit.add_register(reg)
    circuit.cx(0, 1)
    circuit.append(Move(), [1, 2])
    circuit.cx(2, 3)
    circuit.append(Move(), [3, 4])
    circuit.cx(4, 5)

    mapping = [0, 2, 4, 5]

    return circuit, mapping


@fixture
def circuit3() -> QuantumCircuit:
    circuit = QuantumCircuit(4, 4)
    circuit.cx(0, 1)
    circuit.append(CutWire(), [1])
    circuit.cx(1, 2)
    circuit.append(CutWire(), [2])
    circuit.cx(2, 3)
    circuit.append(CutWire(), [2])
    circuit.cx(0, 2)
    circuit.append(CutWire(), [0])
    circuit.cx(0, 1)

    return circuit


@fixture
def resulting_circuit3() -> tuple[QuantumCircuit, list[int]]:
    circuit = QuantumCircuit()
    reg = QuantumRegister(4, "q")
    creg = ClassicalRegister(4, "c")
    circuit.add_register(creg)
    circuit.add_bits(
        [
            Qubit(),
            reg[0],
            Qubit(),
            reg[1],
            Qubit(),
            Qubit(),
            reg[2],
            reg[3],
        ]
    )
    circuit.add_register(reg)
    circuit.cx(0, 2)
    circuit.append(Move(), [2, 3])
    circuit.cx(3, 4)
    circuit.append(Move(), [4, 5])
    circuit.cx(5, 7)
    circuit.append(Move(), [5, 6])
    circuit.cx(0, 6)
    circuit.append(Move(), [0, 1])
    circuit.cx(1, 3)

    mapping = [1, 3, 6, 7]

    return circuit, mapping


@fixture
def circuit4() -> QuantumCircuit:
    circuit = QuantumCircuit()
    reg1 = QuantumRegister(4, "qx")
    reg2 = QuantumRegister(4, "qy")
    creg = ClassicalRegister(8, "c")
    circuit.add_register(creg)
    for i in range(4):
        circuit.add_bits([reg1[i], reg2[i], Qubit()])
    circuit.add_register(reg1)
    circuit.add_register(reg2)
    circuit.cx(0, 1)
    circuit.append(CutWire(), [0])
    circuit.cx(0, 1)
    circuit.append(CutWire(), [1])
    circuit.cx(1, 2)
    circuit.append(CutWire(), [2])

    return circuit


@fixture
def resulting_circuit4() -> tuple[QuantumCircuit, list[int]]:
    circuit = QuantumCircuit()
    reg1 = QuantumRegister(4, "qx")
    reg2 = QuantumRegister(4, "qy")
    creg = ClassicalRegister(8, "c")
    circuit.add_register(creg)
    circuit.add_bits([Qubit(), reg1[0]])
    circuit.add_bits([Qubit(), reg2[0]])
    circuit.add_bits([Qubit(), Qubit()])
    for i in range(1, 4):
        circuit.add_bits([reg1[i], reg2[i], Qubit()])
    circuit.add_register(reg1)
    circuit.add_register(reg2)
    circuit.cx(0, 2)
    circuit.append(Move(), [0, 1])
    circuit.cx(1, 2)
    circuit.append(Move(), [2, 3])
    circuit.cx(3, 4)
    circuit.append(Move(), [4, 5])

    mapping = [1, 3, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]

    return circuit, mapping


@mark.parametrize(
    "sample_circuit, resulting_circuit",
    [
        ("circuit1", "resulting_circuit1"),
        ("circuit2", "resulting_circuit2"),
        ("circuit3", "resulting_circuit3"),
        ("circuit4", "resulting_circuit4"),
    ],
)
def test_transform_cuts_to_moves(request, sample_circuit, resulting_circuit):
    """Tests the transformation of CutWire to Move instruction."""
    assert request.getfixturevalue(resulting_circuit)[0] == transform_cuts_to_moves(
        request.getfixturevalue(sample_circuit)
    )


@mark.parametrize(
    "sample_circuit, resulting_circuit",
    [
        ("circuit1", "resulting_circuit1"),
        ("circuit2", "resulting_circuit2"),
        ("circuit3", "resulting_circuit3"),
        ("circuit4", "resulting_circuit4"),
    ],
)
def test_circuit_mapping(request, sample_circuit, resulting_circuit):
    """Tests the mapping of original and new circuit registers."""
    sample_circuit = request.getfixturevalue(sample_circuit)
    resulting_mapping = request.getfixturevalue(resulting_circuit)[1]

    final_circuit = transform_cuts_to_moves(sample_circuit)
    final_mapping = [
        final_circuit.find_bit(qubit).index for qubit in sample_circuit.qubits
    ]

    assert all(
        final_mapping[index] == resulting_mapping[index]
        for index in range(len(sample_circuit.qubits))
    )


@mark.parametrize(
    "sample_circuit",
    [
        "circuit1",
        "circuit2",
        "circuit3",
        "circuit4",
    ],
)
def test_qreg_name_num(request, sample_circuit):
    """Tests the number and name of qregs in initial and final circuits."""
    sample_circuit = request.getfixturevalue(sample_circuit)
    final_circuit = transform_cuts_to_moves(sample_circuit)

    # Tests number of qregs in initial and final circuits
    assert len(sample_circuit.qregs) == len(final_circuit.qregs)

    for sample_qreg, final_qreg in zip(
        sample_circuit.qregs,
        final_circuit.qregs,
    ):
        assert sample_qreg.name == final_qreg.name


@mark.parametrize(
    "sample_circuit",
    [
        "circuit1",
        "circuit2",
        "circuit3",
        "circuit4",
    ],
)
def test_qreg_size(request, sample_circuit):
    """Tests the size of qregs in initial and final circuits."""
    sample_circuit = request.getfixturevalue(sample_circuit)
    final_circuit = transform_cuts_to_moves(sample_circuit)

    # Tests size of qregs in initial and final circuits
    for sample_qreg, final_qreg in zip(
        sample_circuit.qregs,
        final_circuit.qregs,
    ):
        assert sample_qreg.size == final_qreg.size


@mark.parametrize(
    "sample_circuit",
    [
        "circuit1",
        "circuit2",
        "circuit3",
        "circuit4",
    ],
)
def test_circuit_width(request, sample_circuit):
    """Tests the width of the initial and final circuits."""
    sample_circuit = request.getfixturevalue(sample_circuit)
    final_circuit = transform_cuts_to_moves(sample_circuit)
    total_cut_wire = len(sample_circuit.get_instructions("cut_wire"))

    # Tests width of initial and final circuit
    assert len(sample_circuit.qubits) + total_cut_wire == len(final_circuit.qubits)


@mark.parametrize(
    "sample_circuit",
    [
        "circuit1",
        "circuit2",
        "circuit3",
        "circuit4",
    ],
)
def test_creg(request, sample_circuit):
    """Tests the number and size of cregs in the initial and final circuits."""
    sample_circuit = request.getfixturevalue(sample_circuit)
    final_circuit = transform_cuts_to_moves(sample_circuit)

    # Tests number of cregs in initial and final circuits
    assert len(sample_circuit.cregs) == len(final_circuit.cregs)

    for sample_creg, final_creg in zip(
        sample_circuit.cregs,
        final_circuit.cregs,
    ):
        assert sample_creg.size == final_creg.size
