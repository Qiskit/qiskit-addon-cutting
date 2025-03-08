# This code is a Qiskit project.

# (C) Copyright IBM 2024.

# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Tests for cco_utils module."""

from __future__ import annotations
from typing import Callable

import pytest
from pytest import fixture
from qiskit.circuit.library import efficient_su2
from qiskit import QuantumCircuit, QuantumRegister
from qiskit.circuit import Qubit, Instruction, CircuitInstruction
from qiskit_addon_cutting.cut_finding.cco_utils import (
    qc_to_cco_circuit,
    cco_to_qc_circuit,
)
from qiskit_addon_cutting.cut_finding.circuit_interface import (
    SimpleGateList,
    CircuitElement,
)


def create_test_circuit_1():
    tc_1 = QuantumCircuit(2)
    tc_1.h(1)
    tc_1.barrier(1)
    tc_1.s(0)
    tc_1.barrier()
    tc_1.cx(1, 0)
    return tc_1


def create_test_circuit_2():
    tc_2 = efficient_su2(2, entanglement="linear", reps=2)
    tc_2.assign_parameters([0.4] * len(tc_2.parameters), inplace=True)
    return tc_2


# test circuit 3
@fixture
def internal_test_circuit():
    circuit = [
        CircuitElement(name="cx", params=[], qubits=[0, 1], gamma=3),
        CircuitElement(name="cx", params=[], qubits=[2, 3], gamma=3),
        CircuitElement(name="cx", params=[], qubits=[1, 2], gamma=3),
        CircuitElement(name="cx", params=[], qubits=[0, 1], gamma=3),
        CircuitElement(name="cx", params=[], qubits=[2, 3], gamma=3),
        CircuitElement(name="h", params=[], qubits=[0], gamma=None),
        CircuitElement(name="rx", params=[0.4], qubits=[0], gamma=None),
    ]
    interface = SimpleGateList(circuit)
    interface.insert_gate_cut(2, "LO")
    interface.define_subcircuits([[0, 1], [2, 3]])
    return interface


@pytest.mark.parametrize(
    "create_test_circuit, known_output",
    [
        (
            create_test_circuit_1,
            [
                CircuitElement("h", [], [1], None),
                CircuitElement("barrier", [], [1], None),
                CircuitElement("s", [], [0], None),
                "barrier",
                CircuitElement("cx", [], [1, 0], 3),
            ],
        ),
        (
            create_test_circuit_2,
            [
                CircuitElement("ry", [0.4], [0], None),
                CircuitElement("ry", [0.4], [1], None),
                CircuitElement("rz", [0.4], [0], None),
                CircuitElement("rz", [0.4], [1], None),
                CircuitElement("cx", [], [0, 1], 3),
                CircuitElement("ry", [0.4], [0], None),
                CircuitElement("ry", [0.4], [1], None),
                CircuitElement("rz", [0.4], [0], None),
                CircuitElement("rz", [0.4], [1], None),
                CircuitElement("cx", [], [0, 1], 3),
                CircuitElement("ry", [0.4], [0], None),
                CircuitElement("ry", [0.4], [1], None),
                CircuitElement("rz", [0.4], [0], None),
                CircuitElement("rz", [0.4], [1], None),
            ],
        ),
    ],
)
def test_qc_to_cco_circuit(
    create_test_circuit: Callable[[], QuantumCircuit],
    known_output: list[CircuitElement, str],
):
    test_circuit = create_test_circuit()
    test_circuit_internal = qc_to_cco_circuit(test_circuit)
    assert test_circuit_internal == known_output


def test_cco_to_qc_circuit(internal_test_circuit: SimpleGateList):
    qc_cut = cco_to_qc_circuit(internal_test_circuit)
    assert qc_cut.data == [
        CircuitInstruction(
            operation=Instruction(name="cx", num_qubits=2, num_clbits=0, params=[]),
            qubits=(
                Qubit(QuantumRegister(4, "q"), 0),
                Qubit(QuantumRegister(4, "q"), 1),
            ),
            clbits=(),
        ),
        CircuitInstruction(
            operation=Instruction(name="cx", num_qubits=2, num_clbits=0, params=[]),
            qubits=(
                Qubit(QuantumRegister(4, "q"), 2),
                Qubit(QuantumRegister(4, "q"), 3),
            ),
            clbits=(),
        ),
        CircuitInstruction(
            operation=Instruction(name="cx", num_qubits=2, num_clbits=0, params=[]),
            qubits=(
                Qubit(QuantumRegister(4, "q"), 0),
                Qubit(QuantumRegister(4, "q"), 1),
            ),
            clbits=(),
        ),
        CircuitInstruction(
            operation=Instruction(name="cx", num_qubits=2, num_clbits=0, params=[]),
            qubits=(
                Qubit(QuantumRegister(4, "q"), 2),
                Qubit(QuantumRegister(4, "q"), 3),
            ),
            clbits=(),
        ),
        CircuitInstruction(
            operation=Instruction(name="h", num_qubits=1, num_clbits=0, params=[]),
            qubits=(Qubit(QuantumRegister(4, "q"), 0),),
            clbits=(),
        ),
        CircuitInstruction(
            operation=Instruction(name="rx", num_qubits=1, num_clbits=0, params=[0.4]),
            qubits=(Qubit(QuantumRegister(4, "q"), 0),),
            clbits=(),
        ),
    ]
