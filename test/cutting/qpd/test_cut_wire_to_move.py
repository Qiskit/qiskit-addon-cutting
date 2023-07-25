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

from pytest import fixture
from qiskit.circuit import Qubit, QuantumCircuit, QuantumRegister
from circuit_knitting.cutting.qpd.instructions.move import Move
from circuit_knitting.cutting.qpd.instructions.cut_wire import CutWire
from circuit_knitting.cutting.qpd.cut_wire_to_move import transform_to_move


@fixture
def sample_circuit() -> QuantumCircuit:
    circuit = QuantumCircuit(3)
    circuit.cx(1, 2)
    circuit.append(CutWire(), [1])
    circuit.cx(0, 1)
    circuit.append(CutWire(), [1])
    circuit.cx(1, 2)
    circuit.draw()

    return circuit


@fixture
def resulting_circuit() -> QuantumCircuit:
    circuit = QuantumCircuit(5)
    circuit.cx(1, 4)
    circuit.append(Move(), (1, 2))
    circuit.cx(0, 2)
    circuit.append(Move(), (2, 3))
    circuit.cx(3, 4)
    circuit.draw()

    return circuit


def test_transform_to_move(sample_circuit, resulting_circuit):
    """Tests the transformation of cut_wire to move instruction."""
    assert resulting_circuit == transform_to_move(sample_circuit)
