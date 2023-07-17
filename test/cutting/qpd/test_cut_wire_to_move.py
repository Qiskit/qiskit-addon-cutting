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

from qiskit.circuit import QuantumCircuit
from circuit_knitting.cutting.qpd.instructions.cut_wire import CutWire


def sample_circuit() -> QuantumCircuit:
    circuit = QuantumCircuit(3)
    circuit.x(0)
    circuit.cx(0, 1)
    circuit.append(CutWire(), [1])
    circuit.cx(1, 2)

    return circuit


def test_transform_to_move(sample_circuit):
    """Tests the transformation of cut_wire to move instruction."""
    pass