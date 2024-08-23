# This code is a Qiskit project.

# (C) Copyright IBM 2023.

# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Tests for QPDGate classes."""

import unittest
import copy
import io

import pytest
from qiskit import QuantumCircuit, qpy
from qiskit.circuit.library.standard_gates import CXGate, XGate, YGate, ZGate

from qiskit_addon_cutting.qpd import (
    QPDBasis,
    TwoQubitQPDGate,
    SingleQubitQPDGate,
)


class TestTwoQubitQPDGate(unittest.TestCase):
    def test_qpd_gate_empty(self):
        empty_maps = [([], [])]
        empty_basis = QPDBasis(empty_maps, [1.0])
        empty_gate = TwoQubitQPDGate(empty_basis)
        self.assertEqual(None, empty_gate.basis_id)

    def test_qpd_gate_bad_idx(self):
        empty_maps = [([], [])]
        empty_basis = QPDBasis(empty_maps, [1.0])
        empty_gate = TwoQubitQPDGate(empty_basis)
        with pytest.raises(ValueError) as e_info:
            empty_gate.basis_id = 1
        self.assertEqual("Basis ID out of range", e_info.value.args[0])

    def test_qpd_gate_select_basis(self):
        qpd_maps = [
            ([XGate()], [XGate()]),
            ([YGate()], [YGate()]),
            ([ZGate()], [ZGate()]),
        ]
        qpd_basis = QPDBasis(qpd_maps, [0.5, 0.25, 0.25])
        qpd_gate = TwoQubitQPDGate(qpd_basis)
        qpd_gate.basis_id = 1
        qpd_gate_copy1 = copy.copy(qpd_gate)
        qpd_gate_copy2 = copy.copy(qpd_gate)

        # These will be random if the basis_id isn't working correctly
        self.assertEqual(
            qpd_gate.definition.decompose(),
            qpd_gate_copy1.definition.decompose(),
            qpd_gate_copy2.definition.decompose(),
        )

    def test_qpd_gate_mismatching_basis(self):
        single_qubit_map = [
            ([XGate()],),
        ]
        single_qubit_basis = QPDBasis(single_qubit_map, [1.0])
        with pytest.raises(ValueError) as e_info:
            TwoQubitQPDGate(single_qubit_basis)
        self.assertEqual(
            "TwoQubitQPDGate only supports QPDBasis which act on two qubits.",
            e_info.value.args[0],
        )


class TestSingleQubitQPDGate(unittest.TestCase):
    def test_qpd_gate_empty(self):
        empty_maps = [([],)]
        empty_basis = QPDBasis(empty_maps, [1.0])
        empty_gate = SingleQubitQPDGate(empty_basis, qubit_id=0)
        self.assertEqual(None, empty_gate.basis_id)
        self.assertEqual(0, empty_gate.qubit_id)

    def test_qubit_id_out_of_range(self):
        maps = [([XGate()], [YGate()])]
        basis = QPDBasis(maps, [1.0])
        with pytest.raises(ValueError) as e_info:
            SingleQubitQPDGate(basis, qubit_id=2)
        self.assertEqual(
            "'qubit_id' out of range. 'basis' acts on 2 qubits, but 'qubit_id' is 2.",
            e_info.value.args[0],
        )

    def test_missing_basis_id(self):
        maps = [([XGate()], [YGate()])]
        basis = QPDBasis(maps, [1.0])
        assert SingleQubitQPDGate(basis=basis, qubit_id=0).definition is None

    def test_compare_1q_and_2q(self):
        maps = [([XGate()], [YGate()])]
        basis = QPDBasis(maps, [1.0])
        inst_2q = TwoQubitQPDGate(basis=basis)
        inst_1q = SingleQubitQPDGate(basis=basis, qubit_id=0)
        # Call both eq methods, since single qubit implements a slightly different equivalence
        self.assertFalse(inst_2q == inst_1q)
        self.assertFalse(inst_1q == inst_2q)

    def test_qpy_serialization(self):
        qc = QuantumCircuit(2)
        qc.append(TwoQubitQPDGate.from_instruction(CXGate()), [0, 1])

        f = io.BytesIO()
        qpy.dump(qc, f)
