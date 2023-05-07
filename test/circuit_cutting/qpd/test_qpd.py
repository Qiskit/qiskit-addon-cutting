# This code is a Qiskit project.

# (C) Copyright IBM 2023.

# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Tests for quasiprobability decomposition functions."""

import unittest

import pytest
import numpy as np
from ddt import ddt, data, unpack
from qiskit.circuit import CircuitInstruction
from qiskit.circuit.library import (
    EfficientSU2,
    CXGate,
    CZGate,
    CRXGate,
    CRYGate,
    CRZGate,
    RXXGate,
    RYYGate,
    RZZGate,
)

from circuit_knitting_toolbox.utils.iteration import unique_by_eq
from circuit_knitting_toolbox.circuit_cutting.qpd import (
    QPDBasis,
    TwoQubitQPDGate,
    generate_qpd_samples,
)
from circuit_knitting_toolbox.circuit_cutting.qpd.qpd import *


@ddt
class TestQPDFunctions(unittest.TestCase):
    def setUp(self):
        # Use HWEA for simplicity and easy visualization
        qpd_circuit = EfficientSU2(4, entanglement="linear", reps=2).decompose()

        # We will instantiate 2 QPDBasis objects using from_gate
        rxx_gate = RXXGate(np.pi / 3)
        rxx_decomp = QPDBasis.from_gate(rxx_gate)

        # Create two QPDGates and specify each of their bases
        # Labels are only used for visualizations
        qpd_gate1 = TwoQubitQPDGate(rxx_decomp, label=f"qpd_{rxx_gate.name}")
        qpd_gate2 = TwoQubitQPDGate(rxx_decomp, label=f"qpd_{rxx_gate.name}")
        qpd_gate1.basis_id = 0
        qpd_gate2.basis_id = 0

        # Create the circuit instructions
        qpd_inst1 = CircuitInstruction(qpd_gate1, qubits=[1, 2])
        qpd_inst2 = CircuitInstruction(qpd_gate2, qubits=[1, 2])

        # Hard-coded overwrite of the two CNOTS with our decomposed RXX gates
        qpd_circuit.data[9] = qpd_inst1
        qpd_circuit.data[20] = qpd_inst2

        self.qpd_gate1 = qpd_gate1
        self.qpd_gate2 = qpd_gate2
        self.qpd_circuit = qpd_circuit

    def test_generate_qpd_samples(self):
        with self.subTest("Negative number of samples"):
            with pytest.raises(ValueError) as e_info:
                generate_qpd_samples([], -100)
            assert e_info.value.args[0] == "num_samples must be positive."
        with self.subTest("Zero samples requested"):
            with pytest.raises(ValueError) as e_info:
                generate_qpd_samples([], 0)
            assert e_info.value.args[0] == "num_samples must be positive."
        with self.subTest("Empty case"):
            empty_samples = {(): (1000, WeightType.EXACT)}
            samples = generate_qpd_samples([])
            self.assertEqual(samples, empty_samples)
        with self.subTest("HWEA 100 samples"):
            basis_ids = [9, 20]
            bases = [self.qpd_circuit.data[i].operation.basis for i in basis_ids]
            samples = generate_qpd_samples(bases, num_samples=100)
            self.assertEqual(100, sum(w for w, t in samples.values()))
            for decomp_ids in samples.keys():
                self.assertTrue(0 <= decomp_ids[0] < len(self.qpd_gate1.basis.maps))
                self.assertTrue(0 <= decomp_ids[1] < len(self.qpd_gate2.basis.maps))

    # Optimal values from https://arxiv.org/abs/2205.00016v2 Corollary 4.4 (page 10)
    @data(
        (CXGate(), 3),
        (CZGate(), 3),
        (CRXGate(np.pi / 7), 1 + 2 * np.abs(np.sin(np.pi / 14))),
        (CRYGate(np.pi / 7), 1 + 2 * np.abs(np.sin(np.pi / 14))),
        (CRZGate(np.pi / 7), 1 + 2 * np.abs(np.sin(np.pi / 14))),
        (RXXGate(np.pi / 7), 1 + 2 * np.abs(np.sin(np.pi / 7))),
        (RYYGate(np.pi / 7), 1 + 2 * np.abs(np.sin(np.pi / 7))),
        (RZZGate(np.pi / 7), 1 + 2 * np.abs(np.sin(np.pi / 7))),
    )
    @unpack
    def test_optimal_kappa_for_known_gates(self, instruction, gamma):
        assert np.isclose(qpdbasis_from_gate(instruction).kappa, gamma)

    @data(
        (RXXGate(np.pi / 7), 5, 5),
        (RYYGate(np.pi / 7), 5, 5),
        (RZZGate(np.pi / 7), 5, 5),
        (CRXGate(np.pi / 7), 5, 5),
        (CRYGate(np.pi / 7), 5, 5),
        (CRZGate(np.pi / 7), 5, 5),
        (CXGate(), 5, 5),
        (CZGate(), 5, 5),
        (RZZGate(0), 1, 1),
        (RXXGate(np.pi), 1, 1),
        (CRYGate(np.pi), 5, 5),
    )
    @unpack
    def test_qpdbasis_from_gate_unique_maps(
        self, instruction, q0_num_unique, q1_num_unique
    ):
        """
        Count the number of unique maps with non-zero weight on each qubit.

        Make sure it is as expected based on the instruction provided.
        """
        basis = qpdbasis_from_gate(instruction)
        # Consider only maps with non-zero weight
        relevant_maps = [
            m for m, w in zip(basis.maps, basis.coeffs) if not np.isclose(w, 0)
        ]
        assert len(unique_by_eq(a for (a, b) in relevant_maps)) == q0_num_unique
        assert len(unique_by_eq(b for (a, b) in relevant_maps)) == q1_num_unique
