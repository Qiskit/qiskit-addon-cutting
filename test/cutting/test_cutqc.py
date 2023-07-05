# This code is a Qiskit project.

# (C) Copyright IBM 2022.

# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Tests for circuit_cutting package."""

import unittest
import importlib.util

import numpy as np
from qiskit import QuantumCircuit

from circuit_knitting.cutting.cutqc import (
    cut_circuit_wires,
    evaluate_subcircuits,
    reconstruct_full_distribution,
    create_dd_bin,
    reconstruct_dd_full_distribute,
    verify,
)

cplex_available = importlib.util.find_spec("cplex") is not None


class TestCircuitCutting(unittest.TestCase):
    def setUp(self):
        qc = QuantumCircuit(5)
        for i in range(5):
            qc.h(i)
        qc.cx(0, 1)
        for i in range(2, 5):
            qc.t(i)
        qc.cx(0, 2)
        qc.rx(np.pi / 2, 4)
        qc.rx(np.pi / 2, 0)
        qc.rx(np.pi / 2, 1)
        qc.cx(2, 4)
        qc.t(0)
        qc.t(1)
        qc.cx(2, 3)
        qc.ry(np.pi / 2, 4)
        for i in range(5):
            qc.h(i)

        self.circuit = qc

    @unittest.skipIf(not cplex_available, "cplex is not installed")
    def test_circuit_cutting_automatic(self):
        qc = self.circuit
        cuts = cut_circuit_wires(
            circuit=qc,
            method="automatic",
            max_subcircuit_width=3,
            max_subcircuit_cuts=10,
            max_subcircuit_size=12,
            max_cuts=10,
            num_subcircuits=[2],
        )
        subcircuit_instance_probabilities = evaluate_subcircuits(cuts)
        reconstructed_probabilities = reconstruct_full_distribution(
            qc, subcircuit_instance_probabilities, cuts
        )

        metrics, _ = verify(qc, reconstructed_probabilities)

        self.assertAlmostEqual(0.0, metrics["nearest"]["Mean Squared Error"])

    def test_circuit_cutting_manual(self):
        qc = self.circuit

        cuts = cut_circuit_wires(
            circuit=qc, method="manual", subcircuit_vertices=[[0, 1], [2, 3]]
        )
        subcircuit_instance_probabilities = evaluate_subcircuits(cuts)
        reconstructed_probabilities = reconstruct_full_distribution(
            qc, subcircuit_instance_probabilities, cuts
        )

        metrics, _ = verify(qc, reconstructed_probabilities)

        self.assertAlmostEqual(0.0, metrics["nearest"]["Mean Squared Error"])

    @unittest.skipIf(not cplex_available, "cplex is not installed")
    def test_circuit_cutting_dynamic_definition(self):
        qc = self.circuit

        cuts = cut_circuit_wires(
            circuit=qc,
            method="automatic",
            max_subcircuit_width=3,
            max_subcircuit_cuts=10,
            max_subcircuit_size=12,
            max_cuts=10,
            num_subcircuits=[2],
        )
        subcircuit_instance_probabilities = evaluate_subcircuits(cuts)

        dd_bins = create_dd_bin(subcircuit_instance_probabilities, cuts, 4, 15)

        dd_prob = reconstruct_dd_full_distribute(self.circuit, cuts, dd_bins)
        metrics, _ = verify(qc, dd_prob)

        self.assertAlmostEqual(0.0, metrics["nearest"]["Mean Squared Error"])

    @unittest.skipIf(not cplex_available, "cplex is not installed")
    def test_circuit_cutting_dynamic_definition_ghz(self):
        qc = QuantumCircuit(20, name="ghz")
        qc.h(0)
        for i in range(20 - 1):
            qc.cx(i, i + 1)

        cuts = cut_circuit_wires(
            circuit=qc,
            method="automatic",
            max_subcircuit_width=5,
            max_cuts=8,
            num_subcircuits=[3, 4, 5],
        )

        subcircuit_instance_probabilities = evaluate_subcircuits(cuts)
        dd_bins = create_dd_bin(subcircuit_instance_probabilities, cuts, 10, 5, 4)

        dd_prob = reconstruct_dd_full_distribute(qc, cuts, dd_bins)
        metrics, _ = verify(qc, dd_prob)
        self.assertAlmostEqual(0.0, metrics["nearest"]["Mean Squared Error"])
