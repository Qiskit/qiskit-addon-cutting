# This code is part of Qiskit.
#
# (C) Copyright IBM 2022.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Tests for circuit_cutting package."""
import unittest

import numpy as np
from qiskit import QuantumCircuit

from circuit_knitting_toolbox.circuit_cutting import WireCutter


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

    def test_circuit_cutting(self):
        qc = self.circuit
        cutter = WireCutter(qc)
        cuts = cutter.decompose(
            method="automatic",
            max_subcircuit_width=3,
            max_subcircuit_cuts=10,
            max_subcircuit_size=12,
            max_cuts=10,
            num_subcircuits=[2],
        )
        subcircuit_instance_probabilities = cutter.evaluate(cuts)
        reconstructed_probabilities = cutter.recompose(subcircuit_instance_probabilities, cuts)

        metrics = cutter.verify(reconstructed_probabilities)

        self.assertAlmostEqual(0.0, metrics["nearest"]["Mean Squared Error"])
