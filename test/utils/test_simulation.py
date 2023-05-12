# This code is a Qiskit project.

# (C) Copyright IBM 2023.

# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

import unittest
import pytest

import numpy as np
from qiskit import QuantumCircuit
from circuit_knitting_toolbox.utils.simulation import simulate_statevector_outcomes


class TestSimulationFunctions(unittest.TestCase):
    def test_simulate_statevector_outcomes(self):
        with self.subTest("Normal circuit"):
            qc = QuantumCircuit(2, 1)
            qc.h(0)
            qc.t(0)
            qc.h(0)
            qc.cx(0, 1)
            qc.measure(0, 0)
            qc.h(0)
            r = simulate_statevector_outcomes(qc)
            assert r.keys() == {0, 1}
            assert r[0] == pytest.approx(np.cos(np.pi / 8) ** 2)
            assert r[1] == pytest.approx(1 - np.cos(np.pi / 8) ** 2)

        with self.subTest("Circuit without measurement"):
            qc = QuantumCircuit(2)
            qc.h(0)
            qc.t(0)
            qc.h(0)
            qc.cx(0, 1)
            qc.h(0)
            r = simulate_statevector_outcomes(qc)
            assert r.keys() == {0}
            assert r[0] == pytest.approx(1.0)

        with self.subTest("Overwriting clbits"):
            qc = QuantumCircuit(2, 1)
            qc.h(0)
            qc.measure(0, 0)
            qc.measure(1, 0)
            r = simulate_statevector_outcomes(qc)
            assert r.keys() == {0}
            assert r[0] == pytest.approx(1.0)

        with self.subTest("Bit has probability 1 of being set"):
            qc = QuantumCircuit(1, 1)
            qc.x(0)
            qc.measure(0, 0)
            r = simulate_statevector_outcomes(qc)
            assert r.keys() == {1}

        with self.subTest("Circuit with reset operation"):
            qc = QuantumCircuit(1, 2)
            qc.h(0)
            qc.measure(0, 0)
            qc.reset(0)
            qc.measure(0, 1)
            r = simulate_statevector_outcomes(qc)
            assert r.keys() == {0, 1}
            assert r[0] == pytest.approx(0.5)
            assert r[1] == pytest.approx(0.5)
