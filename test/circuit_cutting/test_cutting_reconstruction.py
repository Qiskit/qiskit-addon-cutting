# This code is a Qiskit project.

# (C) Copyright IBM 2023.

# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

import unittest
from ddt import ddt, data, unpack

import pytest
import numpy as np
from qiskit.result import QuasiDistribution
from qiskit.quantum_info import Pauli, PauliList
from qiskit.circuit import QuantumCircuit, ClassicalRegister

from circuit_knitting_toolbox.utils.observable_grouping import CommutingObservableGroup
from circuit_knitting_toolbox.circuit_cutting.qpd import WeightType
from circuit_knitting_toolbox.circuit_cutting.cutting_reconstruction import (
    _process_outcome,
    reconstruct_expectation_values,
)


@ddt
class TestCuttingReconstruction(unittest.TestCase):
    def setUp(self):
        qc = QuantumCircuit(2)
        qc.h(0)
        qc.cx(0, 1)
        self.qc0 = qc.copy()
        qc.add_register(ClassicalRegister(1, name="qpd_measurements"))
        self.qc1 = qc.copy()
        qc.add_register(ClassicalRegister(2, name="observable_measurements"))
        self.qc2 = qc

        self.cog = CommutingObservableGroup(
            Pauli("XZ"), list(PauliList(["IZ", "XI", "XZ"]))
        )

    def test_cutting_reconstruction(self):
        with self.subTest("Test PauliList observable"):
            quasi_dists = [[[(QuasiDistribution({"0": 1.0}), 0)]]]
            coefficients = [(1.0, WeightType.EXACT)]
            observables = PauliList(["ZZ"])
            expvals = reconstruct_expectation_values(
                quasi_dists, coefficients, observables
            )
            self.assertEqual([1.0], expvals)
        with self.subTest("Test mismatching inputs"):
            quasi_dists = [[[(QuasiDistribution({"0": 1.0}), 0)]]]
            coefficients = [(0.5, WeightType.EXACT), (0.5, WeightType.EXACT)]
            observables = PauliList(["ZZ"])
            with pytest.raises(ValueError) as e_info:
                reconstruct_expectation_values(quasi_dists, coefficients, observables)
            assert (
                e_info.value.args[0]
                == "The number of unique samples in the quasi_dists list (1) does not equal the number of coefficients (2)."
            )
        with self.subTest("Test poorly formatted phases"):
            quasi_dists = [[[(QuasiDistribution({"0": 1.0}), 0)]]]
            coefficients = [(0.5, WeightType.EXACT)]
            observables = PauliList(["ZZ"])
            phases = [1.0, 1.0]
            with pytest.raises(ValueError) as e_info:
                reconstruct_expectation_values(
                    quasi_dists, coefficients, observables, phases=phases
                )
            assert (
                e_info.value.args[0]
                == "The number of phases (2) must equal the number of observables (1)."
            )
            phases = [2.0]
            with pytest.raises(ValueError) as e_info:
                reconstruct_expectation_values(
                    quasi_dists, coefficients, observables, phases=phases
                )
            assert (
                e_info.value.args[0]
                == "All phase magnitudes must equal 1.0. Phase (2.0) encountered."
            )

    @data(
        ("000", [1, 1, 1]),
        ("001", [-1, -1, -1]),
        ("010", [-1, 1, -1]),
        ("011", [1, -1, 1]),
        ("100", [1, -1, -1]),
        ("101", [-1, 1, 1]),
        ("110", [-1, -1, 1]),
        ("111", [1, 1, -1]),
    )
    @unpack
    def test_process_outcome(self, outcome, expected):
        num_qpd_bits = len(self.qc2.cregs[-2])
        for o in (
            outcome,
            f"0b{outcome}",
            int(f"0b{outcome}", 0),
            hex(int(f"0b{outcome}", 0)),
        ):
            assert np.all(_process_outcome(num_qpd_bits, self.cog, o) == expected)
