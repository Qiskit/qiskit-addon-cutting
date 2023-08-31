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
from qiskit.primitives import SamplerResult
from qiskit.quantum_info import Pauli, PauliList
from qiskit.circuit import QuantumCircuit, ClassicalRegister

from circuit_knitting.utils.observable_grouping import CommutingObservableGroup
from circuit_knitting.cutting.qpd import WeightType
from circuit_knitting.cutting.cutting_reconstruction import (
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
            results = SamplerResult(
                quasi_dists=[QuasiDistribution({"0": 1.0})], metadata=[{}]
            )
            results.metadata[0]["num_qpd_bits"] = 1
            weights = [(1.0, WeightType.EXACT)]
            subexperiments = [QuantumCircuit(2)]
            creg1 = ClassicalRegister(1, name="qpd_measurements")
            creg2 = ClassicalRegister(2, name="observable_measurements")
            subexperiments[0].add_register(creg1)
            subexperiments[0].add_register(creg2)
            observables = PauliList(["ZZ"])
            expvals = reconstruct_expectation_values(results, weights, observables)
            self.assertEqual([1.0], expvals)
        with self.subTest("Test mismatching inputs"):
            results = SamplerResult(
                quasi_dists=[QuasiDistribution({"0": 1.0})], metadata=[{}]
            )
            results.metadata[0]["num_qpd_bits"] = 1
            weights = [(0.5, WeightType.EXACT), (0.5, WeightType.EXACT)]
            subexperiments = {"A": QuantumCircuit(2)}
            observables = {"A": PauliList(["Z"]), "B": PauliList(["Z"])}
            with pytest.raises(ValueError) as e_info:
                reconstruct_expectation_values(results, weights, observables)
            assert (
                e_info.value.args[0]
                == "If observables is a dictionary, results must also be a dictionary."
            )
            results2 = {"A": results}
            observables = PauliList(["ZZ"])
            with pytest.raises(ValueError) as e_info:
                reconstruct_expectation_values(results2, weights, observables)
            assert (
                e_info.value.args[0]
                == "If observables is a PauliList, results must be a SamplerResult instance."
            )
        with self.subTest("Test unsupported phase"):
            results = SamplerResult(
                quasi_dists=[QuasiDistribution({"0": 1.0})], metadata=[{}]
            )
            results.metadata[0]["num_qpd_bits"] = 1
            weights = [(0.5, WeightType.EXACT)]
            subexperiments = [QuantumCircuit(2)]
            creg1 = ClassicalRegister(1, name="qpd_measurements")
            creg2 = ClassicalRegister(2, name="observable_measurements")
            subexperiments[0].add_register(creg1)
            subexperiments[0].add_register(creg2)
            observables = PauliList(["iZZ"])
            with pytest.raises(ValueError) as e_info:
                reconstruct_expectation_values(results, weights, observables)
            assert (
                e_info.value.args[0]
                == "An input observable has a phase not equal to 1."
            )
            results = {"A": results}
            subexperiments = {"A": subexperiments}
            observables = {"A": observables}
            with pytest.raises(ValueError) as e_info:
                reconstruct_expectation_values(results, weights, observables)
            assert (
                e_info.value.args[0]
                == "An input observable has a phase not equal to 1."
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
