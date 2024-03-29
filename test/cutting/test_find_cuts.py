# This code is a Qiskit project.

# (C) Copyright IBM 2024.

# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Tests for find_cuts module."""

import unittest

import pytest
import os
import numpy as np
from qiskit import QuantumCircuit

from circuit_knitting.cutting.find_cuts import (
    find_cuts,
    OptimizationParameters,
    DeviceConstraints,
)


class TestCuttingDecomposition(unittest.TestCase):
    def test_find_cuts(self):
        with self.subTest("simple circuit"):
            path_to_circuit = os.path.join(
                os.path.dirname(__file__),
                "..",
                "qasm_circuits",
                "circuit_find_cuts_test.qasm",
            )
            circuit = QuantumCircuit.from_qasm_file(path_to_circuit)
            optimization = OptimizationParameters(seed=111)
            constraints = DeviceConstraints(qubits_per_subcircuit=4)

            _, metadata = find_cuts(
                circuit, optimization=optimization, constraints=constraints
            )
            cut_types = {cut[0] for cut in metadata["cuts"]}

            assert len(metadata["cuts"]) == 2
            assert {"Wire Cut", "Gate Cut"} == cut_types
            assert np.isclose(127.06026169, metadata["sampling_overhead"], atol=1e-8)

        with self.subTest("3-qubit gate"):
            circuit = QuantumCircuit(3)
            circuit.cswap(2, 1, 0)
            circuit.crx(3.57, 1, 0)
            circuit.z(2)
            with pytest.raises(ValueError) as e_info:
                _, metadata = find_cuts(
                    circuit, optimization=optimization, constraints=constraints
                )
            assert e_info.value.args[0] == (
                "The input circuit must contain only single and two-qubits gates. "
                "Found 3-qubit gate: (cswap)."
            )
        with self.subTest(
            "right-wire-cut"
        ):  # tests resolution of https://github.com/Qiskit-Extensions/circuit-knitting-toolbox/issues/508

            circuit = QuantumCircuit(5)
            circuit.cx(0, 3)
            circuit.cx(1, 3)
            circuit.cx(2, 3)
            circuit.h(4)
            circuit.cx(3, 4)
            constraints = DeviceConstraints(qubits_per_subcircuit=3)
            _, metadata = find_cuts(
                circuit, optimization=optimization, constraints=constraints
            )
            cut_types = {cut[0] for cut in metadata["cuts"]}

            assert len(metadata["cuts"]) == 1
            assert {"Wire Cut"} == cut_types
            assert metadata["sampling_overhead"] == 16
