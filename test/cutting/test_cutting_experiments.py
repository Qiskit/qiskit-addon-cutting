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
from qiskit.quantum_info import PauliList
from qiskit.circuit import QuantumCircuit
from qiskit.circuit.library.standard_gates import CXGate

from circuit_knitting.cutting.qpd import (
    SingleQubitQPDGate,
    TwoQubitQPDGate,
    QPDBasis,
)
from circuit_knitting.cutting import generate_cutting_experiments
from circuit_knitting.cutting.qpd import WeightType
from circuit_knitting.cutting import partition_problem


class TestCuttingExperiments(unittest.TestCase):
    def test_generate_cutting_experiments(self):
        with self.subTest("simple circuit and observable"):
            qc = QuantumCircuit(2)
            qc.append(
                TwoQubitQPDGate(QPDBasis.from_gate(CXGate()), label="cut_cx"),
                qargs=[0, 1],
            )
            comp_weights = [
                (0.5, WeightType.EXACT),
                (0.5, WeightType.EXACT),
                (0.5, WeightType.EXACT),
                (-0.5, WeightType.EXACT),
                (0.5, WeightType.EXACT),
                (-0.5, WeightType.EXACT),
            ]
            subexperiments, weights = generate_cutting_experiments(
                qc, PauliList(["ZZ"]), np.inf
            )
            assert weights == comp_weights
            assert len(weights) == len(subexperiments)
            for exp in subexperiments:
                assert isinstance(exp, QuantumCircuit)

        with self.subTest("simple circuit and observable as dict"):
            qc = QuantumCircuit(2)
            qc.append(
                SingleQubitQPDGate(
                    QPDBasis.from_gate(CXGate()), label="cut_cx_0", qubit_id=0
                ),
                qargs=[0],
            )
            qc.append(
                SingleQubitQPDGate(
                    QPDBasis.from_gate(CXGate()), label="cut_cx_0", qubit_id=1
                ),
                qargs=[1],
            )
            comp_weights = [
                (0.5, WeightType.EXACT),
                (0.5, WeightType.EXACT),
                (0.5, WeightType.EXACT),
                (-0.5, WeightType.EXACT),
                (0.5, WeightType.EXACT),
                (-0.5, WeightType.EXACT),
            ]
            subexperiments, weights = generate_cutting_experiments(
                {"A": qc}, {"A": PauliList(["ZY"])}, np.inf
            )
            assert weights == comp_weights
            assert len(weights) == len(subexperiments["A"])
            for circ in subexperiments["A"]:
                assert isinstance(circ, QuantumCircuit)

        with self.subTest("test bad num_samples"):
            qc = QuantumCircuit(4)
            with pytest.raises(ValueError) as e_info:
                generate_cutting_experiments(qc, PauliList(["ZZZZ"]), 0)
            assert e_info.value.args[0] == "num_samples must be at least 1."
            with pytest.raises(ValueError) as e_info:
                generate_cutting_experiments(qc, PauliList(["ZZZZ"]), np.nan)
            assert e_info.value.args[0] == "num_samples must be at least 1."
        with self.subTest("test incompatible inputs"):
            qc = QuantumCircuit(4)
            with pytest.raises(ValueError) as e_info:
                generate_cutting_experiments(qc, {"A": PauliList(["ZZZZ"])}, 4.5)
            assert (
                e_info.value.args[0]
                == "If the input circuits is a QuantumCircuit, the observables must be a PauliList."
            )
            with pytest.raises(ValueError) as e_info:
                generate_cutting_experiments({"A": qc}, PauliList(["ZZZZ"]), 4.5)
            assert (
                e_info.value.args[0]
                == "If the input circuits are contained in a dictionary keyed by partition labels, the input observables must also be represented by such a dictionary."
            )
        with self.subTest("test bad label"):
            qc = QuantumCircuit(2)
            qc.append(
                TwoQubitQPDGate(QPDBasis.from_gate(CXGate()), label="cut_cx"),
                qargs=[0, 1],
            )
            partitioned_problem = partition_problem(
                qc, "AB", observables=PauliList(["ZZ"])
            )
            partitioned_problem.subcircuits["A"].data[0].operation.label = "newlabel"

            with pytest.raises(ValueError) as e_info:
                generate_cutting_experiments(
                    partitioned_problem.subcircuits,
                    partitioned_problem.subobservables,
                    np.inf,
                )
            assert e_info.value.args[0] == (
                "SingleQubitQPDGate instances in input circuit(s) must have their "
                'labels suffixed with "_<id>", where <id> is the index of the gate '
                "relative to the other gates belonging to the same cut. For example, "
                "a two-qubit gate cut can be represented by two SingleQubitQPDGates -- one "
                'labeled "<your_label>_0" and one labeled "<your_label>_1".'
                "  This allows SingleQubitQPDGates belonging to the same cut to be "
                "sampled jointly."
            )
        with self.subTest("test bad observable size"):
            qc = QuantumCircuit(4)
            with pytest.raises(ValueError) as e_info:
                generate_cutting_experiments(qc, PauliList(["ZZ"]), np.inf)
            assert e_info.value.args[0] == (
                "Quantum circuit qubit count (4) does not match qubit count of observable(s) (2)."
                "  Try providing `qubit_locations` explicitly."
            )
        with self.subTest("test single qubit qpd gate in unseparated circuit"):
            qc = QuantumCircuit(2)
            qc.append(
                SingleQubitQPDGate(QPDBasis.from_gate(CXGate()), 0, label="cut_cx_0"),
                qargs=[0],
            )
            with pytest.raises(ValueError) as e_info:
                generate_cutting_experiments(qc, PauliList(["ZZ"]), np.inf)
            assert (
                e_info.value.args[0]
                == "SingleQubitQPDGates are not supported in unseparable circuits."
            )
