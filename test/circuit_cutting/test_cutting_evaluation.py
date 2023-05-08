# This code is a Qiskit project.

# (C) Copyright IBM 2023.

# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

import pytest
import unittest

from qiskit.quantum_info import Pauli, PauliList
from qiskit.circuit import QuantumCircuit, ClassicalRegister

from circuit_knitting_toolbox.utils.observable_grouping import CommutingObservableGroup
from circuit_knitting_toolbox.circuit_cutting.cutting_evaluation import (
    append_measurement_circuit,
)


class TestCuttingEvaluation(unittest.TestCase):
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

    def test_append_measurement_circuit(self):
        qc = self.qc1.copy()
        with self.subTest("In place"):
            qcx = qc.copy()
            assert append_measurement_circuit(qcx, self.cog, inplace=True) is qcx
        with self.subTest("Out of place"):
            assert append_measurement_circuit(qc, self.cog) is not qc
        with self.subTest("Correct measurement circuit"):
            qc2 = self.qc2.copy()
            qc2.measure(0, 1)
            qc2.h(1)
            qc2.measure(1, 2)
            assert append_measurement_circuit(qc, self.cog) == qc2
        with self.subTest("Mismatch between qubit_locations and number of qubits"):
            with pytest.raises(ValueError) as e_info:
                append_measurement_circuit(qc, self.cog, qubit_locations=[0])
            assert (
                e_info.value.args[0]
                == "qubit_locations has 1 element(s) but the observable(s) have 2 qubit(s)."
            )
        with self.subTest("Mismatched qubits, no qubit_locations provided"):
            cog = CommutingObservableGroup(Pauli("X"), [Pauli("X")])
            with pytest.raises(ValueError) as e_info:
                append_measurement_circuit(qc, cog)
            assert (
                e_info.value.args[0]
                == "Quantum circuit qubit count (2) does not match qubit count of observable(s) (1).  Try providing `qubit_locations` explicitly."
            )
