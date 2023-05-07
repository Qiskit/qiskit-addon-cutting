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
from ddt import ddt, data, unpack

import numpy as np
from qiskit.quantum_info import Pauli, PauliList
from qiskit.circuit import QuantumCircuit, ClassicalRegister

from circuit_knitting_toolbox.circuit_cutting.qpd.utils.observable_estimation import *


@ddt
class TestEstimatorUtilities(unittest.TestCase):
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

    def test_most_general_observable(self):
        with self.subTest("No elements"):
            with pytest.raises(ValueError) as e_info:
                most_general_observable([])
            assert (
                e_info.value.args[0]
                == "Empty input sequence: consider performing no experiments rather than an experiment over the identity Pauli."
            )
        with self.subTest("Observable with wrong qubit count"):
            with pytest.raises(ValueError) as e_info:
                most_general_observable([Pauli("X"), Pauli("ZZ")])
            assert (
                e_info.value.args[0]
                == "Observable 1 has incorrect qubit count (2 rather than 1)."
            )
        with self.subTest("Pass strings instead of Paulis"):
            with pytest.raises(ValueError) as e_info:
                most_general_observable(["X", "ZZ"])
            assert (
                e_info.value.args[0]
                == "Input sequence includes something other than a Pauli."
            )
        with self.subTest("Incompatible observables"):
            with pytest.raises(ValueError) as e_info:
                most_general_observable(PauliList(["X", "Z"]))
            assert (
                e_info.value.args[0]
                == "Observables are incompatible; cannot construct a single general observable."
            )
        with self.subTest("Redundancy"):
            assert most_general_observable(PauliList(["ZI", "IZ", "ZZ"])) == Pauli("ZZ")

    @data(
        ((0, 2), PauliList(["IXIX"]), PauliList(["XX"])),
        ((1, 3), PauliList(["ZZII", "iIZIZ"]), PauliList(["ZI", "II"])),
        ((1,), PauliList(["XYZ", "-XXX", "-iZZZ"]), PauliList(["Y", "X", "Z"])),
        (
            (),
            PauliList(["XX", "YY", "ZZ"]),
            PauliList.from_symplectic(np.zeros((3, 0)), np.zeros((3, 0))),
        ),
    )
    @unpack
    def test_observables_restricted_to_subsystem(self, qubits, observables, expected):
        assert (
            observables_restricted_to_subsystem(qubits, PauliList(observables))
            == expected
        )
        assert observables_restricted_to_subsystem(qubits, list(observables)) == list(
            expected
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

    def test_process_outcome_exceptions(self):
        with self.subTest("Invalid or missing first register"):
            with pytest.raises(ValueError) as e_info:
                process_outcome(self.qc0, self.cog, 0)
            assert (
                e_info.value.args[0]
                == "Circuit's first register is expected to be named 'qpd_measurements'."
            )
        with self.subTest("Invalid or missing second register"):
            with pytest.raises(ValueError) as e_info:
                process_outcome(self.qc1, self.cog, 0)
            assert (
                e_info.value.args[0]
                == "Circuit's second register is expected to be named 'observable_measurements'."
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
        for o in (
            outcome,
            f"0b{outcome}",
            int(f"0b{outcome}", 0),
            hex(int(f"0b{outcome}", 0)),
        ):
            assert np.all(process_outcome(self.qc2, self.cog, o) == expected)

    def test_cog_with_nontrivial_phase(self):
        with pytest.raises(ValueError) as e_info:
            CommutingObservableGroup(Pauli("XXI"), [Pauli("iXXI")])
        assert (
            e_info.value.args[0]
            == "CommutingObservableGroup only supports Paulis with phase == 0. (Value provided: 3)"
        )

    def test_observable_collection(self):
        with self.subTest("Initialize with List[Pauli]"):
            oc = ObservableCollection([Pauli("XX"), Pauli("ZZ"), Pauli("XX")])
            assert [len(group.commuting_observables) for group in oc.groups] == [1, 1]

        with self.subTest("Qubit count doesn't match"):
            with pytest.raises(ValueError):
                ObservableCollection([Pauli("XX"), Pauli("ZZZ")])
