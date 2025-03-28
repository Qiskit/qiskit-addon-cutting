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
from qiskit.quantum_info import PauliList, Pauli
from qiskit.circuit import QuantumCircuit, QuantumRegister
from qiskit.circuit.library.standard_gates import CXGate

from qiskit_addon_cutting.qpd import (
    SingleQubitQPDGate,
    TwoQubitQPDGate,
    QPDBasis,
)
from qiskit_addon_cutting.utils.observable_grouping import CommutingObservableGroup
from qiskit_addon_cutting import generate_cutting_experiments
from qiskit_addon_cutting.qpd import WeightType
from qiskit_addon_cutting import partition_problem
from qiskit_addon_cutting.cutting_experiments import (
    _append_measurement_register,
    _append_measurement_circuit,
    _remove_final_resets,
    _consolidate_resets,
    _remove_resets_in_zero_state,
)


class TestCuttingExperiments(unittest.TestCase):
    def test_generate_cutting_experiments(self):
        with self.subTest("simple circuit and observable"):
            qc = QuantumCircuit(2)
            qc.append(
                TwoQubitQPDGate(QPDBasis.from_instruction(CXGate()), label="cut_cx"),
                qargs=[0, 1],
            )
            comp_coeffs = [
                (0.5, WeightType.EXACT),
                (0.5, WeightType.EXACT),
                (0.5, WeightType.EXACT),
                (-0.5, WeightType.EXACT),
                (0.5, WeightType.EXACT),
                (-0.5, WeightType.EXACT),
            ]
            subexperiments, coeffs = generate_cutting_experiments(
                qc, PauliList(["ZZ"]), np.inf
            )
            assert coeffs == comp_coeffs
            assert len(coeffs) == len(subexperiments)
            for exp in subexperiments:
                assert isinstance(exp, QuantumCircuit)

        with self.subTest("simple circuit and observable as dict"):
            qc = QuantumCircuit(2)
            qc.append(
                SingleQubitQPDGate(
                    QPDBasis.from_instruction(CXGate()), label="cut_cx_0", qubit_id=0
                ),
                qargs=[0],
            )
            qc.append(
                SingleQubitQPDGate(
                    QPDBasis.from_instruction(CXGate()), label="cut_cx_0", qubit_id=1
                ),
                qargs=[1],
            )
            comp_coeffs = [
                (0.5, WeightType.EXACT),
                (0.5, WeightType.EXACT),
                (0.5, WeightType.EXACT),
                (-0.5, WeightType.EXACT),
                (0.5, WeightType.EXACT),
                (-0.5, WeightType.EXACT),
            ]
            subexperiments, coeffs = generate_cutting_experiments(
                {"A": qc}, {"A": PauliList(["ZY"])}, np.inf
            )
            assert coeffs == comp_coeffs
            assert len(coeffs) == len(subexperiments["A"])
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
                TwoQubitQPDGate(QPDBasis.from_instruction(CXGate()), label="cut_cx"),
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
                'labels suffixed with "_<id>", where <id> is the index of the cut '
                "relative to the other cuts in the circuit. For example, all "
                "SingleQubitQPDGates belonging to the same cut, N, should have labels "
                ' formatted as "<your_label>_N". This allows SingleQubitQPDGates '
                "belonging to the same cut to be sampled jointly."
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
                SingleQubitQPDGate(
                    QPDBasis.from_instruction(CXGate()), 0, label="cut_cx_0"
                ),
                qargs=[0],
            )
            with pytest.raises(ValueError) as e_info:
                generate_cutting_experiments(qc, PauliList(["ZZ"]), np.inf)
            assert (
                e_info.value.args[0]
                == "SingleQubitQPDGates are not supported in unseparable circuits."
            )

    def test_append_measurement_register(self):
        qc = QuantumCircuit(2)
        qc.h(0)
        qc.cx(0, 1)
        cog = CommutingObservableGroup(Pauli("XZ"), list(PauliList(["IZ", "XI", "XZ"])))
        with self.subTest("In place"):
            qcx = qc.copy()
            assert _append_measurement_register(qcx, cog, inplace=True) is qcx
        with self.subTest("Out of place"):
            assert _append_measurement_register(qc, cog) is not qc
        with self.subTest("Correct number of bits"):
            assert _append_measurement_register(qc, cog).num_clbits == len(
                cog.pauli_indices
            )

    def test_append_measurement_circuit(self):
        qc = QuantumCircuit(2)
        qc.h(0)
        qc.cx(0, 1)
        cog = CommutingObservableGroup(Pauli("XZ"), list(PauliList(["IZ", "XI", "XZ"])))
        _append_measurement_register(qc, cog, inplace=True)
        with self.subTest("In place"):
            qcx = qc.copy()
            assert _append_measurement_circuit(qcx, cog, inplace=True) is qcx
        with self.subTest("Out of place"):
            assert _append_measurement_circuit(qc, cog) is not qc
        with self.subTest("Correct measurement circuit"):
            qc2 = qc.copy()
            qc2.measure(0, 0)
            qc2.h(1)
            qc2.measure(1, 1)
            assert _append_measurement_circuit(qc, cog) == qc2
        with self.subTest("Mismatch between qubit_locations and number of qubits"):
            with pytest.raises(ValueError) as e_info:
                _append_measurement_circuit(qc, cog, qubit_locations=[0])
            assert (
                e_info.value.args[0]
                == "qubit_locations has 1 element(s) but the observable(s) have 2 qubit(s)."
            )
        with self.subTest("No observable_measurements register"):
            with pytest.raises(ValueError) as e_info:
                _append_measurement_circuit(QuantumCircuit(2), cog)
            assert (
                e_info.value.args[0]
                == 'Cannot locate "observable_measurements" register'
            )
        with self.subTest("observable_measurements register has wrong size"):
            cog2 = CommutingObservableGroup(Pauli("XI"), list(PauliList(["XI"])))
            with pytest.raises(ValueError) as e_info:
                _append_measurement_circuit(qc, cog2)
            assert (
                e_info.value.args[0]
                == '"observable_measurements" register is the wrong size for the given commuting observable group (2 != 1)'
            )
        with self.subTest("Mismatched qubits, no qubit_locations provided"):
            cog = CommutingObservableGroup(Pauli("X"), [Pauli("X")])
            with pytest.raises(ValueError) as e_info:
                _append_measurement_circuit(qc, cog)
            assert (
                e_info.value.args[0]
                == "Quantum circuit qubit count (2) does not match qubit count of observable(s) (1).  Try providing `qubit_locations` explicitly."
            )

    def test_consolidate_double_reset(self):
        """Consolidate a pair of resets.
        qr0:--|0>--|0>--   ==>    qr0:--|0>--
        """
        qr = QuantumRegister(1, "qr")
        circuit = QuantumCircuit(qr)
        circuit.reset(qr)
        circuit.reset(qr)

        expected = QuantumCircuit(qr)
        expected.reset(qr)

        _consolidate_resets(circuit)

        self.assertEqual(expected, circuit)

    def test_two_resets(self):
        """Remove two final resets
        qr0:--[H]-|0>-|0>--   ==>    qr0:--[H]--
        """
        qr = QuantumRegister(1, "qr")
        circuit = QuantumCircuit(qr)
        circuit.h(qr[0])
        circuit.reset(qr[0])
        circuit.reset(qr[0])

        expected = QuantumCircuit(qr)
        expected.h(qr[0])

        _remove_final_resets(circuit)

        self.assertEqual(expected, circuit)

    def test_optimize_single_reset_in_diff_qubits(self):
        """Remove a single final reset in different qubits
        qr0:--[H]--|0>--          qr0:--[H]--
                      ==>
        qr1:--[X]--|0>--          qr1:--[X]----
        """
        qr = QuantumRegister(2, "qr")
        circuit = QuantumCircuit(qr)
        circuit.h(0)
        circuit.x(1)
        circuit.reset(qr)

        expected = QuantumCircuit(qr)
        expected.h(0)
        expected.x(1)

        _remove_final_resets(circuit)
        self.assertEqual(expected, circuit)

    def test_optimize_single_final_reset(self):
        """Remove a single final reset
        qr0:--[H]--|0>--   ==>    qr0:--[H]--
        """
        qr = QuantumRegister(1, "qr")
        circuit = QuantumCircuit(qr)
        circuit.h(0)
        circuit.reset(qr)

        expected = QuantumCircuit(qr)
        expected.h(0)

        _remove_final_resets(circuit)

        self.assertEqual(expected, circuit)

    def test_optimize_single_final_reset_2(self):
        """Remove a single final reset on two qubits
        qr0:--[H]--|0>--   ==>    qr0:--[H]-------
            --[X]--[S]--              --[X]--[S]--
        """
        qr = QuantumRegister(2, "qr")
        circuit = QuantumCircuit(qr)
        circuit.h(0)
        circuit.x(1)
        circuit.reset(0)
        circuit.s(1)

        expected = QuantumCircuit(qr)
        expected.h(0)
        expected.x(1)
        expected.s(1)

        _remove_final_resets(circuit)

        self.assertEqual(expected, circuit)

    def test_dont_optimize_non_final_reset(self):
        """Do not remove reset if not final instruction
        qr0:--|0>--[H]--   ==>    qr0:--|0>--[H]--
        """
        qr = QuantumRegister(1, "qr")
        circuit = QuantumCircuit(qr)
        circuit.reset(qr)
        circuit.h(qr)

        expected = QuantumCircuit(qr)
        expected.reset(qr)
        expected.h(qr)

        _remove_final_resets(circuit)

        self.assertEqual(expected, circuit)

    def test_remove_reset_in_zero_state(self):
        """Remove reset if first instruction on qubit
        qr0:--|0>--[H]--   ==>    qr0:--|0>--[H]--
        """
        qr = QuantumRegister(1, "qr")
        circuit = QuantumCircuit(qr)
        circuit.reset(qr)
        circuit.h(qr)

        expected = QuantumCircuit(qr)
        expected.h(qr)

        _remove_resets_in_zero_state(circuit)

        self.assertEqual(expected, circuit)
