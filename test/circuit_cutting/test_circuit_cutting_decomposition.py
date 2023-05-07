# This code is a Qiskit project.

# (C) Copyright IBM 2023.

# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Tests for circuit_cutting package."""
import unittest

import pytest
import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit import CircuitInstruction, Barrier
from qiskit.circuit.library import EfficientSU2, RXXGate

from circuit_knitting_toolbox.circuit_cutting import partition_circuit_qubits
from circuit_knitting_toolbox.circuit_cutting.qpd import QPDBasis, TwoQubitQPDGate


class TestCuttingFunctions(unittest.TestCase):
    def setUp(self):
        # Use HWEA for simplicity and easy visualization
        circuit = EfficientSU2(4, entanglement="linear", reps=2).decompose()
        qpd_circuit = EfficientSU2(4, entanglement="linear", reps=2).decompose()

        # We will instantiate 2 QPDBasis objects using from_gate
        rxx_gate = RXXGate(np.pi / 3)
        rxx_decomp = QPDBasis.from_gate(rxx_gate)

        # Create two QPDGates and specify each of their bases
        # Labels are only used for visualizations
        qpd_gate1 = TwoQubitQPDGate(rxx_decomp, label=f"qpd_{rxx_gate.name}")
        qpd_gate2 = TwoQubitQPDGate(rxx_decomp, label=f"qpd_{rxx_gate.name}")
        qpd_gate1.basis_id = 0
        qpd_gate2.basis_id = 0

        # Create the circuit instructions
        qpd_inst1 = CircuitInstruction(qpd_gate1, qubits=[1, 2])
        qpd_inst2 = CircuitInstruction(qpd_gate2, qubits=[1, 2])
        inst1 = CircuitInstruction(rxx_gate, qubits=[1, 2])
        inst2 = CircuitInstruction(rxx_gate, qubits=[1, 2])

        # Hard-coded overwrite of the two CNOTS with our decomposed RXX gates
        qpd_circuit.data[9] = qpd_inst1
        qpd_circuit.data[20] = qpd_inst2
        circuit.data[9] = inst1
        circuit.data[20] = inst2

        self.qpd_circuit = qpd_circuit
        self.circuit = circuit

    def test_partition_circuit_qubits(self):
        with self.subTest("Empty circuit"):
            compare_circuit = QuantumCircuit()
            partitioned_circuit = partition_circuit_qubits(compare_circuit, [])
            self.assertEqual(partitioned_circuit, compare_circuit)
        with self.subTest("Circuit with parameters"):
            # Split 4q HWEA in middle of qubits
            partition_labels = [0, 0, 1, 1]

            # Get a QPD circuit based on partitions, and set the basis for each gate
            # to match the basis_ids of self.qpd_circuit's QPDGates
            circuit = partition_circuit_qubits(self.circuit, partition_labels)
            for inst in circuit.data:
                if isinstance(inst.operation, TwoQubitQPDGate):
                    inst.operation.basis_id = 0

            # Terra doesn't consider params with same name to be equivalent, so
            # we need to copy the comparison circuit and bind parameters to test
            # equivalence.
            compare_circuit = self.qpd_circuit.copy()
            compare_qpd_circuit = partition_circuit_qubits(
                compare_circuit, partition_labels
            )
            parameter_vals = [np.pi / 4] * len(circuit.parameters)
            circuit.assign_parameters(parameter_vals, inplace=True)
            compare_qpd_circuit.assign_parameters(parameter_vals, inplace=True)
            self.assertEqual(circuit, compare_qpd_circuit)
        with self.subTest("Circuit with barriers"):
            # Split 4q HWEA in middle of qubits
            partition_labels = [0, 0, 1, 1]

            bar1 = CircuitInstruction(Barrier(4), qubits=[0, 1, 2, 3])
            bar2 = CircuitInstruction(Barrier(4), qubits=[0, 1, 2, 3])

            bar_circuit = self.circuit.copy()
            bar_circuit.data.insert(10, bar1)
            bar_circuit.data.insert(22, bar2)

            # Get a QPD circuit based on partitions, and set the basis for each gate
            # to match the basis_ids of self.qpd_circuit's QPDGates
            circuit = partition_circuit_qubits(bar_circuit, partition_labels)
            for inst in circuit.data:
                if isinstance(inst.operation, TwoQubitQPDGate):
                    inst.operation.basis_id = 0

            # Terra doesn't consider params with same name to be equivalent, so
            # we need to copy the comparison circuit and bind parameters to test
            # equivalence.
            compare_circuit = self.qpd_circuit.copy()
            compare_qpd_circuit = partition_circuit_qubits(
                compare_circuit, partition_labels
            )
            bar1 = CircuitInstruction(Barrier(4), qubits=[0, 1, 2, 3])
            bar2 = CircuitInstruction(Barrier(4), qubits=[0, 1, 2, 3])

            compare_qpd_circuit.data.insert(10, bar1)
            compare_qpd_circuit.data.insert(22, bar2)
            parameter_vals = [np.pi / 4] * len(circuit.parameters)
            circuit.assign_parameters(parameter_vals, inplace=True)
            compare_qpd_circuit.assign_parameters(parameter_vals, inplace=True)
            self.assertEqual(circuit, compare_qpd_circuit)
        with self.subTest("Partition IDs the wrong size"):
            compare_circuit = QuantumCircuit()
            with pytest.raises(ValueError) as e_info:
                partition_circuit_qubits(compare_circuit, [0])
            assert (
                e_info.value.args[0]
                == "Length of partition_labels (1) does not equal the number of qubits in the input circuit (0)."
            )
        with self.subTest("Unsupported gate"):
            compare_circuit = QuantumCircuit(3)
            compare_circuit.toffoli(0, 1, 2)
            partitions = [0, 1, 1]
            with pytest.raises(ValueError) as e_info:
                partition_circuit_qubits(compare_circuit, partitions)
            assert (
                e_info.value.args[0]
                == "Decomposition is only supported for two-qubit gates. Cannot decompose (ccx)."
            )
        with self.subTest("Toffoli gate in a single partition"):
            circuit = QuantumCircuit(4)
            circuit.toffoli(0, 1, 2)
            circuit.rzz(np.pi / 7, 2, 3)
            partition_circuit_qubits(circuit, "AAAB")
