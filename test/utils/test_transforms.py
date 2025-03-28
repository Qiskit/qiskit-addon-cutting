# This code is a Qiskit project.

# (C) Copyright IBM 2023.

# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Tests for quantum circuit transformation functions."""
import unittest

import pytest
import numpy as np
from qiskit.circuit import (
    QuantumRegister,
    ClassicalRegister,
    QuantumCircuit,
    CircuitInstruction,
)
from qiskit.circuit.library import efficient_su2, Measure
from qiskit.circuit.library.standard_gates import RZZGate

from qiskit_addon_cutting import partition_circuit_qubits
from qiskit_addon_cutting.qpd import (
    decompose_qpd_instructions,
    TwoQubitQPDGate,
)
from qiskit_addon_cutting.utils.transforms import separate_circuit
from qiskit_addon_cutting.utils.iteration import strict_zip


def prepare_hwea():
    circuit = efficient_su2(4, entanglement="linear", reps=1)

    # Exchange CNOTs with gates we support
    for i, gate in enumerate(circuit.data):
        if len(gate.qubits) == 2:
            q1 = circuit.find_bit(gate.qubits[0])[0]
            q2 = circuit.find_bit(gate.qubits[1])[0]
            circuit.data[i] = CircuitInstruction(RZZGate(np.pi / 4), qubits=[q1, q2])
    return circuit


class TestTransforms(unittest.TestCase):
    """Circuit transform tests."""

    def test_separate_circuit(self):
        with self.subTest("Empty case"):
            circuit = QuantumCircuit()
            separated_circuits = separate_circuit(circuit)
            self.assertEqual({}, separated_circuits.subcircuits)
            self.assertEqual([], separated_circuits.qubit_map)

        with self.subTest("Correct number of components"):
            qreg = QuantumRegister(3, "qr")
            creg = ClassicalRegister(2, "cr")
            circuit = QuantumCircuit(qreg, creg)
            circuit.h([0, 1, 2, 2])

            separated_circuits = separate_circuit(circuit)

            self.assertEqual(3, len(separated_circuits.subcircuits))
            self.assertEqual([(0, 0), (1, 0), (2, 0)], separated_circuits.qubit_map)

            # 2 sets of disconnected qubits
            circuit.cx(1, 2)
            separated_circuits = separate_circuit(circuit)

            self.assertEqual(2, len(separated_circuits.subcircuits))
            self.assertEqual([(0, 0), (1, 0), (1, 1)], separated_circuits.qubit_map)

            # One connected component
            circuit.cx(0, 1)
            separated_circuits = separate_circuit(circuit)

            self.assertEqual(1, len(separated_circuits.subcircuits))
            self.assertEqual([(0, 0), (0, 1), (0, 2)], separated_circuits.qubit_map)

        with self.subTest("Circuit ordering with measurement"):
            qreg = QuantumRegister(3)
            creg = ClassicalRegister(1)
            circuit = QuantumCircuit(qreg, creg)
            circuit.h(0)
            circuit.x(0)
            circuit.x(1)
            circuit.h(1)
            circuit.y(2)
            circuit.h(2)
            circuit.measure(0, 0)

            separated_circuits = separate_circuit(circuit)

            compare1 = QuantumCircuit(1, 1)
            compare1.h(0)
            compare1.x(0)
            compare1.measure(0, 0)
            compare2 = QuantumCircuit(1)
            compare2.x(0)
            compare2.h(0)
            compare3 = QuantumCircuit(1)
            compare3.y(0)
            compare3.h(0)

            for i, operation in enumerate(compare1.data):
                self.assertEqual(
                    operation.operation.name,
                    separated_circuits.subcircuits[0].data[i].operation.name,
                )
            for i, operation in enumerate(compare2.data):
                self.assertEqual(
                    operation.operation.name,
                    separated_circuits.subcircuits[1].data[i].operation.name,
                )
            for i, operation in enumerate(compare3.data):
                self.assertEqual(
                    operation.operation.name,
                    separated_circuits.subcircuits[2].data[i].operation.name,
                )
            self.assertEqual([(0, 0), (1, 0), (2, 0)], separated_circuits.qubit_map)

        with self.subTest("Test bit mapping"):
            # Prepare a HWEA and add some measurements to clbits in a random order
            circuit = prepare_hwea()
            creg = ClassicalRegister(3)
            circuit.add_register(creg)
            circuit.data.insert(
                5, CircuitInstruction(Measure(), qubits=[0], clbits=[2])
            )
            circuit.data.insert(
                10, CircuitInstruction(Measure(), qubits=[2], clbits=[1])
            )
            circuit.measure(0, 0)

            # Create a QPD circuit
            partition_labels = "ABBC"
            qpd_circuit = partition_circuit_qubits(circuit, partition_labels)
            qpd_gate_ids = [
                [i]
                for i, inst in enumerate(qpd_circuit.data)
                if isinstance(inst.operation, TwoQubitQPDGate)
            ]

            # Decompose QPD gates to identities
            decomposed_circuit = decompose_qpd_instructions(
                qpd_circuit, qpd_gate_ids, map_ids=[0, 0]
            )

            # Get separated subcircuits
            separated_circuits = separate_circuit(decomposed_circuit, partition_labels)

            # Make truth qubit map and compare
            qubit_map_truth = [("A", 0), ("B", 0), ("B", 1), ("C", 0)]
            self.assertEqual(qubit_map_truth, separated_circuits.qubit_map)

        with self.subTest("Test bit mapping with unused clbit"):
            # Prepare a HWEA and add some measurements to clbits in a random order
            circuit = prepare_hwea()
            creg = ClassicalRegister(4)
            circuit.add_register(creg)
            circuit.data.insert(
                5, CircuitInstruction(Measure(), qubits=[0], clbits=[3])
            )
            circuit.data.insert(
                10, CircuitInstruction(Measure(), qubits=[2], clbits=[2])
            )
            circuit.measure(0, 1)

            # Create a QPD circuit
            partition_labels = "ABBC"
            qpd_circuit = partition_circuit_qubits(circuit, partition_labels)
            qpd_gate_ids = [
                [i]
                for i, inst in enumerate(qpd_circuit.data)
                if isinstance(inst.operation, TwoQubitQPDGate)
            ]

            # Decompose QPD gates to identities
            decomposed_circuit = decompose_qpd_instructions(
                qpd_circuit, qpd_gate_ids, map_ids=[0, 0]
            )

            # Get separated subcircuits
            separated_circuits = separate_circuit(decomposed_circuit, partition_labels)

            # Make truth qubit map and compare
            qubit_map_truth = [("A", 0), ("B", 0), ("B", 1), ("C", 0)]
            self.assertEqual(qubit_map_truth, separated_circuits.qubit_map)

        with self.subTest("Correct number of components with partition labels"):
            qreg = QuantumRegister(3, "qr")
            creg = ClassicalRegister(2, "cr")
            circuit = QuantumCircuit(qreg, creg)
            circuit.h([0, 1, 2, 2])

            separated_circuits = separate_circuit(circuit, partition_labels="CAB")

            self.assertEqual(3, len(separated_circuits.subcircuits))
            self.assertEqual(
                [("C", 0), ("A", 0), ("B", 0)], separated_circuits.qubit_map
            )

            # 2 sets of disconnected qubits
            circuit.cx(1, 2)
            separated_circuits = separate_circuit(circuit, partition_labels="BAA")

            self.assertEqual(2, len(separated_circuits.subcircuits))
            self.assertEqual(
                [("B", 0), ("A", 0), ("A", 1)], separated_circuits.qubit_map
            )

            # One connected component
            circuit.cx(0, 1)
            separated_circuits = separate_circuit(circuit, partition_labels=[0, 0, 0])

            self.assertEqual(1, len(separated_circuits.subcircuits))
            self.assertEqual([(0, 0), (0, 1), (0, 2)], separated_circuits.qubit_map)

        with self.subTest("Circuit ordering with measurement with partition_labels"):
            qreg = QuantumRegister(3)
            creg = ClassicalRegister(1)
            circuit = QuantumCircuit(qreg, creg)
            circuit.h(0)
            circuit.x(0)
            circuit.x(1)
            circuit.h(1)
            circuit.y(2)
            circuit.h(2)
            circuit.measure(0, 0)

            separated_circuits = separate_circuit(circuit, partition_labels="ABC")

            compare1 = QuantumCircuit(1, 1)
            compare1.h(0)
            compare1.x(0)
            compare1.measure(0, 0)
            compare2 = QuantumCircuit(1)
            compare2.x(0)
            compare2.h(0)
            compare3 = QuantumCircuit(1)
            compare3.y(0)
            compare3.h(0)

            for i, operation in enumerate(compare1.data):
                self.assertEqual(
                    operation.operation.name,
                    separated_circuits.subcircuits["A"].data[i].operation.name,
                )
            for i, operation in enumerate(compare2.data):
                self.assertEqual(
                    operation.operation.name,
                    separated_circuits.subcircuits["B"].data[i].operation.name,
                )
            for i, operation in enumerate(compare3.data):
                self.assertEqual(
                    operation.operation.name,
                    separated_circuits.subcircuits["C"].data[i].operation.name,
                )
            self.assertEqual(
                [("A", 0), ("B", 0), ("C", 0)], separated_circuits.qubit_map
            )

        with self.subTest("Test bit mapping with partition labels"):
            # Prepare a HWEA and add some measurements to clbits in a random order
            circuit = prepare_hwea()
            creg = ClassicalRegister(3)
            circuit.add_register(creg)
            circuit.data.insert(
                5, CircuitInstruction(Measure(), qubits=[0], clbits=[2])
            )
            circuit.data.insert(
                10, CircuitInstruction(Measure(), qubits=[2], clbits=[1])
            )
            circuit.measure(0, 0)

            # Create a QPD circuit
            partition_labels = "ABBC"
            qpd_circuit = partition_circuit_qubits(circuit, partition_labels)
            qpd_gate_ids = [
                [i]
                for i, inst in enumerate(qpd_circuit.data)
                if isinstance(inst.operation, TwoQubitQPDGate)
            ]

            # Decompose QPD gates to identities
            decomposed_circuit = decompose_qpd_instructions(
                qpd_circuit, qpd_gate_ids, map_ids=[0, 0]
            )

            # Get separated subcircuits
            separated_circuits = separate_circuit(decomposed_circuit, partition_labels)

            # Make truth qubit map and compare
            qubit_map_truth = [("A", 0), ("B", 0), ("B", 1), ("C", 0)]
            self.assertEqual(qubit_map_truth, separated_circuits.qubit_map)

        with self.subTest("Test bit mapping with unused clbit with partition_labels"):
            # Prepare a HWEA and add some measurements to clbits in a random order
            circuit = prepare_hwea()
            creg = ClassicalRegister(4)
            circuit.add_register(creg)
            circuit.data.insert(
                5, CircuitInstruction(Measure(), qubits=[0], clbits=[3])
            )
            circuit.data.insert(
                10, CircuitInstruction(Measure(), qubits=[2], clbits=[2])
            )
            circuit.measure(0, 1)

            # Create a QPD circuit
            partition_labels = "ABBC"
            qpd_circuit = partition_circuit_qubits(circuit, partition_labels)
            qpd_gate_ids = [
                [i]
                for i, inst in enumerate(qpd_circuit.data)
                if isinstance(inst.operation, TwoQubitQPDGate)
            ]

            # Decompose QPD gates to identities
            decomposed_circuit = decompose_qpd_instructions(
                qpd_circuit, qpd_gate_ids, map_ids=[0, 0]
            )

            # Get separated subcircuits
            separated_circuits = separate_circuit(
                decomposed_circuit, partition_labels=partition_labels
            )

            # Make truth qubit map and compare
            qubit_map_truth = [("A", 0), ("B", 0), ("B", 1), ("C", 0)]
            self.assertEqual(qubit_map_truth, separated_circuits.qubit_map)

        with self.subTest("Circuit with barriers"):
            qc = QuantumCircuit(3)

            qc.x([0, 1, 2])
            qc.draw()
            qc.barrier()
            qc.draw()
            qc.y([0, 1, 2])
            qc.barrier()
            qc.z([0, 1, 2])
            qc.barrier(0)

            qca = _create_barrier_subcirc()
            qca.barrier(0)
            qcb = _create_barrier_subcirc()
            qcc = _create_barrier_subcirc()

            subcircuits = separate_circuit(qc).subcircuits

            for i, compare_circ in enumerate([qca, qcb, qcc]):
                self.assertEqual(len(subcircuits[i].data), len(compare_circ.data))
                for j, inst in enumerate(compare_circ):
                    self.assertEqual(
                        subcircuits[i].data[j].operation.name, inst.operation.name
                    )

        with self.subTest("Bad partition labels"):
            circuit = QuantumCircuit(2)
            partition_labels = "ABB"
            with pytest.raises(ValueError) as e_info:
                separated_circuits = separate_circuit(
                    circuit, partition_labels=partition_labels
                )
            assert (
                e_info.value.args[0]
                == "The number of partition_labels (3) must equal the number of qubits in the input circuit (2)."
            )

        with self.subTest("Bad partition labels"):
            circuit = QuantumCircuit(2)
            circuit.cx(0, 1)
            partition_labels = "AB"
            with pytest.raises(ValueError) as e_info:
                separated_circuits = separate_circuit(
                    circuit, partition_labels=partition_labels
                )
            assert (
                e_info.value.args[0]
                == "The input circuit cannot be separated along specified partitions. Operation 'cx' at index 0 spans more than one partition."
            )

        with self.subTest("frozenset for each partition label"):
            circuit = QuantumCircuit(4)
            circuit.x(0)
            circuit.cx(1, 2)
            circuit.h(3)
            partition_labels = [
                frozenset([0]),
                frozenset([1]),
                frozenset([1]),
                frozenset([2]),
            ]
            separated_circuits = separate_circuit(
                circuit, partition_labels=partition_labels
            )
            compare = {}
            compare[frozenset([0])] = QuantumCircuit(1)
            compare[frozenset([0])].x(0)
            compare[frozenset([1])] = QuantumCircuit(2)
            compare[frozenset([1])].cx(0, 1)
            compare[frozenset([2])] = QuantumCircuit(1)
            compare[frozenset([2])].h(0)
            assert separated_circuits.subcircuits.keys() == compare.keys()
            for label, subcircuit in separated_circuits.subcircuits.items():
                for op1, op2 in strict_zip(compare[label].data, subcircuit.data):
                    self.assertEqual(op1.operation.name, op2.operation.name)
            assert separated_circuits.qubit_map == [
                (frozenset([0]), 0),
                (frozenset([1]), 0),
                (frozenset([1]), 1),
                (frozenset([2]), 0),
            ]

        with self.subTest("Unused qubit, with partition labels"):
            circuit = QuantumCircuit(2)
            circuit.x(0)
            separated_circuits = separate_circuit(circuit, partition_labels="BA")
            assert separated_circuits.subcircuits.keys() == {"A", "B"}
            assert len(separated_circuits.subcircuits["B"].data) == 1
            assert len(separated_circuits.subcircuits["A"].data) == 0
            assert separated_circuits.qubit_map == [("B", 0), ("A", 0)]

        with self.subTest("Unused qubit, no partition labels"):
            circuit = QuantumCircuit(2)
            circuit.x(1)
            separated_circuits = separate_circuit(circuit)
            assert separated_circuits.subcircuits.keys() == {0}
            assert len(separated_circuits.subcircuits[0].data) == 1
            assert separated_circuits.qubit_map == [(None, None), (0, 0)]

        with self.subTest("Explicit partition label of None on an idle qubit"):
            circuit = QuantumCircuit(2)
            circuit.x(0)
            separated_circuits = separate_circuit(circuit, partition_labels=["A", None])
            assert separated_circuits.subcircuits.keys() == {"A"}
            assert len(separated_circuits.subcircuits["A"].data) == 1
            assert separated_circuits.qubit_map == [("A", 0), (None, None)]

        with self.subTest(
            "Explicit partition label of None on a non-idle qubit should error"
        ):
            circuit = QuantumCircuit(2)
            circuit.h(0)
            circuit.s(0)
            circuit.x(1)
            with pytest.raises(ValueError) as e_info:
                separate_circuit(circuit, partition_labels=["A", None])
            assert (
                e_info.value.args[0]
                == "Operation 'x' at index 2 acts on the 1-th qubit, which was provided a partition label of `None`. If the partition label of a qubit is `None`, then that qubit cannot be used in the circuit."
            )


def _create_barrier_subcirc() -> QuantumCircuit:
    qc = QuantumCircuit(1)
    qc.x(0)
    qc.barrier()
    qc.y(0)
    qc.barrier()
    qc.z(0)

    return qc
