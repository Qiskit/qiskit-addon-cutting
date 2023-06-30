# This code is a Qiskit project.

# (C) Copyright IBM 2023.

# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Tests for quasiprobability decomposition functions."""

import unittest
import math

import pytest
import numpy as np
from ddt import ddt, data, unpack
from qiskit.circuit import CircuitInstruction
from qiskit.circuit.library import (
    EfficientSU2,
    CXGate,
    CZGate,
    CRXGate,
    CRYGate,
    CRZGate,
    RXXGate,
    RYYGate,
    RZZGate,
)

from circuit_knitting.utils.iteration import unique_by_eq
from circuit_knitting.cutting.qpd import (
    QPDBasis,
    SingleQubitQPDGate,
    TwoQubitQPDGate,
    generate_qpd_samples,
)
from circuit_knitting.cutting.qpd.qpd import *


@ddt
class TestQPDFunctions(unittest.TestCase):
    def setUp(self):
        # Use HWEA for simplicity and easy visualization
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

        # Hard-coded overwrite of the two CNOTS with our decomposed RXX gates
        qpd_circuit.data[9] = qpd_inst1
        qpd_circuit.data[20] = qpd_inst2

        self.qpd_gate1 = qpd_gate1
        self.qpd_gate2 = qpd_gate2
        self.qpd_circuit = qpd_circuit

    def test_generate_qpd_samples(self):
        with self.subTest("Negative number of samples"):
            with pytest.raises(ValueError) as e_info:
                generate_qpd_samples([], -100)
            assert e_info.value.args[0] == "num_samples must be at least 1."
        with self.subTest("num_samples == NaN"):
            with pytest.raises(ValueError) as e_info:
                generate_qpd_samples([], math.nan)
            assert e_info.value.args[0] == "num_samples must be at least 1."
        with self.subTest("Zero samples requested"):
            with pytest.raises(ValueError) as e_info:
                generate_qpd_samples([], 0)
            assert e_info.value.args[0] == "num_samples must be at least 1."
        with self.subTest("Empty case"):
            empty_samples = {(): (1000, WeightType.EXACT)}
            samples = generate_qpd_samples([])
            self.assertEqual(samples, empty_samples)
        with self.subTest("HWEA 100 samples"):
            basis_ids = [9, 20]
            bases = [self.qpd_circuit.data[i].operation.basis for i in basis_ids]
            samples = generate_qpd_samples(bases, num_samples=100)
            self.assertEqual(100, sum(w for w, t in samples.values()))
            for decomp_ids in samples.keys():
                self.assertTrue(0 <= decomp_ids[0] < len(self.qpd_gate1.basis.maps))
                self.assertTrue(0 <= decomp_ids[1] < len(self.qpd_gate2.basis.maps))
        with self.subTest("HWEA 100.5 samples"):
            basis_ids = [9, 20]
            bases = [self.qpd_circuit.data[i].operation.basis for i in basis_ids]
            samples = generate_qpd_samples(bases, num_samples=100.5)
            assert sum(w for w, t in samples.values()) == pytest.approx(100.5)
        with self.subTest("HWEA exact weights"):
            # Do the same thing with num_samples above the threshold for exact weights
            basis_ids = [9, 20]
            bases = [self.qpd_circuit.data[i].operation.basis for i in basis_ids]
            samples = generate_qpd_samples(bases, num_samples=1000)
            assert sum(w for w, t in samples.values()) == pytest.approx(1000)
            assert all(t == WeightType.EXACT for w, t in samples.values())
            for decomp_ids in samples.keys():
                self.assertTrue(0 <= decomp_ids[0] < len(self.qpd_gate1.basis.maps))
                self.assertTrue(0 <= decomp_ids[1] < len(self.qpd_gate2.basis.maps))
        with self.subTest("HWEA exact weights via 'infinite' num_samples"):
            basis_ids = [9, 20]
            bases = [self.qpd_circuit.data[i].operation.basis for i in basis_ids]
            samples = generate_qpd_samples(bases, num_samples=math.inf)
            assert sum(w for w, t in samples.values()) == pytest.approx(1)
            assert all(t == WeightType.EXACT for w, t in samples.values())

    def test_decompose_qpd_instructions(self):
        with self.subTest("Empty circuit"):
            circ = QuantumCircuit()
            new_circ = decompose_qpd_instructions(QuantumCircuit(), [])
            circ.add_register(ClassicalRegister(0, name="qpd_measurements"))
            self.assertEqual(circ, new_circ)
        with self.subTest("No QPD circuit"):
            circ = QuantumCircuit(2, 1)
            circ.h(0)
            circ.cx(0, 1)
            circ.measure(1, 0)
            new_circ = decompose_qpd_instructions(circ, [])
            circ.add_register(ClassicalRegister(0, name="qpd_measurements"))
            self.assertEqual(circ, new_circ)
        with self.subTest("Single QPD gate"):
            circ = QuantumCircuit(2)
            circ_compare = circ.copy()
            qpd_basis = QPDBasis.from_gate(RXXGate(np.pi / 3))
            qpd_gate = TwoQubitQPDGate(qpd_basis)
            circ.data.append(CircuitInstruction(qpd_gate, qubits=[0, 1]))
            decomp_circ = decompose_qpd_instructions(circ, [[0]], map_ids=[0])
            circ_compare.add_register(ClassicalRegister(0, name="qpd_measurements"))
            self.assertEqual(decomp_circ, circ_compare)
        with self.subTest("Incorrect map index size"):
            with pytest.raises(ValueError) as e_info:
                decomp_circ = decompose_qpd_instructions(
                    self.qpd_circuit, [[9], [20]], map_ids=[0]
                )
            assert (
                e_info.value.args[0]
                == "The number of map IDs (1) must equal the number of decompositions in the circuit (2)."
            )
        with self.subTest("Test unordered indices"):
            decomp = QPDBasis.from_gate(RXXGate(np.pi / 3))
            qpd_gate1 = TwoQubitQPDGate(basis=decomp)
            qpd_gate2 = TwoQubitQPDGate(basis=decomp)

            qc = QuantumCircuit(2)
            qc.append(CircuitInstruction(qpd_gate1, qubits=[0, 1]))
            qc.x([0, 1])
            qc.y([0, 1])
            qc.append(CircuitInstruction(qpd_gate2, qubits=[0, 1]))
            decompose_qpd_instructions(qc, [[5], [0]], map_ids=[0, 0])
        with self.subTest("Test measurement"):
            qpd_circ = QuantumCircuit(2)
            qpd_inst = CircuitInstruction(self.qpd_gate1, qubits=[0, 1])
            qpd_circ.data.append(qpd_inst)
            dx_circ_truth = QuantumCircuit(2)
            creg = ClassicalRegister(1, name="qpd_measurements")
            dx_circ_truth.add_register(creg)
            dx_circ_truth.h(0)
            dx_circ_truth.rx(np.pi / 2, 1)
            dx_circ_truth.measure(0, 0)
            dx_circ_truth.h(0)
            dx_circ = decompose_qpd_instructions(qpd_circ, [[0]], [2])
            self.assertEqual(dx_circ_truth, dx_circ)
        with self.subTest("test_invalid_map_ids"):
            qc = QuantumCircuit()
            qpd_map_ids = ((),)
            with pytest.raises(ValueError) as e_info:
                decompose_qpd_instructions(qc, qpd_map_ids)
            assert (
                e_info.value.args[0]
                == "Each decomposition must contain either one or two elements. Found a decomposition with (0) elements."
            )
        with self.subTest("test_mismatching_qpd_ids"):
            decomp = QPDBasis.from_gate(RXXGate(np.pi / 3))
            qpd_gate = TwoQubitQPDGate(basis=decomp)
            qc = QuantumCircuit(2)
            qc.h(0)
            qc.append(CircuitInstruction(qpd_gate, qubits=[0, 1]))
            with pytest.raises(ValueError) as e_info:
                decompose_qpd_instructions(qc, [[0]])
            assert (
                e_info.value.args[0]
                == "A circuit data index (0) corresponds to a non-QPDGate (h)."
            )
            qpd_gate1 = SingleQubitQPDGate(basis=decomp, qubit_id=0)
            qpd_gate2 = SingleQubitQPDGate(basis=decomp, qubit_id=1)
            qc.append(CircuitInstruction(qpd_gate1, qubits=[0]))
            qc.h(1)
            qc.append(CircuitInstruction(qpd_gate2, qubits=[1]))
            with pytest.raises(ValueError) as e_info:
                decompose_qpd_instructions(qc, [[1], [2, 3]])
            assert (
                e_info.value.args[0]
                == "A circuit data index (3) corresponds to a non-QPDGate (h)."
            )
        with self.subTest("test_mismatching_qpd_bases"):
            decomp1 = QPDBasis.from_gate(RXXGate(np.pi / 3))
            decomp2 = QPDBasis.from_gate(RXXGate(np.pi / 4))
            qpd_gate1 = SingleQubitQPDGate(basis=decomp1, qubit_id=0)
            qpd_gate2 = SingleQubitQPDGate(basis=decomp2, qubit_id=1)
            qc = QuantumCircuit(2)
            qc.append(CircuitInstruction(qpd_gate1, qubits=[0]))
            qc.append(CircuitInstruction(qpd_gate2, qubits=[1]))
            with pytest.raises(ValueError) as e_info:
                decompose_qpd_instructions(qc, [[0, 1]])
            assert (
                e_info.value.args[0]
                == "Gates within the same decomposition must share an equivalent QPDBasis."
            )
        with self.subTest("test_unspecified_qpd_gates"):
            decomp = QPDBasis.from_gate(RXXGate(np.pi / 3))
            qpd_gate = TwoQubitQPDGate(basis=decomp)
            qpd_gate1 = SingleQubitQPDGate(basis=decomp, qubit_id=0)
            qpd_gate2 = SingleQubitQPDGate(basis=decomp, qubit_id=1)

            qc = QuantumCircuit(2)
            qc.append(CircuitInstruction(qpd_gate1, qubits=[0]))
            qc.append(CircuitInstruction(qpd_gate2, qubits=[1]))
            qc.append(CircuitInstruction(qpd_gate, qubits=[0, 1]))
            with pytest.raises(ValueError) as e_info:
                decompose_qpd_instructions(qc, [[0, 1]])
            assert (
                e_info.value.args[0]
                == "The total number of QPDGates specified in instruction_ids (2) does not equal the number of QPDGates in the circuit (3)."
            )

    # Optimal values from https://arxiv.org/abs/2205.00016v2 Corollary 4.4 (page 10)
    @data(
        (CXGate(), 3),
        (CZGate(), 3),
        (CRXGate(np.pi / 7), 1 + 2 * np.abs(np.sin(np.pi / 14))),
        (CRYGate(np.pi / 7), 1 + 2 * np.abs(np.sin(np.pi / 14))),
        (CRZGate(np.pi / 7), 1 + 2 * np.abs(np.sin(np.pi / 14))),
        (RXXGate(np.pi / 7), 1 + 2 * np.abs(np.sin(np.pi / 7))),
        (RYYGate(np.pi / 7), 1 + 2 * np.abs(np.sin(np.pi / 7))),
        (RZZGate(np.pi / 7), 1 + 2 * np.abs(np.sin(np.pi / 7))),
    )
    @unpack
    def test_optimal_kappa_for_known_gates(self, instruction, gamma):
        assert np.isclose(qpdbasis_from_gate(instruction).kappa, gamma)

    @data(
        (RXXGate(np.pi / 7), 5, 5),
        (RYYGate(np.pi / 7), 5, 5),
        (RZZGate(np.pi / 7), 5, 5),
        (CRXGate(np.pi / 7), 5, 5),
        (CRYGate(np.pi / 7), 5, 5),
        (CRZGate(np.pi / 7), 5, 5),
        (CXGate(), 5, 5),
        (CZGate(), 5, 5),
        (RZZGate(0), 1, 1),
        (RXXGate(np.pi), 1, 1),
        (CRYGate(np.pi), 5, 5),
    )
    @unpack
    def test_qpdbasis_from_gate_unique_maps(
        self, instruction, q0_num_unique, q1_num_unique
    ):
        """
        Count the number of unique maps with non-zero weight on each qubit.

        Make sure it is as expected based on the instruction provided.
        """
        basis = qpdbasis_from_gate(instruction)
        # Consider only maps with non-zero weight
        relevant_maps = [
            m for m, w in zip(basis.maps, basis.coeffs) if not np.isclose(w, 0)
        ]
        assert len(unique_by_eq(a for (a, b) in relevant_maps)) == q0_num_unique
        assert len(unique_by_eq(b for (a, b) in relevant_maps)) == q1_num_unique

    def test_supported_gates(self):
        gates = supported_gates()
        self.assertEqual(
            {"rxx", "ryy", "rzz", "crx", "cry", "crz", "cx", "cy", "cz", "ch", "csx"}, gates
        )
