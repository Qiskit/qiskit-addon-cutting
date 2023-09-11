# This code is a Qiskit project.

# (C) Copyright IBM 2023.

# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.


import unittest
from copy import deepcopy

import pytest
from qiskit.quantum_info import Pauli, PauliList
from qiskit.primitives import Sampler as TerraSampler, SamplerResult
from qiskit_aer.primitives import Sampler as AerSampler
from qiskit.circuit import QuantumCircuit, ClassicalRegister, CircuitInstruction, Clbit
from qiskit.circuit.library.standard_gates import XGate

from circuit_knitting.utils.observable_grouping import CommutingObservableGroup
from circuit_knitting.utils.simulation import ExactSampler
from circuit_knitting.cutting.qpd import (
    SingleQubitQPDGate,
    TwoQubitQPDGate,
    QPDBasis,
)
from circuit_knitting.cutting.cutting_evaluation import execute_experiments
from circuit_knitting.cutting.qpd import WeightType
from circuit_knitting.cutting import partition_problem


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

        self.circuit = QuantumCircuit(2)
        self.circuit.append(
            CircuitInstruction(
                TwoQubitQPDGate(QPDBasis(maps=[([XGate()], [XGate()])], coeffs=[1.0])),
                qubits=[0, 1],
            )
        )
        self.circuit[0].operation.basis_id = 0
        self.observable = PauliList(["ZZ"])
        self.sampler = ExactSampler()

    @pytest.mark.filterwarnings("ignore::DeprecationWarning")
    def test_execute_experiments(self):
        with self.subTest("Basic test"):
            quasi_dists, coefficients = execute_experiments(
                self.circuit, self.observable, num_samples=50, samplers=self.sampler
            )
            self.assertEqual(
                quasi_dists,
                SamplerResult(quasi_dists=[{3: 1.0}], metadata=[{"num_qpd_bits": 0}]),
            )
            self.assertEqual([(1.0, WeightType.EXACT)], coefficients)
        with self.subTest("Basic test with dicts"):
            circ1 = QuantumCircuit(1)
            circ1.append(
                CircuitInstruction(
                    SingleQubitQPDGate(
                        QPDBasis(maps=[([XGate()], [XGate()])], coeffs=[1.0]),
                        qubit_id=0,
                        label="cut_cx_0",
                    ),
                    qubits=[0],
                )
            )
            circ2 = QuantumCircuit(1)
            circ2.append(
                CircuitInstruction(
                    SingleQubitQPDGate(
                        QPDBasis(maps=[([XGate()], [XGate()])], coeffs=[1.0]),
                        qubit_id=1,
                        label="cut_cx_0",
                    ),
                    qubits=[0],
                )
            )
            subcircuits = {"A": circ1, "B": circ2}
            subobservables = {"A": PauliList(["Z"]), "B": PauliList(["Z"])}
            quasi_dists, coefficients = execute_experiments(
                subcircuits,
                subobservables,
                num_samples=50,
                samplers={"A": self.sampler, "B": deepcopy(self.sampler)},
            )
            comp_result = {
                "A": SamplerResult(
                    quasi_dists=[{1: 1.0}], metadata=[{"num_qpd_bits": 0}]
                ),
                "B": SamplerResult(
                    quasi_dists=[{1: 1.0}], metadata=[{"num_qpd_bits": 0}]
                ),
            }
            self.assertEqual(quasi_dists, comp_result)
            self.assertEqual([(1.0, WeightType.EXACT)], coefficients)
        with self.subTest("Terra/Aer samplers with dicts"):
            circ1 = QuantumCircuit(1)
            circ1.append(
                CircuitInstruction(
                    SingleQubitQPDGate(
                        QPDBasis(maps=[([XGate()], [XGate()])], coeffs=[1.0]),
                        qubit_id=0,
                        label="cut_cx_0",
                    ),
                    qubits=[0],
                )
            )
            circ2 = QuantumCircuit(1)
            circ2.append(
                CircuitInstruction(
                    SingleQubitQPDGate(
                        QPDBasis(maps=[([XGate()], [XGate()])], coeffs=[1.0]),
                        qubit_id=1,
                        label="cut_cx_0",
                    ),
                    qubits=[0],
                )
            )
            subcircuits = {"A": circ1, "B": circ2}
            subobservables = {"A": PauliList(["Z"]), "B": PauliList(["Z"])}
            with pytest.raises(ValueError) as e_info:
                execute_experiments(
                    subcircuits,
                    subobservables,
                    num_samples=50,
                    samplers={
                        "A": TerraSampler(),
                        "B": TerraSampler(),
                    },
                )
            assert e_info.value.args[0] == (
                "qiskit.primitives.Sampler does not support mid-circuit measurements. "
                "Use circuit_knitting.utils.simulation.ExactSampler to generate exact "
                "distributions for each subexperiment."
            )
            with pytest.raises(ValueError) as e_info:
                execute_experiments(
                    subcircuits,
                    subobservables,
                    num_samples=50,
                    samplers={
                        "A": AerSampler(run_options={"shots": None}),
                        "B": AerSampler(run_options={"shots": None}),
                    },
                )
            assert e_info.value.args[0] == (
                "qiskit_aer.primitives.Sampler does not support mid-circuit measurements when shots is None. "
                "Use circuit_knitting.utils.simulation.ExactSampler to generate exact distributions "
                "for each subexperiment."
            )
        with self.subTest("Terra sampler"):
            with pytest.raises(ValueError) as e_info:
                execute_experiments(
                    self.circuit,
                    self.observable,
                    num_samples=50,
                    samplers=TerraSampler(),
                )
            assert e_info.value.args[0] == (
                "qiskit.primitives.Sampler does not support mid-circuit measurements. "
                "Use circuit_knitting.utils.simulation.ExactSampler to generate exact "
                "distributions for each subexperiment."
            )
        with self.subTest("Aer sampler no shots"):
            with pytest.raises(ValueError) as e_info:
                execute_experiments(
                    self.circuit,
                    self.observable,
                    num_samples=50,
                    samplers=AerSampler(run_options={"shots": None}),
                )
            assert e_info.value.args[0] == (
                "qiskit_aer.primitives.Sampler does not support mid-circuit measurements when shots is None. "
                "Use circuit_knitting.utils.simulation.ExactSampler to generate exact distributions "
                "for each subexperiment."
            )
        with self.subTest("Bad samplers"):
            with pytest.raises(ValueError) as e_info:
                execute_experiments(
                    self.circuit, self.observable, num_samples=50, samplers=42
                )
            assert e_info.value.args[0] == (
                "The samplers input argument must be either an instance of qiskit.primitives.BaseSampler "
                "or a mapping from partition labels to qiskit.primitives.BaseSampler instances."
            )
            with pytest.raises(ValueError) as e_info:
                execute_experiments(
                    self.circuit, self.observable, num_samples=50, samplers={42: 42}
                )
            assert e_info.value.args[0] == (
                "The samplers input argument must be either an instance of qiskit.primitives.BaseSampler "
                "or a mapping from partition labels to qiskit.primitives.BaseSampler instances."
            )
        with self.subTest("Negative num-samples"):
            with pytest.raises(ValueError) as e_info:
                execute_experiments(
                    self.circuit, self.observable, num_samples=-1, samplers=self.sampler
                )
            assert (
                e_info.value.args[0]
                == "The number of requested samples must be at least 1."
            )
        with self.subTest("Dict of non-unique samplers"):
            qc = QuantumCircuit(2)
            qc.x(0)
            qc.cnot(0, 1)
            subcircuits, _, subobservables = partition_problem(
                circuit=qc, partition_labels="AB", observables=PauliList(["XX"])
            )
            with pytest.raises(ValueError) as e_info:
                execute_experiments(
                    subcircuits,
                    subobservables,
                    num_samples=10,
                    samplers={"A": self.sampler, "B": self.sampler},
                )
            assert (
                e_info.value.args[0]
                == "Currently, if a samplers dict is passed to execute_experiments(), then each sampler must be unique; however, subsystems A and B were passed the same sampler."
            )
        with self.subTest("Incompatible inputs"):
            with pytest.raises(ValueError) as e_info:
                execute_experiments(
                    {"A": self.circuit},
                    self.observable,
                    num_samples=100,
                    samplers=self.sampler,
                )
            assert e_info.value.args[0] == (
                "If a partition mapping (dict[label, subcircuit]) is passed as the circuits argument, a "
                "partition mapping (dict[label, subobservables]) is expected as the subobservables argument."
            )
            with pytest.raises(ValueError) as e_info:
                execute_experiments(
                    self.circuit,
                    {"A": self.observable},
                    num_samples=100,
                    samplers=self.sampler,
                )
            assert e_info.value.args[0] == (
                "If a QuantumCircuit is passed as the circuits argument, a PauliList "
                "is expected as the subobservables argument."
            )
            with pytest.raises(ValueError) as e_info:
                execute_experiments(
                    {"B": self.circuit},
                    {"A": self.observable},
                    num_samples=100,
                    samplers=self.sampler,
                )
            assert (
                e_info.value.args[0]
                == "The keys for the circuits and observables dicts should be equivalent."
            )
            with pytest.raises(ValueError) as e_info:
                execute_experiments(
                    {"A": self.circuit},
                    {"A": self.observable},
                    num_samples=100,
                    samplers={"B": self.sampler},
                )
            assert (
                e_info.value.args[0]
                == "The keys for the circuits and samplers dicts should be equivalent."
            )
        with self.subTest("Single qubit gate in QuantumCircuit input"):
            circuit = QuantumCircuit(1)
            circuit.append(
                CircuitInstruction(
                    SingleQubitQPDGate(
                        QPDBasis(maps=[([XGate()],)], coeffs=[1.0]), qubit_id=0
                    ),
                    qubits=[0],
                )
            )
            observable = PauliList(["Z"])
            with pytest.raises(ValueError) as e_info:
                execute_experiments(
                    circuit,
                    observable,
                    num_samples=50,
                    samplers=self.sampler,
                )
            assert (
                e_info.value.args[0]
                == "SingleQubitQPDGates are not supported in unseparable circuits."
            )
        with self.subTest("Classical regs on input"):
            circuit = self.circuit.copy()
            circuit.add_bits([Clbit()])
            with pytest.raises(ValueError) as e_info:
                execute_experiments(
                    circuit,
                    self.observable,
                    num_samples=50,
                    samplers=self.sampler,
                )
            assert (
                e_info.value.args[0]
                == "Circuits input to execute_experiments should contain no classical registers or bits."
            )
