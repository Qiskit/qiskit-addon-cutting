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
from qiskit.result import QuasiDistribution
from qiskit.primitives import Sampler as TerraSampler
from qiskit_aer.primitives import Sampler as AerSampler
from qiskit.circuit import QuantumCircuit, ClassicalRegister, CircuitInstruction
from qiskit.circuit.library.standard_gates import XGate

from circuit_knitting_toolbox.utils.observable_grouping import CommutingObservableGroup
from circuit_knitting_toolbox.utils.simulation import ExactSampler
from circuit_knitting_toolbox.circuit_cutting.qpd import (
    SingleQubitQPDGate,
    TwoQubitQPDGate,
    QPDBasis,
)
from circuit_knitting_toolbox.circuit_cutting.cutting_evaluation import (
    _append_measurement_circuit,
    execute_experiments,
)
from circuit_knitting_toolbox.circuit_cutting.qpd import WeightType


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

    def test_execute_experiments(self):
        with self.subTest("Basic test"):
            quasi_dists, coefficients = execute_experiments(
                self.circuit, self.observable, num_samples=50, samplers=self.sampler
            )
            self.assertEqual([[[(QuasiDistribution({3: 1.0}), 0)]]], quasi_dists)
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
                samplers={"A": self.sampler, "B": self.sampler},
            )
            self.assertEqual(
                [
                    [
                        [(QuasiDistribution({1: 1.0}), 0)],
                        [(QuasiDistribution({1: 1.0}), 0)],
                    ]
                ],
                quasi_dists,
            )
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
                "Use circuit_knitting_toolbox.utils.simulation.ExactSampler to generate exact "
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
                "Use circuit_knitting_toolbox.utils.simulation.ExactSampler to generate exact distributions "
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
                "Use circuit_knitting_toolbox.utils.simulation.ExactSampler to generate exact "
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
                "Use circuit_knitting_toolbox.utils.simulation.ExactSampler to generate exact distributions "
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
                == "The number of requested samples must be positive."
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
                == "The keys for the circuits and observabes dicts should be equivalent."
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

    def test_append_measurement_circuit(self):
        qc = self.qc1.copy()
        with self.subTest("In place"):
            qcx = qc.copy()
            assert _append_measurement_circuit(qcx, self.cog, inplace=True) is qcx
        with self.subTest("Out of place"):
            assert _append_measurement_circuit(qc, self.cog) is not qc
        with self.subTest("Correct measurement circuit"):
            qc2 = self.qc2.copy()
            qc2.measure(0, 1)
            qc2.h(1)
            qc2.measure(1, 2)
            assert _append_measurement_circuit(qc, self.cog) == qc2
        with self.subTest("Mismatch between qubit_locations and number of qubits"):
            with pytest.raises(ValueError) as e_info:
                _append_measurement_circuit(qc, self.cog, qubit_locations=[0])
            assert (
                e_info.value.args[0]
                == "qubit_locations has 1 element(s) but the observable(s) have 2 qubit(s)."
            )
        with self.subTest("Mismatched qubits, no qubit_locations provided"):
            cog = CommutingObservableGroup(Pauli("X"), [Pauli("X")])
            with pytest.raises(ValueError) as e_info:
                _append_measurement_circuit(qc, cog)
            assert (
                e_info.value.args[0]
                == "Quantum circuit qubit count (2) does not match qubit count of observable(s) (1).  Try providing `qubit_locations` explicitly."
            )
