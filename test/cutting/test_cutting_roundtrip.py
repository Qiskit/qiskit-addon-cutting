# This code is a Qiskit project.

# (C) Copyright IBM 2023.

# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.
from __future__ import annotations

import pytest

import logging

import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit import CircuitInstruction
from qiskit.circuit.library.standard_gates import (
    RXXGate,
    RYYGate,
    RZZGate,
    CHGate,
    CXGate,
    CYGate,
    CZGate,
    CSGate,
    CSdgGate,
    CRXGate,
    CRYGate,
    CRZGate,
    ECRGate,
    CSXGate,
    CPhaseGate,
)
from qiskit.extensions import UnitaryGate
from qiskit.quantum_info import PauliList, random_unitary
from qiskit.primitives import Estimator

from circuit_knitting.utils.simulation import ExactSampler
from circuit_knitting.cutting import (
    partition_problem,
    execute_experiments,
    reconstruct_expectation_values,
)


logger = logging.getLogger(__name__)


def append_random_unitary(circuit: QuantumCircuit, qubits):
    circuit.append(UnitaryGate(random_unitary(2 ** len(qubits))), qubits)


@pytest.fixture(
    params=[
        [CXGate()],
        [CYGate()],
        [CZGate()],
        [CHGate()],
        [ECRGate()],
        [CSXGate()],
        [CSGate()],
        [CSdgGate()],
        [RYYGate(0.0)],
        [RZZGate(np.pi)],
        [RXXGate(np.pi / 3)],
        [RYYGate(np.pi / 7)],
        [RZZGate(np.pi / 11)],
        [CRXGate(0.0)],
        [CRYGate(np.pi)],
        [CRZGate(np.pi / 2)],
        [CRXGate(np.pi / 3)],
        [CRYGate(np.pi / 7)],
        [CRZGate(np.pi / 11)],
        [RXXGate(np.pi / 3), CRYGate(np.pi / 7)],
        [CPhaseGate(np.pi / 3)],
        [RXXGate(np.pi / 3), CPhaseGate(np.pi / 7)],
    ]
)
def example_circuit(
    request,
) -> tuple[QuantumCircuit, QuantumCircuit, list[list[int]]]:
    """Fixture for an example circuit.

    Returns both the original and one with QPDGates as a tuple.

    Except for the parametrized gates, the system can be separated according to
    the partition labels "AAB".

    """
    qc = QuantumCircuit(3)
    cut_indices = []
    for instruction in request.param:
        append_random_unitary(qc, [0, 1])
        append_random_unitary(qc, [2])
        cut_indices.append(len(qc.data))
        qc.append(CircuitInstruction(instruction, [np.random.choice([0, 1]), 2]))
    qc.barrier()
    append_random_unitary(qc, [0, 1])
    qc.barrier()
    append_random_unitary(qc, [2])

    return qc


def test_cutting_exact_reconstruction(example_circuit):
    """Test gate-cut circuit vs original circuit on statevector simulator

    This test uses a statevector simulator to consider the expectation value of
    each of the :math:`2^N` different possible projection operators in the z
    basis at the end of the circuit (or in other words, the precise probability
    of each full-circuit measurement outcome in the limit of infinite shots).
    This test ensures that each such expectation value remains the same under
    the given QPD decomposed gates.
    """
    qc0 = example_circuit
    qc = qc0.copy()

    observables = PauliList(["III", "IIY", "XII", "XYZ", "iZZZ", "-XZI"])
    phases = np.array([(-1j) ** obs.phase for obs in observables])
    observables_nophase = PauliList(["III", "IIY", "XII", "XYZ", "ZZZ", "XZI"])

    estimator = Estimator()
    exact_expvals = (
        estimator.run([qc0] * len(observables), list(observables)).result().values
    )
    sampler = ExactSampler()
    subcircuits, bases, subobservables = partition_problem(
        qc, "AAB", observables=observables_nophase
    )
    quasi_dists, coefficients = execute_experiments(
        circuits=subcircuits,
        subobservables=subobservables,
        num_samples=1500,
        samplers=sampler,
    )
    simulated_expvals = reconstruct_expectation_values(
        quasi_dists, coefficients, subobservables
    )
    simulated_expvals *= phases

    logger.info("Max error: %f", np.max(np.abs(exact_expvals - simulated_expvals)))

    assert np.allclose(exact_expvals, simulated_expvals, atol=1e-8)
