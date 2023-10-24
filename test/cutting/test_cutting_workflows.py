# This code is a Qiskit project.

# (C) Copyright IBM 2023.

# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Tests of various cutting workflows, particularly with regard to flexibility in transpilation."""

import pytest
from copy import deepcopy

from qiskit.circuit import QuantumCircuit
from qiskit.circuit.library import EfficientSU2, CXGate
from qiskit.quantum_info import PauliList
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from qiskit.providers.fake_provider import FakeLagosV2
from qiskit_aer.primitives import Sampler

from circuit_knitting.cutting.qpd.instructions import SingleQubitQPDGate
from circuit_knitting.cutting.qpd import QPDBasis
from circuit_knitting.cutting import (
    partition_problem,
    generate_cutting_experiments,
    reconstruct_expectation_values,
)


def test_transpile_before_realizing_basis_id():
    """Test a workflow where a :class:`.SingleQubitQPDGate` is passed through the transpiler."""
    circuit = EfficientSU2(4, entanglement="linear", reps=2).decompose()
    circuit.assign_parameters([0.8] * len(circuit.parameters), inplace=True)
    observables = PauliList(["ZZII"])
    subcircuits, bases, subobservables = partition_problem(
        circuit=circuit, partition_labels="AABB", observables=observables
    )

    # Create a fake backend, and modify the target gate set so it thinks a
    # SingleQubitQPDGate is allowed.
    backend = FakeLagosV2()
    target = deepcopy(backend.target)
    sample_qpd_instruction = SingleQubitQPDGate(QPDBasis.from_instruction(CXGate()), 1)
    target.add_instruction(
        sample_qpd_instruction,
        {(i,): None for i in range(target.num_qubits)},
    )
    pass_manager = generate_preset_pass_manager(3, target=target)

    # Pass each subcircuit through the pass manager.
    subcircuits = {
        label: pass_manager.run(subcircuits["A"])
        for label, circuit in subcircuits.items()
    }


@pytest.mark.parametrize(
    "label1,label2",
    [
        ("foo", frozenset({1, 2, 4})),
        (42, "bar"),
    ],
)
def test_exotic_labels(label1, label2):
    """Test workflow with labels of non-uniform type."""
    circuit = EfficientSU2(4, entanglement="linear", reps=2).decompose()
    circuit.assign_parameters([0.8] * len(circuit.parameters), inplace=True)
    observables = PauliList(["ZZII", "IZZI", "IIZZ", "XIXI", "ZIZZ", "IXIX"])
    subcircuits, bases, subobservables = partition_problem(
        circuit=circuit,
        partition_labels=[label1, label1, label2, label2],
        observables=observables,
    )
    assert set(subcircuits.keys()) == {label1, label2}

    subexperiments, coefficients = generate_cutting_experiments(
        subcircuits, subobservables, num_samples=1500
    )
    assert subexperiments.keys() == subcircuits.keys()

    samplers = {
        label1: Sampler(run_options={"shots": 10}),
        label2: Sampler(run_options={"shots": 10}),
    }
    results = {
        label: sampler.run(subexperiments[label]).result()
        for label, sampler in samplers.items()
    }

    for label in results:
        for i, subexperiment in enumerate(subexperiments[label]):
            results[label].metadata[i]["num_qpd_bits"] = len(subexperiment.cregs[0])

    reconstructed_expvals = reconstruct_expectation_values(
        results,
        coefficients,
        subobservables,
    )
    assert len(reconstructed_expvals) == len(observables)


def test_workflow_with_unused_qubits():
    """Issue #218"""
    qc = QuantumCircuit(2)
    subcircuits, _, subobservables = partition_problem(
        circuit=qc, partition_labels="AB", observables=PauliList(["XX"])
    )
    generate_cutting_experiments(
        subcircuits,
        subobservables,
        num_samples=10,
    )
