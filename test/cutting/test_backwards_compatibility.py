# This code is a Qiskit project.

# (C) Copyright IBM 2023.

# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Tests for backwards compatibility of circuit cutting workflows.

Tests in this file should *not* be updated as API changes are made.  The whole
point of these tests is to make sure old workflows don't intentionally break.
If a workflow is no longer supported, then that test should be removed.

All imports should be in the test functions themselves, to test the stability
of import locations.

It's okay and encouraged to filter deprecation warnings in this file, because
the entire point is to ensure these workflows continue to work, *not* that they
necessarily work without deprecation warnings.

Additionally, all below tests should be excluded from coverage calculations.

"""

import pytest


@pytest.mark.filterwarnings("ignore::PendingDeprecationWarning")
@pytest.mark.filterwarnings("ignore::DeprecationWarning")
@pytest.mark.skipforcoverage
def test_v0_2_cutting_width_workflow():
    """v0.2 workflow to reduce circuit width through cutting

    docs/circuit_cutting/tutorials/gate_cutting_to_reduce_circuit_width.ipynb
    """
    import numpy as np
    from qiskit.circuit.library import EfficientSU2
    from qiskit.quantum_info import PauliList
    from qiskit_aer.primitives import Sampler

    from circuit_knitting_toolbox.circuit_cutting import (
        partition_problem,
        execute_experiments,
        reconstruct_expectation_values,
    )

    circuit = EfficientSU2(4, entanglement="linear", reps=2).decompose()
    circuit.assign_parameters([0.8] * len(circuit.parameters), inplace=True)
    observables = PauliList(["ZZII", "IZZI", "IIZZ", "XIXI", "ZIZZ", "IXIX"])
    subcircuits, bases, subobservables = partition_problem(
        circuit=circuit, partition_labels="AABB", observables=observables
    )
    assert np.prod([basis.overhead for basis in bases]) == pytest.approx(81)

    samplers = {
        "A": Sampler(run_options={"shots": 1}),
        "B": Sampler(run_options={"shots": 1}),
    }
    quasi_dists, coefficients = execute_experiments(
        circuits=subcircuits,
        subobservables=subobservables,
        num_samples=1500,
        samplers=samplers,
    )
    simulated_expvals = reconstruct_expectation_values(
        quasi_dists,
        coefficients,
        subobservables,
    )
    assert len(simulated_expvals) == len(observables)


@pytest.mark.filterwarnings("ignore::PendingDeprecationWarning")
@pytest.mark.filterwarnings("ignore::DeprecationWarning")
@pytest.mark.skipforcoverage
def test_v0_2_cutting_depth_workflow():
    """v0.2 workflow to reduce circuit depth through cutting

    docs/circuit_cutting/tutorials/gate_cutting_to_reduce_circuit_depth.ipynb
    """
    import numpy as np
    from qiskit import transpile
    from qiskit.circuit.library import EfficientSU2
    from qiskit_aer.primitives import Sampler
    from qiskit.providers.fake_provider import FakeHanoiV2 as FakeHanoi
    from qiskit.quantum_info import PauliList

    from circuit_knitting_toolbox.circuit_cutting import (
        decompose_gates,
        execute_experiments,
        reconstruct_expectation_values,
    )

    circuit = EfficientSU2(num_qubits=4, entanglement="circular").decompose()
    circuit.assign_parameters([0.8] * len(circuit.parameters), inplace=True)
    observables = PauliList(["ZZII", "IZZI", "IIZZ", "XIXI", "ZIZZ", "IXIX"])
    backend = FakeHanoi()
    transpile(circuit, backend=backend, initial_layout=[0, 1, 2, 3])
    cut_indices = [
        i
        for i, instruction in enumerate(circuit.data)
        if {circuit.find_bit(q)[0] for q in instruction.qubits} == {0, 3}
    ]
    # Decompose distant CNOTs into TwoQubitQPDGate instances
    qpd_circuit, bases = decompose_gates(circuit, cut_indices)
    assert np.prod([basis.overhead for basis in bases]) == pytest.approx(729)
    from circuit_knitting_toolbox.circuit_cutting.qpd import decompose_qpd_instructions

    for idx in cut_indices:
        qpd_circuit[idx].operation.basis_id = 2
    qpd_circuit_dx = decompose_qpd_instructions(
        qpd_circuit, [[idx] for idx in cut_indices]
    )
    transpile(qpd_circuit_dx, backend=backend, initial_layout=[0, 1, 2, 3])
    sampler = Sampler(run_options={"shots": 1})
    quasi_dists, coefficients = execute_experiments(
        circuits=qpd_circuit,
        subobservables=observables,
        num_samples=1500,
        samplers=sampler,
    )
    simulated_expvals = reconstruct_expectation_values(
        quasi_dists,
        coefficients,
        observables,
    )
    assert len(simulated_expvals) == len(observables)
