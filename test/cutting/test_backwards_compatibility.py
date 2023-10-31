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
def test_v0_3_cutting_width_workflow():
    """v0.3 workflow to reduce circuit width through cutting

    docs/circuit_cutting/tutorials/01_gate_cutting_to_reduce_circuit_width.ipynb
    """
    import numpy as np
    from qiskit.circuit.library import EfficientSU2
    from qiskit.quantum_info import PauliList
    from qiskit_aer.primitives import Sampler

    from circuit_knitting.cutting import (
        partition_problem,
        execute_experiments,
        reconstruct_expectation_values,
    )

    circuit = EfficientSU2(4, entanglement="linear", reps=2).decompose()
    circuit.assign_parameters([0.4] * len(circuit.parameters), inplace=True)
    observables = PauliList(["ZZII", "IZZI", "IIZZ", "XIXI", "ZIZZ", "IXIX"])
    partitioned_problem = partition_problem(
        circuit=circuit, partition_labels="AABB", observables=observables
    )
    subcircuits = partitioned_problem.subcircuits
    subobservables = partitioned_problem.subobservables
    bases = partitioned_problem.bases
    subcircuits["A"]
    assert np.prod([basis.overhead for basis in bases]) == pytest.approx(81)
    samplers = {
        "A": Sampler(run_options={"shots": 10}),
        "B": Sampler(run_options={"shots": 10}),
    }
    quasi_dists, coefficients = execute_experiments(
        circuits=subcircuits,
        subobservables=subobservables,
        num_samples=np.inf,
        samplers=samplers,
    )
    reconstructed_expvals = reconstruct_expectation_values(
        quasi_dists,
        coefficients,
        subobservables,
    )
    assert len(reconstructed_expvals) == len(observables)


@pytest.mark.filterwarnings("ignore::PendingDeprecationWarning")
@pytest.mark.filterwarnings("ignore::DeprecationWarning")
@pytest.mark.skipforcoverage
def test_v0_3_cutting_depth_workflow():
    """v0.3 workflow to reduce circuit depth through cutting

    docs/circuit_cutting/tutorials/02_gate_cutting_to_reduce_circuit_depth.ipynb
    """
    import numpy as np
    from qiskit import transpile
    from qiskit.circuit.library import EfficientSU2
    from qiskit_aer.primitives import Sampler
    from qiskit.providers.fake_provider import FakeHanoiV2 as FakeHanoi
    from qiskit.quantum_info import PauliList

    from circuit_knitting.cutting import (
        cut_gates,
        execute_experiments,
        reconstruct_expectation_values,
    )

    circuit = EfficientSU2(num_qubits=4, entanglement="circular").decompose()
    circuit.assign_parameters([0.4] * len(circuit.parameters), inplace=True)
    observables = PauliList(["ZZII", "IZZI", "IIZZ", "XIXI", "ZIZZ", "IXIX"])
    backend = FakeHanoi()
    transpiled_qc = transpile(circuit, backend=backend, initial_layout=[0, 1, 2, 3])
    transpiled_qc.depth()

    cut_indices = [
        i
        for i, instruction in enumerate(circuit.data)
        if {circuit.find_bit(q)[0] for q in instruction.qubits} == {0, 3}
    ]
    qpd_circuit, bases = cut_gates(circuit, cut_indices)
    assert np.prod([basis.overhead for basis in bases]) == pytest.approx(729)

    from circuit_knitting.cutting.qpd import decompose_qpd_instructions

    # Set some arbitrary bases to which each of the 3 QPD gates should decompose
    arbitrary_basis_ids = [1, 3, 4]
    for idx, basis_id in zip(cut_indices, arbitrary_basis_ids):
        qpd_circuit[idx].operation.basis_id = basis_id
    # Decompose QPDGates in a circuit into Qiskit operations and measurements
    qpd_circuit_dx = decompose_qpd_instructions(
        qpd_circuit, [[idx] for idx in cut_indices]
    )
    # Transpile the decomposed circuit to the same layout
    transpiled_qpd_circuit = transpile(
        qpd_circuit_dx, backend=backend, initial_layout=[0, 1, 2, 3]
    )
    transpiled_qc.depth()
    transpiled_qpd_circuit.depth()
    sampler = Sampler(run_options={"shots": 10})
    quasi_dists, coefficients = execute_experiments(
        circuits=qpd_circuit,
        subobservables=observables,
        num_samples=np.inf,
        samplers=sampler,
    )
    reconstructed_expvals = reconstruct_expectation_values(
        quasi_dists,
        coefficients,
        observables,
    )
    assert len(reconstructed_expvals) == len(observables)


@pytest.mark.filterwarnings("ignore::PendingDeprecationWarning")
@pytest.mark.filterwarnings("ignore::DeprecationWarning")
@pytest.mark.skipforcoverage
def test_v0_3_wire_cutting_workflow():
    """v0.3 wire cutting workflow

    docs/circuit_cutting/tutorials/03_wire_cutting_via_move_instruction.ipynb
    """
    import numpy as np
    from qiskit import QuantumCircuit
    from qiskit.quantum_info import PauliList
    from qiskit_aer.primitives import Sampler

    from circuit_knitting.cutting.instructions import Move
    from circuit_knitting.cutting import (
        partition_problem,
        execute_experiments,
        reconstruct_expectation_values,
    )

    qc_0 = QuantumCircuit(7)
    for i in range(7):
        qc_0.rx(np.pi / 4, i)
    qc_0.cx(0, 3)
    qc_0.cx(1, 3)
    qc_0.cx(2, 3)
    qc_0.cx(3, 4)
    qc_0.cx(3, 5)
    qc_0.cx(3, 6)
    qc_0.cx(0, 3)
    qc_0.cx(1, 3)
    qc_0.cx(2, 3)
    observables_0 = PauliList(["ZIIIIII", "IIIZIII", "IIIIIIZ"])
    qc_1 = QuantumCircuit(8)
    for i in [*range(4), *range(5, 8)]:
        qc_1.rx(np.pi / 4, i)
    qc_1.cx(0, 3)
    qc_1.cx(1, 3)
    qc_1.cx(2, 3)
    qc_1.append(Move(), [3, 4])
    qc_1.cx(4, 5)
    qc_1.cx(4, 6)
    qc_1.cx(4, 7)
    qc_1.append(Move(), [4, 3])
    qc_1.cx(0, 3)
    qc_1.cx(1, 3)
    qc_1.cx(2, 3)
    observables_1 = PauliList(["ZIIIIIII", "IIIIZIII", "IIIIIIIZ"])
    partitioned_problem = partition_problem(
        circuit=qc_1, partition_labels="AAAABBBB", observables=observables_1
    )
    subcircuits = partitioned_problem.subcircuits
    subobservables = partitioned_problem.subobservables
    bases = partitioned_problem.bases
    subcircuits["A"]
    assert np.prod([basis.overhead for basis in bases]) == pytest.approx(256)
    samplers = {
        "A": Sampler(run_options={"shots": 10}),
        "B": Sampler(run_options={"shots": 10}),
    }
    quasi_dists, coefficients = execute_experiments(
        circuits=subcircuits,
        subobservables=subobservables,
        num_samples=np.inf,
        samplers=samplers,
    )
    reconstructed_expvals = reconstruct_expectation_values(
        quasi_dists,
        coefficients,
        subobservables,
    )
    assert len(reconstructed_expvals) == len(observables_0)
