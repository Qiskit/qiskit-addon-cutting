# This code is a Qiskit project.

# (C) Copyright IBM 2023.

# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Tests of various cutting workflows, particularly with regard to flexibility in transpilation."""

from copy import deepcopy

from qiskit.circuit.library import EfficientSU2, CXGate
from qiskit.quantum_info import PauliList
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from qiskit.providers.fake_provider import FakeLagosV2

from circuit_knitting.cutting.qpd.instructions import SingleQubitQPDGate
from circuit_knitting.cutting.qpd import QPDBasis
from circuit_knitting.cutting import partition_problem


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
    backend = deepcopy(backend)
    sample_qpd_instruction = SingleQubitQPDGate(QPDBasis.from_gate(CXGate()), 1)
    backend.target.add_instruction(
        sample_qpd_instruction,
        {(i,): None for i in range(backend.target.num_qubits)},
    )
    pass_manager = generate_preset_pass_manager(3, backend)

    # Pass each subcircuit through the pass manager.
    subcircuits = {
        label: pass_manager.run(subcircuits["A"])
        for label, circuit in subcircuits.items()
    }
