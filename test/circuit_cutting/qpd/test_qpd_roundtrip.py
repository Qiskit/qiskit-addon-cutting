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

from collections import defaultdict
import itertools
import logging

import numpy as np
from qiskit import QuantumCircuit, execute
from qiskit_aer import AerSimulator
from qiskit.circuit import CircuitInstruction, ClassicalRegister
from qiskit.circuit.library.standard_gates import (
    RXXGate,
    RYYGate,
    RZZGate,
    CXGate,
    CZGate,
    CRXGate,
    CRYGate,
    CRZGate,
)
from qiskit.extensions import UnitaryGate
from qiskit.primitives import Estimator
from qiskit.quantum_info import PauliList, random_unitary

from circuit_knitting_toolbox.circuit_cutting.qpd import (
    QPDBasis,
    TwoQubitQPDGate,
    decompose_qpd_instructions,
)
from circuit_knitting_toolbox.utils.observable_grouping import (
    observables_restricted_to_subsystem,
    ObservableCollection,
)

from circuit_knitting_toolbox.circuit_cutting.cutting_evaluation import (
    append_measurement_circuit,
)
from circuit_knitting_toolbox.circuit_cutting.cutting_reconstruction import (
    process_outcome,
)

from circuit_knitting_toolbox.utils.bitwise import bit_count
from circuit_knitting_toolbox.utils.transforms import separate_circuit
from circuit_knitting_toolbox.utils.iteration import strict_zip
from circuit_knitting_toolbox.circuit_cutting.qpd.utils.simulation import (
    simulate_statevector_outcomes,
)

logger = logging.getLogger(__name__)


def append_random_unitary(circuit: QuantumCircuit, qubits):
    circuit.append(UnitaryGate(random_unitary(2 ** len(qubits))), qubits)


@pytest.fixture(
    params=[
        [CXGate()],
        [CZGate()],
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
    ]
)
def example_circuit_pair(
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

    qc0 = qc.copy()

    # Replace the gates to be cut with instances of QPDGate
    qpd_gate_ids = []
    for idx in cut_indices:
        ci = qc.data[idx]
        basis = QPDBasis.from_gate(ci.operation)
        label = f"qpd_{ci.operation.name}"
        qc.data[idx] = ci.replace(TwoQubitQPDGate(basis, label=label))
        qpd_gate_ids.append([idx])

    # Return both the original circuit and the one with QPDGates
    return qc0, qc, qpd_gate_ids


def test_qpd_exact_reconstruction(example_circuit_pair):
    """Test gate-cut circuit vs original circuit on statevector simulator

    This test uses a statevector simulator to consider the expectation value of
    each of the :math:`2^N` different possible projection operators in the z
    basis at the end of the circuit (or in other words, the precise probability
    of each full-circuit measurement outcome in the limit of infinite shots).
    This test ensures that each such expectation value remains the same under
    the given QPD decomposed gates.
    """
    qc0, qpd_qc, qpd_gate_ids = example_circuit_pair
    qc = qpd_qc.copy()

    sim = AerSimulator()
    qc0.save_statevector("final")
    probs0 = execute(qc0, sim).result().data()["final"].probabilities()

    bases = [qc.data[decomp_ids[0]].operation.basis for decomp_ids in qpd_gate_ids]
    estimated_probs = np.zeros_like(probs0)
    for map_ids in itertools.product(*[range(len(basis.coeffs)) for basis in bases]):
        coeff = np.prod(
            [basis.coeffs[map_id] for basis, map_id in strict_zip(bases, map_ids)]
        )
        logger.info("processing map_ids: %s", map_ids)
        decomp_qc = decompose_qpd_instructions(qpd_qc, qpd_gate_ids, map_ids)
        num_qubits = decomp_qc.num_qubits
        num_qpd_bits = decomp_qc.num_clbits
        decomp_qc.add_register(ClassicalRegister(num_qubits))
        decomp_qc.measure(
            range(num_qubits), range(num_qpd_bits, num_qpd_bits + num_qubits)
        )
        outcomes = simulate_statevector_outcomes(decomp_qc)
        for outcome, prob in outcomes.items():
            qpd_outcomes = outcome & ((1 << num_qpd_bits) - 1)
            meas_outcomes = outcome >> num_qpd_bits
            # qpd_factor will be -1 or +1, depending on the overall parity of qpd
            # measurements.
            qpd_factor = 1 - 2 * (bit_count(qpd_outcomes) & 1)
            estimated_probs[meas_outcomes] += coeff * prob * qpd_factor

    logger.info("Max error: %f", np.max(np.abs(probs0 - estimated_probs)))
    assert np.allclose(probs0, estimated_probs, atol=1e-8)


def test_qpd_exact_expectation_values(example_circuit_pair):
    qc0, qpd_qc, qpd_gate_ids = example_circuit_pair
    qc = qpd_qc.copy()

    # Much of the below logic depends on a specific example circuit, given
    # above by example_circuit_pair.
    N = qc0.num_qubits
    assert N == 3

    # qubits 0 and 1 end up in the first subcircuit, qubit 2 in the second
    partition_labels = "AAB"

    # Translate partition_labels to qubits_by_subsystem.
    qubits_by_subsystem = defaultdict(list)
    for i, label in enumerate(partition_labels):
        qubits_by_subsystem[label].append(i)
    qubits_by_subsystem = dict(qubits_by_subsystem)
    assert qubits_by_subsystem == {"A": [0, 1], "B": [2]}

    # Choose a grab-bag of Pauli observables for which to test.
    observables = PauliList(["III", "IIY", "XII", "XYZ", "iZZZ", "-XZI"])

    # Calculate their "exact" expectation values.
    expvals0 = (
        Estimator().run([qc0] * len(observables), list(observables)).result().values
    )

    # Construct "subobservables", i.e., for each subsystem, a list of
    # the observables (above) restricted to that subsystem.
    subobservables_by_subsystem = {
        label: observables_restricted_to_subsystem(qubits, observables)
        for label, qubits in qubits_by_subsystem.items()
    }
    assert subobservables_by_subsystem == {
        "A": PauliList(["II", "IY", "II", "YZ", "ZZ", "ZI"]),
        "B": PauliList(list("IIXXZX")),
    }

    # Construct the data structure that contains most of what we need in order
    # to take expectation values of our observables within each subsystem.
    subsystem_observables = {
        label: ObservableCollection(subobservables)
        for label, subobservables in subobservables_by_subsystem.items()
    }

    # Iterate over each map_ids possibility and perform the actual simulation.
    expvals = np.zeros(len(observables))
    bases = [qc.data[decomp[0]].operation.basis for decomp in qpd_gate_ids]
    for map_ids in itertools.product(*[range(len(basis.coeffs)) for basis in bases]):
        coeff = np.prod(
            [basis.coeffs[map_id] for basis, map_id in strict_zip(bases, map_ids)]
        )
        logger.info("processing map_ids: %s", map_ids)

        # Decompose and simulate a subexperiment
        decomp_qc = decompose_qpd_instructions(qpd_qc, qpd_gate_ids, map_ids)
        assert len(decomp_qc.cregs) == 1
        subcircuits = separate_circuit(
            decomp_qc, partition_labels=partition_labels
        ).subcircuits
        assert subcircuits.keys() == subsystem_observables.keys()
        # The current contribution to the global observable's expectation value
        # is a *product* of the subsystem expectation values.  So, we
        # initialize an array with all ones and update it by multiplication.
        current_expvals = np.ones((len(expvals),))
        # Loop over subsystems.
        for label, subcircuit in subcircuits.items():
            so = subsystem_observables[label]
            # Create an accumulator for subsystem expectation values.
            subsystem_expvals = [
                np.zeros(len(cog.commuting_observables)) for cog in so.groups
            ]
            # Loop over sets of mutually commuting observables, each of which
            # is represented by its "general subobservable."
            for j, cog in enumerate(so.groups):
                # Append the appropriate measurements to subcircuit, yielding
                meas_qc = append_measurement_circuit(subcircuit, cog)
                assert len(meas_qc.cregs) == 2
                # Perform the actual simulation of meas_qc.
                outcomes = simulate_statevector_outcomes(meas_qc)
                # Iterate over the outcomes.
                for outcome, prob in outcomes.items():
                    num_qpd_bits = len(meas_qc.cregs[-2])
                    subsystem_expvals[j] += prob * process_outcome(
                        num_qpd_bits, cog, outcome
                    )
            # Factor the contribution from the current subsystem into each
            # global expectation value for the current map_ids.
            for k, subobservable in enumerate(subobservables_by_subsystem[label]):
                # NOTE: since this is an exact method, we need only consider
                # index [0] on the following line.
                j, m = so.lookup[subobservable][0]
                current_expvals[k] *= subsystem_expvals[j][m]
        # Accumulate the overall contribution from the current map_ids.
        expvals += coeff * current_expvals

    # Account for the phases in the original Pauli observables.
    phases = np.array([(-1j) ** obs.phase for obs in observables])
    expvals = expvals * phases

    logger.info("Max error: %f", np.max(np.abs(expvals0 - expvals)))

    assert np.allclose(expvals0, expvals, atol=1e-8)
