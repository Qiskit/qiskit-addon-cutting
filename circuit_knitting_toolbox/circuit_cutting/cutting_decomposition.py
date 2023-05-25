# This code is a Qiskit project.

# (C) Copyright IBM 2023.

# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Functions for decomposing circuits and observables for cutting."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence, Hashable
from typing import NamedTuple

from qiskit.circuit import (
    QuantumCircuit,
    CircuitInstruction,
    Barrier,
)
from qiskit.quantum_info import PauliList

from ..utils.observable_grouping import observables_restricted_to_subsystem
from ..utils.transforms import separate_circuit
from .qpd.qpd_basis import QPDBasis
from .qpd.instructions import TwoQubitQPDGate


class PartitionedProblem(NamedTuple):
    """The result of decomposing and separating a circuit and observable(s)."""

    subcircuits: dict[str | int, QuantumCircuit]
    bases: list[QPDBasis]
    subobservables: dict[str | int, QuantumCircuit] | None = None


def partition_circuit_qubits(
    circuit: QuantumCircuit, partition_labels: Sequence[Hashable], inplace: bool = False
) -> QuantumCircuit:
    r"""
    Replace all nonlocal gates belonging to more than one partition with instances of :class:`TwoQubitQPDGate`.

    :class:`TwoQubitQPDGate`\ s belonging to a single partition will not be affected.

    Args:
        circuit: The circuit to partition
        partition_labels: A sequence containing a partition label for each qubit in the
            input circuit. Nonlocal gates belonging to more than one partition
            will be replaced with QPDGates.
        inplace: Flag denoting whether to copy the input circuit before acting on it

    Returns:
        The output circuit with each nonlocal gate spanning two partitions replaced by a
        :class:`TwoQubitQPDGate`

    Raises:
        ValueError: The length of partition_labels does not equal the number of qubits in the circuit.
        ValueError: Input circuit contains unsupported gate.
    """
    if len(partition_labels) != len(circuit.qubits):
        raise ValueError(
            f"Length of partition_labels ({len(partition_labels)}) does not equal the number of qubits "
            f"in the input circuit ({len(circuit.qubits)})."
        )

    if not inplace:
        circuit = circuit.copy()

    # Find 2-qubit gates spanning more than one partition and replace it with a QPDGate.
    for i, instruction in enumerate(circuit.data):
        if instruction.operation.name == "barrier":
            continue
        qubit_indices = [circuit.find_bit(qubit).index for qubit in instruction.qubits]
        partitions_spanned = {partition_labels[idx] for idx in qubit_indices}

        # Ignore local gates and gates that span only one partition
        if (
            len(qubit_indices) <= 1
            or len(partitions_spanned) == 1
            or isinstance(instruction.operation, Barrier)
        ):
            continue

        if len(qubit_indices) > 2:
            raise ValueError(
                "Decomposition is only supported for two-qubit gates. Cannot "
                f"decompose ({instruction.operation.name})."
            )

        # Nonlocal gate exists in two separate partitions
        if isinstance(instruction.operation, TwoQubitQPDGate):
            continue

        decomposition = QPDBasis.from_gate(instruction.operation)
        qpd_gate = TwoQubitQPDGate(
            decomposition, label=f"qpd_{instruction.operation.name}"
        )
        circuit.data[i] = CircuitInstruction(qpd_gate, qubits=qubit_indices)

    return circuit


def decompose_gates(
    circuit: QuantumCircuit, gate_ids: Sequence[int], inplace: bool = False
) -> tuple[QuantumCircuit, list[QPDBasis]]:
    r"""
    Transform specified gates into :class:`TwoQubitQPDGate`\ s.

    Args:
        circuit: The circuit containing gates to be decomposed
        gate_ids: The indices of the gates to decompose
        inplace: Flag denoting whether to copy the input circuit before acting on it

    Returns:
        A copy of the input circuit with the specified gates replaced with :class:`TwoQubitGate`\ s
        and a list of ``QPDBasis`` instances -- one for each decomposed gate.

    Raises:
        ValueError: The input circuit should contain no classical bits or registers.
    """
    if len(circuit.cregs) != 0 or circuit.num_clbits != 0:
        raise ValueError(
            "Circuits input to execute_experiments should contain no classical registers or bits."
        )
    # Replace specified gates with TwoQubitQPDGates
    if not inplace:
        circuit = circuit.copy()

    bases = []
    for gate_id in gate_ids:
        gate = circuit.data[gate_id]
        qubit_indices = [circuit.find_bit(qubit).index for qubit in gate.qubits]
        decomposition = QPDBasis.from_gate(gate.operation)
        bases.append(decomposition)
        qpd_gate = TwoQubitQPDGate(decomposition, label=f"cut_{gate.operation.name}")
        circuit.data[gate_id] = CircuitInstruction(qpd_gate, qubits=qubit_indices)

    return circuit, bases


def partition_problem(
    circuit: QuantumCircuit,
    partition_labels: Sequence[str | int],
    observables: PauliList | None = None,
) -> PartitionedProblem:
    """
    Separate an input circuit and observable(s) along qubit partition labels.

    Circuit qubits with matching partition labels will be grouped together, and non-local
    operations spanning more than one partition will be decomposed and replaced with
    probabilistic local operations.

    If provided, the observables will be separated along the boundaries specified by
    ``partition_labels``.

    Args:
        circuit: The circuit to separate
        partition_labels: A sequence of labels, such that each label corresponds
            to the circuit qubit with the same index
        observables: The observables to separate

    Returns:
        A tuple containing a dictionary mapping a partition label to a subcircuit,
        a list of QPD bases (one for each circuit gate or wire which was decomposed),
        and, optionally, a dictionary mapping a partition label to a list of Pauli observables.


    Raises:
        ValueError: The number of partition labels does not equal the number of qubits in the circuit.
        ValueError: An input observable acts on a different number of qubits than the input circuit.
        ValueError: The input circuit should contain no classical bits or registers.
    """
    if len(partition_labels) != circuit.num_qubits:
        raise ValueError(
            f"The number of partition labels ({len(partition_labels)}) must equal the number "
            f"of qubits in the circuit ({circuit.num_qubits})."
        )
    if observables is not None and any(
        len(obs) != circuit.num_qubits for obs in observables
    ):
        raise ValueError(
            "An input observable acts on a different number of qubits than the input circuit."
        )
    if len(circuit.cregs) != 0 or circuit.num_clbits != 0:
        raise ValueError(
            "Circuits input to execute_experiments should contain no classical registers or bits."
        )

    # Partition the circuit with TwoQubitQPDGates and assign the order via their labels
    qpd_circuit = partition_circuit_qubits(circuit, partition_labels)

    bases = []
    i = 0
    for inst in qpd_circuit.data:
        if isinstance(inst.operation, TwoQubitQPDGate):
            bases.append(inst.operation.basis)
            inst.operation.label = inst.operation.label + f"_{i}"
            i += 1

    # Separate the decomposed circuit into its subcircuits
    qpd_circuit_dx = qpd_circuit.decompose(TwoQubitQPDGate)
    separated_circs = separate_circuit(qpd_circuit_dx, partition_labels)

    # Decompose the observables, if provided
    subobservables_by_subsystem = None
    if observables:
        subobservables_by_subsystem = decompose_observables(
            observables, partition_labels
        )

    return PartitionedProblem(
        separated_circs.subcircuits,
        bases,
        subobservables=subobservables_by_subsystem,
    )


def decompose_observables(
    observables: PauliList, partition_labels: Sequence[str | int]
) -> dict[str | int, PauliList]:
    """
    Decompose a list of observables with respect to some qubit partition labels.

    Args:
        observables: A list of observables to decompose
        partition_labels: A sequence of partition labels, such that each label
            corresponds to the qubit in the same index

    Returns:
        A dictionary mapping a partition to its associated sub-observables
    """
    qubits_by_subsystem = defaultdict(list)
    for i, label in enumerate(partition_labels):
        qubits_by_subsystem[label].append(i)
    qubits_by_subsystem = dict(qubits_by_subsystem)  # type: ignore

    subobservables_by_subsystem = {
        label: observables_restricted_to_subsystem(qubits, observables)
        for label, qubits in qubits_by_subsystem.items()
    }

    return subobservables_by_subsystem
