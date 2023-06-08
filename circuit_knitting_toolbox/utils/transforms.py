# This code is a Qiskit project.

# (C) Copyright IBM 2023.

# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""
Functions for manipulating quantum circuits.

.. currentmodule:: circuit_knitting_toolbox.utils.transforms

.. autosummary::
   :toctree: ../stubs

   separate_circuit
   SeparatedCircuits
"""
from __future__ import annotations

from uuid import uuid4, UUID
from collections import defaultdict
from collections.abc import Sequence, Iterable, Hashable, MutableMapping
from typing import NamedTuple

from rustworkx import PyGraph, connected_components  # type: ignore[attr-defined]
from qiskit.circuit import (
    QuantumCircuit,
    CircuitInstruction,
    QuantumRegister,
    ClassicalRegister,
    Barrier,
    Qubit,
)
from .iteration import unique_by_eq


class SeparatedCircuits(NamedTuple):
    """Named tuple for result of :class:`separate_circuit`."""

    subcircuits: dict[Hashable, QuantumCircuit]
    qubit_map: list[tuple[Hashable, int]]


def separate_circuit(
    circuit: QuantumCircuit,
    partition_labels: Sequence[Hashable] | None = None,
) -> SeparatedCircuits:
    """Separate the circuit into its disconnected components.

    Args:
        circuit: The circuit to separate into disconnected subcircuits

    Returns:
        A named tuple containing the subcircuits and qubit map

    Raises:
        ValueError: The number of partition labels does not equal the number of
            qubits in the input circuit.
    """
    # Split barriers into single-qubit barriers before separating
    new_qc = circuit.copy()
    _split_barriers(new_qc)

    # Generate partition labels based on the connected components of the circuit by default
    if partition_labels is None:
        partition_labels = _partition_labels_from_circuit(new_qc)

    if len(partition_labels) != new_qc.num_qubits:
        raise ValueError(
            f"The number of partition_labels ({len(partition_labels)}) must equal the number of "
            f"qubits in the input circuit ({new_qc.num_qubits})."
        )

    qubit_map, qubits_by_subsystem = _qubit_map_from_partition_labels(partition_labels)

    # Gather instructions corresponding to the same partition together
    subcircuit_data_ids = _separate_instructions_by_partition(new_qc, qubit_map)

    # Create the subcircuits from the instruction subsets
    subcircuits = {}
    for label, subcircuit_data in subcircuit_data_ids.items():
        tmp_data = (new_qc.data[j] for j in subcircuit_data)
        tmp_circ = _circuit_from_instructions(
            tmp_data,
            [new_qc.qubits[j] for j in qubits_by_subsystem[label]],
            new_qc.cregs,
        )
        _combine_barriers(tmp_circ)
        subcircuits[label] = tmp_circ

    return SeparatedCircuits(subcircuits, qubit_map)


def _partition_labels_from_circuit(circuit: QuantumCircuit) -> list[int]:
    """Generate partition labels from the connectivity of a quantum circuit."""
    # Determine connectivity structure of the circuit
    graph: PyGraph = PyGraph()
    graph.add_nodes_from(range(circuit.num_qubits))
    for instruction in circuit.data:
        qubits = instruction.qubits
        for i, q1 in enumerate(qubits):
            for q2 in qubits[i + 1 :]:
                q1_id = circuit.find_bit(q1).index
                q2_id = circuit.find_bit(q2).index
                graph.add_edge(q1_id, q2_id, None)

    # Find the connected subsets of qubits.  For consistency, the subsets
    # should be sorted in ascending order according to the lowest qubit index
    # in each.
    qubit_subsets = connected_components(graph)
    qubit_subsets.sort(key=min)

    # Create partition labels from the connected components
    partition_labels = [-1] * circuit.num_qubits
    for i, subset in enumerate(qubit_subsets):
        for qubit in subset:
            partition_labels[qubit] = i
    assert -1 not in partition_labels

    return partition_labels


def _circuit_from_instructions(
    instructions: Iterable[CircuitInstruction],
    qubits: Sequence[Qubit],
    cregs: Iterable[ClassicalRegister],
) -> QuantumCircuit:
    """
    Create a circuit from instructions.

    This pipeline is designed to pass all the classical register(s) from the
    uncut circuit to each subcircuit, so we add them here.
    """
    circuit = QuantumCircuit()
    circuit.add_register(QuantumRegister(bits=qubits))
    for register in cregs:
        circuit.add_register(register)
    for data in instructions:
        circuit.append(data)

    return circuit


def _qubit_map_from_partition_labels(
    partition_labels: Sequence[Hashable],
) -> tuple[list[tuple[Hashable, int]], dict[Hashable, list[int]]]:
    """Generate a qubit map given a qubit partitioning."""
    qubit_map: list[tuple[Hashable, int]] = []
    qubits_by_subsystem: MutableMapping[Hashable, list[int]] = defaultdict(list)
    for i, qubit_label in enumerate(partition_labels):
        current_label_qubits = qubits_by_subsystem[qubit_label]
        qubit_map.append((qubit_label, len(current_label_qubits)))
        current_label_qubits.append(i)
    return qubit_map, dict(qubits_by_subsystem)


def _separate_instructions_by_partition(
    circuit: QuantumCircuit,
    qubit_map: Sequence[tuple[Hashable, int]],
) -> dict[Hashable, list[int]]:
    """Generate a list of instructions for each partition of the circuit."""
    unique_labels = unique_by_eq(label for label, _ in qubit_map)
    subcircuit_data_ids: dict[Hashable, list[int]] = {
        label: [] for label in unique_labels
    }

    for i, gate in enumerate(circuit.data):
        partitions_spanned = {
            qubit_map[circuit.find_bit(qubit).index][0] for qubit in gate.qubits
        }
        # All qubits for any gate should belong to one subset
        if len(partitions_spanned) != 1:
            assert len(partitions_spanned) != 0
            raise ValueError(
                "The input circuit cannot be separated along specified partitions. "
                f"Operation ({gate.operation.name}) in index ({i}) spans more than "
                "one partition."
            )
        partition_id = partitions_spanned.pop()
        subcircuit_data_ids[partition_id].append(i)

    # Return a regular dict rather than defaultdict
    return dict(subcircuit_data_ids)


def _split_barriers(circuit: QuantumCircuit):
    """Mutate an input circuit to split barriers into single qubit barriers."""
    for i, inst in enumerate(circuit):
        num_qubits = len(inst.qubits)
        if num_qubits == 1 or inst.operation.name != "barrier":
            continue
        barrier_uuid = uuid4()

        # Replace the N-qubit barrier with a single-qubit barrier
        circuit.data[i] = CircuitInstruction(
            Barrier(1, label=barrier_uuid), qubits=[inst.qubits[0]]
        )
        # Insert the remaining single-qubit barriers directly behind the first
        for j in range(1, num_qubits):
            circuit.data.insert(
                i + j,
                CircuitInstruction(
                    Barrier(1, label=barrier_uuid), qubits=[inst.qubits[j]]
                ),
            )


def _combine_barriers(circuit: QuantumCircuit):
    """Mutate input circuit to combine barriers with common UUID labels into a single barrier."""
    # Gather the indices to each group of barriers with common UUID labels
    uuid_map = defaultdict(list)
    for i, inst in enumerate(circuit):
        if (
            inst.operation.name != "barrier"
            or len(inst.qubits) != 1
            or not isinstance(inst.operation.label, UUID)
        ):
            continue
        barrier_uuid = inst.operation.label
        uuid_map[barrier_uuid].append(i)

    # Replace the first single-qubit barrier in each group with the full-sized barrier
    cleanup_inst = []
    for barrier_indices in uuid_map.values():
        qubits = [circuit.data[barrier_id].qubits[0] for barrier_id in barrier_indices]
        new_barrier = CircuitInstruction(Barrier(len(barrier_indices)), qubits=qubits)
        circuit.data[barrier_indices[0]] = new_barrier
        # Gather indices of single-qubit barriers we need to clean up
        cleanup_inst.extend(barrier_indices[1:])

    # Clean up old, single-qubit barriers with uuid labels
    cleanup_inst = sorted(cleanup_inst)
    for shift, inst in enumerate(cleanup_inst):
        del circuit.data[inst - shift]
