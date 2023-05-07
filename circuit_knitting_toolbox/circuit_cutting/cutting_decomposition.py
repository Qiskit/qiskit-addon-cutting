# This code is a Qiskit project.

# (C) Copyright IBM 2023.

# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Functions for conducting circuit cutting."""

from __future__ import annotations

from collections.abc import Sequence, Hashable

from qiskit.circuit import (
    QuantumCircuit,
    CircuitInstruction,
    Barrier,
)
from qiskit.quantum_info import PauliList

from .qpd.qpd_basis import QPDBasis
from .qpd.instructions import TwoQubitQPDGate
from ..utils.transforms import separate_circuit


def partition_circuit_qubits(
    circuit: QuantumCircuit, partition_labels: Sequence[Hashable]
) -> QuantumCircuit:
    r"""
    Replace all nonlocal gates belonging to more than one partition with instances of :class:`TwoQubitQPDGate`.

    :class:`TwoQubitQPDGate`\ s belonging to a single partition will not be affected.

    Args:
        circuit: The circuit to partition
        partition_labels: A sequence containing a partition label for each qubit in the
            input circuit. Nonlocal gates belonging to more than one partition
            will be replaced with QPDGates.

    Returns:
        (:class:`~qiskit.QuantumCircuit`): The output circuit with each nonlocal gate spanning
            two partitions replaced by a :class:`TwoQubitQPDGate`.

    Raises:
        ValueError:
            Length of partition_labels does not equal the number of qubits in the circuit
        ValueError:
            Cannot decompose gate which acts on more than 2 qubits
    """
    if len(partition_labels) != len(circuit.qubits):
        raise ValueError(
            f"Length of partition_labels ({len(partition_labels)}) does not equal the number of qubits "
            f"in the input circuit ({len(circuit.qubits)})."
        )

    new_qc = circuit.copy()

    # Find 2-qubit gates spanning more than one partition and replace it with a QPDGate.
    for i, gate in enumerate(new_qc.data):
        if gate.operation.name == "barrier":
            continue
        qubit_indices = [new_qc.find_bit(qubit).index for qubit in gate.qubits]
        partitions_spanned = {partition_labels[idx] for idx in qubit_indices}
        # Ignore local gates and gates that span only one partition
        if (
            len(qubit_indices) <= 1
            or len(partitions_spanned) == 1
            or isinstance(gate.operation, Barrier)
        ):
            continue

        if len(qubit_indices) > 2:
            raise ValueError(
                f"Decomposition is only supported for two-qubit gates. Cannot decompose ({gate.operation.name})."
            )

        # Nonlocal gate exists in two separate partitions
        if isinstance(gate.operation, TwoQubitQPDGate):
            continue

        decomposition = QPDBasis.from_gate(gate.operation)
        qpd_gate = TwoQubitQPDGate(decomposition, label=f"qpd_{gate.operation.name}")
        new_qc.data[i] = CircuitInstruction(qpd_gate, qubits=qubit_indices)

    return new_qc
