# This code is a Qiskit project.

# (C) Copyright IBM 2023.

# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Simulation of precise measurement outcome probabilities."""
from __future__ import annotations

import numpy as np
from qiskit.circuit import QuantumCircuit
from qiskit.quantum_info import Statevector, Operator


_TOLERANCE = 1e-16


def simulate_statevector_outcomes(qc: QuantumCircuit, /) -> dict[int, float]:
    """Return each classical outcome along with its precise probability.

    Circuit can contain mid-circuit, projective measurements, but classical bits cannot be written more than once.
    """
    current = {0: (1.0, Statevector.from_int(0, 2**qc.num_qubits))}
    processed_clbit_indices = set()
    for inst in qc.data:
        if len(inst.clbits) == 0:
            # Evolve each statevector according to the current instruction
            for _, sv in current.values():
                # Calling `_evolve_instruction` rather than `evolve` allows us
                # to avoid a copy.
                Statevector._evolve_instruction(
                    sv, inst.operation, [qc.find_bit(q)[0] for q in inst.qubits]
                )
        else:
            if inst.operation.name != "measure":
                raise ValueError(
                    "Circuit cannot contain a non-measurement operation on classical bit(s)."
                )  # pragma: no cover
            qubit_idx = qc.find_bit(inst.qubits[0])[0]
            clbit_idx = qc.find_bit(inst.clbits[0])[0]
            if clbit_idx in processed_clbit_indices:
                raise ValueError(
                    "Circuits that overwrite a classical bit are not supported."
                )
            processed_clbit_indices.add(clbit_idx)
            # The current instruction is a measurement, so we need to keep
            # track of the statevector and corresponding probability of *both*
            # possible outcomes (although, we truncate states if their
            # probability becomes less than _TOLERANCE).  In the following, we
            # loop through each outcome so far and prepare to update the state.
            pending_delete: list[int] = []
            pending_insert: list[tuple[int, tuple[float, Statevector]]] = []
            for k, (prob, sv) in current.items():
                (prob0, prob1) = sv.probabilities([qubit_idx])
                dims = sv.dims([qubit_idx])  # always going to be (2,) for a qubit
                if np.isclose(prob0, 0, atol=_TOLERANCE):
                    pending_delete.append(k)
                else:
                    proj0 = np.diag([1 / np.sqrt(prob0), 0.0])
                    sv0 = sv.evolve(
                        Operator(proj0, input_dims=dims, output_dims=dims),
                        qargs=[qubit_idx],
                    )
                    current[k] = (prob * prob0, sv0)
                if not np.isclose(prob1, 0, atol=_TOLERANCE):
                    proj1 = np.diag([0.0, 1 / np.sqrt(prob1)])
                    sv1 = sv.evolve(
                        Operator(proj1, input_dims=dims, output_dims=dims),
                        qargs=[qubit_idx],
                    )
                    pending_insert.append((k | (1 << clbit_idx), (prob * prob1, sv1)))
            # A dict's keys cannot be changed while iterating it, so we perform
            # all such updates now that iteration over the dict is complete.
            for k in pending_delete:
                del current[k]
            for k, v in pending_insert:
                current[k] = v

    return {outcome: prob for outcome, (prob, sv) in current.items()}
