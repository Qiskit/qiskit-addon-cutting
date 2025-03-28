# This code is a Qiskit project.

# (C) Copyright IBM 2023.

# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

# Reminder: update the RST file in docs/apidocs when adding new interfaces.
"""Simulation of precise measurement outcome probabilities."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence
from typing import Any

import numpy as np
from qiskit.circuit import ControlFlowOp, QuantumCircuit
from qiskit.quantum_info import Statevector, Operator
from qiskit.primitives.base import BaseSamplerV1, SamplerResult
from qiskit.primitives.primitive_job import PrimitiveJob
from qiskit.result import QuasiDistribution

from .iteration import strict_zip


_TOLERANCE = 1e-16


def simulate_statevector_outcomes(qc: QuantumCircuit, /) -> dict[int, float]:
    """Return each classical outcome along with its precise probability.

    Circuit can contain mid-circuit, projective measurements.

    All gates are supported, along with measurements and reset operations.
    """
    current = defaultdict(list)
    current[0].append((1.0, Statevector.from_int(0, 2**qc.num_qubits)))
    for inst in qc.data:
        if isinstance(inst.operation, ControlFlowOp):
            raise ValueError(
                "Operations conditioned on classical bits are currently not supported."
            )
        opname = inst.operation.name
        if opname in ("measure", "reset"):
            # The current instruction is not unitary: it's either a measurement
            # or a reset.
            qubit_idx = qc.find_bit(inst.qubits[0])[0]
            if opname == "measure":
                # We will need to set a classical bit depending on the
                # measurement result.  `k_flipper` locates that bit.
                k_flipper = 1 << qc.find_bit(inst.clbits[0])[0]
            else:
                # It's a reset operation, so we will not be modifying any
                # classical bits.
                k_flipper = 0
            # We need to keep track of the statevector and corresponding
            # probability of *both* possible outcomes (although, we truncate
            # states if their probability becomes less than _TOLERANCE).  In
            # the following, we loop through each outcome so far and prepare to
            # update the state.
            pending_delete: list[tuple[int, int]] = []
            pending_insert: list[tuple[int, tuple[float, Statevector]]] = []
            for k, svs in current.items():
                k0 = k ^ (k & k_flipper)  # like k, but k_flipper bit will NOT be set
                k1 = k | k_flipper  # like k, but k_flipper bit (if any) will be set
                for i, (prob, sv) in enumerate(svs):
                    (prob0, prob1) = sv.probabilities([qubit_idx])
                    dims = sv.dims([qubit_idx])  # always going to be (2,) for a qubit
                    pending_delete.append((k, i))
                    # Handle the 0 branch of the wave function
                    if not np.isclose(prob0, 0, atol=_TOLERANCE):
                        proj0 = np.diag([1 / np.sqrt(prob0), 0.0])
                        sv0 = sv.evolve(
                            Operator(proj0, input_dims=dims, output_dims=dims),
                            qargs=[qubit_idx],
                        )
                        pending_insert.append((k0, (prob * prob0, sv0)))
                    # Handle the 1 branch of the wave function
                    if not np.isclose(prob1, 0, atol=_TOLERANCE):
                        proj1 = np.diag([0.0, 1 / np.sqrt(prob1)])
                        if k_flipper == 0:
                            # It's a reset operation, so we need to rotate the 1
                            # result back to 0 by applying the same rotation as
                            # the X gate.
                            proj1 = np.array([(0, 1), (1, 0)]) @ proj1
                        sv1 = sv.evolve(
                            Operator(proj1, input_dims=dims, output_dims=dims),
                            qargs=[qubit_idx],
                        )
                        pending_insert.append((k1, (prob * prob1, sv1)))
            # A dict's keys cannot be changed while iterating it, so we perform
            # all such updates now that iteration over the dict is complete.
            for k, i in reversed(pending_delete):
                del current[k][i]
            for k, v in pending_insert:
                current[k].append(v)
            # We might as well clean up empty lists, too.
            for k in [k for k, v in current.items() if not v]:
                del current[k]
        else:
            # The current instruction is a unitary operation (i.e., a gate).
            if len(inst.clbits) != 0:  # pragma: no cover
                raise ValueError(
                    "Circuit cannot contain a non-measurement operation on classical bit(s)."
                )
            # Evolve each statevector according to the current instruction
            for svs in current.values():
                for _, sv in svs:
                    # Calling `_evolve_instruction` rather than `evolve` allows
                    # us to avoid a copy.
                    Statevector._evolve_instruction(
                        sv, inst.operation, [qc.find_bit(q)[0] for q in inst.qubits]
                    )

    return {outcome: sum(prob for prob, _ in svs) for outcome, svs in current.items()}


class ExactSampler(BaseSamplerV1):
    """Sampler which returns exact probabilities for each possible outcome.

    This sampler supports:

    - all unitary gates
    - projective measurements, anywhere in the circuit
    - reset operations, anywhere in the circuit
    - some (or all) classical bits can remain unused
    - classical bits can be written more than once

    The samplers provided by :mod:`qiskit.primitives` and
    :mod:`qiskit_aer.primitives` do not currently support all of the above
    functionality.  Related upstream issues:

    - https://github.com/Qiskit/qiskit/issues/9657
    - https://github.com/Qiskit/qiskit-aer/issues/1810
    - https://github.com/Qiskit/qiskit-aer/issues/1811
    """

    def _call(
        self,
        circuits: tuple[QuantumCircuit, ...],
        parameter_values: Sequence[Sequence[float]],
        **ignored_run_options,
    ) -> SamplerResult:
        metadata: list[dict[str, Any]] = [{} for _ in range(len(circuits))]
        bound_circuits = [
            circuit if len(value) == 0 else circuit.assign_parameters(value)
            for circuit, value in strict_zip(circuits, parameter_values)
        ]
        probabilities = [simulate_statevector_outcomes(qc) for qc in bound_circuits]
        quasis = [QuasiDistribution(p) for p in probabilities]
        return SamplerResult(quasis, metadata)

    def _run(
        self,
        circuits: tuple[QuantumCircuit, ...],
        parameter_values: tuple[tuple[float, ...], ...],
        **run_options,
    ):
        job = PrimitiveJob(self._call, circuits, parameter_values, **run_options)
        # The public submit method was removed in Qiskit 1.0
        (job.submit if hasattr(job, "submit") else job._submit)()
        return job
