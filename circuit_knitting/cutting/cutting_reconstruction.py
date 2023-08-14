# This code is a Qiskit project.

# (C) Copyright IBM 2023.

# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Functions for reconstructing the results of circuit cutting experiments."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from qiskit.quantum_info import PauliList
from qiskit.result import QuasiDistribution

from ..utils.observable_grouping import CommutingObservableGroup, ObservableCollection
from ..utils.bitwise import bit_count
from .cutting_decomposition import decompose_observables, PartitionedCuttingProblem
from .qpd import WeightType


def reconstruct_expectation_values(
    partitioned_problem: PartitionedCuttingProblem,
    quasi_dists: dict[str | int, Sequence[QuasiDistribution]],
) -> list[float]:
    r"""
    Reconstruct an expectation value from the results of the sub-experiments.

    Args:
        partitioned_problem: The results from cutting gates and wires in a circuit
        quasi_dists: The results from running the cutting subexperiments using the
            Qiskit Sampler primitive.

    Returns:
        A ``list`` of ``float``\ s, such that each float is a simulated expectation
        value corresponding to the input observable in the same position

    Raises:
        ValueError: An input observable has a phase not equal to 1.
    """
    observables = partitioned_problem.observables
    weights = partitioned_problem.weights
    # Create the commuting observable groups
    if isinstance(observables, PauliList):
        if any(obs.phase != 0 for obs in observables):
            raise ValueError("An input observable has a phase not equal to 1.")
        subobservables_by_subsystem = decompose_observables(
            observables, "A" * len(observables[0])
        )
        expvals = np.zeros(len(observables))

    else:
        for label, subobservable in observables.items():
            if any(obs.phase != 0 for obs in subobservable):
                raise ValueError("An input observable has a phase not equal to 1.")
        subobservables_by_subsystem = observables
        expvals = np.zeros(len(list(observables.values())[0]))

    subsystem_observables = {
        label: ObservableCollection(subobservables)
        for label, subobservables in subobservables_by_subsystem.items()
    }
    sorted_subsystems = sorted(subsystem_observables.keys())  # type: ignore

    # Count the number of midcircuit measurements in each subexperiment
    num_qpd_bits = {}
    for i, label in enumerate(sorted_subsystems):
        nums_bits = []
        for j, circ in enumerate(partitioned_problem.subexperiments[label]):
            nums_bits.append(len(circ.cregs[0]))
        num_qpd_bits[label] = nums_bits

    key0 = sorted(partitioned_problem.subexperiments.keys())[0]
    assert (
        len(partitioned_problem.subexperiments[key0])
        % len(subsystem_observables[key0].groups)
        == 0
    )
    num_unique_samples = len(partitioned_problem.weights)
    for i in range(num_unique_samples):
        current_expvals = np.ones((len(expvals),))
        for label in sorted_subsystems:
            so = subsystem_observables[label]
            weight = weights[i]
            subsystem_expvals = [
                np.zeros(len(cog.commuting_observables)) for cog in so.groups
            ]
            for k, cog in enumerate(so.groups):
                quasi_probs = quasi_dists[label][i * len(so.groups) + k]
                for outcome, quasi_prob in quasi_probs.items():
                    subsystem_expvals[k] += quasi_prob * _process_outcome(
                        num_qpd_bits[label][i * len(so.groups) + k], cog, outcome
                    )
            for k, subobservable in enumerate(subobservables_by_subsystem[label]):
                current_expvals[k] *= np.mean(
                    [subsystem_expvals[m][n] for m, n in so.lookup[subobservable]]
                )
        expvals += weight[0] * current_expvals

    return list(expvals)


def _process_outcome(
    num_qpd_bits: int, cog: CommutingObservableGroup, outcome: int | str, /
) -> np.typing.NDArray[np.float64]:
    """
    Process a single outcome of a QPD experiment with observables.

    Args:
        num_qpd_bits: The number of QPD measurements in the circuit. It is
            assumed that the second to last creg in the generating circuit
            is the creg  containing the QPD measurements, and the last
            creg is associated with the observable measurements.
        cog: The observable set being measured by the current experiment
        outcome: The outcome of the classical bits

    Returns:
        A 1D array of the observable measurements.  The elements of
        this vector correspond to the elements of ``cog.commuting_observables``,
        and each result will be either +1 or -1.
    """
    outcome = _outcome_to_int(outcome)
    qpd_outcomes = outcome & ((1 << num_qpd_bits) - 1)
    meas_outcomes = outcome >> num_qpd_bits

    # qpd_factor will be -1 or +1, depending on the overall parity of qpd
    # measurements.
    qpd_factor = 1 - 2 * (bit_count(qpd_outcomes) & 1)

    rv = np.zeros(len(cog.pauli_bitmasks))
    for i, mask in enumerate(cog.pauli_bitmasks):
        # meas will be -1 or +1, depending on the measurement
        # of the current operator.
        meas = 1 - 2 * (bit_count(meas_outcomes & mask) & 1)
        rv[i] = qpd_factor * meas

    return rv


def _outcome_to_int(outcome: int | str) -> int:
    if isinstance(outcome, int):
        return outcome
    outcome = outcome.replace(" ", "")
    if len(outcome) < 2 or outcome[1] in ("0", "1"):
        outcome = outcome.replace(" ", "")
        return int(f"0b{outcome}", 0)
    return int(outcome, 0)
