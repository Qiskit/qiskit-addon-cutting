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

from collections import defaultdict
from collections.abc import Sequence

import numpy as np
from qiskit.quantum_info import PauliList
from qiskit.result import QuasiDistribution

from ..utils.observable_grouping import CommutingObservableGroup, ObservableCollection
from ..utils.bitwise import bit_count
from .cutting_decomposition import decompose_observables
from .qpd import WeightType


def reconstruct_expectation_values(
    observables: PauliList | dict[str | int, PauliList],
    weights: Sequence[tuple[float, WeightType]],
    results: Sequence[QuasiDistribution] | dict[str | int, Sequence[QuasiDistribution]],
) -> list[float]:
    r"""
    Reconstruct an expectation value from the results of the sub-experiments.

    Args:
        results: The results from running the cutting subexperiments using the
            Qiskit Sampler primitive.

    Returns:
        A ``list`` of ``float``\ s, such that each float is a simulated expectation
        value corresponding to the input observable in the same position

    Raises:
        ValueError: ``observables``, and ``quasi-dists`` are of incompatible types.
        ValueError: An input observable has a phase not equal to 1.
    """
    if isinstance(observables, PauliList) and not isinstance(results, Sequence):
        raise ValueError(
            "If observables is a PauliList, results must be a QuasiDistribution."
        )
    if isinstance(observables, dict) and not isinstance(results, dict):
        raise ValueError(
            "If observables is a dictionary, results must also be a dictionary."
        )
    # If circuit was not separated, transform input data structures to dictionary format
    if isinstance(observables, PauliList):
        if any(obs.phase != 0 for obs in observables):
            raise ValueError("An input observable has a phase not equal to 1.")
        subobservables_by_subsystem = decompose_observables(
            observables, "A" * len(observables[0])
        )
        assert isinstance(results, Sequence)
        results_dict: dict[str | int, Sequence[QuasiDistribution]] = {"A": results}
        expvals = np.zeros(len(observables))

    else:
        assert isinstance(results, dict)
        results_dict = results
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

    # Get the number of QPD bits for each partition's subexperiments

    # QuasiDistribution._num_bits is not guaranteed to reflect the total number of
    # qubit measurements on the input circuit. It may only reflect the number of bits
    # needed to represent the outcomes of the sampled distribution.
    # More info in QuasiDistribution __init__ comments:
    # https://github.com/Qiskit/qiskit-terra/blob/0388d543dee1fe59f07b257fa218fe99511397c8/qiskit/result/distributions/quasi.py

    # If the most significant bit(s) of the observable were never sampled positively,
    # the QuasiDistribution._num_bits would report an erroneously low number of bits,
    # resulting in an erroneously low estimation of the number of QPD bits. We mitigate
    # this by leveraging the fact that all QPD registers are the same size for a
    # given partition, and we take the max estimation of the number of QPD bits
    # across all experiments for a given partition.
    qpd_bits_by_partition = defaultdict(lambda: 0)
    for i in range(len(weights)):
        for label in sorted_subsystems:
            so = subsystem_observables[label]
            for k, cog in enumerate(so.groups):
                num_obs_bits = len(
                    [char for char in cog.general_observable.to_label() if char != "I"]
                )
                quasi_probs: QuasiDistribution = results_dict[label][i * len(so.groups) + k]  # type: ignore
                ###################################################################################
                # Accessing private QuasiDistribution._num_bits field here. Switch to public field
                # when Qiskit issue # 10648 is resolved and released.
                ###################################################################################
                qpd_bits_by_partition[label] = max(
                    qpd_bits_by_partition[label], quasi_probs._num_bits - num_obs_bits
                )

    # Reconstruct the expectation values
    for i in range(len(weights)):
        current_expvals = np.ones((len(expvals),))
        for label in sorted_subsystems:
            so = subsystem_observables[label]
            weight = weights[i]
            subsystem_expvals = [
                np.zeros(len(cog.commuting_observables)) for cog in so.groups
            ]
            for k, cog in enumerate(so.groups):
                num_obs_bits = len(
                    [char for char in cog.general_observable.to_label() if char != "I"]
                )
                quasi_probs: QuasiDistribution = results_dict[label][i * len(so.groups) + k]  # type: ignore

                for outcome, quasi_prob in quasi_probs.items():  # type: ignore
                    subsystem_expvals[k] += quasi_prob * _process_outcome(
                        qpd_bits_by_partition[label], cog, outcome
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
