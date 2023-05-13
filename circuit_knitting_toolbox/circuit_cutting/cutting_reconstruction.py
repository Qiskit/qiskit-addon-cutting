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

import numpy as np

from ..utils.observable_grouping import CommutingObservableGroup
from ..utils.bitwise import bit_count


def process_outcome(
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
