# This code is a Qiskit project.

# (C) Copyright IBM 2024.

# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""
Bootstrapping utilities.

.. currentmodule:: circuit_knitting.utils.bootstrap

.. autosummary::
   :toctree: ../stubs/

   bootstrap_sampler_result
"""

from __future__ import annotations

import numpy as np
from qiskit.primitives import PrimitiveResult, SamplerPubResult, DataBin, BitArray


def _bootstrap_pubresult(pr: SamplerPubResult, /, rng: np.random.Generator):
    num_shots = next(iter(pr.data.values())).num_shots
    indices = rng.choice(range(num_shots), num_shots, replace=True)
    new_data: dict[str, BitArray] = {}
    for k, v in pr.data.items():
        new_data[k] = BitArray(v.array[indices, ...], v.num_bits)
    return SamplerPubResult(DataBin(**new_data))


def bootstrap_sampler_result(
    result: PrimitiveResult[SamplerPubResult],
    /,
    seed: int | np.random.Generator | None = None,
):
    """Construct a single synthetic dataset from a :class:`PrimitiveResult`."""
    if isinstance(seed, np.random.Generator):
        rng = seed
    else:
        rng = np.random.default_rng(seed)

    return PrimitiveResult(
        [_bootstrap_pubresult(pubresult, rng) for pubresult in result],
        {**result.metadata, "bootstrapped": True},
    )
