# This code is a Qiskit project.

# (C) Copyright IBM 2022.

# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Code to initialize the wire cutting imports."""
from .wire_cutting_evaluation import run_subcircuit_instances
from .wire_cutting_post_processing import generate_summation_terms, build
from .wire_cutting_verification import verify
from .wire_cutting import (
    cut_circuit_wires,
    evaluate_subcircuits,
    reconstruct_full_distribution,
)

from warnings import warn

warn(
    f"The package namespace {__name__} is deprecated. It has been renamed circuit_knitting_toolbox.circuit_cutting.cutqc",
    DeprecationWarning,
)

__all__ = [
    "run_subcircuit_instances",
    "generate_summation_terms",
    "build",
    "verify",
    "cut_circuit_wires",
    "evaluate_subcircuits",
    "reconstruct_full_distribution",
]
