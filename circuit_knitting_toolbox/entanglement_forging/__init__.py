# This code is a Qiskit project.

# (C) Copyright IBM 2023.

# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Deprecated import location ``circuit_knitting_toolbox.entanglement_forging``."""

import sys
from warnings import warn

from circuit_knitting.forging import (
    EntanglementForgingAnsatz,
    EntanglementForgingKnitter,
    EntanglementForgingOperator,
    EntanglementForgingGroundStateSolver,
    cholesky_decomposition,
    convert_cholesky_operator,
    entanglement_forging_ansatz,
    entanglement_forging_ground_state_solver,
    entanglement_forging_knitter,
    entanglement_forging_operator,
)
import circuit_knitting.forging.cholesky_decomposition as cholesky_module

__all__ = [
    "EntanglementForgingAnsatz",
    "EntanglementForgingKnitter",
    "EntanglementForgingOperator",
    "EntanglementForgingGroundStateSolver",
    "cholesky_decomposition",
    "convert_cholesky_operator",
]

sys.modules[
    "circuit_knitting_toolbox.entanglement_forging.cholesky_decomposition"
] = cholesky_module
sys.modules[
    "circuit_knitting_toolbox.entanglement_forging.entanglement_forging_ansatz"
] = entanglement_forging_ansatz
sys.modules[
    "circuit_knitting_toolbox.entanglement_forging.entanglement_forging_ground_state_solver"
] = entanglement_forging_ground_state_solver
sys.modules[
    "circuit_knitting_toolbox.entanglement_forging.entanglement_forging_knitter"
] = entanglement_forging_knitter
sys.modules[
    "circuit_knitting_toolbox.entanglement_forging.entanglement_forging_operator"
] = entanglement_forging_operator

warn(
    f"The package namespace {__name__} is deprecated and will be removed "
    "no sooner than Circuit Knitting Toolbox 0.4.0. Use namespace "
    "circuit_knitting.forging instead.",
    DeprecationWarning,
    stacklevel=2,
)
