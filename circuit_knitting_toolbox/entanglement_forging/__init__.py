# This code is a Qiskit project.

# (C) Copyright IBM 2022.

# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""
Entanglement Forging (:mod:`circuit_knitting_toolbox.entanglement_forging`).

.. currentmodule:: circuit_knitting_toolbox.entanglement_forging

Classes
=======

.. autosummary::
   :toctree: ../stubs/
   :nosignatures:

   EntanglementForgingKnitter
   EntanglementForgingOperator
   EntanglementForgingAnsatz
   EntanglementForgingGroundStateSolver

Decomposition Functions
=======================

.. autosummary::
   :toctree: ../stubs/
   :nosignatures:

   cholesky_decomposition
   convert_cholesky_operator

"""

from .entanglement_forging_ansatz import EntanglementForgingAnsatz
from .entanglement_forging_knitter import EntanglementForgingKnitter
from .entanglement_forging_operator import EntanglementForgingOperator
from .entanglement_forging_ground_state_solver import (
    EntanglementForgingGroundStateSolver,
)
from .cholesky_decomposition import cholesky_decomposition, convert_cholesky_operator

__all__ = [
    "EntanglementForgingAnsatz",
    "EntanglementForgingKnitter",
    "EntanglementForgingOperator",
    "EntanglementForgingGroundStateSolver",
    "cholesky_decomposition",
    "convert_cholesky_operator",
]

from qiskit_nature import settings

settings.use_pauli_sum_op = False
