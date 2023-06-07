# This code is a Qiskit project.

# (C) Copyright IBM 2022.

# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""
Entanglement Forging (:mod:`circuit_knitting.forging`).

.. currentmodule:: circuit_knitting.forging

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
from qiskit_nature import settings

# From now forward, the entanglement forging module requires that SparsePauliOp
# be used rather than PauliSumOp, as the latter is deprecated in Qiskit Terra 0.24.
# However, Qiskit Nature still is not expected to change this default until Qiskit
# Nature 0.7. Here, we modify the global state to opt into the new behavior early.
# Unfortunately, this means that any code that calls Qiskit Nature in the same
# process as entanglement forging will need to be updated to use SparsePauliOp as well.
settings.use_pauli_sum_op = False

__all__ = [
    "EntanglementForgingAnsatz",
    "EntanglementForgingKnitter",
    "EntanglementForgingOperator",
    "EntanglementForgingGroundStateSolver",
    "cholesky_decomposition",
    "convert_cholesky_operator",
]
