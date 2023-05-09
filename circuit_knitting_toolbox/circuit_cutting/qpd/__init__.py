# This code is a Qiskit project.

# (C) Copyright IBM 2023.

# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""
Main quasiprobability decomposition functionality.

.. autosummary::
   :toctree: ../stubs/
   :nosignatures:

   QPDBasis
   generate_qpd_samples

Instructions (:mod:`circuit_knitting_toolbox.qpd.instructions`)
======================================

.. automodule:: circuit_knitting_toolbox.qpd.instructions
"""

from .qpd_basis import QPDBasis
from .qpd import generate_qpd_samples
from .instructions import (
    BaseQPDGate,
    SingleQubitQPDGate,
    TwoQubitQPDGate,
    QPDMeasure,
)

__all__ = [
    "qpdbasis_from_gate",
    "generate_qpd_samples",
    "QPDBasis",
    "BaseQPDGate",
    "TwoQubitQPDGate",
    "SingleQubitQPDGate",
    "QPDMeasure",
]
