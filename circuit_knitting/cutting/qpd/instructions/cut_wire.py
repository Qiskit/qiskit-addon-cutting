# This code is part of Qiskit.
#
# (C) Copyright IBM 2023.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Single-qubit instruction to denote a wire cut location."""
from __future__ import annotations

from qiskit.circuit import Instruction


class CutWire(Instruction):
    """An instruction for denoting a wire cut location."""

    def __init__(self, label: str | None = None):
        """"""
        super().__init__("cut_wire", 1, 0, [], label=label)
