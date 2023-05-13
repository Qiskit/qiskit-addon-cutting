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

"""Two-qubit instruction representing a swap + single-qubit reset."""
from __future__ import annotations

from qiskit.circuit import QuantumCircuit, Instruction


class Move(Instruction):
    """A two-qubit instruction representing a reset of the second qubit followed by a swap.

    The desired effect of this instruction, typically, is to move the state of
    the first qubit to the second qubit.  For this to work as expected, the
    second incoming qubit must share no entanglement with the remainder of the
    system.  If this qubit *is* entangled, then performing the reset operation
    will in turn implement a quantum channel on the other qubit(s) with which
    it is entangled, resulting in the partial collapse of those qubits.

    Generally speaking, the most straightforward way to ensure that the second
    qubit is not entangled is to prepare it in one of the following ways before
    implementing the :class:`Move`:

    - No operations since initialization.
    - A :class:`~qiskit.Measure` is the preceding operation.
    - A :class:`~qiskit.Reset` is the preceding operation.
    - The preceding use of the qubit is such that it is the first output qubit of another
      :class:`Move` operation.

    """

    def __init__(self, label: str | None = None):
        """Create a Move instruction."""
        super().__init__("move", 2, 0, [], label=label)

    def _define(self):
        """Set definition to equivalent circuit."""
        qc = QuantumCircuit(2, name=self.name)
        qc.reset(1)
        qc.swap(0, 1)
        self.definition = qc
