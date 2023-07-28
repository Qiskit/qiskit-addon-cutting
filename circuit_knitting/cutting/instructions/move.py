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

    The simplest way to ensure that the second (i.e., destination) qubit shares
    no entanglement with the remainder of the system is to use a fresh qubit
    which has not been used since initialization.

    Another valid way is to use, as a desination qubit, a qubit whose immediate
    prior use was as the source (i.e., first) qubit of a preceding
    :class:`Move` operation.

    See `the tutorial on wire cutting using the Move instruction
    <../circuit_cutting/tutorials/03_wire_cutting_via_move_instruction.ipynb>`__
    for an example whose two :class:`Move` instructions correspond to each of
    the aforementioned cases.
    """

    def __init__(self, label: str | None = None):
        """Create a :class:`Move` instruction."""
        super().__init__("move", 2, 0, [], label=label)

    def _define(self):
        """Set definition to equivalent circuit."""
        qc = QuantumCircuit(2, name=self.name)
        qc.reset(1)
        qc.swap(0, 1)
        self.definition = qc
