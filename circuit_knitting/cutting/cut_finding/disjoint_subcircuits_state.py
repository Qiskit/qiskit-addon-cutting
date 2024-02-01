# This code is a Qiskit project.

# (C) Copyright IBM 2024.

# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Class needed for representing search-space states when cutting circuits."""

import copy
import numpy as np
from collections import Counter


class DisjointSubcircuitsState:

    """Class for representing search-space states when cutting
    circuits to construct disjoint subcircuits.  Only minimally
    sufficient information is stored in order to minimize the
    memory footprint.

    Each wire cut introduces a new wire.  A mapping from qubit IDs
    in QASM-like statements to wire IDs is therefore created
    and maintained.  Groups of wires form subcircuits, and these
    subcircuits can then be merged via search actions.  The mapping
    from wires to subcircuits is represented using an up-tree data
    structure over wires.  The number of wires (width) in each
    subcircuit is also tracked to ensure subcircuits will fit on
    target quantum devices.

    Member Variables:

    wiremap (int Numpy array) provides the mapping from qubit IDs
    to wire IDs.

    num_wires (int) is the number of wires in the cut circuit.

    uptree (int Numpy array) contains the uptree data structure that
    defines groups of wires that form subcircuits.  The uptree array
    map wire IDs to parent wire IDs in a subcircuit.  If a wire points
    to itself, then that wire is the root wire in the corresponding
    subcircuit.  Otherwise, you need to follow the parent links to find
    the root wire that corresponds to that subcircuit.

    width (int Numpy array) contains the number of wires in each
    subcircuit.  The values of width are valid only for root wire IDs.

    bell_pairs (list) is a list of pairs of subcircuits (wires) that
    define the virtual Bell pairs that would need to be constructed in
    order to implement optimal LOCC wire and gate cuts using ancillas.

    gamma_LB (float) is the cumulative lower-bound gamma for circuit cuts
    that cannot be constructed using Bell pairs, such as LO gate cuts
    for small-angled rotations.

    gamma_UB (float) is the cumulative upper-bound gamma for all circuit
    cuts assuming all cuts are LO.

    no_merge (list) contains a list of subcircuit merging constaints.
    Each constraint can either be a pair of wire IDs or a list of pairs
    of wire IDs.  In the case of a pair of wire IDs, the constraint is
    that the subcircuits that contain those wire IDs cannot be merged
    by subsequent search actions.  In the case of a list of pairs of
    wire IDs, the constraint is that at least one pair of corresponding
    subcircuits cannot be merged.

    actions (list) contains a list of circuit-cutting actions that have
    been performed on the circuit.  Elements of the list have the form

        [<action_object>, <gate_specification>, (<arg_1>, ..., <arg_n>)]

    The <action_object> is the object that was used to generate the
    circuit cut.  The <gate_specification> is the specification of the
    cut gate using the format defined in the CircuitInterface class
    description.  The trailing entries are the arguments needed by the
    <action_object> to apply further search-space generating objects
    in Stage Two in order to explore the space of QPD assignments to
    the circuit-cutting action.

    level (int) is the level in the search tree at which this search
    state resides, with 0 being the root of the search tree.
    """

    def __init__(self, num_qubits=None, max_wire_cuts=None):
        """An instance of :class:`DisjointSubcircuitsState` must be initialized with
        a specification of the number of qubits in the circuit and the
        maximum number of wire cuts that can be performed."""

        if not (
            num_qubits is None or (isinstance(num_qubits, int) and num_qubits >= 0)
        ):
            raise ValueError("num_qubits must be either be None or a positive integer.")

        if not (
            max_wire_cuts is None
            or (isinstance(max_wire_cuts, int) and max_wire_cuts >= 0)
        ):
            raise ValueError(
                "max_wire_cuts must be either be None or a positive integer."
            )

        if num_qubits is None or max_wire_cuts is None:
            self.wiremap = None
            self.num_wires = None

            self.uptree = None
            self.width = None

            self.bell_pairs = None
            self.gamma_LB = None
            self.gamma_UB = None

            self.no_merge = None
            self.actions = None
            self.level = None
            self.cut_actions_list = None

        else:
            max_wires = num_qubits + max_wire_cuts

            self.wiremap = np.arange(num_qubits)
            self.num_wires = num_qubits

            self.uptree = np.arange(max_wires)
            self.width = np.ones(max_wires, dtype=int)

            self.bell_pairs = list()
            self.gamma_LB = 1.0
            self.gamma_UB = 1.0

            self.no_merge = list()
            self.actions = list()
            self.cut_actions_list = list()
            self.level = 0

    def __copy__(self):
        new_state = DisjointSubcircuitsState()

        new_state.wiremap = self.wiremap.copy()
        new_state.num_wires = self.num_wires

        new_state.uptree = self.uptree.copy()
        new_state.width = self.width.copy()

        new_state.bell_pairs = self.bell_pairs.copy()
        new_state.gamma_LB = self.gamma_LB
        new_state.gamma_UB = self.gamma_UB

        new_state.no_merge = self.no_merge.copy()
        new_state.actions = self.actions.copy()
        new_state.cut_actions_list = self.cut_actions_list.copy()
        new_state.level = None

        return new_state

    def copy(self):
        """Make shallow copy."""

        return copy.copy(self)

    def CutActionsList(self):
        """Create a formatted list containing the actions carried out on a DisjointSubcircuitState
        along with the locations of these actions which are specified in terms of
        gate and wire references."""

        cut_actions = PrintActionListWithNames(self.actions)

        # Output formatting for LO gate and wire cuts.
        for i in range(len(cut_actions)):
            if (cut_actions[i][0] == "CutLeftWire") or (
                cut_actions[i][0] == "CutRightWire"
            ):
                self.cut_actions_list.append(
                    {
                        "Cut action": cut_actions[i][0],
                        "Cut location:": {
                            "Gate": [cut_actions[i][1][0], cut_actions[i][1][1]]
                        },
                        "Input wire": cut_actions[i][2][0][0],
                    }
                )
            elif cut_actions[i][0] == "CutTwoQubitGate":
                self.cut_actions_list.append(
                    {
                        "Cut action": cut_actions[i][0],
                        "Cut Gate": [cut_actions[i][1][0], cut_actions[i][1][1]],
                    }
                )
            if not self.cut_actions_list:
                self.cut_actions_list = cut_actions

            return self.cut_actions_list

    def print(self, simple=False):  # pragma: no cover
        """Print the various properties of a DisjointSubcircuitState."""

        cut_actions_list = self.CutActionsList()
        if simple:
            print(cut_actions_list)
        else:
            print("wiremap", self.wiremap)
            print("num_wires", self.num_wires)
            print("uptree", self.uptree)
            print("width", self.width)
            print("bell_pairs", self.bell_pairs)
            print("gamma_LB", self.gamma_LB)
            print("lowerBound", self.lowerBoundGamma())
            print("gamma_UB", self.gamma_UB)
            print("no_merge", self.no_merge)
            print("actions", PrintActionListWithNames(self.actions))
            print("level", self.level)

    def getNumQubits(self):
        """Return the number of qubits in the circuit."""

        if self.wiremap is not None:
            return self.wiremap.shape[0]

    def getMaxWidth(self):
        """Return the maximum width across subcircuits."""

        if self.width is not None:
            return np.amax(self.width)

    def getSubCircuitIndices(self):
        """Return a list of root indices for the subcircuits in
        the current cut circuit.
        """

        if self.uptree is not None:
            return [i for i, j in enumerate(self.uptree[: self.num_wires]) if i == j]

    def getWireRootMapping(self):
        """Return a list of root wires for each wire in
        the current cut circuit.
        """

        return [self.findWireRoot(i) for i in range(self.num_wires)]

    def findRootBellPair(self, bell_pair):
        """Find the root wires for a Bell pair (represented as a pair
        of wires) and returns a sorted tuple representing the Bell pair.
        """

        r0 = self.findWireRoot(bell_pair[0])
        r1 = self.findWireRoot(bell_pair[1])
        return (r0, r1) if (r0 < r1) else (r1, r0)

    def lowerBoundGamma(self):
        """Calculate a lower bound for gamma using the current
        counts for the different types of circuit cuts.
        """

        root_bell_pairs = map(lambda x: self.findRootBellPair(x), self.bell_pairs)

        return self.gamma_LB * calcRootBellPairsGamma(root_bell_pairs)

    def upperBoundGamma(self):
        """Calculate an upper bound for gamma using the current
        counts for the different types of circuit cuts.
        """

        return self.gamma_UB

    def canAddWires(self, num_wires: int) -> bool:
        """Return True if an additional num_wires can be cut
        without exceeding the maximum allowed number of wire cuts.
        """

        return self.num_wires + num_wires <= self.uptree.shape[0]

    def canExpandSubcircuit(self, root: int, num_wires: int, max_width: int) -> bool:
        """Return True if num_wires can be added to subcircuit root
        without exceeding the maximum allowed number of qubits.
        """

        return self.width[root] + num_wires <= max_width

    def newWire(self, qubit):
        """Cut the wire associated with qubit and returns
        the ID of the new wire now associated with qubit.
        """

        assert self.num_wires < self.uptree.shape[0], (
            "Max new wires exceeded " + f"{self.num_wires}, {self.uptree.shape[0]}"
        )

        self.wiremap[qubit] = self.num_wires
        self.num_wires += 1

        return self.wiremap[qubit]

    def getWire(self, qubit):
        """Return the ID of the wire currently associated with qubit."""

        return self.wiremap[qubit]

    def findWireRoot(self, wire):
        """Return the ID of the root wire in the subcircuit
        that contains wire and collapses the path to the root.
        """

        # Find the root wire in the subcircuit
        root = wire
        while root != self.uptree[root]:
            root = self.uptree[root]

        # Collapse the path to the root
        while wire != root:
            parent = self.uptree[wire]
            self.uptree[wire] = root
            wire = parent

        return root

    def findQubitRoot(self, qubit):
        """Return the ID of the root wire in the subcircuit currently
        associated with qubit and collapses the path to the root.
        """

        return self.findWireRoot(self.wiremap[qubit])

    def checkDoNotMergeRoots(self, root_1, root_2):
        """Return True if the subcircuits represented by
        root wire IDs root_1 and root_2 should not be merged.
        """

        assert root_1 == self.uptree[root_1] and root_2 == self.uptree[root_2], (
            "Arguments must be roots: "
            + f"{root_1} != {self.uptree[root_1]} "
            + f"or {root_2} != {self.uptree[root_2]}"
        )

        for clause in self.no_merge:
            r1 = self.findWireRoot(clause[0])
            r2 = self.findWireRoot(clause[1])

            assert r1 != r2, "Do-Not-Merge clauses must not be identical"

            if (r1 == root_1 and r2 == root_2) or (r1 == root_2 and r2 == root_1):
                return True

        return False

    def verifyMergeConstraints(self):
        """Return True if all merge constraints are satisfied."""

        for clause in self.no_merge:
            r1 = self.findWireRoot(clause[0])
            r2 = self.findWireRoot(clause[1])
            if r1 == r2:
                return False

        return True

    def assertDoNotMergeRoots(self, wire_1, wire_2):
        """Add a constraint that the subcircuits associated
        with wires IDs wire_1 and wire_2 should not be merged.
        """

        assert self.findWireRoot(wire_1) != self.findWireRoot(
            wire_2
        ), f"{wire_1} cannot be the same subcircuit as {wire_2}"

        self.no_merge.append((wire_1, wire_2))

    def mergeRoots(self, root_1, root_2):
        """Merge the subcircuits associated with root wire IDs root_1
        and root_2, and updates the statistics (i.e., width)
        associated with the newly merged subcircuit.
        """

        assert root_1 == self.uptree[root_1] and root_2 == self.uptree[root_2], (
            "Arguments must be roots: "
            + f"{root_1} != {self.uptree[root_1]} "
            + f"or {root_2} != {self.uptree[root_2]}"
        )

        assert root_1 != root_2, f"Cannot merge root {root_1} with itself"

        merged_root = min(root_1, root_2)
        other_root = max(root_1, root_2)
        self.uptree[other_root] = merged_root
        self.width[merged_root] += self.width[other_root]

    def addAction(self, action_obj, gate_spec, *args):
        """Append the specified action to the list of search-space
        actions that have been performed.
        """

        if action_obj.getName() is not None:
            self.actions.append([action_obj, gate_spec, args])

    def getSearchLevel(self):
        """Return the search level."""

        return self.level

    def setNextLevel(self, state):
        """Set the search level of self to one plus the search
        level of the input state.
        """

        self.level = state.level + 1

    def exportCuts(self, circuit_interface):
        """Export LO cuts into the input circuit_interface for each of
        the cutting decisions made.
        """

        # This wire map assumes no reuse of measured qubits that
        # result from wire cuts
        wire_map = np.arange(self.num_wires)

        for action, gate_spec, cut_args in self.actions:
            action.exportCuts(circuit_interface, wire_map, gate_spec, cut_args)

        root_list = self.getSubCircuitIndices()
        wires_to_roots = self.getWireRootMapping()

        subcircuits = [
            list({wire_map[w] for w, r in enumerate(wires_to_roots) if r == root})
            for root in root_list
        ]

        circuit_interface.defineSubcircuits(subcircuits)


def calcRootBellPairsGamma(root_bell_pairs):
    """Calculate the minimum-achievable LOCC gamma for circuit
    cuts that utilize virtual Bell pairs. The input can be a list
    or iterator over hashable identifiers that represent Bell pairs
    across disconnected subcircuits in a cut circuit. There must be
    a one-to-one mapping between identifiers and pairs of subcircuits.
    Repeated identifiers are interpreted as mutiple Bell pairs across
    the same pair of subcircuits, and the counts of such repeats are
    used to calculate gamma.
    """

    gamma = 1.0
    for n in Counter(root_bell_pairs).values():
        gamma *= 2 ** (n + 1) - 1

    return gamma


def PrintActionListWithNames(action_list):
    """Replace the action objects that appear in action lists
    in DisjointSubcircuitsState objects with the corresponding
    action names for readability, and print.
    """

    return [[x[0].getName()] + x[1:] for x in action_list]
