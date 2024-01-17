# This code is a Qiskit project.

# (C) Copyright IBM 2024.

# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Classes needed to implement the actions involved in circuit cutting."""

import numpy as np
from abc import ABC, abstractmethod
from .search_space_generator import ActionNames

# Object that holds action names for constructing disjoint subcircuits
disjoint_subcircuit_actions = ActionNames()


class DisjointSearchAction(ABC):

    """Base class for search actions for constructing disjoint subcircuits."""

    @abstractmethod
    def getName(self):
        """Derived classes must return the look-up name of the action"""

        assert False, "Derived classes must override getName()"

    @abstractmethod
    def getGroupNames(self):
        """Derived classes must return a list of group names"""

        assert False, "Derived classes must override getGroupNames()"

    @abstractmethod
    def nextStatePrimitive(self, state, gate_spec, max_width):
        """Derived classes must return a list of search states that
        result from applying all variations of the action to gate_spec
        in the specified DisjointSubcircuitsState state, subject to the
        constraint that the number of resulting qubits (wires) in each
        subcircuit cannot exceed max_width"""

        assert False, "Derived classes must override nextState()"

    def nextState(self, state, gate_spec, max_width):
        """Return a list of search states that result from applying the
        action to gate_spec in the specified DisjointSubcircuitsState
        state, subject to the constraint that the number of resulting
        qubits (wires) in each subcircuit cannot exceed max_width.
        """

        next_list = self.nextStatePrimitive(state, gate_spec, max_width)

        for next_state in next_list:
            next_state.setNextLevel(state)

        return next_list

    def registerCut(self, assignment_settings, gate_spec, cut_args):
        """Derived classes must register the action in the specified
        AssignmentSettings object, where the action was applied to gate_spec
        with the action arguments cut_args"""

        assert False, "Derived classes must override registerCut()"

    def initializeCut(self, assignment_settings, gate_spec, cut_args):
        """Derived classes must initialize the action in the specified
        AssignmentSettings object, where the action was applied to gate_spec
        with the action arguments cut_args. Intialization is performed after
        all actions have been registered."""

        assert False, "Derived classes must override initializeCut()"

    def nextAssignment(
        self, assign_state, constraint_obj, gate_spec, cut_args, assign_actions
    ):
        """Return a list of next assignment states that result from
        applying assignment actions to the input assignment state.
        """

        next_list = self.nextAssignmentPrimitive(
            assign_state, constraint_obj, gate_spec, cut_args, assign_actions
        )

        for next_state in next_list:
            next_state.setNextLevel(assign_state)

        return next_list

    def nextAssignmentPrimitive(
        self, assign_state, constraint_obj, gate_spec, cut_args, assign_actions
    ):
        """Derived classes must retrieve the appropriate group of QPD
        assignment actions from assign_actions, and then collect and
        return the combined list of next assignment states that result
        from applying those actions to the input assignment state, with
        the constraint object, gate_spec, and cut_args provided as inputs
        to the nextState() methods of those assignment actions."""

        assert False, "Derived classes must override initializeCut()"


class ActionApplyGate(DisjointSearchAction):

    """Action class that implements the action of
    applying a two-qubit gate without decomposition"""

    def getName(self):
        """Return the look-up name of ActionApplyGate."""

        return None

    def getGroupNames(self):
        """Return the group name of ActionApplyGate."""

        return [None, "TwoQubitGates", "MultiqubitGates"]

    def nextStatePrimitive(self, state, gate_spec, max_width):
        """Return the new state that results from applying
        ActionApplyGate to state given the two-qubit gate
        specification: gate_spec.
        """

        gate = gate_spec[1]  # extract the gate from gate specification.
        if len(gate.qubits) > 2:
            # The function multiqubitNextState handles
            # gates that act on 3 or more qubits.
            return self.multiqubitNextState(state, gate_spec, max_width)

        r1 = state.findQubitRoot(gate.qubits[0])  # extract the root wire for the first qubit
        # acted on by the given 2-qubit gate.
        r2 = state.findQubitRoot(gate.qubits[1])  # extract the root wire for the second qubit
        # acted on by the given 2-qubit gate.

        # If applying the gate would cause the number of qubits to exceed
        # the qubit limit, then do not apply the gate
        if r1 != r2 and state.width[r1] + state.width[r2] > max_width:
            return list()

        # If the gate cannot be applied because it would violate the
        # merge constraints, then do not apply the gate
        if state.checkDoNotMergeRoots(r1, r2):
            return list()

        new_state = state.copy()

        if r1 != r2:
            new_state.mergeRoots(r1, r2)

        new_state.addAction(self, gate_spec)

        return [new_state]

    def multiqubitNextState(self, state, gate_spec, max_width):
        """Return the new state that results from applying
        ActionApplyGate to state given a multiqubit gate specification: gate_spec.
        """

        gate = gate_spec[1]
        roots = list(set([state.findQubitRoot(q) for q in gate.qubits]))
        new_width = sum([state.width[r] for r in roots])

        # If applying the gate would cause the number of qubits to exceed
        # the qubit limit, then do not apply the gate
        if new_width > max_width:
            return list()

        new_state = state.copy()

        r0 = roots[0]
        for r in roots[1:]:
            new_state.mergeRoots(r, r0)
            r0 = new_state.findWireRoot(r0)

        # If the gate cannot be applied because it would violate the
        # merge constraints, then do not apply the gate
        if not new_state.verifyMergeConstraints():
            return list()

        new_state.addAction(self, gate_spec)

        return [new_state]


### Adds ActionApplyGate to the object disjoint_subcircuit_actions
disjoint_subcircuit_actions.defineAction(ActionApplyGate())


class ActionCutTwoQubitGate(DisjointSearchAction):

    """Action class that implements the action of
        cutting a two-qubit gate.
    .

        TODO: The list of supported gates needs to be expanded.
    """

    def __init__(self):
        """The values in gate_dict are tuples in (gamma_LB, num_bell_pairs, gamma_UB) format.
        lowerBoundGamma is computed from gamma_LB using the DisjointSubcircuitsState.lowerBoundGamma() method.
        """

        self.gate_dict = {
            "cx": (1, 1, 3),
            "swap": (1, 2, 7),
            "iswap": (1, 2, 7),
            "crx": (
                lambda t: (
                    1 + 2 * np.abs(np.sin(t[1] / 2)),
                    0,
                    1 + 2 * np.abs(np.sin(t[1] / 2)),
                )
            ),
            "cry": (
                lambda t: (
                    1 + 2 * np.abs(np.sin(t[1] / 2)),
                    0,
                    1 + 2 * np.abs(np.sin(t[1] / 2)),
                )
            ),
            "crz": (
                lambda t: (
                    1 + 2 * np.abs(np.sin(t[1] / 2)),
                    0,
                    1 + 2 * np.abs(np.sin(t[1] / 2)),
                )
            ),
            "rxx": (
                lambda t: (
                    1 + 2 * np.abs(np.sin(t[1])),
                    0,
                    1 + 2 * np.abs(np.sin(t[1])),
                )
            ),
            "ryy": (
                lambda t: (
                    1 + 2 * np.abs(np.sin(t[1])),
                    0,
                    1 + 2 * np.abs(np.sin(t[1])),
                )
            ),
            "rzz": (
                lambda t: (
                    1 + 2 * np.abs(np.sin(t[1])),
                    0,
                    1 + 2 * np.abs(np.sin(t[1])),
                )
            ),
        }

    def getName(self):
        """Return the look-up name of ActionCutTwoQubitGate."""

        return "CutTwoQubitGate"

    def getGroupNames(self):
        """Return the group name of ActionCutTwoQubitGate."""

        return ["GateCut", "TwoQubitGates"]

    def nextStatePrimitive(self, state, gate_spec, max_width):
        """Return the new state that results from applying
        ActionCutTwoQubitGate to state given the gate_spec.
        """

        # If the gate is not a two-qubit gate, then return the empty list
        if len(gate_spec[1].qubits) != 2:
            return list()

        gamma_LB, num_bell_pairs, gamma_UB = self.getCostParams(gate_spec)

        if gamma_LB is None:
            return list()

        gate = gate_spec[1]
        q1 = gate.qubits[0]
        q2 = gate.qubits[1]
        w1 = state.getWire(q1)
        w2 = state.getWire(q2)
        r1 = state.findQubitRoot(q1)
        r2 = state.findQubitRoot(q2)

        if r1 == r2:
            return list()

        new_state = state.copy()

        new_state.assertDoNotMergeRoots(r1, r2)

        new_state.gamma_LB *= gamma_LB

        for k in range(num_bell_pairs):
            new_state.bell_pairs.append((r1, r2))

        new_state.gamma_UB *= gamma_UB

        new_state.addAction(self, gate_spec, (1, w1), (2, w2))

        return [new_state]

    def getCostParams(self, gate_spec):
        return lookupCostParams(self.gate_dict, gate_spec, (None, None, None))

    def registerCut(self, assignment_settings, gate_spec, cut_args):
        """Register the gate cuts made by a ActionCutTwoQubitGate action
        in an AssignmentSettings object.
        """

        assignment_settings.registerGateCut(gate_spec, cut_args[0][0])
        assignment_settings.registerGateCut(gate_spec, cut_args[1][0])

    def initializeCut(self, assignment_settings, gate_spec, cut_args):
        """Initialize the gate cuts made by a ActionCutTwoQubitGate action
        in an AssignmentSettings object.
        """

        assignment_settings.initGateCut(gate_spec, cut_args[0][0])
        assignment_settings.initGateCut(gate_spec, cut_args[1][0])

    def nextAssignmentPrimitive(
        self, assign_state, constraint_obj, gate_spec, cut_args, assign_actions
    ):
        action_list = assign_actions.getGroup("TwoQubitGateCut")

        new_list = list()
        for action in action_list:
            new_list.extend(
                action.nextState(assign_state, constraint_obj, gate_spec, cut_args)
            )

        return new_list

    def exportCuts(self, circuit_interface, wire_map, gate_spec, args):
        """Insert an LO gate cut into the input circuit for the specified gate
        and cut arguments.
        """

        circuit_interface.insertGateCut(gate_spec[0], "LO")


### Adds ActionCutTwoQubitGate to the object disjoint_subcircuit_actions
disjoint_subcircuit_actions.defineAction(ActionCutTwoQubitGate())


def lookupCostParams(gate_dict, gate_spec, default_value):
    gate_name = gate_spec[1].name

    if gate_name in gate_dict:
        return gate_dict[gate_name]

    # DO WE NEED THIS LOGIC? WHY WOULD THE NAME BE A TUPLE OR LIST?
    elif isinstance(gate_name, tuple) or isinstance(gate_name, list):
        if gate_name[0] in gate_dict:
            return gate_dict[gate_name[0]](gate_name)

    return default_value


class ActionCutLeftWire(DisjointSearchAction):

    """Action class that implements the action of
    cutting the left (first) wire of a two-qubit gate"""

    def getName(self):
        """Return the look-up name of ActionCutLeftWire."""

        return "CutLeftWire"

    def getGroupNames(self):
        """Return the group name of ActionCutLeftWire."""

        return ["WireCut", "TwoQubitGates"]

    def nextStatePrimitive(self, state, gate_spec, max_width):
        """Return the new state that results from applying
        ActionCutLeftWire to state given the gate_spec.
        """

        # If the gate is not a two-qubit gate, then return the empty list
        if len(gate_spec[1].qubits) != 2:
            return list()

        # If the wire-cut limit would be exceeded, return the empty list
        if not state.canAddWires(1):
            return list()

        gate = gate_spec[1]
        q1 = gate.qubits[0]
        q2 = gate.qubits[1]
        w1 = state.getWire(q1)
        w2 = state.getWire(q2)
        r1 = state.findQubitRoot(q1)
        r2 = state.findQubitRoot(q2)

        if r1 == r2:
            return list()

        if not state.canExpandSubcircuit(r2, 1, max_width):
            return list()

        new_state = state.copy()

        rnew = new_state.newWire(q1)
        new_state.mergeRoots(rnew, r2)
        new_state.assertDoNotMergeRoots(r1, r2)  # Because r2 < rnew

        new_state.bell_pairs.append((r1, r2))
        new_state.gamma_UB *= 4

        new_state.addAction(self, gate_spec, (1, w1, rnew))

        return [new_state]

    def registerCut(self, assignment_settings, gate_spec, cut_args):
        """Register the wire cuts made by a ActionCutLeftWire action
        in an AssignmentSettings object.
        """

        registerAllWireCuts(assignment_settings, gate_spec, cut_args)

    def initializeCut(self, assignment_settings, gate_spec, cut_args):
        """Initialize the wire cuts made by a ActionCutLeftWire action
        in an AssignmentSettings object.
        """

        for gate_input in [pair[0] for pair in cut_args]:
            assignment_settings.initWireCut(gate_spec, gate_input)

        assignment_settings.initApplyGate(gate_spec)

    def nextAssignmentPrimitive(
        self, assign_state, constraint_obj, gate_spec, cut_args, assign_actions
    ):
        action_list = assign_actions.getGroup("WireCut")

        return assignWireCuts(
            action_list, assign_state, constraint_obj, gate_spec, cut_args
        )

    def exportCuts(self, circuit_interface, wire_map, gate_spec, cut_args):
        """Insert an LO wire cut into the input circuit for the specified
        gate and cut arguments.
        """

        insertAllLOWireCuts(circuit_interface, wire_map, gate_spec, cut_args)


### Adds ActionCutLeftWire to the object disjoint_subcircuit_actions
disjoint_subcircuit_actions.defineAction(ActionCutLeftWire())


def registerAllWireCuts(assignment_settings, gate_spec, cut_args):
    """Register a list of wire cuts in an AssignmentSettings object."""

    for cut_triple in cut_args:
        assignment_settings.registerWireCut(gate_spec, cut_triple)


def assignWireCuts(action_list, assign_state, constraint_obj, gate_spec, tuple_list):
    if len(tuple_list) <= 0:
        return [
            assign_state,
        ]

    wire_cut = tuple_list[0]
    new_states = list()
    for action in action_list:
        new_states.extend(
            action.nextState(assign_state, constraint_obj, gate_spec, wire_cut)
        )

    final_states = list()
    for state in new_states:
        final_states.extend(
            assignWireCuts(
                action_list, state, constraint_obj, gate_spec, tuple_list[1:]
            )
        )

    return final_states


def insertAllLOWireCuts(circuit_interface, wire_map, gate_spec, cut_args):
    """Insert LO wire cuts into the input circuit for the specified
    gate and all cut arguments.
    """

    gate_ID = gate_spec[0]
    for input_ID, wire_ID, new_wire_ID in cut_args:
        circuit_interface.insertWireCut(
            gate_ID, input_ID, wire_map[wire_ID], wire_map[new_wire_ID], "LO"
        )


class ActionCutRightWire(DisjointSearchAction):

    """Action class that implements the action of
    cutting the right (second) wire of a two-qubit gate"""

    def getName(self):
        """Return the look-up name of ActionCutRightWire."""

        return "CutRightWire"

    def getGroupNames(self):
        """Return the group name of ActionCutRightWire."""

        return ["WireCut", "TwoQubitGates"]

    def nextStatePrimitive(self, state, gate_spec, max_width):
        """Return the new state that results from applying
        ActionCutRightWire to state given the gate_spec.
        """

        # If the gate is not a two-qubit gate, then return the empty list
        if len(gate_spec[1].qubits) != 2:
            return list()

        # If the wire-cut limit would be exceeded, return the empty list
        if not state.canAddWires(1):
            return list()

        gate = gate_spec[1]
        q1 = gate.qubits[0]
        q2 = gate.qubits[1]
        w1 = state.getWire(q1)
        w2 = state.getWire(q2)
        r1 = state.findQubitRoot(q1)
        r2 = state.findQubitRoot(q2)

        if r1 == r2:
            return list()

        if not state.canExpandSubcircuit(r1, 1, max_width):
            return list()

        new_state = state.copy()

        rnew = new_state.newWire(q2)
        new_state.mergeRoots(r1, rnew)
        new_state.assertDoNotMergeRoots(r1, r2)  # Because r1 < rnew

        new_state.bell_pairs.append((r1, r2))
        new_state.gamma_UB *= 4

        new_state.addAction(self, gate_spec, (2, w2, rnew))

        return [new_state]

    def registerCut(self, assignment_settings, gate_spec, cut_args):
        """Register the wire cuts made by a ActionCutRightWire action
        in an AssignmentSettings object.
        """

        registerAllWireCuts(assignment_settings, gate_spec, cut_args)

    def initializeCut(self, assignment_settings, gate_spec, cut_args):
        """Initialize the wire cuts made by a ActionCutRightWire action
        in an AssignmentSettings object.
        """

        for gate_input in [pair[0] for pair in cut_args]:
            assignment_settings.initWireCut(gate_spec, gate_input)

        assignment_settings.initApplyGate(gate_spec)

    def nextAssignmentPrimitive(
        self, assign_state, constraint_obj, gate_spec, cut_args, assign_actions
    ):
        action_list = assign_actions.getGroup("WireCut")

        return assignWireCuts(
            action_list, assign_state, constraint_obj, gate_spec, cut_args
        )

    def exportCuts(self, circuit_interface, wire_map, gate_spec, cut_args):
        """Insert an LO wire cut into the input circuit for the specified
        gate and cut arguments.
        """

        insertAllLOWireCuts(circuit_interface, wire_map, gate_spec, cut_args)


### Adds ActionCutRightWire to the object disjoint_subcircuit_actions
disjoint_subcircuit_actions.defineAction(ActionCutRightWire())


class ActionCutBothWires(DisjointSearchAction):

    """Action class that implements the action of
    cutting both wires of a two-qubit gate"""

    def getName(self):
        """Return the look-up name of ActionCutBothWires."""

        return "CutBothWires"

    def getGroupNames(self):
        """Return the group name of ActionCutBothWires."""

        return ["WireCut", "TwoQubitGates"]

    def nextStatePrimitive(self, state, gate_spec, max_width):
        """Return the new state that results from applying
        ActionCutBothWires to state given the gate_spec.
        """

        # If the gate is not a two-qubit gate, then return the empty list
        if len(gate_spec[1].qubits) != 2:
            return list()

        # If the wire-cut limit would be exceeded, return the empty list
        if not state.canAddWires(2):
            return list()

        # If the maximum width is less than two, return the empty list
        if max_width < 2:
            return list()

        gate = gate_spec[1]
        q1 = gate.qubits[0]
        q2 = gate.qubits[1]
        w1 = state.getWire(q1)
        w2 = state.getWire(q2)
        r1 = state.findQubitRoot(q1)
        r2 = state.findQubitRoot(q2)

        new_state = state.copy()

        rnew_1 = new_state.newWire(q1)
        rnew_2 = new_state.newWire(q2)
        new_state.mergeRoots(rnew_1, rnew_2)
        new_state.assertDoNotMergeRoots(r1, rnew_1)  # Because r1 < rnew_1
        new_state.assertDoNotMergeRoots(r2, rnew_2)  # Because r2 < rnew_2

        new_state.bell_pairs.append((r1, rnew_1))
        new_state.bell_pairs.append((r2, rnew_2))
        new_state.gamma_UB *= 16

        new_state.addAction(self, gate_spec, (1, w1, rnew_1), (2, w2, rnew_2))

        return [new_state]

    def registerCut(self, assignment_settings, gate_spec, cut_args):
        """Register the wire cuts made by a ActionCutBothWires action
        in an AssignmentSettings object.
        """

        registerAllWireCuts(assignment_settings, gate_spec, cut_args)

    def initializeCut(self, assignment_settings, gate_spec, cut_args):
        """Initialize the wire cuts made by a ActionCutBothWires action
        in an AssignmentSettings object.
        """

        for gate_input in [pair[0] for pair in cut_args]:
            assignment_settings.initWireCut(gate_spec, gate_input)

        assignment_settings.initApplyGate(gate_spec)

    def nextAssignmentPrimitive(
        self, assign_state, constraint_obj, gate_spec, cut_args, assign_actions
    ):
        action_list = assign_actions.getGroup("WireCut")

        return assignWireCuts(
            action_list, assign_state, constraint_obj, gate_spec, cut_args
        )

    def exportCuts(self, circuit_interface, wire_map, gate_spec, cut_args):
        """Insert an LO wire cut into the input circuit for the specified
        gate and cut arguments.
        """

        insertAllLOWireCuts(circuit_interface, wire_map, gate_spec, cut_args)


### Adds ActionCutBothWires to the object disjoint_subcircuit_actions
disjoint_subcircuit_actions.defineAction(ActionCutBothWires())


class ActionMultiWireCut(DisjointSearchAction):

    """Action class that implements search over wire cuts
    for gates (protected subcircuits) with more that two inputs"""

    def getName(self):
        """Return the look-up name of ActionMultiWireCut."""

        return "MultiWireCut"

    def getGroupNames(self):
        """Return the group name of ActionMultiWireCut."""

        return ["WireCut", "MultiqubitGates"]

    def nextStatePrimitive(self, state, gate_spec, max_width):
        """Return the new state that results from applying
        ActionMultiWireCut to state given the gate_spec.
        """

        gate = gate_spec[1]

        # If the gate is applied to two or fewer qubits, return the empty list
        if len(gate.qubits) <= 2:
            return list()

        input_pairs = [(i + 1, state.findQubitRoot(q)) for i, q in enumerate(gate.qubits)]
        subcircuits = list(set([pair[1] for pair in input_pairs]))

        return self.nextStateRecurse(
            state, gate_spec, max_width, input_pairs, subcircuits
        )

    def nextStateRecurse(
        self, state, gate_spec, max_width, input_pairs, subcircuits, cuts=[], merges=[]
    ):
        """Recursive implementation of nextState()"""

        # If the limit on the total number of wire cuts would
        # be exceeded, then return the empty list
        if not state.canAddWires(len(cuts)):
            return list()

        # Base case of the recursion
        if len(subcircuits) <= 0:
            # If there are no wire cuts, then return the empty list
            if len(cuts) <= 0:
                return list()

            # Case: all wires are cut
            elif len(merges) <= 0:
                new_state = state.copy()

                gate = gate_spec[1]
                r0 = None

                cut_triples = self.addCutsToNewState(new_state, gate, cuts, r0)

                new_state.addAction(self, gate_spec, *cut_triples)

                return [new_state]

            # Case: at least one wire is not cut
            else:
                new_width = len(cuts) + sum([state.width[r] for r in merges])

                # If applying the gate would cause the number of qubits to
                # exceed the qubit limit even with the wire cuts, then
                # return the empty list
                if new_width > max_width:
                    return list()

                new_state = state.copy()

                r0 = merges[0]
                for r in merges[1:]:
                    new_state.mergeRoots(r0, r)

                # If the gate cannot be applied because it would violate the
                # merge constraints, then do not apply the gate
                if not new_state.verifyMergeConstraints():
                    return list()

                gate = gate_spec[1]
                r0 = new_state.findWireRoot(r0)

                cut_triples = self.addCutsToNewState(new_state, gate, cuts, r0)

                new_state.addAction(self, gate_spec, *cut_triples)

                return [new_state]

        # Recursive step
        else:
            root = subcircuits[0]

            # Case A: all input wires from subcircuit root are cut
            new_cuts = [pair for pair in input_pairs if (pair[1] == root)]

            cut_case = self.nextStateRecurse(
                state,
                gate_spec,
                max_width,
                input_pairs,
                subcircuits[1:],
                cuts + new_cuts,
                merges,
            )

            # Case B: all input wires from subcircuit root are left uncut
            uncut_case = self.nextStateRecurse(
                state,
                gate_spec,
                max_width,
                input_pairs,
                subcircuits[1:],
                cuts,
                merges + [root],
            )

            return cut_case + uncut_case

    def addCutsToNewState(self, new_state, gate, cuts, downstream_root):
        """Updates the new_state to incorporate a list of wire cuts"""

        cut_triples = list()

        for i, root in cuts:
            qubit = gate.qubits[i]
            wire = new_state.getWire(qubit)
            rnew = new_state.newWire(qubit)
            cut_triples.append((i, wire, rnew))
            if downstream_root is None:
                downstream_root = rnew
            else:
                new_state.mergeRoots(rnew, downstream_root)
            new_state.assertDoNotMergeRoots(root, downstream_root)
            new_state.bell_pairs.append((root, downstream_root))
            new_state.gamma_UB *= 4

        return cut_triples

    def registerCut(self, assignment_settings, gate_spec, cut_args):
        """Register the wire cuts made by a ActionMultiWireCut action
        in an AssignmentSettings object.
        """

        registerAllWireCuts(assignment_settings, gate_spec, cut_args)

    def initializeCut(self, assignment_settings, gate_spec, cut_args):
        """Initialize the wire cuts made by a ActionMultiWireCut action
        in an AssignmentSettings object.
        """

        for gate_input in [pair[0] for pair in cut_args]:
            assignment_settings.initWireCut(gate_spec, gate_input)

        assignment_settings.initApplyGate(gate_spec)

    def nextAssignmentPrimitive(
        self, assign_state, constraint_obj, gate_spec, cut_args, assign_actions
    ):
        action_list = assign_actions.getGroup("WireCut")

        return assignWireCuts(
            action_list, assign_state, constraint_obj, gate_spec, cut_args
        )

    def exportCuts(self, circuit_interface, wire_map, gate_spec, cut_args):
        """Insert an LO wire cut into the input circuit for the specified
        gate and cut arguments.
        """

        insertAllLOWireCuts(circuit_interface, wire_map, gate_spec, cut_args)


### Adds ActionMultiWireCut to the object disjoint_subcircuit_actions
disjoint_subcircuit_actions.defineAction(ActionMultiWireCut())
