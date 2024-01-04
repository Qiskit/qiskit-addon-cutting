""" File containing classes needed to implement the actions involved in circuit cutting."""
import numpy as np
from abc import ABC, abstractmethod
from .search_space_generator import ActionNames

### This is an object that holds action names for constructing disjoint subcircuits
disjoint_subcircuit_actions = ActionNames()


class DisjointSearchAction(ABC):

    """Base class for search actions for constructing disjoint subcircuits."""

    @abstractmethod
    def getName(self):
        """Derived classes must return the look-up name of the action."""

        assert False, "Derived classes must override getName()"

    @abstractmethod
    def getGroupNames(self):
        """Derived classes must return a list of group names."""

        assert False, "Derived classes must override getGroupNames()"

    @abstractmethod
    def nextStatePrimitive(self, state, gate_spec, max_width):
        """Derived classes must return a list of search states that
        result from applying all variations of the action to gate_spec
        in the specified DisjointSubcircuitsState state, subject to the
        constraint that the number of resulting qubits (wires) in each
        subcircuit cannot exceed max_width.
        """

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
        with the action arguments cut_args.
        """

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
        """Returns a list of next assignment states that result from
        applying assignment actions to the input assignment state.
        """

        next_list = self.nextAssignmentPrimitive(
            assign_state, constraint_obj, gate_spec, cut_args, assign_actions
        )

        for next_state in next_list:
            next_state.setNextLevel(assign_state)

        return next_list


class ActionApplyGate(DisjointSearchAction):

    """Action class that implements the action of
    applying a two-qubit gate without decomposition"""

    def getName(self):
        """Return the look-up name of ActionApplyGate"""

        return None

    def getGroupNames(self):
        """Return the group name of ActionApplyGate"""

        return [None, "TwoQubitGates", "MultiqubitGates"]

    def nextStatePrimitive(self, state, gate_spec, max_width):
        """Return the new state that results from applying
        ActionApplyGate to state given the two-qubit gate
        specification: gate_spec.
        """

        if len(gate_spec[1]) > 3:
            # The function multiqubitNextState handles
            # gates that act on 3 or more qubits.
            return self.multiqubitNextState(state, gate_spec, max_width)

        gate = gate_spec[1]  # extract the gate from gate specification.
        r1 = state.findQubitRoot(gate[1])  # extract the root wire for the first qubit
        # acted on by the given 2-qubit gate.
        r2 = state.findQubitRoot(gate[2])  # extract the root wire for the second qubit
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
        roots = list(set([state.findQubitRoot(q) for q in gate[1:]]))
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
        """The values in gate_dict are tuples in
        (gamma_LB, num_bell_pairs, gamma_UB) format.
        lowerBoundGamma is computed from gamma_LB using the
        DisjointSubcircuitsState.lowerBoundGamma() method.
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
        if len(gate_spec[1]) != 3:
            return list()

        gamma_LB, num_bell_pairs, gamma_UB = self.getCostParams(gate_spec)

        if gamma_LB is None:
            return list()

        gate = gate_spec[1]
        q1 = gate[1]
        q2 = gate[2]
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
        """Call lookupCostParams function."""
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

    def exportCuts(self, circuit_interface, wire_map, gate_spec, args):
        """Insert an LO gate cut into the input circuit for the specified gate
        and cut arguments.
        """

        circuit_interface.insertGateCut(gate_spec[0], "LO")


### Adds ActionCutTwoQubitGate to the object disjoint_subcircuit_actions
disjoint_subcircuit_actions.defineAction(ActionCutTwoQubitGate())


def lookupCostParams(gate_dict, gate_spec, default_value):
    gate_name = gate_spec[1][0]

    if gate_name in gate_dict:
        return gate_dict[gate_name]

    elif isinstance(gate_name, tuple) or isinstance(gate_name, list):
        if gate_name[0] in gate_dict:
            return gate_dict[gate_name[0]](gate_name)

    return default_value
