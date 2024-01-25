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
        """Derived classes must return the look-up name of the action."""

    @abstractmethod
    def getGroupNames(self):
        """Derived classes must return a list of group names."""

    @abstractmethod
    def nextStatePrimitive(self, state, gate_spec, max_width):
        """Derived classes must return a list of search states that
        result from applying all variations of the action to gate_spec
        in the specified DisjointSubcircuitsState state, subject to the
        constraint that the number of resulting qubits (wires) in each
        subcircuit cannot exceed max_width.
        """

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

        r1 = state.findQubitRoot(
            gate.qubits[0]
        )  # extract the root wire for the first qubit
        # acted on by the given 2-qubit gate.
        r2 = state.findQubitRoot(
            gate.qubits[1]
        )  # extract the root wire for the second qubit
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


### Adds ActionApplyGate to the object disjoint_subcircuit_actions
disjoint_subcircuit_actions.defineAction(ActionApplyGate())


class ActionCutTwoQubitGate(DisjointSearchAction):

    """Action class that implements the action of
        cutting a two-qubit gate.

        TODO: The list of supported gates needs to be expanded.
    """

    def __init__(self):
        """The values in gate_dict are tuples in (gamma_LB, num_bell_pairs, gamma_UB) format.
        lowerBoundGamma is computed from gamma_LB using the DisjointSubcircuitsState.lowerBoundGamma() method.
        """

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

    @staticmethod
    def getCostParams(gate_spec):
        """
        Get the cost parameters.

        This method returns a tuple of the form:
            (gamma_lower_bound, num_bell_pairs, gamma_upper_bound)

        Since CKT only supports single-cut LO, these tuples will be of
        the form (gamma, 0, gamma).
        """
        gamma = gate_spec[1].gamma
        return (gamma, 0, gamma)


    def exportCuts(self, circuit_interface, wire_map, gate_spec, args):
        """Insert an LO gate cut into the input circuit for the specified gate
        and cut arguments.
        """

        circuit_interface.insertGateCut(gate_spec[0], "LO")


### Adds ActionCutTwoQubitGate to the object disjoint_subcircuit_actions
disjoint_subcircuit_actions.defineAction(ActionCutTwoQubitGate())


def lookupCostParams(gate_dict, gate_spec, default_value):
    gate_name = gate_spec[1].name
    params = gate_spec[1].params
    if len(params) == 0:
        return gate_dict[gate_name]

    else:
        if gate_name in gate_dict:
            return gate_dict[gate_name]((gate_name, *params))

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


    def exportCuts(self, circuit_interface, wire_map, gate_spec, cut_args):
        """Insert an LO wire cut into the input circuit for the specified
        gate and cut arguments.
        """

        insertAllLOWireCuts(circuit_interface, wire_map, gate_spec, cut_args)


### Adds ActionCutLeftWire to the object disjoint_subcircuit_actions
disjoint_subcircuit_actions.defineAction(ActionCutLeftWire())



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

    def exportCuts(self, circuit_interface, wire_map, gate_spec, cut_args):
        """Insert an LO wire cut into the input circuit for the specified
        gate and cut arguments.
        """

        insertAllLOWireCuts(circuit_interface, wire_map, gate_spec, cut_args)


### Adds ActionCutBothWires to the object disjoint_subcircuit_actions
disjoint_subcircuit_actions.defineAction(ActionCutBothWires())
