# This code is a Qiskit project.

# (C) Copyright IBM 2024.

# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Classes required to search for optimal cut locations."""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from typing import cast
from numpy.typing import NDArray
from .search_space_generator import ActionNames
from .cco_utils import selectSearchEngine, greedyBestFirstSearch
from .cutting_actions import disjoint_subcircuit_actions
from .search_space_generator import (
    getActionSubset,
    SearchFunctions,
    SearchSpaceGenerator,
)
from .disjoint_subcircuits_state import DisjointSubcircuitsState
from .circuit_interface import SimpleGateList, CircuitElement, Sequence
from .optimization_settings import OptimizationSettings
from .quantum_device_constraints import DeviceConstraints


@dataclass
class CutOptimizationFuncArgs:

    """Class for passing relevant arguments to the CutOptimization
    search-space generating functions.
    """

    entangling_gates: Sequence[
        Sequence[int | CircuitElement | None | list]
    ] | None = None
    search_actions: ActionNames | None = None
    max_gamma: float | int | None = None
    qpu_width: int | None = None


def CutOptimizationCostFunc(
    state: DisjointSubcircuitsState, func_args: CutOptimizationFuncArgs
) -> tuple[int | float, int]:
    """Return the cost function. The particular cost function chosen here
    aims to minimize the gamma while also (secondarily) giving preference to
    circuit partitionings that balance the sizes of the resulting partitions,
    by minimizing the maximum width across subcircuits.
    """

    return (state.lowerBoundGamma(), state.getMaxWidth())


def CutOptimizationUpperBoundCostFunc(
    goal_state, func_args: CutOptimizationFuncArgs
) -> tuple[int | float, int | float]:
    """Return the gamma upper bound."""
    return (goal_state.upperBoundGamma(), np.inf)


def CutOptimizationMinCostBoundFunc(
    func_args: CutOptimizationFuncArgs,
) -> tuple[int | float, int | float] | None:
    """Return an a priori min-cost bound defined in the optimization settings."""

    if func_args.max_gamma is None:  # pragma: no cover
        return None

    return (func_args.max_gamma, np.inf)


def CutOptimizationNextStateFunc(
    state: DisjointSubcircuitsState, func_args: CutOptimizationFuncArgs
) -> list[DisjointSubcircuitsState]:
    """Generate a list of next states from the input state."""

    # Get the entangling gate spec that is to be processed next based
    # on the search level of the input state
    assert func_args.entangling_gates is not None
    assert func_args.search_actions is not None

    gate_spec = func_args.entangling_gates[state.getSearchLevel()]

    # Determine which search actions can be performed, taking into
    # account any user-specified constraints that might have been
    # placed on how the current entangling gate is to be handled
    # in the search
    gate = gate_spec[1]
    gate = cast(CircuitElement, gate)
    if len(gate.qubits) == 2:
        action_list = func_args.search_actions.getGroup("TwoQubitGates")
    else:
        raise ValueError(
            "In the current version, only the cutting of two qubit gates is supported."
        )

    gate_actions = gate_spec[2]
    gate_actions = cast(list, gate_actions)
    action_list = getActionSubset(action_list, gate_actions)
    # Apply the search actions to generate a list of next states
    next_state_list = []
    assert action_list is not None
    for action in action_list:
        func_args.qpu_width = cast(int, func_args.qpu_width)
        next_state_list.extend(action.nextState(state, gate_spec, func_args.qpu_width))
    return next_state_list


def CutOptimizationGoalStateFunc(
    state: DisjointSubcircuitsState, func_args: CutOptimizationFuncArgs
) -> bool:
    """Return True if the input state is a goal state (i.e., the cutting decisions made satisfy
    the device constraints and the optimization settings).
    """
    func_args.entangling_gates = cast(list[list], func_args.entangling_gates)
    return state.getSearchLevel() >= len(func_args.entangling_gates)


### Global variable that holds the search-space functions for generating
### the cut optimization search space
cut_optimization_search_funcs = SearchFunctions(
    cost_func=CutOptimizationCostFunc,
    upperbound_cost_func=CutOptimizationUpperBoundCostFunc,
    next_state_func=CutOptimizationNextStateFunc,
    goal_state_func=CutOptimizationGoalStateFunc,
    mincost_bound_func=CutOptimizationMinCostBoundFunc,
)


def greedyCutOptimization(
    circuit_interface: SimpleGateList,
    optimization_settings: OptimizationSettings,
    device_constraints: DeviceConstraints,
    search_space_funcs: SearchFunctions = cut_optimization_search_funcs,
    search_actions: ActionNames = disjoint_subcircuit_actions,
) -> DisjointSubcircuitsState | None:
    func_args = CutOptimizationFuncArgs()
    func_args.entangling_gates = circuit_interface.getMultiQubitGates()
    func_args.search_actions = search_actions
    func_args.max_gamma = optimization_settings.getMaxGamma()
    func_args.qpu_width = device_constraints.getQPUWidth()

    start_state = DisjointSubcircuitsState(
        circuit_interface.getNumQubits(), maxWireCutsCircuit(circuit_interface)
    )
    return greedyBestFirstSearch(start_state, search_space_funcs, func_args)


################################################################################


class CutOptimization:

    """Class that implements cut optimization whereby qubits are not reused
    via circuit folding (i.e., when mid-circuit measurement and active
    reset are not available).

    CutOptimization focuses on using circuit cutting to create disjoint subcircuits.
    It then uses upper and lower bounds on the resulting
    gamma in order to decide where and how to cut while deferring the exact
    choices of quasiprobability decompositions.

    Member Variables:

    circuit (:class:`CircuitInterface`) is the interface object for the circuit
    to be cut.

    settings (:class:`OptimizationSettings`) is an object that contains the settings
    that control the optimization process.

    constraints (:class:`DeviceConstraints`) is an object that contains the device
    constraints that solutions must obey.

    search_funcs (:class:`SearchFunctions`) is an object that holds the functions
    needed to generate and explore the cut optimization search space.

    func_args (:class:`CutOptimizationFuncArgs`) is an object that contains the
    necessary device constraints and optimization settings parameters that
    aree needed by the cut optimization search-space function.

    search_actions (:class:`ActionNames`) is an object that contains the allowed
    actions that are used to generate the search space.

    search_engine (:class`BestFirstSearch`) is an object that implements the
    search algorithm.
    """

    def __init__(
        self,
        circuit_interface,
        optimization_settings,
        device_constraints,
        search_engine_config={
            "CutOptimization": SearchSpaceGenerator(
                functions=cut_optimization_search_funcs,
                actions=disjoint_subcircuit_actions,
            )
        },
    ):
        """A CutOptimization object must be initialized with
        a specification of all of the parameters of the optimization to be
        performed: i.e., the circuit to be cut, the optimization settings,
        the target-device constraints, the functions for generating the
        search space, and the allowed search actions.
        """

        generator = search_engine_config["CutOptimization"]
        search_space_funcs = generator.functions
        search_space_actions = generator.actions

        # Extract the subset of allowed actions as defined in the settings object
        cut_groups = optimization_settings.getCutSearchGroups()
        cut_actions = search_space_actions.copy(cut_groups)

        self.circuit = circuit_interface
        self.settings = optimization_settings
        self.constraints = device_constraints
        self.search_funcs = search_space_funcs
        self.search_actions = cut_actions

        self.func_args = CutOptimizationFuncArgs()
        self.func_args.entangling_gates = self.circuit.getMultiQubitGates()
        self.func_args.search_actions = self.search_actions
        self.func_args.max_gamma = self.settings.getMaxGamma()
        self.func_args.qpu_width = self.constraints.getQPUWidth()

        # Perform an initial greedy best-first search to determine an upper
        # bound for the optimal gamma
        self.greedy_goal_state = greedyCutOptimization(
            self.circuit,
            self.settings,
            self.constraints,
            search_space_funcs=self.search_funcs,
            search_actions=self.search_actions,
        )
        ################################################################################

        # Use the upper bound for the optimal gamma to determine the maximum
        # number of wire cuts that can be performed when allocating the
        # data structures in the actual state.
        max_wire_cuts = maxWireCutsCircuit(self.circuit)

        if self.greedy_goal_state is not None:
            mwc = maxWireCutsGamma(self.greedy_goal_state.upperBoundGamma())
            max_wire_cuts = min(max_wire_cuts, mwc)

        elif self.func_args.max_gamma is not None:
            mwc = maxWireCutsGamma(self.func_args.max_gamma)
            max_wire_cuts = min(max_wire_cuts, mwc)

        # Push the start state onto the search_engine
        start_state = DisjointSubcircuitsState(
            self.circuit.getNumQubits(), max_wire_cuts
        )

        sq = selectSearchEngine(
            "CutOptimization",
            self.settings,
            self.search_funcs,
            stop_at_first_min=False,
        )
        sq.initialize([start_state], self.func_args)

        # Use the upper bound for the optimal gamma to constrain the search
        if self.greedy_goal_state is not None:
            sq.updateUpperBoundGoalState(self.greedy_goal_state, self.func_args)

        self.search_engine = sq
        self.goal_state_returned = False

    def optimizationPass(self) -> tuple[DisjointSubcircuitsState, int | float]:
        """Produce, at each call, a goal state representing a distinct
        set of cutting decisions. None is returned once no additional choices
        of cuts can be made without exceeding the minimum upper bound across
        all cutting decisions previously returned, given the optimization settings.
        """
        state, cost = self.search_engine.optimizationPass(self.func_args)
        if state is None and not self.goal_state_returned:
            state = self.greedy_goal_state
            cost = self.search_funcs.cost_func(state, self.func_args)

        self.goal_state_returned = True

        return state, cost

    def minimumReached(self) -> bool:
        """Return True if the optimization reached a global minimum."""

        return self.search_engine.minimumReached()

    def getStats(self, penultimate: bool = False) -> NDArray[np.int_]:
        """Return the search-engine statistics."""

        return self.search_engine.getStats(penultimate=penultimate)

    def getUpperBoundCost(self) -> tuple[int | float, int | float]:
        """Return the current upperbound cost."""

        return self.search_engine.getUpperBoundCost()

    def updateUpperBoundCost(self, cost_bound: tuple[int | float, int | float]) -> None:
        """Update the cost upper bound based on an input cost bound."""

        self.search_engine.updateUpperBoundCost(cost_bound)


def maxWireCutsCircuit(circuit_interface: SimpleGateList) -> int:
    """Calculate an upper bound on the maximum number of wire cuts
    that can be made given the total number of inputs to multiqubit
    gates in the circuit.

    NOTE: There is no advantage gained by cutting wires that
    only have single qubit gates acting on them, so without
    loss of generality we can assume that wire cutting is
    performed only on the inputs to multiqubit gates.
    """

    multiqubit_wires = [len(x[1].qubits) for x in circuit_interface.getMultiQubitGates()]  # type: ignore
    return sum(multiqubit_wires)


def maxWireCutsGamma(max_gamma: float | int) -> int:
    """Calculate an upper bound on the maximum number of wire cuts
    that can be made given the maximum allowed gamma.
    """

    return int(np.ceil(np.log2(max_gamma + 1) - 1))
