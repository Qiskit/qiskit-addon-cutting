# This code is a Qiskit project.

# (C) Copyright IBM 2024.

# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Tests for best_first_search module."""

from __future__ import annotations

from pytest import fixture
from numpy import inf
from qiskit_addon_cutting.cut_finding.circuit_interface import (
    SimpleGateList,
    CircuitElement,
    GateSpec,
)
from qiskit_addon_cutting.cut_finding.cut_optimization import (
    cut_optimization_next_state_func,
    cut_optimization_min_cost_bound_func,
    cut_optimization_cost_func,
    cut_optimization_goal_state_func,
    cut_optimization_upper_bound_cost_func,
    CutOptimizationFuncArgs,
    CutOptimization,
)
from qiskit_addon_cutting.cut_finding.optimization_settings import (
    OptimizationSettings,
)
from qiskit_addon_cutting.automated_cut_finding import DeviceConstraints
from qiskit_addon_cutting.cut_finding.disjoint_subcircuits_state import (
    get_actions_list,
)
from qiskit_addon_cutting.cut_finding.cutting_actions import (
    disjoint_subcircuit_actions,
    DisjointSubcircuitsState,
)

from qiskit_addon_cutting.cut_finding.best_first_search import (
    BestFirstSearch,
    SearchFunctions,
)


@fixture
def test_circuit():
    circuit = [
        CircuitElement(name="cx", params=[], qubits=[0, 1], gamma=3),
        CircuitElement(name="cx", params=[], qubits=[0, 2], gamma=3),
        CircuitElement(name="cx", params=[], qubits=[1, 2], gamma=3),
        CircuitElement(name="cx", params=[], qubits=[0, 3], gamma=3),
        CircuitElement(name="cx", params=[], qubits=[1, 3], gamma=3),
        CircuitElement(name="cx", params=[], qubits=[2, 3], gamma=3),
        CircuitElement(name="cx", params=[], qubits=[4, 5], gamma=3),
        CircuitElement(name="cx", params=[], qubits=[4, 6], gamma=3),
        CircuitElement(name="cx", params=[], qubits=[5, 6], gamma=3),
        CircuitElement(name="cx", params=[], qubits=[4, 7], gamma=3),
        CircuitElement(name="cx", params=[], qubits=[5, 7], gamma=3),
        CircuitElement(name="cx", params=[], qubits=[6, 7], gamma=3),
        CircuitElement(name="cx", params=[], qubits=[3, 4], gamma=3),
        CircuitElement(name="cx", params=[], qubits=[3, 5], gamma=3),
        CircuitElement(name="cx", params=[], qubits=[3, 6], gamma=3),
        CircuitElement(name="cx", params=[], qubits=[0, 1], gamma=3),
        CircuitElement(name="cx", params=[], qubits=[0, 2], gamma=3),
        CircuitElement(name="cx", params=[], qubits=[1, 2], gamma=3),
        CircuitElement(name="cx", params=[], qubits=[0, 3], gamma=3),
        CircuitElement(name="cx", params=[], qubits=[1, 3], gamma=3),
        CircuitElement(name="cx", params=[], qubits=[2, 3], gamma=3),
        CircuitElement(name="cx", params=[], qubits=[4, 5], gamma=3),
        CircuitElement(name="cx", params=[], qubits=[4, 6], gamma=3),
        CircuitElement(name="cx", params=[], qubits=[5, 6], gamma=3),
        CircuitElement(name="cx", params=[], qubits=[4, 7], gamma=3),
        CircuitElement(name="cx", params=[], qubits=[5, 7], gamma=3),
        CircuitElement(name="cx", params=[], qubits=[6, 7], gamma=3),
    ]
    interface = SimpleGateList(circuit)
    return interface


def test_best_first_search(test_circuit: SimpleGateList):
    settings = OptimizationSettings(seed=12345)

    settings.set_engine_selection("CutOptimization", "BestFirst")

    constraint_obj = DeviceConstraints(qubits_per_subcircuit=4)

    op = CutOptimization(test_circuit, settings, constraint_obj)

    out, _ = op.optimization_pass()
    assert op.search_engine.get_stats(penultimate=True) is not None
    assert op.search_engine.get_stats() is not None
    assert op.get_upperbound_cost() == (27, inf)
    assert op.minimum_reached() is True
    assert out is not None
    assert (out.lower_bound_gamma(), out.gamma_UB, out.get_max_width()) == (
        27,
        27,
        4,
    )  # lower and upper bounds are the same in the absence of LOCC.
    actions_sublist = []
    for i, action in enumerate(get_actions_list(out.actions)):
        assert action.action.get_name() == "CutTwoQubitGate"
        actions_sublist.append(get_actions_list(out.actions)[i][1:])
    assert actions_sublist == [
        (
            GateSpec(
                instruction_id=12,
                gate=CircuitElement(name="cx", params=[], qubits=[3, 4], gamma=3),
                cut_constraints=None,
            ),
            (((1, 3), (2, 4)),),
        ),
        (
            GateSpec(
                instruction_id=13,
                gate=CircuitElement(name="cx", params=[], qubits=[3, 5], gamma=3),
                cut_constraints=None,
            ),
            (((1, 3), (2, 5)),),
        ),
        (
            GateSpec(
                instruction_id=14,
                gate=CircuitElement(name="cx", params=[], qubits=[3, 6], gamma=3),
                cut_constraints=None,
            ),
            (((1, 3), (2, 6)),),
        ),
    ]

    out, _ = op.optimization_pass()

    assert op.search_engine.get_stats(penultimate=True) is not None
    assert op.search_engine.get_stats() is not None
    assert op.get_upperbound_cost() == (27, inf)
    assert op.minimum_reached() is True
    assert out is None


def test_best_first_search_termination():
    """Test that if the best first search is run multiple times, it terminates once no further feasible cut states can be found,
    in which case None is returned for both the cost and the state. This test also serves to describe the workflow of the optimizer
    at a granular level."""

    # Specify circuit
    circuit = [
        CircuitElement(name="cx", params=[], qubits=[0, 1], gamma=3),
        CircuitElement(name="cx", params=[], qubits=[2, 3], gamma=3),
        CircuitElement(name="cx", params=[], qubits=[1, 2], gamma=3),
    ]

    interface = SimpleGateList(circuit)

    # Specify optimization settings, search engine, and device constraints.
    settings = OptimizationSettings(seed=123)
    settings.set_engine_selection("CutOptimization", "BestFirst")

    constraints = DeviceConstraints(qubits_per_subcircuit=3)

    # Initialize and pass arguments to search space generating object.
    func_args = CutOptimizationFuncArgs()
    func_args.entangling_gates = interface.get_multiqubit_gates()
    func_args.search_actions = disjoint_subcircuit_actions
    func_args.max_gamma = settings.get_max_gamma
    func_args.qpu_width = constraints.get_qpu_width()

    # Initialize search functions object, needed to explore a search space.
    cut_optimization_search_funcs = SearchFunctions(
        cost_func=cut_optimization_cost_func,
        upperbound_cost_func=cut_optimization_upper_bound_cost_func,
        next_state_func=cut_optimization_next_state_func,
        goal_state_func=cut_optimization_goal_state_func,
        mincost_bound_func=cut_optimization_min_cost_bound_func,
    )

    # Initialize disjoint subcircuits state object
    # while specifying number of qubits and max allowed wire cuts.
    state = DisjointSubcircuitsState(interface.get_num_qubits(), 2)

    # Initialize bfs object.
    bfs = BestFirstSearch(
        optimization_settings=settings, search_functions=cut_optimization_search_funcs
    )

    # Push an input state.
    bfs.initialize([state], func_args)

    counter = 0

    cut_state = state
    while cut_state is not None:
        cut_state, cut_cost = bfs.optimization_pass(func_args)
        counter += 1

    # There are 5 possible cut states that can be found for this circuit,
    # given that there need to be 3 qubits per subcircuit. These correspond
    # to 3 gate cuts (i.e cutting any of the 3 gates) and cutting either of
    # the input wires to the CNOT between qubits 1 and 2.
    # After these 5 possible cuts are returned, at the 6th iteration, None
    # is returned for both the state and the cost.
    assert counter == 6 and cut_cost is None
