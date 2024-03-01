from __future__ import annotations

from pytest import fixture
from typing import Callable
from circuit_knitting.cutting.cut_finding.circuit_interface import (
    CircuitElement,
    SimpleGateList,
)
from circuit_knitting.cutting.cut_finding.cutting_actions import (
    ActionApplyGate,
    ActionCutTwoQubitGate,
    ActionCutLeftWire,
    ActionCutRightWire,
)
from circuit_knitting.cutting.cut_finding.disjoint_subcircuits_state import (
    DisjointSubcircuitsState,
    print_actions_list,
)
from circuit_knitting.cutting.cut_finding.search_space_generator import ActionNames


@fixture
def test_circuit():
    circuit = [
        CircuitElement(name="h", params=[], qubits=["q1"], gamma=None),
        CircuitElement(name="s", params=[], qubits=["q0"], gamma=None),
        CircuitElement(name="cx", params=[], qubits=["q1", "q0"], gamma=3),
    ]

    interface = SimpleGateList(circuit)

    # initialize DisjointSubcircuitsState object.
    state = DisjointSubcircuitsState(interface.get_num_qubits(), 2)

    two_qubit_gate = interface.get_multiqubit_gates()[0]

    return interface, state, two_qubit_gate


def test_action_apply_gate(
    test_circuit: Callable[
        [],
        tuple[
            SimpleGateList, DisjointSubcircuitsState, list[int | CircuitElement | None]
        ],
    ]
):
    """Test the application of a gate without any cutting actions."""

    _, state, two_qubit_gate = test_circuit
    apply_gate = ActionApplyGate()
    assert apply_gate.get_name() is None
    assert apply_gate.get_group_names() == [None, "TwoQubitGates"]

    updated_state = apply_gate.next_state_primitive(state, two_qubit_gate, 2)
    actions_list = []
    for state in updated_state:
        actions_list.extend(state.actions)
    assert actions_list == []  # no actions when the gate is simply applied.


def test_cut_two_qubit_gate(
    test_circuit: Callable[
        [],
        tuple[
            SimpleGateList, DisjointSubcircuitsState, list[int | CircuitElement | None]
        ],
    ]
):
    """Test the action of cutting a two qubit gate."""

    interface, state, two_qubit_gate = test_circuit
    cut_gate = ActionCutTwoQubitGate()
    assert cut_gate.get_name() == "CutTwoQubitGate"
    assert cut_gate.get_group_names() == ["GateCut", "TwoQubitGates"]

    updated_state = cut_gate.next_state_primitive(state, two_qubit_gate, 2)
    actions_list = []
    for state in updated_state:
        actions_list.extend(print_actions_list(state.actions))
    assert actions_list == [
        [
            "CutTwoQubitGate",
            [2, CircuitElement(name="cx", params=[], qubits=[0, 1], gamma=3), None],
            ((1, 0), (2, 1)),
        ]
    ]

    assert cut_gate.get_cost_params(two_qubit_gate) == (
        3,
        0,
        3,
    )  # reproduces the parameters for a CNOT when only LO is enabled.

    cut_gate.export_cuts(
        interface, None, two_qubit_gate, None
    )  # insert cut in circuit interface.
    assert interface.cut_type[2] == "LO"


def test_cut_left_wire(
    test_circuit: Callable[
        [],
        tuple[
            SimpleGateList, DisjointSubcircuitsState, list[int | CircuitElement | None]
        ],
    ]
):
    """Test the action of cutting the first (left) input wire to a two qubit gate."""
    _, state, two_qubit_gate = test_circuit
    cut_left_wire = ActionCutLeftWire()
    assert cut_left_wire.get_name() == "CutLeftWire"
    assert cut_left_wire.get_group_names() == ["WireCut", "TwoQubitGates"]

    updated_state = cut_left_wire.next_state_primitive(state, two_qubit_gate, 3)
    actions_list = []
    for state in updated_state:
        actions_list.extend(print_actions_list(state.actions))
    # TO-DO: Consider replacing actions_list with a NamedTuple.
    assert actions_list[0][0] == "CutLeftWire"
    assert actions_list[0][1][1] == CircuitElement(
        name="cx", params=[], qubits=[0, 1], gamma=3
    )
    assert actions_list[0][2][0][0] == 1  # the first input ('left') wire is cut.


def test_cut_right_wire(
    test_circuit: Callable[
        [],
        tuple[
            SimpleGateList, DisjointSubcircuitsState, list[int | CircuitElement | None]
        ],
    ]
):
    """Test the action of cutting the second (right) input wire to a two qubit gate."""
    _, state, two_qubit_gate = test_circuit
    cut_right_wire = ActionCutRightWire()
    assert cut_right_wire.get_name() == "CutRightWire"
    assert cut_right_wire.get_group_names() == ["WireCut", "TwoQubitGates"]

    updated_state = cut_right_wire.next_state_primitive(state, two_qubit_gate, 3)
    actions_list = []
    for state in updated_state:
        actions_list.extend(print_actions_list(state.actions))
    assert actions_list[0][0] == "CutRightWire"
    assert actions_list[0][1][1] == CircuitElement(
        name="cx", params=[], qubits=[0, 1], gamma=3
    )
    assert actions_list[0][2][0][0] == 2  # the second input ('right') wire is cut


def test_defined_actions():
    # Check that unsupported cutting actions return None
    # when the action or corresponding group is requested.

    assert ActionNames().get_action("LOCCGateCut") is None

    assert ActionNames().get_group("LOCCCUTS") is None
