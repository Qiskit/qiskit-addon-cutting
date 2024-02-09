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
def testCircuit():
    circuit = [
        CircuitElement(name="h", params=[], qubits=["q1"], gamma=None),
        CircuitElement(name="s", params=[], qubits=["q0"], gamma=None),
        CircuitElement(name="cx", params=[], qubits=["q1", "q0"], gamma=3),
    ]

    interface = SimpleGateList(circuit)

    # initialize DisjointSubcircuitsState object.
    state = DisjointSubcircuitsState(interface.getNumQubits(), 2)

    two_qubit_gate = interface.getMultiQubitGates()[0]

    return interface, state, two_qubit_gate


def test_ActionApplyGate(
    testCircuit: Callable[
        [],
        tuple[
            SimpleGateList, DisjointSubcircuitsState, list[int | CircuitElement | None]
        ],
    ]
):
    """Test the application of a gate without any cutting actions."""

    _, state, two_qubit_gate = testCircuit
    apply_gate = ActionApplyGate()
    assert apply_gate.getName() is None
    assert apply_gate.getGroupNames() == [None, "TwoQubitGates"]

    updated_state = apply_gate.nextStatePrimitive(state, two_qubit_gate, 2)
    actions_list = []
    for state in updated_state:
        actions_list.extend(state.actions)
    assert actions_list == []  # no actions when the gate is simply applied.


def test_CutTwoQubitGate(
    testCircuit: Callable[
        [],
        tuple[
            SimpleGateList, DisjointSubcircuitsState, list[int | CircuitElement | None]
        ],
    ]
):
    """Test the action of cutting a two qubit gate."""

    interface, state, two_qubit_gate = testCircuit
    cut_gate = ActionCutTwoQubitGate()
    assert cut_gate.getName() == "CutTwoQubitGate"
    assert cut_gate.getGroupNames() == ["GateCut", "TwoQubitGates"]

    updated_state = cut_gate.nextStatePrimitive(state, two_qubit_gate, 2)
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

    assert cut_gate.getCostParams(two_qubit_gate) == (
        3,
        0,
        3,
    )  # reproduces the parameters for a CNOT when only LO is enabled.

    cut_gate.exportCuts(
        interface, None, two_qubit_gate, None
    )  # insert cut in circuit interface.
    assert interface.cut_type[2] == "LO"


def test_CutLeftWire(
    testCircuit: Callable[
        [],
        tuple[
            SimpleGateList, DisjointSubcircuitsState, list[int | CircuitElement | None]
        ],
    ]
):
    """Test the action of cutting the first (left) input wire to a two qubit gate."""
    _, state, two_qubit_gate = testCircuit
    cut_left_wire = ActionCutLeftWire()
    assert cut_left_wire.getName() == "CutLeftWire"
    assert cut_left_wire.getGroupNames() == ["WireCut", "TwoQubitGates"]

    updated_state = cut_left_wire.nextStatePrimitive(state, two_qubit_gate, 3)
    actions_list = []
    for state in updated_state:
        actions_list.extend(print_actions_list(state.actions))
    assert actions_list[0][0] == "CutLeftWire"
    assert actions_list[0][1][1] == CircuitElement(
        name="cx", params=[], qubits=[0, 1], gamma=3
    )
    assert actions_list[0][2][0][0] == 1  # the first input ('left') wire is cut.


def test_CutRightWire(
    testCircuit: Callable[
        [],
        tuple[
            SimpleGateList, DisjointSubcircuitsState, list[int | CircuitElement | None]
        ],
    ]
):
    """Test the action of cutting the second (right) input wire to a two qubit gate."""
    _, state, two_qubit_gate = testCircuit
    cut_right_wire = ActionCutRightWire()
    assert cut_right_wire.getName() == "CutRightWire"
    assert cut_right_wire.getGroupNames() == ["WireCut", "TwoQubitGates"]

    updated_state = cut_right_wire.nextStatePrimitive(state, two_qubit_gate, 3)
    actions_list = []
    for state in updated_state:
        actions_list.extend(print_actions_list(state.actions))
    assert actions_list[0][0] == "CutRightWire"
    assert actions_list[0][1][1] == CircuitElement(
        name="cx", params=[], qubits=[0, 1], gamma=3
    )
    assert actions_list[0][2][0][0] == 2  # the second input ('right') wire is cut


def test_DefinedActions():
    # Check that unsupported cutting actions return None
    # when the action or corresponding group is requested.

    assert ActionNames().getAction("LOCCGateCut") is None

    assert ActionNames().getGroup("LOCCCUTS") is None
