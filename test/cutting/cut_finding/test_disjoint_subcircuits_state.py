from pytest import mark, raises, fixture
from circuit_knitting.cutting.cut_finding.circuit_interface import SimpleGateList
from circuit_knitting.cutting.cut_finding.disjoint_subcircuits_state import (
    DisjointSubcircuitsState,
)
from circuit_knitting.cutting.cut_finding.cut_optimization import (
    disjoint_subcircuit_actions,
)


@mark.parametrize("num_qubits, max_wire_cuts", [(2.1, 1.2), (None, -1), (-1, None)])
def test_StateInitialization(num_qubits, max_wire_cuts):
    """Test device constraints for being valid data types."""

    with raises(ValueError):
        _ = DisjointSubcircuitsState(num_qubits, max_wire_cuts)


@fixture
def testCircuit():
    circuit = [
        ("h", "q1"),
        ("barrier", "q1"),
        ("s", "q0"),
        "barrier",
        ("cx", "q1", "q0"),
    ]

    interface = SimpleGateList(circuit)

    # initialize DisjointSubcircuitsState object.
    state = DisjointSubcircuitsState(interface.getNumQubits(), 2)

    two_qubit_gate = interface.getMultiQubitGates()[0]

    return state, two_qubit_gate


def test_StateUncut(testCircuit):
    state, _ = testCircuit

    assert list(state.wiremap) == [0, 1]

    assert state.num_wires == 2

    assert state.getNumQubits() == 2

    assert list(state.uptree) == [0, 1, 2, 3]

    assert list(state.width) == [1, 1, 1, 1]

    assert list(state.no_merge) == []

    assert state.getSearchLevel() == 0

    # print_output = test_prints(state.print(simple=True))

    # assert print_output == []


def test_ApplyGate(testCircuit):
    state, two_qubit_gate = testCircuit

    next_state = disjoint_subcircuit_actions.getAction(None).nextState(
        state, two_qubit_gate, 10
    )[0]

    assert list(next_state.wiremap) == [0, 1]

    assert next_state.num_wires == 2

    assert next_state.findQubitRoot(1) == 0

    assert next_state.getWireRootMapping() == [0, 0]

    assert list(next_state.uptree) == [0, 0, 2, 3]

    assert list(next_state.width) == [2, 1, 1, 1]

    assert list(next_state.no_merge) == []

    assert next_state.getSearchLevel() == 1


def test_CutGate(testCircuit):
    state, two_qubit_gate = testCircuit

    next_state = disjoint_subcircuit_actions.getAction("CutTwoQubitGate").nextState(
        state, two_qubit_gate, 10
    )[0]

    assert list(next_state.wiremap) == [0, 1]

    assert next_state.checkDoNotMergeRoots(0, 1) is True

    assert next_state.num_wires == 2

    assert state.getNumQubits() == 2

    assert next_state.getWireRootMapping() == [0, 1]

    assert list(next_state.uptree) == [0, 1, 2, 3]

    assert list(next_state.width) == [1, 1, 1, 1]

    assert list(next_state.no_merge) == [(0, 1)]

    assert next_state.getSearchLevel() == 1

    assert next_state.lowerBoundGamma() == 3  # one CNOT cut.

    assert (
        next_state.upperBoundGamma() == 3
    )  # equal to lowerBoundGamma for single gate cuts.


def test_CutLeftWire(testCircuit):
    state, two_qubit_gate = testCircuit

    next_state = disjoint_subcircuit_actions.getAction("CutLeftWire").nextState(
        state, two_qubit_gate, 10
    )[0]

    assert list(next_state.wiremap) == [
        2,
        1,
    ]  # qubit 0 is mapped onto wire ID 2 after cut.

    assert next_state.num_wires == 3

    assert state.getNumQubits() == 2

    assert next_state.canExpandSubcircuit(1, 1, 2) is False

    assert next_state.canExpandSubcircuit(1, 1, 3) is True

    assert next_state.canAddWires(2) is False

    assert next_state.getWireRootMapping() == [0, 1, 1]

    assert next_state.checkDoNotMergeRoots(0, 1) is True

    assert list(next_state.uptree) == [0, 1, 1, 3]

    assert list(next_state.width) == [1, 2, 1, 1]

    assert list(next_state.no_merge) == [(0, 1)]

    assert next_state.getMaxWidth() == 2

    assert next_state.findQubitRoot(0) == 1

    assert next_state.getSearchLevel() == 1

    assert next_state.lowerBoundGamma() == 3

    assert next_state.upperBoundGamma() == 4


def test_CutRightWire(testCircuit):
    state, two_qubit_gate = testCircuit

    next_state = disjoint_subcircuit_actions.getAction("CutRightWire").nextState(
        state, two_qubit_gate, 10
    )[0]

    assert list(next_state.wiremap) == [
        0,
        2,
    ]  # qubit 1 is mapped onto wire ID 2 after cut.

    assert next_state.num_wires == 3

    assert state.getNumQubits() == 2

    assert next_state.canAddWires(1) is True

    assert next_state.getWireRootMapping() == [0, 1, 0]

    assert next_state.checkDoNotMergeRoots(0, 1) is True

    assert list(next_state.uptree) == [0, 1, 0, 3]

    assert list(next_state.width) == [2, 1, 1, 1]

    assert list(next_state.no_merge) == [(0, 1)]

    assert next_state.findQubitRoot(1) == 0

    assert next_state.getSearchLevel() == 1


def test_CutBothWires(testCircuit):
    state, two_qubit_gate = testCircuit

    next_state = disjoint_subcircuit_actions.getAction("CutBothWires").nextState(
        state, two_qubit_gate, 10
    )[0]

    assert list(next_state.wiremap) == [2, 3]

    assert next_state.canAddWires(1) is False

    assert next_state.num_wires == 4

    assert state.getNumQubits() == 2

    assert next_state.getWireRootMapping() == [0, 1, 2, 2]

    assert (
        next_state.checkDoNotMergeRoots(0, 2)
        == next_state.checkDoNotMergeRoots(1, 2)
        is True
    )

    assert list(next_state.uptree) == [0, 1, 2, 2]

    assert list(next_state.width) == [1, 1, 2, 1]

    assert list(next_state.no_merge) == [(0, 2), (1, 3)]

    assert next_state.findQubitRoot(0) == 2  # maps to third wire initialized after cut.

    assert (
        next_state.findQubitRoot(1) == 2
    )  # maps to third wire because of the entangling gate.

    assert next_state.getSearchLevel() == 1

    assert next_state.lowerBoundGamma() == 9  # 3^n scaling.

    assert next_state.upperBoundGamma() == 16  # 4^n scaling.

    assert next_state.verifyMergeConstraints() is True
