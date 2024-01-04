from pytest import mark, raises, fixture
from circuit_cutting_optimizer.circuit_interface import SimpleGateList
from circuit_cutting_optimizer.disjoint_subcircuits_state import (
    DisjointSubcircuitsState,
)
from circuit_cutting_optimizer.cut_optimization import (
    disjoint_subcircuit_actions,
)


@mark.parametrize("num_qubits", [2.1, -1])
def test_StateInitialization(num_qubits):
    """Test that initialization values are valid data types."""

    with raises(ValueError):
        _ = DisjointSubcircuitsState(num_qubits)


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

    state = DisjointSubcircuitsState(
        interface.getNumQubits()
    )  # initialization of DisjoingSubcircuitsState object.

    two_qubit_gate = interface.getMultiQubitGates()[0]

    return state, two_qubit_gate


def test_StateUnCut(testCircuit):
    state, _ = testCircuit

    assert list(state.wiremap) == [0, 1]

    assert state.num_wires == 2

    assert list(state.uptree) == [0, 1]

    assert list(state.width) == [1, 1]

    assert list(state.no_merge) == []

    assert state.getSearchLevel() == 0


def test_ApplyGate(testCircuit):
    state, two_qubit_gate = testCircuit

    next_state = disjoint_subcircuit_actions.getAction(None).nextState(
        state, two_qubit_gate, 10
    )[0]

    assert list(next_state.wiremap) == [0, 1]

    assert next_state.num_wires == 2

    assert next_state.findQubitRoot(1) == 0

    assert list(next_state.uptree) == [0, 0]

    assert list(next_state.width) == [2, 1]

    assert list(next_state.no_merge) == []

    assert next_state.getSearchLevel() == 1


def test_CutGate(testCircuit):
    state, two_qubit_gate = testCircuit

    next_state = disjoint_subcircuit_actions.getAction("CutTwoQubitGate").nextState(
        state, two_qubit_gate, 10
    )[0]

    assert list(next_state.wiremap) == [0, 1]

    assert next_state.num_wires == 2

    assert list(next_state.uptree) == [0, 1]

    assert list(next_state.width) == [1, 1]

    assert list(next_state.no_merge) == [(0, 1)]

    assert next_state.getSearchLevel() == 1
