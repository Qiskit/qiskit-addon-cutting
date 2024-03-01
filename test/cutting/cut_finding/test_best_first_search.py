from __future__ import annotations

from pytest import fixture
from numpy import inf
from circuit_knitting.cutting.cut_finding.circuit_interface import (
    SimpleGateList,
    CircuitElement,
)
from circuit_knitting.cutting.cut_finding.cut_optimization import CutOptimization
from circuit_knitting.cutting.cut_finding.optimization_settings import (
    OptimizationSettings,
)
from circuit_knitting.cutting.cut_finding.quantum_device_constraints import (
    DeviceConstraints,
)
from circuit_knitting.cutting.cut_finding.disjoint_subcircuits_state import (
    print_actions_list,
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
    settings = OptimizationSettings(rand_seed=12345)

    settings.set_engine_selection("CutOptimization", "BestFirst")

    constraint_obj = DeviceConstraints(qubits_per_QPU=4, num_QPUs=2)

    op = CutOptimization(test_circuit, settings, constraint_obj)

    out, _ = op.optimization_pass()

    assert op.search_engine.get_stats(penultimate=True) is not None
    assert op.search_engine.get_stats() is not None
    assert op.get_upperbound_cost() == (27, inf)
    assert op.minimum_reached() is False
    assert out is not None
    assert (out.lower_bound_gamma(), out.gamma_UB, out.get_max_width()) == (
        27,
        27,
        4,
    )  # lower and upper bounds are the same in the absence of LOCC.
    assert print_actions_list(out.actions) == [
        [
            "CutTwoQubitGate",
            [12, CircuitElement(name="cx", params=[], qubits=[3, 4], gamma=3), None],
            ((1, 3), (2, 4)),
        ],
        [
            "CutTwoQubitGate",
            [13, CircuitElement(name="cx", params=[], qubits=[3, 5], gamma=3), None],
            ((1, 3), (2, 5)),
        ],
        [
            "CutTwoQubitGate",
            [14, CircuitElement(name="cx", params=[], qubits=[3, 6], gamma=3), None],
            ((1, 3), (2, 6)),
        ],
    ]

    out, _ = op.optimization_pass()

    assert op.search_engine.get_stats(penultimate=True) is not None
    assert op.search_engine.get_stats() is not None
    assert op.get_upperbound_cost() == (27, inf)
    assert op.minimum_reached() is True
    assert out is None