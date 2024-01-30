from pytest import fixture
from numpy import inf
from circuit_knitting.cutting.cut_finding.circuit_interface import SimpleGateList
from circuit_knitting.cutting.cut_finding.cut_optimization import CutOptimization
from circuit_knitting.cutting.cut_finding.optimization_settings import OptimizationSettings
from circuit_knitting.cutting.cut_finding.quantum_device_constraints import DeviceConstraints
from circuit_knitting.cutting.cut_finding.disjoint_subcircuits_state import (
    PrintActionListWithNames,
)


@fixture
def testCircuit():
    circuit = [
        ("cx", 0, 1),
        ("cx", 0, 2),
        ("cx", 1, 2),
        ("cx", 0, 3),
        ("cx", 1, 3),
        ("cx", 2, 3),
        ("cx", 4, 5),
        ("cx", 4, 6),
        ("cx", 5, 6),
        ("cx", 4, 7),
        ("cx", 5, 7),
        ("cx", 6, 7),
        ("cx", 3, 4),
        ("cx", 3, 5),
        ("cx", 3, 6),
        ("cx", 0, 1),
        ("cx", 0, 2),
        ("cx", 1, 2),
        ("cx", 0, 3),
        ("cx", 1, 3),
        ("cx", 2, 3),
        ("cx", 4, 5),
        ("cx", 4, 6),
        ("cx", 5, 6),
        ("cx", 4, 7),
        ("cx", 5, 7),
        ("cx", 6, 7),
    ]
    interface = SimpleGateList(circuit)
    return interface


def test_BestFirstSearch(testCircuit):
    settings = OptimizationSettings(rand_seed=12345)

    settings.setEngineSelection("CutOptimization", "BestFirst")

    constraint_obj = DeviceConstraints(qubits_per_QPU=4, num_QPUs=2)

    op = CutOptimization(testCircuit, settings, constraint_obj)

    out, _ = op.optimizationPass()

    assert op.search_engine.getStats(penultimate = True) is not None
    assert op.search_engine.getStats() is not None
    assert op.getUpperBoundCost() == (27, inf)
    assert op.minimumReached() == False
    assert out is not None
    assert (out.lowerBoundGamma(), out.gamma_UB, out.getMaxWidth()) == (15, 27, 4)
    assert PrintActionListWithNames(out.actions) == [
        ["CutTwoQubitGate", [12, ["cx", 3, 4], None], ((1, 3), (2, 4))],
        ["CutTwoQubitGate", [13, ["cx", 3, 5], None], ((1, 3), (2, 5))],
        ["CutTwoQubitGate", [14, ["cx", 3, 6], None], ((1, 3), (2, 6))],
    ]

    out, _ = op.optimizationPass()

    assert op.search_engine.getStats(penultimate = True) is not None
    assert op.search_engine.getStats() is not None
    assert op.getUpperBoundCost() == (27, inf)
    assert op.minimumReached() == True
    assert out is None
