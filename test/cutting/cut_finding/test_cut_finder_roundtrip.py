import numpy as np
from numpy import array
from pytest import fixture, raises
from qiskit import QuantumCircuit
from qiskit.circuit.library import EfficientSU2
from circuit_knitting.cutting.cut_finding.utils import QCtoCCOCircuit
from circuit_knitting.cutting.cut_finding.circuit_interface import SimpleGateList
from circuit_knitting.cutting.cut_finding.optimization_settings import OptimizationSettings
from circuit_knitting.cutting.cut_finding.quantum_device_constraints import DeviceConstraints
from circuit_knitting.cutting.cut_finding.disjoint_subcircuits_state import (
    PrintActionListWithNames,
)
from circuit_knitting.cutting.cut_finding.lo_cuts_optimizer import LOCutsOptimizer


@fixture
def gate_cut_test_setup():
    qc = EfficientSU2(4, entanglement="linear", reps=2).decompose()
    qc.assign_parameters([0.4] * len(qc.parameters), inplace=True)
    circuit_internal = QCtoCCOCircuit(qc)
    interface = SimpleGateList(circuit_internal)
    settings = OptimizationSettings(rand_seed=12345)
    settings.setEngineSelection("CutOptimization", "BestFirst")
    return interface, settings


@fixture
def wire_cut_test_setup():
    qc = QuantumCircuit(7)
    for i in range(7):
        qc.rx(np.pi / 4, i)
    qc.cx(0, 3)
    qc.cx(1, 3)
    qc.cx(2, 3)
    qc.cx(3, 4)
    qc.cx(3, 5)
    qc.cx(3, 6)
    circuit_internal = QCtoCCOCircuit(qc)
    interface = SimpleGateList(circuit_internal)
    settings = OptimizationSettings(rand_seed=12345)
    settings.setEngineSelection("CutOptimization", "BestFirst")
    return interface, settings


def test_no_cuts(gate_cut_test_setup):
    # QPU with 4 qubits requires no cutting.
    qubits_per_QPU = 4
    num_QPUs = 2

    interface, settings = gate_cut_test_setup

    constraint_obj = DeviceConstraints(qubits_per_QPU, num_QPUs)

    optimization_pass = LOCutsOptimizer(interface, settings, constraint_obj)

    output = optimization_pass.optimize(interface, settings, constraint_obj)

    print(optimization_pass.best_result)

    assert PrintActionListWithNames(output.actions) == [] #no cutting.

    assert interface.exportSubcircuitsAsString(name_mapping="default") == "AAAA"


def test_GateCuts(gate_cut_test_setup):
    # QPU with 2 qubits requires cutting.
    qubits_per_QPU = 2
    num_QPUs = 2

    interface, settings = gate_cut_test_setup

    constraint_obj = DeviceConstraints(qubits_per_QPU, num_QPUs)

    optimization_pass = LOCutsOptimizer(interface, settings, constraint_obj)

    output = optimization_pass.optimize()

    cut_actions_list = output.CutActionsList()

    assert cut_actions_list == [
        {"Cut action": "CutTwoQubitGate", "Cut Gate": [9, ["cx", 1, 2]]}
    ]

    best_result = optimization_pass.getResults()

    assert output.upperBoundGamma() == best_result.gamma_UB == 9  # 2 LO cnot cuts.

    assert optimization_pass.minimumReached() == True  # matches optimal solution.

    assert (
        interface.exportSubcircuitsAsString(name_mapping="default") == "AABB"
    )  # circuit separated into 2 subcircuits.

    assert (
        optimization_pass.getStats()["CutOptimization"] == array([15, 46, 15, 6])
    ).all()  # matches known stats.


def test_WireCuts(wire_cut_test_setup):
    qubits_per_QPU = 4
    num_QPUs = 2

    interface, settings = wire_cut_test_setup

    constraint_obj = DeviceConstraints(qubits_per_QPU, num_QPUs)

    optimization_pass = LOCutsOptimizer(interface, settings, constraint_obj)

    output = optimization_pass.optimize()

    cut_actions_list = output.CutActionsList()

    assert cut_actions_list == [
        {
            "Cut action": "CutLeftWire",
            "Cut location:": {"Gate": [10, ["cx", 3, 4]]},
            "Input wire": 1,
        }
    ]

    best_result = optimization_pass.getResults()

    assert output.upperBoundGamma() == best_result.gamma_UB == 4  # One LO wire cut.

    assert optimization_pass.minimumReached() == True  # matches optimal solution


def test_selectSearchEngine(gate_cut_test_setup):
    qubits_per_QPU = 4
    num_QPUs = 2

    interface, settings = gate_cut_test_setup

    # check if unsupported search engine is flagged.
    settings.setEngineSelection("CutOptimization", "BeamSearch")

    constraint_obj = DeviceConstraints(qubits_per_QPU, num_QPUs)

    optimization_pass = LOCutsOptimizer(interface, settings, constraint_obj)

    with raises(ValueError):
        _ = optimization_pass.optimize()
