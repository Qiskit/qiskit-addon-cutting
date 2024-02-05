import numpy as np
from numpy import array
from pytest import fixture, raises
from qiskit import QuantumCircuit
from qiskit.circuit.library import EfficientSU2
from circuit_knitting.cutting.cut_finding.cco_utils import QCtoCCOCircuit
from circuit_knitting.cutting.cut_finding.circuit_interface import (
    SimpleGateList,
    CircuitElement,
)
from circuit_knitting.cutting.cut_finding.optimization_settings import (
    OptimizationSettings,
)
from circuit_knitting.cutting.cut_finding.quantum_device_constraints import (
    DeviceConstraints,
)
from circuit_knitting.cutting.cut_finding.disjoint_subcircuits_state import (
    PrintActionListWithNames,
    DisjointSubcircuitsState,
)
from circuit_knitting.cutting.cut_finding.lo_cuts_optimizer import (
    LOCutsOptimizer,
    cut_optimization_search_funcs,
)
from circuit_knitting.cutting.cut_finding.cut_optimization import (
    CutOptimizationFuncArgs,
)
from circuit_knitting.cutting.cut_finding.cutting_actions import (
    disjoint_subcircuit_actions,
)
from circuit_knitting.cutting.cut_finding.best_first_search import BestFirstSearch


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


@fixture
def multiqubit_test_setup():
    qc = QuantumCircuit(3)
    qc.ccx(0, 1, 2)
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

    assert PrintActionListWithNames(output.actions) == []  # no cutting.

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
        {
            "Cut action": "CutTwoQubitGate",
            "Cut Gate": [
                9,
                CircuitElement(name="cx", params=[], qubits=[1, 2], gamma=3),
            ],
        }
    ]

    best_result = optimization_pass.getResults()

    assert output.upperBoundGamma() == best_result.gamma_UB == 9  # 2 LO cnot cuts.

    assert optimization_pass.minimumReached() is True  # matches optimal solution.

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
            "Cut location:": {
                "Gate": [
                    10,
                    CircuitElement(name="cx", params=[], qubits=[3, 4], gamma=3),
                ]
            },
            "Input wire": 1,
        }
    ]

    best_result = optimization_pass.getResults()

    assert output.upperBoundGamma() == best_result.gamma_UB == 4  # One LO wire cut.

    assert optimization_pass.minimumReached() is True  # matches optimal solution


# check if unsupported search engine is flagged.
def test_SelectSearchEngine(gate_cut_test_setup):
    qubits_per_QPU = 4
    num_QPUs = 2

    interface, settings = gate_cut_test_setup

    settings.setEngineSelection("CutOptimization", "BeamSearch")

    search_engine = settings.getEngineSelection("CutOptimization")

    constraint_obj = DeviceConstraints(qubits_per_QPU, num_QPUs)

    optimization_pass = LOCutsOptimizer(interface, settings, constraint_obj)

    with raises(ValueError) as e_info:
        _ = optimization_pass.optimize()
    assert e_info.value.args[0] == f"Search engine {search_engine} is not supported."


# The cutting of multiqubit gates is not supported at present.
def test_MultiqubitCuts(multiqubit_test_setup):
    # QPU with 2 qubits requires cutting.
    qubits_per_QPU = 2
    num_QPUs = 2

    interface, settings = multiqubit_test_setup

    constraint_obj = DeviceConstraints(qubits_per_QPU, num_QPUs)

    optimization_pass = LOCutsOptimizer(interface, settings, constraint_obj)

    with raises(ValueError) as e_info:
        _ = optimization_pass.optimize()
    assert (
        e_info.value.args[0]
        == "At present, only the cutting of two qubit gates is supported."
    )


def test_UpdatedCostBounds(gate_cut_test_setup):
    qubits_per_QPU = 3
    num_QPUs = 2

    interface, settings = gate_cut_test_setup

    constraint_obj = DeviceConstraints(qubits_per_QPU, num_QPUs)

    func_args = CutOptimizationFuncArgs()
    func_args.entangling_gates = interface.getMultiQubitGates()
    func_args.search_actions = disjoint_subcircuit_actions
    func_args.max_gamma = settings.getMaxGamma()
    func_args.qpu_width = constraint_obj.getQPUWidth()
    state = DisjointSubcircuitsState(interface.getNumQubits(), 2)
    bfs = BestFirstSearch(settings, cut_optimization_search_funcs)
    bfs.initialize([state], func_args)

    # Perform cut finding with the default cost upper bound.
    state, _ = bfs.optimizationPass(func_args)
    assert state is not None

    # Update and lower cost upper bound.
    bfs.updateUpperBoundCost((2, 4))
    state, _ = bfs.optimizationPass(func_args)
    assert (
        state is None
    )  # Since any cut has a cost of at least 3, the returned state must be None.
