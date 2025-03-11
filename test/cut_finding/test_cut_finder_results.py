# This code is a Qiskit project.

# (C) Copyright IBM 2024.

# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""End to end tests for the cut finder workflow."""

from __future__ import annotations

import numpy as np
import unittest
from pytest import raises
from qiskit import QuantumCircuit
from qiskit.circuit.library import efficient_su2
from qiskit_addon_cutting.cut_finding.cco_utils import qc_to_cco_circuit
from qiskit_addon_cutting.cut_finding.circuit_interface import (
    SimpleGateList,
)
from qiskit_addon_cutting.cut_finding.optimization_settings import (
    OptimizationSettings,
)
from qiskit_addon_cutting.automated_cut_finding import DeviceConstraints
from qiskit_addon_cutting.cut_finding.disjoint_subcircuits_state import (
    get_actions_list,
    SingleWireCutIdentifier,
    WireCutLocation,
    CutIdentifier,
    CutLocation,
)
from qiskit_addon_cutting.cut_finding.lo_cuts_optimizer import (
    LOCutsOptimizer,
)
from qiskit_addon_cutting.cut_finding.cut_optimization import CutOptimization


class TestCuttingFourQubitCircuit(unittest.TestCase):
    def setUp(self):
        qc = efficient_su2(4, entanglement="linear", reps=2)
        qc.assign_parameters([0.4] * len(qc.parameters), inplace=True)
        self.circuit_internal = qc_to_cco_circuit(qc)

    def test_four_qubit_cutting_workflow(self):

        with self.subTest("No cuts needed"):

            qubits_per_subcircuit = 4

            interface = SimpleGateList(self.circuit_internal)

            settings = OptimizationSettings(seed=12345, gate_lo=True, wire_lo=True)

            settings.set_engine_selection("CutOptimization", "BestFirst")

            constraint_obj = DeviceConstraints(qubits_per_subcircuit)

            optimization_pass = LOCutsOptimizer(interface, settings, constraint_obj)

            output = optimization_pass.optimize(interface, settings, constraint_obj)

            assert get_actions_list(output.actions) == []  # no cutting.

            assert (
                interface.export_subcircuits_as_string(name_mapping="default") == "AAAA"
            )

        with self.subTest("No cuts found when all flags set to False"):

            qubits_per_subcircuit = 3

            interface = SimpleGateList(self.circuit_internal)

            settings = OptimizationSettings(seed=12345, gate_lo=False, wire_lo=False)

            settings.set_engine_selection("CutOptimization", "BestFirst")

            constraint_obj = DeviceConstraints(qubits_per_subcircuit)

            optimization_pass = LOCutsOptimizer(interface, settings, constraint_obj)

            with raises(ValueError) as e_info:
                optimization_pass.optimize(interface, settings, constraint_obj)
            assert (
                e_info.value.args[0]
                == "None state encountered: no cut state satisfying the specified constraints and settings could be found."
            )

        with self.subTest(
            "No separating cuts possible if one qubit per qpu and only wire cuts allowed"
        ):

            settings = OptimizationSettings(seed=12345, gate_lo=False, wire_lo=True)

            settings.set_engine_selection("CutOptimization", "BestFirst")

            interface = SimpleGateList(self.circuit_internal)

            qubits_per_subcircuit = 1
            constraint_obj = DeviceConstraints(qubits_per_subcircuit)

            optimization_pass = LOCutsOptimizer(interface, settings, constraint_obj)

            with raises(ValueError) as e_info:
                optimization_pass.optimize(interface, settings, constraint_obj)
            assert (
                e_info.value.args[0]
                == "None state encountered: no cut state satisfying the specified constraints and settings could be found."
            )

        with self.subTest("Gate cuts to get three qubits per subcircuit"):
            # QPU with 3 qubits for a 4 qubit circuit enforces cutting.
            qubits_per_subcircuit = 3

            interface = SimpleGateList(self.circuit_internal)

            settings = OptimizationSettings(seed=12345, gate_lo=True, wire_lo=True)

            settings.set_engine_selection("CutOptimization", "BestFirst")

            constraint_obj = DeviceConstraints(qubits_per_subcircuit)

            optimization_pass = LOCutsOptimizer(interface, settings, constraint_obj)

            output = optimization_pass.optimize()

            cut_actions_list = output.cut_actions_sublist()

            assert cut_actions_list == [
                CutIdentifier(
                    cut_action="CutTwoQubitGate",
                    cut_location=CutLocation(
                        instruction_id=10, gate_name="cx", qubits=[2, 3]
                    ),
                ),
                CutIdentifier(
                    cut_action="CutTwoQubitGate",
                    cut_location=CutLocation(
                        instruction_id=21, gate_name="cx", qubits=[2, 3]
                    ),
                ),
            ]
            best_result = optimization_pass.get_results()

            assert (
                output.upper_bound_gamma() == best_result.gamma_UB == 9
            )  # 2 LO cnot cuts.

            assert (
                optimization_pass.minimum_reached() is True
            )  # matches optimal solution.

            assert (
                interface.export_subcircuits_as_string(name_mapping="default") == "AAAB"
            )  # circuit separated into 2 subcircuits.

        with self.subTest("Gate cuts to get two qubits per subcircuit"):

            qubits_per_subcircuit = 2

            interface = SimpleGateList(self.circuit_internal)

            settings = OptimizationSettings(seed=12345, gate_lo=True, wire_lo=True)

            settings.set_engine_selection("CutOptimization", "BestFirst")

            constraint_obj = DeviceConstraints(qubits_per_subcircuit)

            optimization_pass = LOCutsOptimizer(interface, settings, constraint_obj)

            output = optimization_pass.optimize()

            cut_actions_list = output.cut_actions_sublist()

            assert cut_actions_list == [
                CutIdentifier(
                    cut_action="CutTwoQubitGate",
                    cut_location=CutLocation(
                        instruction_id=9, gate_name="cx", qubits=[1, 2]
                    ),
                ),
                CutIdentifier(
                    cut_action="CutTwoQubitGate",
                    cut_location=CutLocation(
                        instruction_id=20, gate_name="cx", qubits=[1, 2]
                    ),
                ),
            ]

        best_result = optimization_pass.get_results()

        assert (
            output.upper_bound_gamma() == best_result.gamma_UB == 9
        )  # 2 LO cnot cuts.

        assert optimization_pass.minimum_reached() is True  # matches optimal solution.

        assert (
            interface.export_subcircuits_as_string(name_mapping="default") == "AABB"
        )  # circuit separated into 2 subcircuits.

        assert (
            optimization_pass.get_stats()["CutOptimization"].backjumps
            <= settings.max_backjumps
        )

        with self.subTest("Cut both wires instance"):

            qubits_per_subcircuit = 2

            interface = SimpleGateList(self.circuit_internal)

            settings = OptimizationSettings(seed=12345, gate_lo=False, wire_lo=True)

            settings.set_engine_selection("CutOptimization", "BestFirst")

            constraint_obj = DeviceConstraints(qubits_per_subcircuit)

            optimization_pass = LOCutsOptimizer(interface, settings, constraint_obj)

            output = optimization_pass.optimize()

            cut_actions_list = output.cut_actions_sublist()

            assert cut_actions_list == [
                SingleWireCutIdentifier(
                    cut_action="CutLeftWire",
                    wire_cut_location=WireCutLocation(
                        instruction_id=9, gate_name="cx", qubits=[1, 2], input=1
                    ),
                ),
                SingleWireCutIdentifier(
                    cut_action="CutLeftWire",
                    wire_cut_location=WireCutLocation(
                        instruction_id=10, gate_name="cx", qubits=[2, 3], input=1
                    ),
                ),
                CutIdentifier(
                    cut_action="CutBothWires",
                    cut_location=CutLocation(
                        instruction_id=19, gate_name="cx", qubits=[0, 1]
                    ),
                ),
                CutIdentifier(
                    cut_action="CutBothWires",
                    cut_location=CutLocation(
                        instruction_id=20, gate_name="cx", qubits=[1, 2]
                    ),
                ),
                CutIdentifier(
                    cut_action="CutBothWires",
                    cut_location=CutLocation(
                        instruction_id=21, gate_name="cx", qubits=[2, 3]
                    ),
                ),
            ]

        best_result = optimization_pass.get_results()

        assert output.upper_bound_gamma() == best_result.gamma_UB == 65536

        assert (
            interface.export_subcircuits_as_string(name_mapping="default")
            == "ADABDEBCEFCF"
        )

        with self.subTest("Wire cuts to get to 3 qubits per subcircuit"):

            qubits_per_subcircuit = 3

            interface = SimpleGateList(self.circuit_internal)

            settings = OptimizationSettings(seed=12345, gate_lo=False, wire_lo=True)

            settings.set_engine_selection("CutOptimization", "BestFirst")

            constraint_obj = DeviceConstraints(qubits_per_subcircuit)

            optimization_pass = LOCutsOptimizer(interface, settings, constraint_obj)

            output = optimization_pass.optimize()

            cut_actions_list = output.cut_actions_sublist()

            assert cut_actions_list == [
                SingleWireCutIdentifier(
                    cut_action="CutLeftWire",
                    wire_cut_location=WireCutLocation(
                        instruction_id=10, gate_name="cx", qubits=[2, 3], input=1
                    ),
                ),
                SingleWireCutIdentifier(
                    cut_action="CutLeftWire",
                    wire_cut_location=WireCutLocation(
                        instruction_id=20, gate_name="cx", qubits=[1, 2], input=1
                    ),
                ),
            ]

        best_result = optimization_pass.get_results()

        assert (
            output.upper_bound_gamma() == best_result.gamma_UB == 16
        )  # 2 LO wire cuts.

        assert (
            interface.export_subcircuits_as_string(name_mapping="default") == "AABABB"
        )  # circuit separated into 2 subcircuits.

        with self.subTest("Search engine not supported"):
            # Check if unspported search engine is flagged

            qubits_per_subcircuit = 4

            interface = SimpleGateList(self.circuit_internal)

            settings = OptimizationSettings(seed=12345, gate_lo=True, wire_lo=True)

            settings.set_engine_selection("CutOptimization", "BeamSearch")

            search_engine = settings.get_engine_selection("CutOptimization")

        constraint_obj = DeviceConstraints(qubits_per_subcircuit)

        optimization_pass = LOCutsOptimizer(interface, settings, constraint_obj)

        with raises(ValueError) as e_info:
            _ = optimization_pass.optimize()
        assert (
            e_info.value.args[0] == f"Search engine {search_engine} is not supported."
        )

        with self.subTest("Greedy search gate cut warm start test"):
            # Even if the input cost bounds are too stringent, greedy_cut_optimization
            # is able to return a solution.

            qubits_per_subcircuit = 3

            interface = SimpleGateList(self.circuit_internal)

            settings = OptimizationSettings(seed=12345, gate_lo=True, wire_lo=False)

            settings.set_engine_selection("CutOptimization", "BestFirst")

            constraint_obj = DeviceConstraints(qubits_per_subcircuit)

            # Impose a stringent cost upper bound, insist gamma <=2.
            cut_opt = CutOptimization(interface, settings, constraint_obj)
            cut_opt.update_upperbound_cost((2, 4))
            state, cost = cut_opt.optimization_pass()

            # 2 cnot cuts are still found
            assert state is not None
            assert cost[0] == 9

        with self.subTest("Greedy search wire cut warm start test"):
            # Even if the input cost bounds are too stringent, greedy_cut_optimization
            # is able to return a solution.

            qubits_per_subcircuit = 3

            interface = SimpleGateList(self.circuit_internal)

            settings = OptimizationSettings(seed=12345, gate_lo=False, wire_lo=True)

            settings.set_engine_selection("CutOptimization", "BestFirst")

            constraint_obj = DeviceConstraints(qubits_per_subcircuit)

            # Impose a stringent cost upper bound, insist gamma <=2.
            cut_opt = CutOptimization(interface, settings, constraint_obj)
            cut_opt.update_upperbound_cost((2, 4))
            state, cost = cut_opt.optimization_pass()

            # 2 LO wire cuts are still found
            assert state is not None
            assert cost[0] == 16


class TestCuttingSevenQubitCircuit(unittest.TestCase):
    def setUp(self):
        qc = QuantumCircuit(7)
        for i in range(7):
            qc.rx(np.pi / 4, i)
        qc.cx(0, 3)
        qc.cx(1, 3)
        qc.cx(2, 3)
        qc.cx(3, 4)
        qc.cx(3, 5)
        qc.cx(3, 6)
        self.circuit_internal = qc_to_cco_circuit(qc)

    def test_seven_qubit_workflow(self):
        with self.subTest("Two qubits per subcircuit"):

            qubits_per_subcircuit = 2

            interface = SimpleGateList(self.circuit_internal)

            settings = OptimizationSettings(seed=12345, gate_lo=True, wire_lo=True)

            settings.set_engine_selection("CutOptimization", "BestFirst")

            constraint_obj = DeviceConstraints(qubits_per_subcircuit)

            optimization_pass = LOCutsOptimizer(interface, settings, constraint_obj)

            output = optimization_pass.optimize()

            cut_actions_list = output.cut_actions_sublist()

            assert cut_actions_list == [
                CutIdentifier(
                    cut_action="CutTwoQubitGate",
                    cut_location=CutLocation(
                        instruction_id=7, gate_name="cx", qubits=[0, 3]
                    ),
                ),
                CutIdentifier(
                    cut_action="CutTwoQubitGate",
                    cut_location=CutLocation(
                        instruction_id=8, gate_name="cx", qubits=[1, 3]
                    ),
                ),
                CutIdentifier(
                    cut_action="CutTwoQubitGate",
                    cut_location=CutLocation(
                        instruction_id=9, gate_name="cx", qubits=[2, 3]
                    ),
                ),
                CutIdentifier(
                    cut_action="CutTwoQubitGate",
                    cut_location=CutLocation(
                        instruction_id=11, gate_name="cx", qubits=[3, 5]
                    ),
                ),
                CutIdentifier(
                    cut_action="CutTwoQubitGate",
                    cut_location=CutLocation(
                        instruction_id=12, gate_name="cx", qubits=[3, 6]
                    ),
                ),
            ]

            best_result = optimization_pass.get_results()

            assert (
                output.upper_bound_gamma() == best_result.gamma_UB == 243
            )  # 5 LO cnot cuts.

            assert (
                optimization_pass.minimum_reached() is True
            )  # matches optimal solution.

            assert (
                interface.export_subcircuits_as_string(name_mapping="default")
                == "ABCDDEF"
            )  # circuit separated into 2 subcircuits.

        with self.subTest("Single wire cut"):

            qubits_per_subcircuit = 4

            interface = SimpleGateList(self.circuit_internal)

            settings = OptimizationSettings(seed=12345, gate_lo=True, wire_lo=True)

            settings.set_engine_selection("CutOptimization", "BestFirst")

            constraint_obj = DeviceConstraints(qubits_per_subcircuit)

            optimization_pass = LOCutsOptimizer(interface, settings, constraint_obj)

            output = optimization_pass.optimize()

            cut_actions_list = output.cut_actions_sublist()

            assert cut_actions_list == [
                SingleWireCutIdentifier(
                    cut_action="CutLeftWire",
                    wire_cut_location=WireCutLocation(
                        instruction_id=10, gate_name="cx", qubits=[3, 4], input=1
                    ),
                )
            ]

            assert (
                interface.export_subcircuits_as_string(name_mapping="default")
                == "AAAABBBB"
            )  # extra wires because of wire cuts
            # and no qubit reuse.

            best_result = optimization_pass.get_results()

            assert (
                output.upper_bound_gamma() == best_result.gamma_UB == 4
            )  # One LO wire cut.

            assert (
                optimization_pass.minimum_reached() is True
            )  # matches optimal solution

        with self.subTest("Two single wire cuts"):

            qubits_per_subcircuit = 3

            interface = SimpleGateList(self.circuit_internal)

            settings = OptimizationSettings(seed=12345, gate_lo=True, wire_lo=True)

            settings.set_engine_selection("CutOptimization", "BestFirst")

            constraint_obj = DeviceConstraints(qubits_per_subcircuit)

            optimization_pass = LOCutsOptimizer(interface, settings, constraint_obj)

            output = optimization_pass.optimize()

            cut_actions_list = output.cut_actions_sublist()

            assert cut_actions_list == [
                SingleWireCutIdentifier(
                    cut_action="CutRightWire",
                    wire_cut_location=WireCutLocation(
                        instruction_id=9, gate_name="cx", qubits=[2, 3], input=2
                    ),
                ),
                SingleWireCutIdentifier(
                    cut_action="CutLeftWire",
                    wire_cut_location=WireCutLocation(
                        instruction_id=11, gate_name="cx", qubits=[3, 5], input=1
                    ),
                ),
            ]

            assert (
                interface.export_subcircuits_as_string(name_mapping="default")
                == "AABABCBCC"
            )  # extra wires because of wire cuts
        # and no qubit reuse. In the string above,
        # {A: wire 0, A:wire 1, B:wire 2, A: wire 3,
        # B: first cut on wire 3, C: second cut on wire 3,
        # B: wire 4, C: wire 5, C: wire 6}.

        best_result = optimization_pass.get_results()

        assert (
            output.upper_bound_gamma() == best_result.gamma_UB == 16
        )  # Two LO wire cuts.

        assert optimization_pass.minimum_reached() is True  # matches optimal solution


class TestCuttingMultiQubitGates(unittest.TestCase):
    def setUp(self):
        qc = QuantumCircuit(3)
        qc.ccx(0, 1, 2)
        circuit_internal = qc_to_cco_circuit(qc)
        self.interface = SimpleGateList(circuit_internal)
        self.settings = OptimizationSettings(seed=12345)
        self.settings.set_engine_selection("CutOptimization", "BestFirst")

    def no_cutting_multiqubit_gates(self):

        # The cutting of multiqubit gates is not supported at present.
        qubits_per_subcircuit = 2

        constraint_obj = DeviceConstraints(qubits_per_subcircuit)

        optimization_pass = LOCutsOptimizer(
            self.interface, self.settings, constraint_obj
        )

        with raises(ValueError) as e_info:
            _ = optimization_pass.optimize()
        assert e_info.value.args[0] == (
            "The input circuit must contain only single and two-qubits gates. "
            "Found 3-qubit gate: (ccx)."
        )
