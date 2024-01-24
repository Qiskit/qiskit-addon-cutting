import numpy as np
from circuit_knitting.cutting.cut_finding.circuit_interface import SimpleGateList


class TestCircuitInterface:
    def test_CircuitConversion(self):
        """Test conversion of circuits to the internal representation
        used by the circuit-cutting optimizer.
        """

        trial_circuit = [
            ("h", "q1"),
            ("barrier", "q1"),
            ("s", "q0"),
            "barrier",
            ("cx", "q1", "q0"),
        ]
        circuit_converted = SimpleGateList(trial_circuit)

        assert circuit_converted.getNumQubits() == 2
        assert circuit_converted.getNumWires() == 2
        assert circuit_converted.qubit_names.item_dict == {"q1": 0, "q0": 1}
        assert circuit_converted.getMultiQubitGates() == [[4, ["cx", 0, 1], None]]
        assert circuit_converted.circuit == [
            [["h", 0], None],
            [["barrier", 0], None],
            [["s", 1], None],
            ["barrier", None],
            [["cx", 0, 1], None],
        ]

    def test_GateCutInterface(self):
        """Test the internal representation of LO gate cuts."""

        trial_circuit = [
            ("cx", 0, 1),
            ("cx", 2, 3),
            ("cx", 1, 2),
            ("cx", 0, 1),
            ("cx", 2, 3),
        ]
        circuit_converted = SimpleGateList(trial_circuit)
        circuit_converted.insertGateCut(2, "LO")
        circuit_converted.defineSubcircuits([[0, 1], [2, 3]])

        assert list(circuit_converted.new_gate_ID_map) == [0, 1, 2, 3, 4]
        assert circuit_converted.cut_type == [None, None, "LO", None, None]
        assert (
            circuit_converted.exportSubcircuitsAsString(name_mapping="default")
            == "AABB"
        )
        assert circuit_converted.exportCutCircuit(name_mapping="default") == [
            ["cx", 0, 1],
            ["cx", 2, 3],
            ["cx", 1, 2],
            ["cx", 0, 1],
            ["cx", 2, 3],
        ]

        # the following two methods are the same in the absence of wire cuts.
        assert (
            list(circuit_converted.exportOutputWires(name_mapping="default"))
            == list(circuit_converted.exportOutputWires(name_mapping=None))
            == [0, 1, 2, 3]
        )

    def test_WireCutInterface(self):
        """Test the internal representation of LO wire cuts."""

        trial_circuit = [
            ("cx", 0, 1),
            ("cx", 2, 3),
            ("cx", 1, 2),
            ("cx", 0, 1),
            ("cx", 2, 3),
        ]
        circuit_converted = SimpleGateList(trial_circuit)
        circuit_converted.insertWireCut(
            2, 1, 1, 4, "LO"
        )  # cut first input wire of trial_circuit[2] and map it to wire id 4.
        assert list(circuit_converted.output_wires) == [0, 4, 2, 3]

        assert circuit_converted.cut_type[2] == "LO"

        # the missing gate 2 corresponds to a move operation
        assert list(circuit_converted.new_gate_ID_map) == [0, 1, 3, 4, 5]

        assert circuit_converted.exportCutCircuit(name_mapping=None) == [
            ["cx", 0, 1],
            ["cx", 2, 3],
            ["move", 1, ("cut", 1)],
            ["cx", ("cut", 1), 2],
            ["cx", 0, ("cut", 1)],
            ["cx", 2, 3],
        ]

        # relabel wires after wire cuts according to 'None' name_mapping.
        assert circuit_converted.exportOutputWires(name_mapping=None) == {
            0: 0,
            1: ("cut", 1),
            2: 2,
            3: 3,
        }

        assert circuit_converted.exportCutCircuit(name_mapping="default") == [
            ["cx", 0, 1],
            ["cx", 3, 4],
            ["move", 1, 2],
            ["cx", 2, 3],
            ["cx", 0, 2],
            ["cx", 3, 4],
        ]

        # relabel wires after wire cuts according to 'default' name_mapping.
        assert circuit_converted.exportOutputWires(name_mapping="default") == {
            0: 0,
            1: 2,
            2: 3,
            3: 4,
        }
