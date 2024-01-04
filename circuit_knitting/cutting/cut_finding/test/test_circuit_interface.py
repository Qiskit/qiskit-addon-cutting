from circuit_cutting_optimizer.circuit_interface import SimpleGateList


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

    def test_CutInterface(self):
        """Test the internal representation of circuit cuts."""

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
        ]  # in the absence of any wire cuts.
        assert (
            circuit_converted.makeWireMapping(name_mapping="default")
            == circuit_converted.makeWireMapping(name_mapping=None)
            == [0, 1, 2, 3]
        )  # the two methods are the same in the absence of wire cuts.
