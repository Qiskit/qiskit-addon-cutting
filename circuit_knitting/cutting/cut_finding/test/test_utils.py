import pytest
from qiskit import QuantumCircuit
from qiskit.circuit.library import EfficientSU2
from circuit_cutting_optimizer.utils import QCtoCCOCircuit

# test circuit 1.
qc1 = QuantumCircuit(2)
qc1.h(1)
qc1.barrier(1)
qc1.s(0)
qc1.barrier()
qc1.cx(1, 0)

# test circuit 2
qc2 = EfficientSU2(2, entanglement="linear", reps=2).decompose()
qc2.assign_parameters([0.4] * len(qc2.parameters), inplace=True)


@pytest.mark.parametrize(
    "input, output",
    [
        (qc1, [("h", 1), ("barrier", 1), ("s", 0), "barrier", ("cx", 1, 0)]),
        (
            qc2,
            [
                (("ry", 0.4), 0),
                (("rz", 0.4), 0),
                (("ry", 0.4), 1),
                (("rz", 0.4), 1),
                ("cx", 0, 1),
                (("ry", 0.4), 0),
                (("rz", 0.4), 0),
                (("ry", 0.4), 1),
                (("rz", 0.4), 1),
                ("cx", 0, 1),
                (("ry", 0.4), 0),
                (("rz", 0.4), 0),
                (("ry", 0.4), 1),
                (("rz", 0.4), 1),
            ],
        ),
    ],
)
def test_QCtoCCOCircuit(input, output):
    circuit_internal = QCtoCCOCircuit(input)
    assert circuit_internal == output
