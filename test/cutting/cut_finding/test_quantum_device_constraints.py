import pytest
from circuit_knitting.cutting.cut_finding.quantum_device_constraints import DeviceConstraints


@pytest.mark.parametrize(
    "qubits_per_QPU, num_QPUs", [(2.1, 1.2), (1.2, 0), (-1, 1), (1, 0)]
)
def test_DeviceConstraints(qubits_per_QPU, num_QPUs):
    """Test device constraints for being valid data types."""

    with pytest.raises(ValueError):
        _ = DeviceConstraints(qubits_per_QPU, num_QPUs)


@pytest.mark.parametrize("qubits_per_QPU, num_QPUs", [(2, 4), (1, 3)])
def test_getQPUWidth(qubits_per_QPU, num_QPUs):
    """Test that getQPUWidth returns number of qubits per qpu."""

    assert DeviceConstraints(qubits_per_QPU, num_QPUs).getQPUWidth() == qubits_per_QPU
