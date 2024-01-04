import pytest
from circuit_cutting_optimizer.quantum_device_constraints import DeviceConstraints


@pytest.mark.parametrize("qubits_per_QPU, num_QPUs", [(2.1, 1.2), (1.2, 0), (-1, 1)])
def test_DeviceConstraints(qubits_per_QPU, num_QPUs):
    """Test device constraints for being valid data types."""

    with pytest.raises(ValueError):
        _ = DeviceConstraints(qubits_per_QPU, num_QPUs)
