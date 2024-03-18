from __future__ import annotations

import pytest
from circuit_knitting.cutting.cut_finding.quantum_device_constraints import (
    DeviceConstraints,
)


@pytest.mark.parametrize("qubits_per_qpu, max_subcircuits", [(1, -1), (-1, 1), (1, 0)])
def test_device_constraints(qubits_per_qpu: int, max_subcircuits: int):
    """Test device constraints for being valid data types."""

    with pytest.raises(ValueError):
        _ = DeviceConstraints(qubits_per_qpu, max_subcircuits)


@pytest.mark.parametrize("qubits_per_qpu, max_subcircuits", [(2, 4), (1, 3)])
def test_get_qpu_width(qubits_per_qpu: int, max_subcircuits: int):
    """Test that get_qpu_width returns number of qubits per qpu."""

    assert (
        DeviceConstraints(qubits_per_qpu, max_subcircuits).get_qpu_width()
        == qubits_per_qpu
    )
