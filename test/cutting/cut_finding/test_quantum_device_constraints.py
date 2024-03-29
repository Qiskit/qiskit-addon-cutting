# This code is a Qiskit project.

# (C) Copyright IBM 2024.

# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Tests for the quantum_device_constraints module."""

from __future__ import annotations

import pytest
from circuit_knitting.cutting.cut_finding.quantum_device_constraints import (
    DeviceConstraints,
)


@pytest.mark.parametrize("qubits_per_subcircuit", [-1, 0])
def test_device_constraints(qubits_per_subcircuit: int):
    """Test device constraints for being valid data types."""

    with pytest.raises(ValueError):
        _ = DeviceConstraints(qubits_per_subcircuit)


@pytest.mark.parametrize("qubits_per_subcircuit", [2, 1])
def test_get_qpu_width(qubits_per_subcircuit: int):
    """Test that get_qpu_width returns number of qubits per qpu."""

    assert (
        DeviceConstraints(qubits_per_subcircuit).get_qpu_width()
        == qubits_per_subcircuit
    )
