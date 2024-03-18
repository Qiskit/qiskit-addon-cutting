# This code is a Qiskit project.

# (C) Copyright IBM 2024.

# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Class used for specifying characteristics of the target QPU."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class DeviceConstraints:
    """Specify the constraints (qubits per QPU and maximum number of subcircuits) that must be respected."""

    qubits_per_qpu: int
    max_subcircuits: int

    def __post_init__(self):
        """Post-init method for data class."""
        if self.qubits_per_qpu < 1 or self.max_subcircuits < 1:
            raise ValueError(
                "qubits_per_QPU and num_QPUs must be positive definite integers."
            )

    def get_qpu_width(self) -> int:
        """Return the number of qubits supported on each individual QPU."""
        return self.qubits_per_qpu
