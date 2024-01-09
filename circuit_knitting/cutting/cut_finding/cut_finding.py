# This code is a Qiskit project.

# (C) Copyright IBM 2024.

# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Automatically find cut locations in a quantum circuit."""

from __future__ import annotations

from qiskit import QuantumCircuit

from .optimization_settings import OptimizationSettings
from .quantum_device_constraints import DeviceConstraints


def find_cuts(
    circuit: QuantumCircuit,
    optimization: OptimizationSettings | dict[str, str | int],
    constraints: DeviceConstraints | dict[str, int],
):
    pass
