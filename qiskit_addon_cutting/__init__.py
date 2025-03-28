# This code is a Qiskit project.

# (C) Copyright IBM 2022.

# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Circuit cutting."""

from __future__ import annotations
from importlib.metadata import version, PackageNotFoundError

from .cutting_decomposition import (
    partition_circuit_qubits,
    partition_problem,
    cut_gates,
    PartitionedCuttingProblem,
)
from .cutting_experiments import generate_cutting_experiments
from .cutting_reconstruction import reconstruct_expectation_values
from .wire_cutting_transforms import cut_wires, expand_observables
from .automated_cut_finding import find_cuts, DeviceConstraints, OptimizationParameters

try:
    __version__ = version("qiskit-addon-cutting")
except PackageNotFoundError:  # pragma: no cover
    # Package is not installed
    pass

__all__ = [
    "partition_circuit_qubits",
    "partition_problem",
    "cut_gates",
    "generate_cutting_experiments",
    "reconstruct_expectation_values",
    "PartitionedCuttingProblem",
    "cut_wires",
    "expand_observables",
    "find_cuts",
    "DeviceConstraints",
    "OptimizationParameters",
]
