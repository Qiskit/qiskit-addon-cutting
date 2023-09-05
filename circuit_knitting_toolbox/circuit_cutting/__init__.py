# This code is a Qiskit project.

# (C) Copyright IBM 2023.

# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Deprecated import location ``circuit_knitting_toolbox.circuit_cutting``."""

import sys
from warnings import warn

from circuit_knitting.cutting import (
    partition_circuit_qubits,
    partition_problem,
    cut_gates,
    decompose_gates,
    execute_experiments,
    reconstruct_expectation_values,
    PartitionedCuttingProblem,
    qpd,
    cutqc,
    cutting_decomposition,
    cutting_evaluation,
    cutting_reconstruction,
    wire_cutting,
)

__all__ = [
    "partition_circuit_qubits",
    "partition_problem",
    "cut_gates",
    "decompose_gates",
    "execute_experiments",
    "reconstruct_expectation_values",
    "PartitionedCuttingProblem",
]

sys.modules["circuit_knitting_toolbox.circuit_cutting.qpd"] = qpd
sys.modules["circuit_knitting_toolbox.circuit_cutting.qpd.qpd"] = qpd.qpd
sys.modules["circuit_knitting_toolbox.circuit_cutting.qpd.qpd_basis"] = qpd.qpd_basis
sys.modules[
    "circuit_knitting_toolbox.circuit_cutting.qpd.instructions"
] = qpd.instructions
sys.modules[
    "circuit_knitting_toolbox.circuit_cutting.qpd.instructions.qpd_gate"
] = qpd.instructions.qpd_gate
sys.modules[
    "circuit_knitting_toolbox.circuit_cutting.qpd.instructions.qpd_measure"
] = qpd.instructions.qpd_measure
sys.modules["circuit_knitting_toolbox.circuit_cutting.cutqc"] = cutqc
sys.modules["circuit_knitting_toolbox.circuit_cutting.wire_cutting"] = wire_cutting
sys.modules[
    "circuit_knitting_toolbox.circuit_cutting.cutting_decomposition"
] = cutting_decomposition
sys.modules[
    "circuit_knitting_toolbox.circuit_cutting.cutting_evaluation"
] = cutting_evaluation
sys.modules[
    "circuit_knitting_toolbox.circuit_cutting.cutting_reconstruction"
] = cutting_reconstruction

warn(
    f"The package namespace {__name__} is deprecated and will be removed "
    "no sooner than Circuit Knitting Toolbox 0.4.0. Use namespace "
    "circuit_knitting.cutting instead.",
    DeprecationWarning,
    stacklevel=2,
)
