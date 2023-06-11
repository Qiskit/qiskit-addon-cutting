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

"""
Circuit Cutting
===============

.. autosummary::
    :toctree: ../stubs/
    :nosignatures:

    partition_circuit_qubits
    partition_problem
    cut_gates
    execute_experiments
    reconstruct_expectation_values

.. autosummary::
    :toctree: ../stubs/
    :nosignatures:
    :template: autosummary/class_no_inherited_members.rst

    PartitionedCuttingProblem
    CuttingExperimentResults

Quasi-Probability Decomposition (QPD)
=====================================

.. autosummary::
    :toctree: ../stubs/
    :nosignatures:
    :template: autosummary/class_no_inherited_members.rst

    qpd.QPDBasis
    qpd.BaseQPDGate
    qpd.SingleQubitQPDGate
    qpd.TwoQubitQPDGate
    qpd.WeightType

.. autosummary::
    :toctree: ../stubs/
    :nosignatures:

    qpd.generate_qpd_samples
    qpd.decompose_qpd_instructions

CutQC
=====

.. autosummary::
   :toctree: ../stubs/
   :nosignatures:

    cutqc.run_subcircuit_instances
    cutqc.generate_summation_terms
    cutqc.build
    cutqc.verify
    cutqc.cut_circuit_wires
    cutqc.evaluate_subcircuits
    cutqc.reconstruct_full_distribution
"""

from .cutting_decomposition import (
    partition_circuit_qubits,
    partition_problem,
    cut_gates,
    decompose_gates,
    execute_experiments,
    reconstruct_expectation_values,
    PartitionedCuttingProblem,
    CuttingExperimentResults,
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
    "CuttingExperimentResults",
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
