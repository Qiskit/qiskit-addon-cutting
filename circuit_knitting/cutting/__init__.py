# This code is a Qiskit project.

# (C) Copyright IBM 2022.

# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""
Circuit Cutting (:mod:`circuit_knitting.cutting`).

.. currentmodule:: circuit_knitting.cutting

Circuit Cutting
===============

.. autosummary::
    :toctree: ../stubs/
    :nosignatures:

    transform_cuts_to_moves
    expand_observables
    partition_circuit_qubits
    partition_problem
    cut_gates
    decompose_gates
    execute_experiments
    reconstruct_expectation_values

.. autosummary::
    :toctree: ../stubs/
    :nosignatures:
    :template: autosummary/class_no_inherited_members.rst

    PartitionedCuttingProblem
    CuttingExperimentResults
    instructions.CutWire
    instructions.Move

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

    qpd.generate_qpd_weights
    qpd.generate_qpd_samples
    qpd.decompose_qpd_instructions
    qpd.qpdbasis_from_instruction

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
    PartitionedCuttingProblem,
)
from .cutting_evaluation import execute_experiments, CuttingExperimentResults
from .cutting_reconstruction import reconstruct_expectation_values
from .cut_wire_to_move import transform_cuts_to_moves, expand_observables

__all__ = [
    "partition_circuit_qubits",
    "partition_problem",
    "cut_gates",
    "decompose_gates",
    "execute_experiments",
    "reconstruct_expectation_values",
    "PartitionedCuttingProblem",
    "CuttingExperimentResults",
    "transform_cuts_to_moves",
    "expand_observables",
]
