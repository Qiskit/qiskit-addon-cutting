# This code is a Qiskit project.

# (C) Copyright IBM 2023.

# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Tests for quasiprobability decomposition functions."""

import unittest

from ddt import ddt, data, unpack
import numpy as np
from qiskit.circuit.library import (
    CXGate,
    CZGate,
    CRXGate,
    CRYGate,
    CRZGate,
    RXXGate,
    RYYGate,
    RZZGate,
)

from circuit_knitting_toolbox.utils.iteration import unique_by_eq
from circuit_knitting_toolbox.circuit_cutting.qpd.qpd import qpdbasis_from_gate


@ddt
class TestQPDFunctions(unittest.TestCase):
    # Optimal values from https://arxiv.org/abs/2205.00016v2 Corollary 4.4 (page 10)
    @data(
        (CXGate(), 3),
        (CZGate(), 3),
        (CRXGate(np.pi / 7), 1 + 2 * np.abs(np.sin(np.pi / 14))),
        (CRYGate(np.pi / 7), 1 + 2 * np.abs(np.sin(np.pi / 14))),
        (CRZGate(np.pi / 7), 1 + 2 * np.abs(np.sin(np.pi / 14))),
        (RXXGate(np.pi / 7), 1 + 2 * np.abs(np.sin(np.pi / 7))),
        (RYYGate(np.pi / 7), 1 + 2 * np.abs(np.sin(np.pi / 7))),
        (RZZGate(np.pi / 7), 1 + 2 * np.abs(np.sin(np.pi / 7))),
    )
    @unpack
    def test_optimal_kappa_for_known_gates(self, instruction, gamma):
        assert np.isclose(qpdbasis_from_gate(instruction).kappa, gamma)

    @data(
        (RXXGate(np.pi / 7), 5, 5),
        (RYYGate(np.pi / 7), 5, 5),
        (RZZGate(np.pi / 7), 5, 5),
        (CRXGate(np.pi / 7), 5, 5),
        (CRYGate(np.pi / 7), 5, 5),
        (CRZGate(np.pi / 7), 5, 5),
        (CXGate(), 5, 5),
        (CZGate(), 5, 5),
        (RZZGate(0), 1, 1),
        (RXXGate(np.pi), 1, 1),
        (CRYGate(np.pi), 5, 5),
    )
    @unpack
    def test_qpdbasis_from_gate_unique_maps(
        self, instruction, q0_num_unique, q1_num_unique
    ):
        """
        Count the number of unique maps with non-zero weight on each qubit.

        Make sure it is as expected based on the instruction provided.
        """
        basis = qpdbasis_from_gate(instruction)
        # Consider only maps with non-zero weight
        relevant_maps = [
            m for m, w in zip(basis.maps, basis.coeffs) if not np.isclose(w, 0)
        ]
        assert len(unique_by_eq(a for (a, b) in relevant_maps)) == q0_num_unique
        assert len(unique_by_eq(b for (a, b) in relevant_maps)) == q1_num_unique
