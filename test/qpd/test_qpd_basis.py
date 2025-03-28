# This code is a Qiskit project.

# (C) Copyright IBM 2023.

# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Tests for QPDBasis class."""

import unittest
import copy

import numpy as np
import pytest
from qiskit.circuit import Parameter
from qiskit.circuit.library.standard_gates import *

from qiskit_addon_cutting.qpd import QPDBasis, QPDMeasure


class TestQPDBasis(unittest.TestCase):
    def setUp(self):
        # RXX decomp
        x_gate = XGate()
        x_r_plus = SXGate()
        x_r_minus = SXdgGate()
        x_measure = [HGate(), QPDMeasure(), HGate()]
        self.truth_rxx_maps = [
            ([], []),
            ([x_gate], [x_gate]),
            (x_measure, [x_r_plus]),
            (x_measure, [x_r_minus]),
            ([x_r_plus], x_measure),
            ([x_r_minus], x_measure),
        ]
        # Hard-code the coeffs to test against
        self.truth_rxx_coeffs = [
            0.7500000000000001,
            0.24999999999999994,
            0.4330127018922193,
            -0.4330127018922193,
            0.4330127018922193,
            -0.4330127018922193,
        ]

        # RYY decomp
        y_gate = YGate()
        y_r_plus = RYGate(1 * np.pi / 2)
        y_r_minus = RYGate(-1 * np.pi / 2)
        y_measure = [SXGate(), QPDMeasure(), SXdgGate()]
        self.truth_ryy_maps = [
            ([], []),
            ([y_gate], [y_gate]),
            (y_measure, [y_r_plus]),
            (y_measure, [y_r_minus]),
            ([y_r_plus], y_measure),
            ([y_r_minus], y_measure),
        ]
        # Hard-code the coeffs to test against
        self.truth_ryy_coeffs = [
            0.9045084971874736,
            0.09549150281252627,
            0.2938926261462365,
            -0.2938926261462365,
            0.2938926261462365,
            -0.2938926261462365,
        ]

        # RZZ decomp
        z_gate = ZGate()
        z_r_plus = SGate()
        z_r_minus = SdgGate()
        z_measure = [QPDMeasure()]
        self.truth_rzz_maps = [
            ([], []),
            ([z_gate], [z_gate]),
            (z_measure, [z_r_plus]),
            (z_measure, [z_r_minus]),
            ([z_r_plus], z_measure),
            ([z_r_minus], z_measure),
        ]
        # Hard-code the coeffs to test against
        self.truth_rzz_coeffs = [
            0.9330127018922194,
            0.06698729810778066,
            0.24999999999999997,
            -0.24999999999999997,
            0.24999999999999997,
            -0.24999999999999997,
        ]

    def test_from_instruction(self):
        rxx_gate = RXXGate(np.pi / 3)
        ryy_gate = RYYGate(np.pi / 5)
        rzz_gate = RZZGate(np.pi / 6)

        rxx_decomp: QPDBasis = QPDBasis.from_instruction(rxx_gate)
        ryy_decomp: QPDBasis = QPDBasis.from_instruction(ryy_gate)
        rzz_decomp: QPDBasis = QPDBasis.from_instruction(rzz_gate)

        rxx_truth = QPDBasis(self.truth_rxx_maps, self.truth_rxx_coeffs)
        ryy_truth = QPDBasis(self.truth_ryy_maps, self.truth_ryy_coeffs)
        rzz_truth = QPDBasis(self.truth_rzz_maps, self.truth_rzz_coeffs)

        self.assertEqual(rxx_truth.maps, rxx_decomp.maps)
        self.assertEqual(ryy_truth.maps, ryy_decomp.maps)
        self.assertEqual(rzz_truth.maps, rzz_decomp.maps)

        np.testing.assert_allclose(rxx_truth.coeffs, rxx_decomp.coeffs)
        np.testing.assert_allclose(ryy_truth.coeffs, ryy_decomp.coeffs)
        np.testing.assert_allclose(rzz_truth.coeffs, rzz_decomp.coeffs)

    def test_eq(self):
        basis = QPDBasis.from_instruction(RXXGate(np.pi / 7))
        self.assertEqual(copy.deepcopy(basis), basis)

    def test_unsupported_gate(self):
        with pytest.raises(ValueError) as e_info:
            QPDBasis.from_instruction(C3XGate())
        assert e_info.value.args[0] == "Instruction not supported: mcx"

    def test_unbound_parameter(self):
        with self.subTest("Explicitly supported gate"):
            # For explicitly support gates, we can give a specific error
            # message due to unbound parameters.
            with pytest.raises(ValueError) as e_info:
                QPDBasis.from_instruction(RZZGate(Parameter("θ")))
            assert (
                e_info.value.args[0]
                == "Cannot decompose (rzz) instruction with unbound parameters."
            )
        with self.subTest("Implicitly supported gate"):
            # For implicitly supported gates, we can detect that `to_matrix`
            # failed, but there are other possible explanations, too.  See
            # https://github.com/Qiskit/qiskit/issues/10396
            with pytest.raises(ValueError) as e_info:
                QPDBasis.from_instruction(XXPlusYYGate(Parameter("θ")))
            assert (
                e_info.value.args[0]
                == "`to_matrix` conversion of two-qubit gate (xx_plus_yy) failed. Often, this can be caused by unbound parameters."
            )

    def test_erroneous_compare(self):
        rxx_truth = QPDBasis(self.truth_rxx_maps, self.truth_rxx_coeffs)
        self.assertFalse(rxx_truth == "rxx")

        new_qpd = copy.deepcopy(rxx_truth)

        new_qpd.coeffs[0] = 0.0
        self.assertFalse(new_qpd == rxx_truth)

        new_qpd.maps[0] = (None, "M")
        self.assertFalse(new_qpd == rxx_truth)

        new_qpd._set_maps(new_qpd.maps[0:-1])
        self.assertFalse(new_qpd == rxx_truth)

    def test_properties(self):
        rxx_truth = QPDBasis(self.truth_rxx_maps, self.truth_rxx_coeffs)
        new_qpd = copy.deepcopy(rxx_truth)
        new_qpd2 = copy.deepcopy(new_qpd)
        new_qpd.coeffs = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
        new_qpd2.coeffs = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]

        self.assertTrue(all(new_qpd.probabilities == new_qpd2.probabilities))
        self.assertTrue(new_qpd.kappa == new_qpd2.kappa)
        self.assertTrue(new_qpd.overhead == new_qpd2.kappa**2)

    def test_mismatching_indices(self):
        maps = [(None, "M")]
        coeffs = [1.0, 2.0]
        with pytest.raises(ValueError) as e_info:
            QPDBasis(maps=maps, coeffs=coeffs)
        assert e_info.value.args[0] == "Coefficients must be same length as maps."

    def test_invalid_maps(self):
        with self.subTest("Empty maps"):
            with pytest.raises(ValueError) as e_info:
                QPDBasis([], [])
            assert (
                e_info.value.args[0]
                == "Number of maps passed to QPDBasis must be nonzero."
            )
        with self.subTest("Maps with inconsistent qubit count"):
            with pytest.raises(ValueError) as e_info:
                QPDBasis([([], []), ([],)], [0.5, 0.5])
            assert (
                e_info.value.args[0]
                == "All maps passed to QPDBasis must act on the same number of qubits. (Index 1 contains a 1-tuple but should contain a 2-tuple.)"
            )
        with self.subTest("Maps on three qubits"):
            with pytest.raises(ValueError) as e_info:
                QPDBasis([([], [], [])], [1.0])
            assert e_info.value.args[0] == "QPDBasis supports at most two qubits."
