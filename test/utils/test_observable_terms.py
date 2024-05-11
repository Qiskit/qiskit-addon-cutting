# This code is a Qiskit project.

# (C) Copyright IBM 2024.

# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

import unittest
import pytest

from qiskit.quantum_info import Pauli, PauliList, SparsePauliOp

from circuit_knitting.utils.observable_terms import *


class TestObservableTerms(unittest.TestCase):
    def test_gather(self):
        with self.subTest("Normal usage"):
            terms = gather_unique_observable_terms(
                [Pauli("XX"), SparsePauliOp(PauliList(["iXY", "ZZ", "XX"]))]
            )
            assert terms.num_qubits == 2
            assert len(terms) == 3
        with self.subTest("Mismatching observables"):
            with pytest.raises(ValueError) as e_info:
                gather_unique_observable_terms([Pauli("XX"), Pauli("XYZ")])
            assert (
                e_info.value.args[0]
                == "Cannot construct PauliList.  Do provided observables all have the same number of qubits?"
            )
        with self.subTest("Empty PauliList should work"):
            terms = gather_unique_observable_terms(PauliList(["XXXX"])[:0])
            assert terms == PauliList(["IIII"])[:0]
        with self.subTest("Empty list should not (unable to infer qubit count)"):
            with pytest.raises(ValueError) as e_info:
                gather_unique_observable_terms([])
            assert e_info.value.args[0] == "observables list cannot be empty"
        with self.subTest("All zero coefficients"):
            unique_terms = gather_unique_observable_terms(
                SparsePauliOp(PauliList(["XYZ"]), [0])
            )
            assert unique_terms == PauliList(["III"])[:0]

    def test_reconstruct(self):
        with self.subTest("Normal usage"):
            evs = reconstruct_observable_expvals_from_terms(
                [Pauli("XX"), SparsePauliOp(PauliList(["iXY", "ZZ", "XX"]))],
                {Pauli("XX"): 7, Pauli("XY"): 11, Pauli("ZZ"): 13},
            )
            assert len(evs) == 2
            assert evs[0] == 7
            assert evs[1] == 20 + 11j
        with self.subTest("No observables"):
            evs = reconstruct_observable_expvals_from_terms(
                [], {Pauli("XX"): 7, Pauli("XY"): 11, Pauli("ZZ"): 13}
            )
            assert len(evs) == 0
        with self.subTest("No observables or terms"):
            evs = reconstruct_observable_expvals_from_terms([], {})
            assert len(evs) == 0
        with self.subTest("Missing term"):
            with pytest.raises(ValueError) as e_info:
                reconstruct_observable_expvals_from_terms(
                    [Pauli("XY")], {Pauli("XX"): 5, Pauli("ZZ"): 7}
                )
            assert (
                e_info.value.args[0]
                == "An observable contains a Pauli term whose expectation value was not provided."
            )
        with self.subTest("SparsePauliOp including a term with zero coeffient"):
            evs = reconstruct_observable_expvals_from_terms(
                [SparsePauliOp(["XX", "-XY"], [0, 1])], {Pauli("XY"): 3}
            )
            assert len(evs) == 1
            assert evs[0] == -3
