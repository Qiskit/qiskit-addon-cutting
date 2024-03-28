# This code is a Qiskit project.

# (C) Copyright IBM 2024.

# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Tests for CKT equivalence libraries."""

import unittest

import numpy as np
from qiskit.circuit import EquivalenceLibrary
from qiskit.circuit.library.standard_gates import SdgGate


from circuit_knitting.utils.equivalence import equivalence_libraries


class TestEquivalenceLibraries(unittest.TestCase):
    def setUp(self):
        self.heron_lib = equivalence_libraries["heron"]
        self.eagle_lib = equivalence_libraries["eagle"]
        self.standard_lib = equivalence_libraries["standard"]

    def test_equivalence_library_dict(self):
        assert isinstance(self.heron_lib, EquivalenceLibrary)
        assert isinstance(self.eagle_lib, EquivalenceLibrary)
        assert self.standard_lib == None

    def test_equivalence_heron(self):
        heron_equivalence = self.heron_lib.get_entry(SdgGate())[0]
        assert heron_equivalence.data[0].operation.name == "rz"
        assert heron_equivalence.data[0].operation.params == [-np.pi / 2]

    def test_equivalence_eagle(self):
        eagle_equivalence = self.eagle_lib.get_entry(SdgGate())[0]
        assert eagle_equivalence.data[0].operation.name == "rz"
        assert eagle_equivalence.data[0].operation.params == [-np.pi / 2]
