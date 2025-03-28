# This code is a Qiskit project.

# (C) Copyright IBM 2023.

# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

from qiskit_addon_cutting.utils.iteration import (
    unique_by_id,
    unique_by_eq,
    strict_zip,
)


def test_unique_by_id():
    a = [1, 3, 5]
    for x, y in strict_zip(
        unique_by_id([9, a, None, 5, a, 5, False, 7, None, "b"]),
        [9, a, None, 5, False, 7, "b"],
    ):
        assert x is y
    assert list(unique_by_id([a, a.copy()])) == [a, a]
    assert list(unique_by_id([a, a])) == [a]
    assert list(unique_by_id([a, 5, None, a])) == [a, 5, None]


def test_unique_by_eq():
    a = [1, 3, 5]
    b = [2, 4, 6]
    assert unique_by_eq([a, b, 5, b, a]) == [a, b, 5]
    assert unique_by_eq([a, b, 5, b.copy(), a.copy()]) == [a, b, 5]
