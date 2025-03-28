# This code is a Qiskit project.

# (C) Copyright IBM 2023.

# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

# Reminder: update the RST file in docs/apidocs when adding new interfaces.
"""Module for bitwise utilities."""

if hasattr(0, "bit_count"):

    def bit_count(x: int, /):  # pragma: no cover
        """Count number of set bits."""
        # New in Python 3.10
        return x.bit_count()  # type: ignore[attr-defined]

else:

    def bit_count(x: int, /):  # pragma: no cover
        """Count number of set bits."""
        # Slower fallback for Python 3.9 and earlier
        return bin(x).count("1")
