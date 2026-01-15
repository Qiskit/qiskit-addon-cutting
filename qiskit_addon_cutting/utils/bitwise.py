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

from qiskit.utils.deprecation import deprecate_func  # pragma: no cover


@deprecate_func(
    removal_timeline="no sooner than qiskit-addon-cutting v0.12.0",
    since="0.11.0",
    package_name="qiskit-addon-cutting",
    additional_msg="Use ``x.bit_count()`` method instead of ``bit_count(x)``. ",
)
def bit_count(x: int, /):  # pragma: no cover
    """Count number of set bits."""
    # New in Python 3.10
    return x.bit_count()  # type: ignore[attr-defined]
