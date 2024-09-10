# This code is a Qiskit project.

# (C) Copyright IBM 2022.

# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""
Utility functions.

=================================================================
Bitwise utilities (:mod:`circuit_knitting.utils.bitwise`)
=================================================================

.. automodule:: circuit_knitting.utils.bitwise

=====================================================================
Iteration utilities (:mod:`circuit_knitting.utils.iteration`)
=====================================================================

.. automodule:: circuit_knitting.utils.iteration

===============================================================================
Observable grouping (:mod:`circuit_knitting.utils.observable_grouping`)
===============================================================================

.. automodule:: circuit_knitting.utils.observable_grouping

=============================================================
Simulation (:mod:`circuit_knitting.utils.simulation`)
=============================================================

.. automodule:: circuit_knitting.utils.simulation

=============================================================
Transforms (:mod:`circuit_knitting.utils.transforms`)
=============================================================

.. automodule:: circuit_knitting.utils.transforms

===================================================================
Transpiler passes (:mod:`circuit_knitting.utils.transpiler_passes`)
===================================================================

.. automodule:: circuit_knitting.utils.transpiler_passes
"""

from warnings import warn

warn(
    "The `circuit_knitting.utils` import location has been deprecated and "
    "has been moved to `qiskit_addon_cutting.utils` in the "
    "qiskit-addon-cutting package.  Users are encouraged to migrate to the "
    "new package name and import locations to receive further updates.",
    category=DeprecationWarning,
    stacklevel=1,
)
