# This code is a Qiskit project.

# (C) Copyright IBM 2023.

# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

# Reminder: update the RST file in docs/apidocs when adding new interfaces.
"""Iteration utilities."""

import sys
from collections.abc import Iterable, ValuesView

# Certain calls to `zip` should be strict, but the `strict` kwarg was not added
# until Python 3.10.  The following allows us to begin marking today which
# calls should be strict, even though this will only be enforced on Python 3.10
# and later.
if sys.version_info >= (3, 10, 0):  # pragma: no cover

    def strict_zip(*args, **kwargs):
        """Equivalent to ``zip([...], strict=True)`` where supported."""
        return zip(*args, strict=True, **kwargs)

else:  # pragma: no cover
    strict_zip = zip  # type: ignore


def unique_by_id(iterable: Iterable, /) -> ValuesView:
    """Return unique objects in ``iterable``, by identity.

    >>> a = {0}
    >>> list(unique_by_id([a, a]))
    [{0}]
    >>> list(unique_by_id([a, a.copy()]))
    [{0}, {0}]
    """
    return {id(x): x for x in iterable}.values()


def unique_by_eq(iterable: Iterable, /) -> list:
    """Return unique objects in ``iterable``, by equality.

    This function is only appropriate if (i) there are a small number of
    objects, and (ii) the objects are not guaranteed to be hashable.
    Otherwise, a ``dict`` or ``set`` is a better choice.

    This function may potentially make a comparison between all pairs of
    elements, so it executes in :math:`O(n^2)` time in the worst case, in
    contrast to a ``dict`` or ``set``, both of which can be constructed in
    :math:`O(n)` time.

    >>> a = {0}
    >>> list(unique_by_eq([a, a]))
    [{0}]
    >>> list(unique_by_eq([a, a.copy()]))
    [{0}]
    """
    rv = []
    for item in unique_by_id(iterable):
        if item not in rv:
            rv.append(item)
    return rv
