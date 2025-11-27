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

from collections.abc import Iterable, ValuesView

from qiskit.utils.deprecation import deprecate_func


@deprecate_func(
    removal_timeline="no sooner than qiskit-addon-cutting v0.12.0",
    since="0.11.0",
    package_name="qiskit-addon-cutting",
    additional_msg="Use ``zip([...], strict=True)``. ",
)
def strict_zip(*args, **kwargs):  # pragma: no cover
    """Equivalent to ``zip([...], strict=True)``."""
    return zip(*args, strict=True, **kwargs)


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
