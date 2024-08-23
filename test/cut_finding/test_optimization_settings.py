# This code is a Qiskit project.

# (C) Copyright IBM 2024.

# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Tests for the optimization_settings module."""

from __future__ import annotations

import pytest
from qiskit_addon_cutting.cut_finding.optimization_settings import (
    OptimizationSettings,
)


@pytest.mark.parametrize(
    "max_gamma, max_backjumps ",
    [(0, 1), (-1, 0), (1, -1)],
)
def test_optimization_parameters(max_gamma: int, max_backjumps: int):
    """Test optimization parameters for being valid data types."""

    with pytest.raises(ValueError):
        _ = OptimizationSettings(max_gamma=max_gamma, max_backjumps=max_backjumps)


def test_gate_cut_types(gate_lo: bool = True, gate_locc_ancillas: bool = False):
    """Test default gate cut types."""
    op = OptimizationSettings(gate_lo, gate_locc_ancillas)
    op.set_gate_cut_types()
    assert op.gate_cut_lo is True
    assert op.gate_cut_locc_with_ancillas is False


def test_wire_cut_types(
    wire_lo: bool = True,
    wire_locc_ancillas: bool = False,
    wire_locc_no_ancillas: bool = False,
):
    """Test default wire cut types."""
    op = OptimizationSettings(wire_lo, wire_locc_ancillas, wire_locc_no_ancillas)
    op.set_wire_cut_types()
    assert op.wire_cut_lo
    assert op.wire_cut_locc_with_ancillas is False
    assert op.wire_cut_locc_no_ancillas is False


def test_all_cut_search_groups():
    """Test for the existence of all cut search groups."""
    assert OptimizationSettings(
        gate_lo=True,
        gate_locc_ancillas=True,
        wire_lo=True,
        wire_locc_ancillas=True,
        wire_locc_no_ancillas=True,
    ).get_cut_search_groups() == [None, "GateCut", "WireCut"]
