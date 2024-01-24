import pytest
from circuit_knitting.cutting.cut_finding.optimization_settings import OptimizationSettings


@pytest.mark.parametrize(
    "max_gamma, max_backjumps ",
    [(2.1, 1), (2, 1.2), (0, 1), (-1, 0)],
)
def test_OptimizationParameters(max_gamma, max_backjumps):
    """Test optimization parameters for being valid data types."""

    with pytest.raises(ValueError):
        _ = OptimizationSettings(max_gamma=max_gamma, max_backjumps=max_backjumps)


def test_GateCutTypes(LO=True, LOCC_ancillas=False, LOCC_no_ancillas=False):
    """Test default gate cut types."""
    op = OptimizationSettings()
    op.setGateCutTypes()
    assert op.gate_cut_LO == True
    assert op.gate_cut_LOCC_with_ancillas == False


def test_WireCutTypes(LO=True, LOCC_ancillas=False, LOCC_no_ancillas=False):
    """Test default wire cut types."""
    op = OptimizationSettings()
    op.setWireCutTypes()
    assert op.wire_cut_LO
    assert op.wire_cut_LOCC_with_ancillas == False
    assert op.wire_cut_LOCC_no_ancillas == False


def test_AllCutSearchGroups():
    """Test for the existence of all Cut search groups."""
    assert OptimizationSettings(
        LO=True, LOCC_ancillas=True, LOCC_no_ancillas=True
    ).getCutSearchGroups() == [None, "GateCut", "WireCut"]
