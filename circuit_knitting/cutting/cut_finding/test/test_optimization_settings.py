import pytest
from circuit_cutting_optimizer.optimization_settings import OptimizationSettings


@pytest.mark.parametrize(
    "max_gamma, max_backjumps, beam_width ",
    [(2.1, 1.2, -1.4), (1.2, 1.5, 2.3), (0, 1, 1), (1, 1, 0)],
)
def test_OptimizationParameters(max_gamma, max_backjumps, beam_width):
    """Test optimization parameters for being valid data types."""

    with pytest.raises(ValueError):
        _ = OptimizationSettings(
            max_gamma=max_gamma, max_backjumps=max_backjumps, beam_width=beam_width
        )


def test_AllCutSearchGroups():
    """Test for the existence of all enabled cutting search groups."""

    assert OptimizationSettings(LO=True).getCutSearchGroups() == [None, "GateCut"]
