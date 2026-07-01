import pandas as pd
import pytest

from interactive_portfolio import confidence_at_strike, strike_at_confidence


def sample_curve():
    return pd.DataFrame(
        {
            "market_cap_to_current": [0.5, 0.75, 1.0, 1.25, 1.5],
            "win_probability": [0.01, 0.10, 0.50, 0.80, 0.99],
        }
    )


def test_win_boundary_confidence_maps_to_normalized_strike():
    strike = strike_at_confidence(
        sample_curve(),
        0.80,
        boundary_type="Win boundary",
        normalized_spot=100.0,
    )

    assert strike == pytest.approx(125.0)
    assert confidence_at_strike(
        sample_curve(),
        strike,
        boundary_type="Win boundary",
        normalized_spot=100.0,
    ) == pytest.approx(0.80)


def test_loss_boundary_uses_complementary_probability():
    strike = strike_at_confidence(
        sample_curve(),
        0.90,
        boundary_type="Loss boundary",
        normalized_spot=100.0,
    )

    assert strike == pytest.approx(75.0)
    assert confidence_at_strike(
        sample_curve(),
        strike,
        boundary_type="Loss boundary",
        normalized_spot=100.0,
    ) == pytest.approx(0.90)
