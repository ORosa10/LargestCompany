import numpy as np
import pandas as pd
import pytest

from manual_portfolio import manual_option_payoffs_and_analytics, resolve_manual_option_legs


def sample_boundaries():
    return pd.DataFrame(
        [
            {
                "Ticker": "AAA",
                "Confidence level": 0.80,
                "Lower loss boundary / current": 0.75,
                "Upper win boundary / current": 1.30,
            }
        ]
    )


def test_resolve_manual_option_legs_uses_normalized_boundary_strikes():
    editor = pd.DataFrame(
        [
            {
                "Active": True,
                "Ticker": "AAA",
                "Option type": "Put",
                "Position": "Long",
                "Quantity": 0.2,
                "Strike source": "Loss boundary",
                "Boundary confidence (%)": 80,
                "Manual strike": 90.0,
                "Pricing IV": 0.30,
            },
            {
                "Active": True,
                "Ticker": "AAA",
                "Option type": "Call",
                "Position": "Short",
                "Quantity": 0.1,
                "Strike source": "Win boundary",
                "Boundary confidence (%)": 80,
                "Manual strike": 110.0,
                "Pricing IV": 0.30,
            },
        ]
    )

    legs = resolve_manual_option_legs(
        editor,
        sample_boundaries(),
        time_to_expiry=1.0,
        risk_free_rate=0.04,
        normalized_spot=100.0,
    )

    assert legs["Strike"].tolist() == pytest.approx([75.0, 130.0])
    assert legs["Strike / spot"].tolist() == pytest.approx([0.75, 1.30])
    assert legs["Theoretical premium"].gt(0).all()


def test_manual_strike_is_used_directly():
    editor = pd.DataFrame(
        [
            {
                "Active": True,
                "Ticker": "AAA",
                "Option type": "Call",
                "Position": "Long",
                "Quantity": 1.0,
                "Strike source": "Manual strike",
                "Boundary confidence (%)": 80,
                "Manual strike": 115.0,
                "Pricing IV": 0.25,
            }
        ]
    )

    legs = resolve_manual_option_legs(
        editor,
        sample_boundaries(),
        time_to_expiry=1.0,
        risk_free_rate=0.04,
    )

    assert legs.iloc[0]["Strike"] == pytest.approx(115.0)
    assert legs.iloc[0]["Strike / spot"] == pytest.approx(1.15)


def test_manual_portfolio_analytics_sum_leg_payoffs():
    legs = pd.DataFrame(
        [
            {
                "Instrument": "Long AAA Put",
                "Ticker": "AAA",
                "Option type": "Put",
                "Position": "Long",
                "Quantity": 1.0,
                "Strike": 90.0,
                "Theoretical premium": 2.0,
            },
            {
                "Instrument": "Short AAA Call",
                "Ticker": "AAA",
                "Option type": "Call",
                "Position": "Short",
                "Quantity": 1.0,
                "Strike": 110.0,
                "Theoretical premium": 3.0,
            },
        ]
    )
    terminal_prices = pd.DataFrame({"AAA": [80.0, 100.0, 120.0]})

    total, analytics = manual_option_payoffs_and_analytics(
        legs,
        terminal_prices,
        contract_multiplier=1.0,
        include_premiums=True,
    )

    assert total.tolist() == pytest.approx([11.0, 1.0, -9.0])
    assert len(analytics) == 2
    assert analytics["Initial premium cashflow"].tolist() == pytest.approx([-2.0, 3.0])
