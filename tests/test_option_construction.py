import numpy as np
import pandas as pd
import pytest

from option_construction import (
    attach_theoretical_premiums,
    black_scholes_price,
    boundary_cap_to_strike,
    construct_candidate_option_structure,
    option_payoff,
    payoff_grid_for_leg,
)


def test_boundary_cap_to_strike_scales_with_market_cap_ratio():
    strike = boundary_cap_to_strike(
        boundary_market_cap=120.0,
        current_market_cap=100.0,
        spot_price=50.0,
    )

    assert strike == pytest.approx(60.0)


def test_construct_candidate_option_structure_creates_four_legs():
    boundaries = pd.DataFrame(
        [
            {
                "Ticker": "AAA",
                "Confidence level": 0.99,
                "Lower loss boundary": 80.0,
                "Upper win boundary": 120.0,
                "Lower loss boundary / current": 0.8,
                "Upper win boundary / current": 1.2,
            },
            {
                "Ticker": "BBB",
                "Confidence level": 0.99,
                "Lower loss boundary": 90.0,
                "Upper win boundary": 130.0,
                "Lower loss boundary / current": 0.9,
                "Upper win boundary / current": 1.3,
            },
        ]
    )
    results = pd.DataFrame(
        [
            {"Ticker": "AAA", "Model probability": 0.6},
            {"Ticker": "BBB", "Model probability": 0.4},
        ]
    )
    current_caps = pd.Series({"AAA": 100.0, "BBB": 100.0})
    spots = pd.Series({"AAA": 50.0, "BBB": 40.0})

    structure = construct_candidate_option_structure(
        boundaries,
        results,
        current_caps,
        spots,
        selected_ticker="AAA",
        competitor_ticker="BBB",
        confidence_level=0.99,
    )

    assert structure["Instrument"].tolist() == [
        "Short AAA Call",
        "Long AAA Put",
        "Long BBB Call",
        "Short BBB Put",
    ]
    assert structure.loc[structure["Instrument"] == "Short AAA Call", "Strike"].iloc[0] == pytest.approx(60.0)
    assert structure.loc[structure["Instrument"] == "Long AAA Put", "Strike"].iloc[0] == pytest.approx(40.0)
    assert structure.loc[structure["Instrument"] == "Long BBB Call", "Strike"].iloc[0] == pytest.approx(52.0)
    assert structure.loc[structure["Instrument"] == "Short BBB Put", "Strike"].iloc[0] == pytest.approx(36.0)


def test_option_payoff_long_and_short_signs():
    terminal_prices = np.array([80.0, 100.0, 120.0])

    long_call = option_payoff("Call", "Long", 100.0, terminal_prices)
    short_call = option_payoff("Call", "Short", 100.0, terminal_prices)
    long_put = option_payoff("Put", "Long", 100.0, terminal_prices)
    short_put = option_payoff("Put", "Short", 100.0, terminal_prices)

    assert long_call.tolist() == pytest.approx([0.0, 0.0, 20.0])
    assert short_call.tolist() == pytest.approx([0.0, 0.0, -20.0])
    assert long_put.tolist() == pytest.approx([20.0, 0.0, 0.0])
    assert short_put.tolist() == pytest.approx([-20.0, 0.0, 0.0])


def test_black_scholes_price_returns_positive_call_and_put_values():
    call = black_scholes_price(spot=100.0, strike=100.0, time_to_expiry=1.0, volatility=0.25, risk_free_rate=0.04, option_type="Call")
    put = black_scholes_price(spot=100.0, strike=100.0, time_to_expiry=1.0, volatility=0.25, risk_free_rate=0.04, option_type="Put")

    assert call > 0.0
    assert put > 0.0
    assert call > put


def test_attach_theoretical_premiums_adds_debit_credit_fields():
    structure = pd.DataFrame(
        [
            {
                "Instrument": "Long AAA Call",
                "Ticker": "AAA",
                "Option type": "Call",
                "Position": "Long",
                "Strike": 110.0,
                "Boundary used": "99% win boundary",
                "Boundary market cap": 120.0,
                "Boundary / current cap": 1.2,
                "Spot": 100.0,
                "Purpose": "test",
            },
            {
                "Instrument": "Short BBB Put",
                "Ticker": "BBB",
                "Option type": "Put",
                "Position": "Short",
                "Strike": 90.0,
                "Boundary used": "99% loss boundary",
                "Boundary market cap": 80.0,
                "Boundary / current cap": 0.8,
                "Spot": 100.0,
                "Purpose": "test",
            },
        ]
    )

    valued = attach_theoretical_premiums(
        structure,
        pd.Series({"AAA": 0.30, "BBB": 0.20}),
        time_to_expiry=0.5,
        risk_free_rate=0.04,
    )

    assert valued["Theoretical premium"].gt(0.0).all()
    assert valued["Premium direction"].tolist() == ["Debit", "Credit"]
    assert valued["Model IV"].tolist() == pytest.approx([0.30, 0.20])


def test_payoff_grid_uses_theoretical_premium_when_available():
    leg = pd.Series(
        {
            "Instrument": "Long AAA Call",
            "Option type": "Call",
            "Position": "Long",
            "Strike": 100.0,
            "Spot": 100.0,
            "Theoretical premium": 5.0,
        }
    )

    payoff_with_premium = payoff_grid_for_leg(leg, points=3)
    payoff_without_premium = payoff_grid_for_leg(leg, points=3, premium=0.0)

    assert payoff_with_premium["Premium"].iloc[0] == pytest.approx(5.0)
    assert payoff_without_premium["Premium"].iloc[0] == pytest.approx(0.0)
