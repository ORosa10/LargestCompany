import pandas as pd
import pytest

from boundaries import (
    calculate_boundaries_for_all_tickers,
    calculate_conditional_win_curve,
    find_probability_boundaries,
    pairwise_market_cap_boundary,
    pairwise_probability,
    winner_probability_at_market_cap,
)


def test_conditional_win_curve_and_boundaries_from_scenarios():
    simulated_caps = pd.DataFrame(
        {
            "AAA": [80, 85, 90, 95, 100, 105, 110, 115, 120, 125],
            "BBB": [120, 115, 110, 105, 100, 95, 90, 85, 80, 75],
            "CCC": [88, 89, 90, 91, 92, 93, 94, 95, 96, 97],
        }
    )

    curve = calculate_conditional_win_curve(simulated_caps, "AAA", current_market_cap=100.0, n_bins=5)
    boundaries = find_probability_boundaries(curve, [0.80], current_market_cap=100.0, ticker="AAA")

    assert set(["win_probability", "loss_probability", "average_rank", "scenario_count"]).issubset(curve.columns)
    assert curve["scenario_count"].sum() == len(simulated_caps)
    assert boundaries.loc[0, "Ticker"] == "AAA"
    assert boundaries.loc[0, "Upper win boundary"] > boundaries.loc[0, "Lower loss boundary"]


def test_all_ticker_conditional_boundaries_returns_each_ticker():
    simulated_caps = pd.DataFrame(
        {
            "AAA": [80, 90, 100, 110, 120, 130],
            "BBB": [130, 120, 110, 100, 90, 80],
            "CCC": [85, 95, 105, 115, 125, 135],
        }
    )
    current_caps = pd.Series({"AAA": 100.0, "BBB": 100.0, "CCC": 95.0})

    table = calculate_boundaries_for_all_tickers(simulated_caps, current_caps, [0.80], n_bins=3)

    assert set(table["Ticker"]) == {"AAA", "BBB", "CCC"}
    assert (table["Confidence level"] == 0.80).all()


def test_pairwise_boundary_hits_target_probability():
    selected_cap = 100.0
    competitor_cap = 95.0
    selected_iv = 0.30
    competitor_iv = 0.25
    correlation = 0.40
    days_to_target = 180
    target_probability = 0.70

    boundary_cap = pairwise_market_cap_boundary(
        selected_cap,
        competitor_cap,
        selected_iv,
        competitor_iv,
        correlation,
        days_to_target,
        target_probability,
    )
    probability = pairwise_probability(
        boundary_cap,
        competitor_cap,
        selected_iv,
        competitor_iv,
        correlation,
        days_to_target,
    )

    assert probability == pytest.approx(target_probability)


def test_pairwise_boundary_increases_with_target_probability():
    low_target_cap = pairwise_market_cap_boundary(100.0, 95.0, 0.30, 0.25, 0.40, 180, 0.55)
    high_target_cap = pairwise_market_cap_boundary(100.0, 95.0, 0.30, 0.25, 0.40, 180, 0.80)

    assert high_target_cap > low_target_cap


def test_winner_probability_rises_when_selected_market_cap_rises():
    inputs = pd.DataFrame(
        [
            {"Ticker": "AAA", "Current market cap": 100.0, "Implied volatility": 0.25, "Polymarket YES price": 0.50},
            {"Ticker": "BBB", "Current market cap": 100.0, "Implied volatility": 0.25, "Polymarket YES price": 0.50},
            {"Ticker": "CCC", "Current market cap": 90.0, "Implied volatility": 0.25, "Polymarket YES price": 0.10},
        ]
    )
    corr = pd.DataFrame(
        [[1.0, 0.3, 0.3], [0.3, 1.0, 0.3], [0.3, 0.3, 1.0]],
        index=["AAA", "BBB", "CCC"],
        columns=["AAA", "BBB", "CCC"],
    )

    low_probability = winner_probability_at_market_cap(
        inputs,
        corr,
        ticker="AAA",
        market_cap=90.0,
        days_to_target=90,
        simulations=3_000,
        seed=42,
    )
    high_probability = winner_probability_at_market_cap(
        inputs,
        corr,
        ticker="AAA",
        market_cap=120.0,
        days_to_target=90,
        simulations=3_000,
        seed=42,
    )

    assert high_probability > low_probability
