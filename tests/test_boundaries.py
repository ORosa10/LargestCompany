import pandas as pd
import pytest

from boundaries import pairwise_market_cap_boundary, pairwise_probability, winner_probability_at_market_cap
from model import run_probability_engine


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
