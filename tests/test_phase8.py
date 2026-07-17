import numpy as np
import pandas as pd
import pytest

from model import default_correlation_matrix, run_probability_engine
from phase7 import PortfolioSpec
from phase8 import (
    budget_scaling,
    breakeven_bands,
    capital_metrics,
    kelly_sizing,
    return_distribution,
    risk_metrics,
)


def _result(seed=3, simulations=20000):
    company = pd.DataFrame(
        [
            {"Ticker": "NVDA", "Current market cap": 4_300e9, "Implied volatility": 0.42, "Polymarket YES price": 0.83},
            {"Ticker": "AAPL", "Current market cap": 3_100e9, "Implied volatility": 0.24, "Polymarket YES price": 0.123},
            {"Ticker": "GOOGL", "Current market cap": 2_100e9, "Implied volatility": 0.28, "Polymarket YES price": 0.046},
        ]
    )
    corr = default_correlation_matrix(company["Ticker"].tolist())
    return run_probability_engine(company, corr, days_to_target=90, simulations=simulations, seed=seed)


def _legs():
    return pd.DataFrame(
        [
            {"Instrument": "NVDA long put 160", "Ticker": "NVDA", "Option type": "Put", "Position": "Long", "Strike": 160.0, "Theoretical premium": 6.0, "Quantity": 1.0},
            {"Instrument": "NVDA short put 140", "Ticker": "NVDA", "Option type": "Put", "Position": "Short", "Strike": 140.0, "Theoretical premium": 3.0, "Quantity": 1.0},
            {"Instrument": "NVDA long call 200", "Ticker": "NVDA", "Option type": "Call", "Position": "Long", "Strike": 200.0, "Theoretical premium": 7.0, "Quantity": 1.0},
            {"Instrument": "NVDA short call 230", "Ticker": "NVDA", "Option type": "Call", "Position": "Short", "Strike": 230.0, "Theoretical premium": 3.5, "Quantity": 1.0},
        ]
    )


def _portfolio(side="NO", entry=0.17, qty=100.0, ticker="NVDA"):
    current_caps = pd.Series({"NVDA": 4_300e9, "AAPL": 3_100e9, "GOOGL": 2_100e9})
    spots = pd.Series({"NVDA": 175.0, "AAPL": 230.0, "GOOGL": 180.0})
    return PortfolioSpec(
        option_legs=_legs(), current_market_caps=current_caps, spot_prices=spots,
        selected_ticker=ticker, polymarket_side=side, polymarket_entry_price=entry,
        polymarket_quantity=qty, contract_multiplier=100.0, include_option_premiums=True,
    )


def test_capital_metrics_exact():
    capital = capital_metrics(_portfolio())
    assert capital["Polymarket cost"] == pytest.approx(100.0 * 0.17)
    assert capital["Long option premium"] == pytest.approx((6.0 + 7.0) * 100.0)
    assert capital["Short option premium (credit)"] == pytest.approx((3.0 + 3.5) * 100.0)
    assert capital["Net option debit"] == pytest.approx(650.0)
    assert capital["Net cash outlay"] == pytest.approx(667.0)
    assert capital["Gross premium paid"] == pytest.approx(1317.0)


def test_risk_metrics_definitions():
    result = _result()
    metrics = risk_metrics(result, _portfolio())
    max_loss = float(metrics["Max loss (capital at risk)"])
    expected = float(metrics["Expected profit"])
    assert max_loss > 0
    assert metrics["Return on capital-at-risk"] == pytest.approx(expected / max_loss)
    assert metrics["Capital deployed (cash)"] == pytest.approx(667.0)
    assert 0.0 <= metrics["Probability of profit"] <= 1.0
    assert metrics["Probability of profit"] + metrics["Probability of loss"] <= 1.0 + 1e-9


def test_return_distribution_is_monotone():
    result = _result()
    dist = return_distribution(result, _portfolio())
    assert list(dist.columns) == ["Percentile", "Profit", "Return on capital-at-risk"]
    profits = dist["Profit"].to_numpy()
    assert np.all(np.diff(profits) >= -1e-6)


def test_kelly_favorable_is_positive_and_fractional_monotone():
    result = _result()
    kelly = kelly_sizing(result, _portfolio())
    full = float(kelly["Full Kelly fraction"])
    assert 0.0 < full <= 1.0
    assert kelly["0.5x Kelly fraction"] == pytest.approx(full * 0.5)
    assert kelly["0.25x Kelly fraction"] == pytest.approx(full * 0.25)
    assert full >= kelly["0.5x Kelly fraction"] >= kelly["0.25x Kelly fraction"]


def test_kelly_declines_bet_when_expectation_is_negative():
    # A YES bet on GOOGL (which almost never wins) at a rich price is a clear
    # negative-expectation trade, so Kelly must recommend zero.
    result = _result()
    current_caps = pd.Series({"NVDA": 4_300e9, "AAPL": 3_100e9, "GOOGL": 2_100e9})
    spots = pd.Series({"NVDA": 175.0, "AAPL": 230.0, "GOOGL": 180.0})
    losing = PortfolioSpec(
        option_legs=_legs().assign(Quantity=0.0), current_market_caps=current_caps, spot_prices=spots,
        selected_ticker="GOOGL", polymarket_side="YES", polymarket_entry_price=0.50,
        polymarket_quantity=100.0, contract_multiplier=100.0, include_option_premiums=True,
    )
    kelly = kelly_sizing(result, losing)
    assert float(kelly["Full Kelly fraction"]) == 0.0


def test_budget_scaling_is_linear():
    result = _result()
    metrics = risk_metrics(result, _portfolio())
    max_loss = float(metrics["Max loss (capital at risk)"])
    scaled = budget_scaling(metrics, 50_000.0, basis="capital-at-risk")
    factor = 50_000.0 / max_loss
    assert scaled["Scale factor"] == pytest.approx(factor)
    assert scaled["Scaled expected profit"] == pytest.approx(float(metrics["Expected profit"]) * factor)
    assert scaled["Scaled max loss"] == pytest.approx(50_000.0)
    with pytest.raises(ValueError):
        budget_scaling(metrics, 1000.0, basis="bogus")


def test_breakeven_bands_structure():
    result = _result()
    bands = breakeven_bands(result, _portfolio())
    for column in ["Breakeven selected ratio", "Breakeven stock price", "Direction"]:
        assert column in bands.columns
