import pandas as pd

from event_universe import apply_event_prices


def test_deleted_event_ticker_remains_in_simulation_universe_with_zero_price():
    company_inputs = pd.DataFrame(
        {
            "Ticker": ["NVDA", "AAPL", "GOOGL", "MSFT"],
            "Current market cap": [4.0, 3.0, 3.5, 2.8],
            "Implied volatility": [0.4, 0.3, 0.35, 0.25],
            "Polymarket YES price": [0.8, 0.1, 0.1, 0.0],
        }
    )
    event_prices = pd.DataFrame(
        {"Ticker": ["NVDA", "GOOGL"], "Polymarket YES price": [0.831, 0.046]}
    )

    updated, visible, unknown = apply_event_prices(company_inputs, event_prices)

    assert updated["Ticker"].tolist() == ["NVDA", "AAPL", "GOOGL", "MSFT"]
    prices = updated.set_index("Ticker")["Polymarket YES price"]
    assert prices["NVDA"] == 0.831
    assert prices["GOOGL"] == 0.046
    assert prices["AAPL"] == 0.0
    assert prices["MSFT"] == 0.0
    assert visible["Ticker"].tolist() == ["NVDA", "GOOGL"]
    assert unknown == []


def test_unknown_event_ticker_is_reported_not_added():
    company_inputs = pd.DataFrame(
        {
            "Ticker": ["NVDA"],
            "Polymarket YES price": [0.0],
        }
    )
    event_prices = pd.DataFrame(
        {"Ticker": ["NVDA", "UNKNOWN"], "Polymarket YES price": [0.83, 0.17]}
    )

    updated, visible, unknown = apply_event_prices(company_inputs, event_prices)

    assert updated["Ticker"].tolist() == ["NVDA"]
    assert visible["Ticker"].tolist() == ["NVDA"]
    assert unknown == ["UNKNOWN"]
