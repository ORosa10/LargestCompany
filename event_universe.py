from __future__ import annotations

import pandas as pd


JULY_2026_EVENT_PRICES = pd.DataFrame(
    {
        "Ticker": ["NVDA", "AAPL", "GOOGL"],
        "Polymarket YES price": [0.830, 0.123, 0.046],
    }
)


def apply_event_prices(
    company_inputs: pd.DataFrame,
    event_prices: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    """Map visible Polymarket outcomes onto the full simulation universe.

    Firms omitted from ``event_prices`` remain in the Monte Carlo universe but
    receive a zero Polymarket price. Unknown event tickers are returned for a UI
    warning and are not added without market-cap and model inputs.
    """
    required_company = {"Ticker", "Polymarket YES price"}
    required_event = {"Ticker", "Polymarket YES price"}
    if not required_company.issubset(company_inputs.columns):
        raise ValueError("Company inputs must contain Ticker and Polymarket YES price.")
    if not required_event.issubset(event_prices.columns):
        raise ValueError("Event prices must contain Ticker and Polymarket YES price.")

    universe = company_inputs.copy()
    universe["Ticker"] = universe["Ticker"].astype(str).str.strip().str.upper()
    valid_tickers = set(universe["Ticker"])

    visible = event_prices.copy()
    visible["Ticker"] = visible["Ticker"].astype(str).str.strip().str.upper()
    visible["Polymarket YES price"] = pd.to_numeric(
        visible["Polymarket YES price"], errors="coerce"
    )
    visible = visible.dropna(subset=["Ticker", "Polymarket YES price"])
    visible = visible[visible["Ticker"] != ""]
    visible["Polymarket YES price"] = visible["Polymarket YES price"].clip(0.0, 1.0)

    unknown = sorted(set(visible["Ticker"]) - valid_tickers)
    visible = visible[visible["Ticker"].isin(valid_tickers)]
    visible = visible.drop_duplicates("Ticker", keep="last").reset_index(drop=True)

    price_map = visible.set_index("Ticker")["Polymarket YES price"]
    universe["Polymarket YES price"] = universe["Ticker"].map(price_map).fillna(0.0)
    return universe, visible, unknown
