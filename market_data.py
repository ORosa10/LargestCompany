"""Market data helpers for current market capitalization and spot prices."""

from __future__ import annotations

import pandas as pd
import yfinance as yf

from correlations import yahoo_symbol


def fetch_market_caps(tickers: list[str]) -> pd.DataFrame:
    """Fetch current market capitalizations from Yahoo Finance.

    Uses yfinance fast_info when available and falls back to info['marketCap'].
    """

    rows = []
    for ticker in tickers:
        yahoo_ticker = yahoo_symbol(ticker)
        yft = yf.Ticker(yahoo_ticker)
        market_cap = None
        source = None

        try:
            fast_info = yft.fast_info
            market_cap = fast_info.get("market_cap") if hasattr(fast_info, "get") else None
            if market_cap:
                source = "Yahoo fast_info.market_cap"
        except Exception:
            market_cap = None

        if not market_cap:
            try:
                info = yft.info
                market_cap = info.get("marketCap")
                if market_cap:
                    source = "Yahoo info.marketCap"
            except Exception:
                market_cap = None

        if not market_cap or market_cap <= 0:
            raise ValueError(f"Could not fetch a valid market cap for {ticker} ({yahoo_ticker}).")

        rows.append(
            {
                "ticker": ticker,
                "yahoo_ticker": yahoo_ticker,
                "market_cap": float(market_cap),
                "source": source or "Yahoo Finance",
            }
        )

    return pd.DataFrame(rows)


def fetch_spot_prices(tickers: list[str]) -> pd.DataFrame:
    """Fetch current spot prices from Yahoo Finance."""

    rows = []
    for ticker in tickers:
        yahoo_ticker = yahoo_symbol(ticker)
        yft = yf.Ticker(yahoo_ticker)
        spot = None
        source = None

        try:
            fast_info = yft.fast_info
            for key in ["last_price", "regular_market_price", "previous_close"]:
                spot = fast_info.get(key) if hasattr(fast_info, "get") else None
                if spot:
                    source = f"Yahoo fast_info.{key}"
                    break
        except Exception:
            spot = None

        if not spot:
            try:
                history = yft.history(period="5d", auto_adjust=False)
                if not history.empty:
                    spot = float(history["Close"].dropna().iloc[-1])
                    source = "Yahoo recent close"
            except Exception:
                spot = None

        if not spot or spot <= 0:
            raise ValueError(f"Could not fetch a valid spot price for {ticker} ({yahoo_ticker}).")

        rows.append(
            {
                "ticker": ticker,
                "yahoo_ticker": yahoo_ticker,
                "spot_price": float(spot),
                "source": source or "Yahoo Finance",
            }
        )

    return pd.DataFrame(rows)


def apply_market_caps(company_inputs: pd.DataFrame, market_caps: pd.DataFrame) -> pd.DataFrame:
    """Replace company input market caps with fetched values by ticker."""

    updated = company_inputs.copy()
    cap_by_ticker = market_caps.set_index("ticker")["market_cap"]
    updated["Current market cap"] = updated["Ticker"].map(cap_by_ticker).fillna(updated["Current market cap"])
    return updated
