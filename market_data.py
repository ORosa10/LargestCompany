"""Market data helpers with an explicit refresh policy and last-valid cache."""

from __future__ import annotations

import pandas as pd
import yfinance as yf

from correlations import yahoo_symbol
from simulation_store import load_phase_artifact, save_phase_artifact


def _data_mode() -> str:
    policy = load_phase_artifact("data_policy") or {}
    return str(policy.get("mode", "refresh"))


def _cached_table(key: str, tickers: list[str]) -> pd.DataFrame | None:
    payload = load_phase_artifact("market_data_snapshot") or {}
    table = payload.get(key)
    if not isinstance(table, pd.DataFrame) or table.empty:
        return None
    indexed = table.drop_duplicates("ticker", keep="last").set_index("ticker")
    if any(ticker not in indexed.index for ticker in tickers):
        return None
    return indexed.loc[tickers].reset_index()


def _save_table(key: str, table: pd.DataFrame) -> None:
    payload = load_phase_artifact("market_data_snapshot") or {}
    previous = payload.get(key)
    if isinstance(previous, pd.DataFrame) and not previous.empty:
        table = pd.concat([previous, table], ignore_index=True).drop_duplicates("ticker", keep="last")
    payload[key] = table
    save_phase_artifact("market_data_snapshot", payload)


def _fetch_market_cap(ticker: str) -> dict:
    symbol = yahoo_symbol(ticker)
    yft = yf.Ticker(symbol)
    market_cap = None
    source = None
    try:
        fast_info = yft.fast_info
        market_cap = fast_info.get("market_cap") if hasattr(fast_info, "get") else None
        if market_cap:
            source = "Yahoo fast_info.market_cap"
    except Exception:
        pass
    if not market_cap:
        try:
            market_cap = yft.info.get("marketCap")
            if market_cap:
                source = "Yahoo info.marketCap"
        except Exception:
            pass
    if not market_cap or market_cap <= 0:
        raise ValueError(f"Could not fetch a valid market cap for {ticker} ({symbol}).")
    return {"ticker": ticker, "yahoo_ticker": symbol, "market_cap": float(market_cap), "source": source or "Yahoo Finance"}


def fetch_market_caps(tickers: list[str]) -> pd.DataFrame:
    """Return live or last-saved market caps according to the Phase 0 policy."""
    if _data_mode() == "saved":
        cached = _cached_table("market_caps", tickers)
        if cached is not None:
            cached = cached.copy()
            cached["source"] = cached["source"].astype(str) + " (last saved)"
            return cached

    rows = []
    cached_all = _cached_table("market_caps", tickers)
    cached_by_ticker = cached_all.set_index("ticker") if cached_all is not None else pd.DataFrame()
    for ticker in tickers:
        try:
            rows.append(_fetch_market_cap(ticker))
        except Exception:
            if not cached_by_ticker.empty and ticker in cached_by_ticker.index:
                row = cached_by_ticker.loc[ticker].to_dict()
                row["ticker"] = ticker
                row["source"] = str(row.get("source", "cached")) + " (fallback)"
                rows.append(row)
            else:
                raise
    result = pd.DataFrame(rows)
    _save_table("market_caps", result)
    return result


def _fetch_spot(ticker: str) -> dict:
    symbol = yahoo_symbol(ticker)
    yft = yf.Ticker(symbol)
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
        pass
    if not spot:
        try:
            history = yft.history(period="5d", auto_adjust=False)
            if not history.empty:
                spot = float(history["Close"].dropna().iloc[-1])
                source = "Yahoo recent close"
        except Exception:
            pass
    if not spot or spot <= 0:
        raise ValueError(f"Could not fetch a valid spot price for {ticker} ({symbol}).")
    return {"ticker": ticker, "yahoo_ticker": symbol, "spot_price": float(spot), "source": source or "Yahoo Finance"}


def fetch_spot_prices(tickers: list[str]) -> pd.DataFrame:
    """Return live or last-saved spots according to the Phase 0 policy."""
    if _data_mode() == "saved":
        cached = _cached_table("spot_prices", tickers)
        if cached is not None:
            cached = cached.copy()
            cached["source"] = cached["source"].astype(str) + " (last saved)"
            return cached

    rows = []
    cached_all = _cached_table("spot_prices", tickers)
    cached_by_ticker = cached_all.set_index("ticker") if cached_all is not None else pd.DataFrame()
    for ticker in tickers:
        try:
            rows.append(_fetch_spot(ticker))
        except Exception:
            if not cached_by_ticker.empty and ticker in cached_by_ticker.index:
                row = cached_by_ticker.loc[ticker].to_dict()
                row["ticker"] = ticker
                row["source"] = str(row.get("source", "cached")) + " (fallback)"
                rows.append(row)
            else:
                raise
    result = pd.DataFrame(rows)
    _save_table("spot_prices", result)
    return result


def apply_market_caps(company_inputs: pd.DataFrame, market_caps: pd.DataFrame) -> pd.DataFrame:
    updated = company_inputs.copy()
    cap_by_ticker = market_caps.set_index("ticker")["market_cap"]
    updated["Current market cap"] = updated["Ticker"].map(cap_by_ticker).fillna(updated["Current market cap"])
    return updated
