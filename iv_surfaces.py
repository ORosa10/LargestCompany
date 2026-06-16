"""MVP implied-volatility extraction from Yahoo Finance option chains.

This module intentionally uses a simple near-ATM IV estimate. It does not yet calibrate
an entire volatility surface or risk-neutral density.
"""

from __future__ import annotations

from dataclasses import dataclass
+from datetime import date

import numpy as np
import pandas as pd
import yfinance as yf

from correlations import yahoo_symbol


@dataclass(frozen=True)
class TickerIVEstimate:
    ticker: str
    yahoo_ticker: str
    expiry: str
    target_date: str
    spot: float
    atm_strike: float
    implied_volatility: float
    call_iv: float | None
    put_iv: float | None
    source: str


def _parse_expiry(expiry: str) -> date:
    return date.fromisoformat(expiry)


def nearest_expiry(available_expiries: list[str], target_date: date) -> str:
    if not available_expiries:
        raise ValueError("No option expiries available.")
    return min(available_expiries, key=lambda expiry: abs((_parse_expiry(expiry) - target_date).days))


def _clean_option_chain_side(options: pd.DataFrame) -> pd.DataFrame:
    required = ["strike", "impliedVolatility"]
    missing = [column for column in required if column not in options.columns]
    if missing:
        raise ValueError(f"Option chain missing columns: {missing}.")

    clean = options.copy()
    clean["strike"] = pd.to_numeric(clean["strike"], errors="coerce")
    clean["impliedVolatility"] = pd.to_numeric(clean["impliedVolatility"], errors="coerce")
    clean = clean.dropna(subset=["strike", "impliedVolatility"])
    clean = clean[(clean["strike"] > 0) & (clean["impliedVolatility"] > 0)]
    return clean


def estimate_atm_iv_for_ticker(ticker: str, target_date: date) -> TickerIVEstimate:
    """Estimate near-ATM IV from Yahoo option chain closest to target date."""

    yahoo_ticker = yahoo_symbol(ticker)
    yft = yf.Ticker(yahoo_ticker)
    expiries = list(yft.options or [])
    expiry = nearest_expiry(expiries, target_date)

    history = yft.history(period="5d", auto_adjust=False)
    if history.empty or "Close" not in history.columns:
        raise ValueError(f"No recent spot price available for {ticker}.")
    spot = float(history["Close"].dropna().iloc[-1])

    chain = yft.option_chain(expiry)
    calls = _clean_option_chain_side(chain.calls)
    puts = _clean_option_chain_side(chain.puts)
    if calls.empty and puts.empty:
        raise ValueError(f"No usable options for {ticker} expiry {expiry}.")

    candidate_strikes = pd.concat([calls[["strike"]], puts[["strike"]]], ignore_index=True)
    atm_strike = float(candidate_strikes.iloc[(candidate_strikes["strike"] - spot).abs().argsort().iloc[0]]["strike"])

    call_iv = None
    if not calls.empty:
        call_row = calls.iloc[(calls["strike"] - atm_strike).abs().argsort().iloc[0]]
        call_iv = float(call_row["impliedVolatility"])

    put_iv = None
    if not puts.empty:
        put_row = puts.iloc[(puts["strike"] - atm_strike).abs().argsort().iloc[0]]
        put_iv = float(put_row["impliedVolatility"])

    iv_values = [value for value in [call_iv, put_iv] if value is not None and np.isfinite(value)]
    if not iv_values:
        raise ValueError(f"No valid ATM implied volatility for {ticker} expiry {expiry}.")

    return TickerIVEstimate(
        ticker=ticker,
        yahoo_ticker=yahoo_ticker,
        expiry=expiry,
        target_date=target_date.isoformat(),
        spot=spot,
        atm_strike=atm_strike,
        implied_volatility=float(np.mean(iv_values)),
        call_iv=call_iv,
        put_iv=put_iv,
        source="Yahoo Finance option chain near-ATM IV",
    )


def estimate_atm_ivs(tickers: list[str], target_date: date) -> pd.DataFrame:
    """Estimate near-ATM IV for all tickers and return a tidy table."""

    estimates = [estimate_atm_iv_for_ticker(ticker, target_date) for ticker in tickers]
    return pd.DataFrame([estimate.__dict__ for estimate in estimates])


def apply_iv_estimates(company_inputs: pd.DataFrame, iv_estimates: pd.DataFrame) -> pd.DataFrame:
    """Replace company input IVs with estimated IVs by ticker."""

    updated = company_inputs.copy()
    iv_by_ticker = iv_estimates.set_index("ticker")["implied_volatility"]
    updated["Implied volatility"] = updated["Ticker"].map(iv_by_ticker).fillna(updated["Implied volatility"])
    return updated
