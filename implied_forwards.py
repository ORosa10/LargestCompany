"""Estimate target-date equity forwards from option put-call parity."""

from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd
import yfinance as yf

from correlations import yahoo_symbol


def _quote_mid(frame: pd.DataFrame) -> pd.Series:
    bid = pd.to_numeric(frame.get("bid"), errors="coerce")
    ask = pd.to_numeric(frame.get("ask"), errors="coerce")
    valid = (bid > 0) & (ask > 0) & (ask >= bid)
    return ((bid + ask) / 2.0).where(valid)


def _forward_for_expiry(
    yft: yf.Ticker,
    expiry: str,
    *,
    spot: float,
    risk_free_rate: float,
    as_of: date,
    strikes_each_side: int,
) -> dict | None:
    expiry_date = date.fromisoformat(expiry)
    years = (expiry_date - as_of).days / 365.0
    if years <= 0:
        return None
    chain = yft.option_chain(expiry)
    calls = chain.calls.copy()
    puts = chain.puts.copy()
    if calls.empty or puts.empty:
        return None
    calls["call_mid"] = _quote_mid(calls)
    puts["put_mid"] = _quote_mid(puts)
    pairs = calls[["strike", "call_mid"]].merge(
        puts[["strike", "put_mid"]], on="strike", how="inner"
    ).dropna()
    pairs["strike"] = pd.to_numeric(pairs["strike"], errors="coerce")
    pairs = pairs[(pairs["strike"] > 0) & (pairs["call_mid"] > 0) & (pairs["put_mid"] > 0)]
    pairs = pairs[(pairs["strike"] / spot).between(0.80, 1.20)]
    if pairs.empty:
        return None
    pairs["distance"] = (pairs["strike"] - spot).abs()
    pairs = pairs.nsmallest(max(int(strikes_each_side), 1), "distance")
    pairs["implied_forward"] = pairs["strike"] + np.exp(risk_free_rate * years) * (
        pairs["call_mid"] - pairs["put_mid"]
    )
    pairs = pairs[np.isfinite(pairs["implied_forward"]) & (pairs["implied_forward"] > 0)]
    if pairs.empty:
        return None
    forward = float(pairs["implied_forward"].median())
    dispersion = float((pairs["implied_forward"] / forward - 1.0).abs().median())
    return {
        "expiry": expiry_date,
        "years": years,
        "forward": forward,
        "forward_to_spot": forward / spot,
        "annualized_carry": np.log(forward / spot) / years,
        "pair_count": len(pairs),
        "forward_dispersion": dispersion,
    }


def estimate_implied_forward_for_ticker(
    ticker: str,
    target_date: date,
    *,
    risk_free_rate: float = 0.04,
    strikes_each_side: int = 7,
) -> dict:
    """Estimate target forward by interpolating option-implied annualized carry."""
    as_of = date.today()
    target_years = (target_date - as_of).days / 365.0
    if target_years <= 0:
        raise ValueError("Target date must be after today.")
    symbol = yahoo_symbol(ticker)
    yft = yf.Ticker(symbol)
    history = yft.history(period="5d", auto_adjust=False)
    if history.empty:
        raise ValueError(f"No spot price available for {ticker}.")
    spot = float(history["Close"].dropna().iloc[-1])
    estimates = []
    for expiry in list(yft.options or []):
        estimate = _forward_for_expiry(
            yft,
            expiry,
            spot=spot,
            risk_free_rate=float(risk_free_rate),
            as_of=as_of,
            strikes_each_side=strikes_each_side,
        )
        if estimate is not None:
            estimates.append(estimate)
    if not estimates:
        raise ValueError(f"No valid call-put quote pairs available for {ticker}.")
    estimates.sort(key=lambda row: row["years"])
    times = np.array([row["years"] for row in estimates], dtype=float)
    carries = np.array([row["annualized_carry"] for row in estimates], dtype=float)
    target_carry = float(np.interp(target_years, times, carries, left=carries[0], right=carries[-1]))
    target_forward = spot * np.exp(target_carry * target_years)
    lower = max((row for row in estimates if row["years"] <= target_years), key=lambda row: row["years"], default=estimates[0])
    upper = min((row for row in estimates if row["years"] >= target_years), key=lambda row: row["years"], default=estimates[-1])
    return {
        "ticker": ticker,
        "yahoo_ticker": symbol,
        "as_of": as_of.isoformat(),
        "target_date": target_date.isoformat(),
        "spot": spot,
        "implied_forward": target_forward,
        "forward_to_spot": target_forward / spot,
        "annualized_implied_carry": target_carry,
        "lower_expiry": lower["expiry"].isoformat(),
        "upper_expiry": upper["expiry"].isoformat(),
        "quote_pairs": int(lower["pair_count"] + (0 if upper is lower else upper["pair_count"])),
        "forward_dispersion": max(lower["forward_dispersion"], upper["forward_dispersion"]),
        "source": "Yahoo option bid/ask mids via put-call parity",
    }


def estimate_implied_forwards(
    tickers: list[str],
    target_date: date,
    *,
    risk_free_rate: float = 0.04,
) -> pd.DataFrame:
    return pd.DataFrame([
        estimate_implied_forward_for_ticker(
            ticker, target_date, risk_free_rate=risk_free_rate
        )
        for ticker in tickers
    ])


def apply_implied_forwards(
    company_inputs: pd.DataFrame,
    forward_estimates: pd.DataFrame,
) -> pd.DataFrame:
    updated = company_inputs.copy()
    ratios = forward_estimates.set_index("ticker")["forward_to_spot"]
    updated["Forward / spot"] = updated["Ticker"].map(ratios)
    if updated["Forward / spot"].isna().any() or (updated["Forward / spot"] <= 0).any():
        raise ValueError("Missing or invalid implied forward ratio.")
    return updated
