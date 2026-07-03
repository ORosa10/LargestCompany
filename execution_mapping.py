"""Phase 6 helpers for mapping normalized research legs to executable terms."""

from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd
import yfinance as yf

from correlations import yahoo_symbol
from option_construction import black_scholes_price


def default_strike_step(spot: float) -> float:
    """Provide an editable display grid until listed strikes arrive in Phase 7."""
    if spot < 25:
        return 0.5
    if spot < 100:
        return 1.0
    if spot < 250:
        return 2.5
    if spot < 500:
        return 5.0
    return 10.0


def round_to_strike_grid(strike: float, step: float) -> float:
    if strike <= 0 or step <= 0:
        raise ValueError("strike and strike step must be positive.")
    return float(max(step, np.round(strike / step) * step))


def fetch_option_expirations(tickers: list[str]) -> dict[str, list[date]]:
    """Fetch currently listed expiration dates without downloading option quotes."""
    output: dict[str, list[date]] = {}
    for ticker in tickers:
        values = yf.Ticker(yahoo_symbol(ticker)).options
        output[ticker] = sorted({pd.Timestamp(value).date() for value in values})
    return output


def choose_expiration(
    expirations: list[date],
    target_date: date,
    policy: str,
) -> date | None:
    if not expirations:
        return None
    ordered = sorted(expirations)
    if policy == "First expiry on/after target":
        candidates = [value for value in ordered if value >= target_date]
        return candidates[0] if candidates else ordered[-1]
    if policy == "Last expiry on/before target":
        candidates = [value for value in ordered if value <= target_date]
        return candidates[-1] if candidates else ordered[0]
    if policy == "Nearest listed expiry":
        return min(ordered, key=lambda value: abs((value - target_date).days))
    raise ValueError(f"Unknown expiration policy: {policy}")


def map_normalized_legs(
    legs: pd.DataFrame,
    spot_by_ticker: pd.Series,
    strike_step_by_ticker: pd.Series,
) -> pd.DataFrame:
    """Map Phase 5 strikes (spot=100) into current-dollar strike grids."""
    rows = []
    for index, leg in legs.reset_index(drop=True).iterrows():
        ticker = str(leg["Ticker"])
        spot = float(spot_by_ticker.loc[ticker])
        step = float(strike_step_by_ticker.loc[ticker])
        normalized_strike = float(leg["Strike"])
        raw_strike = spot * normalized_strike / 100.0
        executable_strike = round_to_strike_grid(raw_strike, step)
        mapped_normalized = executable_strike / spot * 100.0
        rows.append(
            {
                "Leg": index + 1,
                "Use": True,
                "Ticker": ticker,
                "Option type": str(leg["Option type"]),
                "Position": str(leg["Position"]),
                "Quantity": float(leg["Quantity"]),
                "Phase 5 normalized strike": normalized_strike,
                "Current spot": spot,
                "Raw real strike": raw_strike,
                "Strike step": step,
                "Executable strike": executable_strike,
                "Mapped normalized strike": mapped_normalized,
                "Strike mapping error": mapped_normalized - normalized_strike,
                "Model IV": float(leg.get("Model IV", np.nan)),
            }
        )
    return pd.DataFrame(rows)


def rebuild_normalized_legs(
    original_legs: pd.DataFrame,
    mapping: pd.DataFrame,
    *,
    time_to_expiry: float,
    risk_free_rate: float,
) -> pd.DataFrame:
    """Rebuild Phase 5-compatible legs after real-strike editing and rounding."""
    rows = []
    original = original_legs.reset_index(drop=True)
    for _, mapped in mapping[mapping["Use"]].iterrows():
        source = original.iloc[int(mapped["Leg"]) - 1]
        strike = float(mapped["Executable strike"]) / float(mapped["Current spot"]) * 100.0
        volatility = float(mapped["Model IV"])
        premium = black_scholes_price(
            spot=100.0,
            strike=strike,
            time_to_expiry=time_to_expiry,
            volatility=volatility,
            risk_free_rate=risk_free_rate,
            option_type=str(mapped["Option type"]),
        )
        rows.append(
            {
                "Instrument": f"{mapped['Position']} {mapped['Ticker']} {mapped['Option type']} {mapped['Executable strike']:.2f}",
                "Ticker": str(mapped["Ticker"]),
                "Option type": str(mapped["Option type"]),
                "Position": str(mapped["Position"]),
                "Quantity": float(mapped["Quantity"]),
                "Strike": strike,
                "Strike / spot": strike / 100.0,
                "Strike source": "Phase 6 real-strike mapping",
                "Boundary used": source.get("Boundary used", "Mapped from Phase 5"),
                "Spot": 100.0,
                "Model IV": volatility,
                "Risk-free rate": risk_free_rate,
                "Time to expiry": time_to_expiry,
                "Theoretical premium": premium,
            }
        )
    return pd.DataFrame(rows)
