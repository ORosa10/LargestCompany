"""Market-consistent option valuation helpers shared by downstream phases.

Distribution IV belongs to the Phase 1 probability model. Option pricing IV is
strike-specific and comes from the calibrated smile whenever the saved Phase 1
run uses the matching surface expiry. Forward carry is inherited from Phase 1.
"""

from __future__ import annotations

from math import erf, exp, log, sqrt

import numpy as np
import pandas as pd

from iv_surface_model import SURFACE_EXPIRY, default_surface_nodes
from option_construction import VALUED_OPTION_COLUMNS


EXTRA_VALUATION_COLUMNS = ["IV source", "Forward / spot", "Implied dividend yield"]


def normal_cdf(value: float) -> float:
    return 0.5 * (1.0 + erf(value / sqrt(2.0)))


def surface_iv_for_strike(
    ticker: str,
    strike: float,
    spot: float,
    *,
    surface_nodes: pd.DataFrame | None = None,
) -> float | None:
    """Interpolate smile IV at the leg's current strike/spot moneyness."""

    if strike <= 0 or spot <= 0:
        raise ValueError("strike and spot must be positive.")
    nodes = default_surface_nodes() if surface_nodes is None else surface_nodes.copy()
    ticker_nodes = nodes[nodes["Ticker"].astype(str) == str(ticker)].sort_values("Moneyness")
    if ticker_nodes.empty:
        return None
    moneyness = float(strike) / float(spot)
    x = np.log(ticker_nodes["Moneyness"].to_numpy(dtype=float))
    y = ticker_nodes["IV"].to_numpy(dtype=float)
    return float(np.clip(np.interp(log(moneyness), x, y, left=y[0], right=y[-1]), 0.01, 4.0))


def implied_dividend_yield(
    forward_ratio: float,
    time_to_expiry: float,
    risk_free_rate: float,
) -> float:
    """Return q implied by F/S = exp((r-q)T)."""

    if forward_ratio <= 0:
        raise ValueError("forward_ratio must be positive.")
    if time_to_expiry <= 0:
        return 0.0
    return float(risk_free_rate - log(forward_ratio) / time_to_expiry)


def black_scholes_price_with_carry(
    *,
    spot: float,
    strike: float,
    time_to_expiry: float,
    volatility: float,
    risk_free_rate: float,
    dividend_yield: float,
    option_type: str,
) -> float:
    """Black-Scholes value with continuous carry consistent with Phase 1."""

    if spot <= 0 or strike <= 0:
        raise ValueError("spot and strike must be positive.")
    option = option_type.lower()
    if time_to_expiry <= 0:
        if option == "call":
            return float(max(spot - strike, 0.0))
        if option == "put":
            return float(max(strike - spot, 0.0))
        raise ValueError("option_type must be Call or Put.")
    if volatility <= 0:
        raise ValueError("volatility must be positive.")

    root_t = sqrt(time_to_expiry)
    sigma_root_t = volatility * root_t
    d1 = (
        log(spot / strike)
        + (risk_free_rate - dividend_yield + 0.5 * volatility**2) * time_to_expiry
    ) / sigma_root_t
    d2 = d1 - sigma_root_t
    spot_discount = exp(-dividend_yield * time_to_expiry)
    strike_discount = exp(-risk_free_rate * time_to_expiry)
    if option == "call":
        return float(spot * spot_discount * normal_cdf(d1) - strike * strike_discount * normal_cdf(d2))
    if option == "put":
        return float(strike * strike_discount * normal_cdf(-d2) - spot * spot_discount * normal_cdf(-d1))
    raise ValueError("option_type must be Call or Put.")


def attach_market_consistent_premiums(
    structure: pd.DataFrame,
    fallback_ivs: pd.Series | dict[str, float],
    *,
    forward_ratios: pd.Series | dict[str, float] | None,
    time_to_expiry: float,
    risk_free_rate: float,
    use_surface: bool,
    surface_nodes: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Attach strike-specific IVs and Phase 1-consistent carry to option legs."""

    if structure.empty:
        return structure.copy()
    iv_fallback = pd.Series(fallback_ivs, dtype=float)
    forwards = pd.Series(forward_ratios, dtype=float) if forward_ratios is not None else pd.Series(dtype=float)
    valued = structure.copy()
    premiums: list[float] = []
    model_ivs: list[float] = []
    iv_sources: list[str] = []
    forward_values: list[float] = []
    dividend_yields: list[float] = []
    premium_directions: list[str] = []

    for _, leg in valued.iterrows():
        ticker = str(leg["Ticker"])
        surface_iv = surface_iv_for_strike(
            ticker,
            float(leg["Strike"]),
            float(leg["Spot"]),
            surface_nodes=surface_nodes,
        ) if use_surface else None
        if surface_iv is None:
            iv = float(iv_fallback.loc[ticker])
            iv_source = "Phase 1 ATM fallback"
        else:
            iv = float(surface_iv)
            iv_source = f"Calibrated smile ({SURFACE_EXPIRY})"

        default_forward = exp(float(risk_free_rate) * float(time_to_expiry))
        forward_ratio = float(forwards.get(ticker, default_forward))
        if not np.isfinite(forward_ratio) or forward_ratio <= 0:
            forward_ratio = default_forward
        dividend_yield = implied_dividend_yield(
            forward_ratio,
            float(time_to_expiry),
            float(risk_free_rate),
        )
        premium = black_scholes_price_with_carry(
            spot=float(leg["Spot"]),
            strike=float(leg["Strike"]),
            time_to_expiry=float(time_to_expiry),
            volatility=iv,
            risk_free_rate=float(risk_free_rate),
            dividend_yield=dividend_yield,
            option_type=str(leg["Option type"]),
        )
        model_ivs.append(iv)
        iv_sources.append(iv_source)
        forward_values.append(forward_ratio)
        dividend_yields.append(dividend_yield)
        premiums.append(premium)
        premium_directions.append("Credit" if str(leg["Position"]).lower() == "short" else "Debit")

    valued["Model IV"] = model_ivs
    valued["IV source"] = iv_sources
    valued["Forward / spot"] = forward_values
    valued["Implied dividend yield"] = dividend_yields
    valued["Risk-free rate"] = float(risk_free_rate)
    valued["Time to expiry"] = float(time_to_expiry)
    valued["Theoretical premium"] = premiums
    valued["Premium direction"] = premium_directions
    return valued[VALUED_OPTION_COLUMNS + EXTRA_VALUATION_COLUMNS]
