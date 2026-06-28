"""Option construction engine for Phase 3.

Phase 3 does not optimize hedge ratios or combine positions. It translates
Phase 2 probability boundaries into natural vanilla option building blocks.
"""

from __future__ import annotations

from math import erf, exp, log, sqrt

import numpy as np
import pandas as pd


OPTION_COLUMNS = [
    "Instrument",
    "Ticker",
    "Option type",
    "Position",
    "Strike",
    "Boundary used",
    "Boundary market cap",
    "Boundary / current cap",
    "Spot",
    "Purpose",
]

VALUED_OPTION_COLUMNS = OPTION_COLUMNS + [
    "Model IV",
    "Risk-free rate",
    "Time to expiry",
    "Theoretical premium",
    "Premium direction",
]

CONSTRUCTION_MODES = [
    "selected_only",
    "single_competitor",
    "full_universe",
]


def normal_cdf(value: float) -> float:
    return 0.5 * (1.0 + erf(value / sqrt(2.0)))


def black_scholes_price(
    spot: float,
    strike: float,
    time_to_expiry: float,
    volatility: float,
    risk_free_rate: float,
    option_type: str,
) -> float:
    """Black-Scholes option value with no dividends.

    This is a simplified Phase 3 theoretical quote. It deliberately uses one IV
    per ticker rather than a full volatility surface.
    """

    if spot <= 0 or strike <= 0:
        raise ValueError("spot and strike must be positive.")
    if time_to_expiry <= 0:
        option = option_type.lower()
        if option == "call":
            return float(max(spot - strike, 0.0))
        if option == "put":
            return float(max(strike - spot, 0.0))
        raise ValueError("option_type must be Call or Put.")
    if volatility <= 0:
        raise ValueError("volatility must be positive.")

    sigma_sqrt_t = volatility * sqrt(time_to_expiry)
    d1 = (log(spot / strike) + (risk_free_rate + 0.5 * volatility**2) * time_to_expiry) / sigma_sqrt_t
    d2 = d1 - sigma_sqrt_t
    option = option_type.lower()
    if option == "call":
        return float(spot * normal_cdf(d1) - strike * exp(-risk_free_rate * time_to_expiry) * normal_cdf(d2))
    if option == "put":
        return float(strike * exp(-risk_free_rate * time_to_expiry) * normal_cdf(-d2) - spot * normal_cdf(-d1))
    raise ValueError("option_type must be Call or Put.")


def attach_theoretical_premiums(
    structure: pd.DataFrame,
    implied_volatilities: pd.Series | dict[str, float],
    *,
    time_to_expiry: float,
    risk_free_rate: float,
) -> pd.DataFrame:
    """Attach simplified Black-Scholes premiums to candidate option legs."""

    ivs = pd.Series(implied_volatilities, dtype=float)
    valued = structure.copy()
    premiums = []
    model_ivs = []
    premium_directions = []
    for _, leg in valued.iterrows():
        ticker = str(leg["Ticker"])
        iv = float(ivs.loc[ticker])
        premium = black_scholes_price(
            spot=float(leg["Spot"]),
            strike=float(leg["Strike"]),
            time_to_expiry=float(time_to_expiry),
            volatility=iv,
            risk_free_rate=float(risk_free_rate),
            option_type=str(leg["Option type"]),
        )
        model_ivs.append(iv)
        premiums.append(premium)
        premium_directions.append("Credit" if str(leg["Position"]).lower() == "short" else "Debit")

    valued["Model IV"] = model_ivs
    valued["Risk-free rate"] = float(risk_free_rate)
    valued["Time to expiry"] = float(time_to_expiry)
    valued["Theoretical premium"] = premiums
    valued["Premium direction"] = premium_directions
    return valued[VALUED_OPTION_COLUMNS]


def boundary_cap_to_strike(boundary_market_cap: float, current_market_cap: float, spot_price: float) -> float:
    """Convert a market-cap boundary into an equivalent stock-price strike."""

    if not np.isfinite(boundary_market_cap):
        return np.nan
    if current_market_cap <= 0:
        raise ValueError("current_market_cap must be positive.")
    if spot_price <= 0:
        raise ValueError("spot_price must be positive.")
    return float(spot_price * boundary_market_cap / current_market_cap)


def _boundary_row(boundaries: pd.DataFrame, ticker: str, confidence_level: float) -> pd.Series:
    rows = boundaries[(boundaries["Ticker"] == ticker) & np.isclose(boundaries["Confidence level"], confidence_level)]
    if rows.empty:
        raise ValueError(f"No boundary row for {ticker} at confidence {confidence_level:.0%}.")
    return rows.iloc[0]


def construct_selected_ticker_legs(
    boundaries: pd.DataFrame,
    *,
    ticker: str,
    current_market_cap: float,
    spot_price: float,
    confidence_level: float,
) -> list[dict]:
    """Construct natural option legs for the selected Polymarket candidate."""

    row = _boundary_row(boundaries, ticker, confidence_level)
    upper_cap = float(row["Upper win boundary"])
    lower_cap = float(row["Lower loss boundary"])
    upper_strike = boundary_cap_to_strike(upper_cap, current_market_cap, spot_price)
    lower_strike = boundary_cap_to_strike(lower_cap, current_market_cap, spot_price)

    return [
        {
            "Instrument": f"Short {ticker} Call",
            "Ticker": ticker,
            "Option type": "Call",
            "Position": "Short",
            "Strike": upper_strike,
            "Boundary used": f"{confidence_level:.0%} win boundary",
            "Boundary market cap": upper_cap,
            "Boundary / current cap": upper_cap / current_market_cap,
            "Spot": spot_price,
            "Purpose": "Sell upside beyond the selected ticker's high-confidence win zone.",
        },
        {
            "Instrument": f"Long {ticker} Put",
            "Ticker": ticker,
            "Option type": "Put",
            "Position": "Long",
            "Strike": lower_strike,
            "Boundary used": f"{confidence_level:.0%} loss boundary",
            "Boundary market cap": lower_cap,
            "Boundary / current cap": lower_cap / current_market_cap,
            "Spot": spot_price,
            "Purpose": "Protect downside near the selected ticker's high-confidence loss zone.",
        },
    ]


def construct_competitor_legs(
    boundaries: pd.DataFrame,
    *,
    competitor_ticker: str,
    current_market_cap: float,
    spot_price: float,
    confidence_level: float,
    include_short_put: bool = True,
) -> list[dict]:
    """Construct natural option legs for a competing ticker."""

    row = _boundary_row(boundaries, competitor_ticker, confidence_level)
    upper_cap = float(row["Upper win boundary"])
    lower_cap = float(row["Lower loss boundary"])
    upper_strike = boundary_cap_to_strike(upper_cap, current_market_cap, spot_price)
    lower_strike = boundary_cap_to_strike(lower_cap, current_market_cap, spot_price)

    legs = [
        {
            "Instrument": f"Long {competitor_ticker} Call",
            "Ticker": competitor_ticker,
            "Option type": "Call",
            "Position": "Long",
            "Strike": upper_strike,
            "Boundary used": f"{confidence_level:.0%} competitor win boundary",
            "Boundary market cap": upper_cap,
            "Boundary / current cap": upper_cap / current_market_cap,
            "Spot": spot_price,
            "Purpose": "Protect against this competitor winning the ranking event.",
        }
    ]
    if include_short_put:
        legs.append(
            {
                "Instrument": f"Short {competitor_ticker} Put",
                "Ticker": competitor_ticker,
                "Option type": "Put",
                "Position": "Short",
                "Strike": lower_strike,
                "Boundary used": f"{confidence_level:.0%} competitor loss boundary",
                "Boundary market cap": lower_cap,
                "Boundary / current cap": lower_cap / current_market_cap,
                "Spot": spot_price,
                "Purpose": "Optional income leg in competitor downside zones where this competitor is less threatening.",
            }
        )
    return legs


def strongest_competitor(results: pd.DataFrame, selected_ticker: str) -> str:
    """Pick the competitor with the highest unconditional P(#1)."""

    competitors = results[results["Ticker"] != selected_ticker].copy()
    if competitors.empty:
        raise ValueError("At least one competitor is required.")
    return str(competitors.sort_values("Model probability", ascending=False).iloc[0]["Ticker"])


def ordered_competitors(results: pd.DataFrame, selected_ticker: str) -> list[str]:
    """Return competitors sorted by unconditional win probability."""

    competitors = results[results["Ticker"] != selected_ticker].copy()
    if competitors.empty:
        return []
    return competitors.sort_values("Model probability", ascending=False)["Ticker"].astype(str).tolist()


def construct_candidate_option_structure(
    boundaries: pd.DataFrame,
    results: pd.DataFrame,
    current_market_caps: pd.Series | dict[str, float],
    spot_prices: pd.Series | dict[str, float],
    *,
    selected_ticker: str,
    competitor_ticker: str | None,
    confidence_level: float,
    construction_mode: str = "single_competitor",
    include_competitor_short_puts: bool = True,
) -> pd.DataFrame:
    """Create the Phase 3 candidate option building blocks.

    Modes:
    - selected_only: hedge only the ticker underlying the Polymarket YES bet.
    - single_competitor: add diagnostic legs for one chosen competitor.
    - full_universe: add competitor protection legs for every other ticker.
    """

    if construction_mode not in CONSTRUCTION_MODES:
        raise ValueError(f"construction_mode must be one of {CONSTRUCTION_MODES}.")

    current_caps = pd.Series(current_market_caps, dtype=float)
    spots = pd.Series(spot_prices, dtype=float)

    legs = []
    legs.extend(
        construct_selected_ticker_legs(
            boundaries,
            ticker=selected_ticker,
            current_market_cap=float(current_caps.loc[selected_ticker]),
            spot_price=float(spots.loc[selected_ticker]),
            confidence_level=confidence_level,
        )
    )

    if construction_mode == "selected_only":
        return pd.DataFrame(legs)[OPTION_COLUMNS]

    if construction_mode == "single_competitor":
        competitors = [competitor_ticker or strongest_competitor(results, selected_ticker)]
    else:
        competitors = ordered_competitors(results, selected_ticker)

    for competitor in competitors:
        legs.extend(
            construct_competitor_legs(
                boundaries,
                competitor_ticker=competitor,
                current_market_cap=float(current_caps.loc[competitor]),
                spot_price=float(spots.loc[competitor]),
                confidence_level=confidence_level,
                include_short_put=include_competitor_short_puts,
            )
        )
    return pd.DataFrame(legs)[OPTION_COLUMNS]


def option_payoff(option_type: str, position: str, strike: float, terminal_prices: np.ndarray, premium: float = 0.0) -> np.ndarray:
    """Calculate standalone option payoff at expiry for one contract/share.

    Premium is included as a cash cost for long options and a cash credit for
    short options.
    """

    option = option_type.lower()
    side = position.lower()
    if option == "call":
        intrinsic = np.maximum(terminal_prices - strike, 0.0)
    elif option == "put":
        intrinsic = np.maximum(strike - terminal_prices, 0.0)
    else:
        raise ValueError("option_type must be Call or Put.")

    if side == "long":
        return intrinsic - premium
    if side == "short":
        return -intrinsic + premium
    raise ValueError("position must be Long or Short.")


def payoff_grid_for_leg(leg: pd.Series, *, price_min: float | None = None, price_max: float | None = None, points: int = 200, premium: float | None = None) -> pd.DataFrame:
    """Generate standalone payoff curve for a single option leg."""

    spot = float(leg["Spot"])
    strike = float(leg["Strike"])
    leg_premium = float(leg.get("Theoretical premium", 0.0)) if premium is None else float(premium)
    low = price_min if price_min is not None else max(0.01, min(spot, strike) * 0.5)
    high = price_max if price_max is not None else max(spot, strike) * 1.8
    terminal_prices = np.linspace(low, high, points)
    payoffs = option_payoff(str(leg["Option type"]), str(leg["Position"]), strike, terminal_prices, premium=leg_premium)
    return pd.DataFrame(
        {
            "Instrument": leg["Instrument"],
            "Terminal price": terminal_prices,
            "Payoff": payoffs,
            "Strike": strike,
            "Spot": spot,
            "Premium": leg_premium,
        }
    )
