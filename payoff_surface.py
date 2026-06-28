"""Payoff surface engine for Phase 4.

Phase 4 combines Polymarket event payoff with candidate option legs from Phase 3
and evaluates the combined payoff across Monte Carlo scenarios. It does not
optimize quantities or strike choices.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from option_construction import option_payoff


def polymarket_payoff(
    winner_tickers: pd.Series,
    *,
    selected_ticker: str,
    side: str,
    entry_price: float,
    quantity: float,
) -> pd.Series:
    """Calculate Polymarket YES/NO payoff per scenario.

    A YES share pays 1 if the selected ticker wins and 0 otherwise. A NO share
    pays 1 if the selected ticker does not win and 0 otherwise. Payoff is net of
    entry cost.
    """

    if not 0.0 <= entry_price <= 1.0:
        raise ValueError("entry_price must be between 0 and 1.")
    side_clean = side.upper()
    selected_wins = winner_tickers.astype(str) == selected_ticker
    if side_clean == "YES":
        payoff = np.where(selected_wins, 1.0 - entry_price, -entry_price)
    elif side_clean == "NO":
        payoff = np.where(selected_wins, -entry_price, 1.0 - entry_price)
    else:
        raise ValueError("side must be YES or NO.")
    return pd.Series(payoff * float(quantity), index=winner_tickers.index, name="Polymarket payoff")


def winner_from_ranks(ranks: pd.DataFrame) -> pd.Series:
    """Return the winning ticker for each scenario from a rank matrix."""

    if ranks.empty:
        raise ValueError("ranks cannot be empty.")
    return ranks.idxmin(axis=1).rename("Winner")


def terminal_stock_prices(
    terminal_market_caps: pd.DataFrame,
    current_market_caps: pd.Series | dict[str, float],
    spot_prices: pd.Series | dict[str, float],
) -> pd.DataFrame:
    """Convert terminal market caps into implied terminal stock prices."""

    current_caps = pd.Series(current_market_caps, dtype=float)
    spots = pd.Series(spot_prices, dtype=float)
    prices = pd.DataFrame(index=terminal_market_caps.index)
    for ticker in terminal_market_caps.columns:
        if ticker not in current_caps.index or ticker not in spots.index:
            raise ValueError(f"Missing current cap or spot price for {ticker}.")
        prices[ticker] = terminal_market_caps[ticker].astype(float) / float(current_caps.loc[ticker]) * float(spots.loc[ticker])
    return prices


def option_leg_scenario_payoff(
    leg: pd.Series,
    terminal_prices: pd.Series,
    *,
    quantity: float,
    contract_multiplier: float,
    include_premium: bool,
) -> pd.Series:
    """Calculate scenario payoff for one option leg."""

    strike = float(leg["Strike"])
    if not np.isfinite(strike):
        raise ValueError(f"Option leg {leg['Instrument']} has no valid strike. Lower the boundary confidence level or choose another construction mode.")
    premium = float(leg.get("Theoretical premium", 0.0)) if include_premium else 0.0
    if not np.isfinite(premium):
        raise ValueError(f"Option leg {leg['Instrument']} has no valid theoretical premium.")
    payoff = option_payoff(
        str(leg["Option type"]),
        str(leg["Position"]),
        strike,
        terminal_prices.to_numpy(dtype=float),
        premium=premium,
    )
    return pd.Series(payoff * float(quantity) * float(contract_multiplier), index=terminal_prices.index, name=str(leg["Instrument"]))


def calculate_scenario_payoffs(
    terminal_market_caps: pd.DataFrame,
    ranks: pd.DataFrame,
    current_market_caps: pd.Series | dict[str, float],
    spot_prices: pd.Series | dict[str, float],
    option_legs: pd.DataFrame,
    *,
    selected_ticker: str,
    polymarket_side: str,
    polymarket_entry_price: float,
    polymarket_quantity: float,
    contract_multiplier: float = 100.0,
    include_option_premiums: bool = True,
) -> pd.DataFrame:
    """Combine Polymarket and option payoffs for every Monte Carlo scenario."""

    winners = winner_from_ranks(ranks)
    terminal_prices = terminal_stock_prices(terminal_market_caps, current_market_caps, spot_prices)
    scenario = pd.DataFrame(index=terminal_market_caps.index)
    scenario["Winner"] = winners
    scenario["Selected terminal market cap"] = terminal_market_caps[selected_ticker].astype(float)
    scenario["Polymarket payoff"] = polymarket_payoff(
        winners,
        selected_ticker=selected_ticker,
        side=polymarket_side,
        entry_price=float(polymarket_entry_price),
        quantity=float(polymarket_quantity),
    )

    option_total = pd.Series(0.0, index=terminal_market_caps.index, name="Option payoff")
    legs = option_legs.copy()
    if "Quantity" not in legs.columns:
        legs["Quantity"] = 0.0
    for _, leg in legs.iterrows():
        quantity = float(leg.get("Quantity", 0.0))
        if quantity == 0.0:
            continue
        ticker = str(leg["Ticker"])
        leg_payoff = option_leg_scenario_payoff(
            leg,
            terminal_prices[ticker],
            quantity=quantity,
            contract_multiplier=contract_multiplier,
            include_premium=include_option_premiums,
        )
        option_total = option_total.add(leg_payoff, fill_value=0.0)

    scenario["Option payoff"] = option_total
    scenario["Total payoff"] = scenario["Polymarket payoff"] + scenario["Option payoff"]
    return scenario


def payoff_summary(scenario_payoffs: pd.DataFrame, *, shortfall_probability: float = 0.05) -> pd.Series:
    """Summary metrics for a scenario payoff distribution."""

    payoff = scenario_payoffs["Total payoff"].astype(float)
    threshold = payoff.quantile(shortfall_probability)
    tail = payoff[payoff <= threshold]
    return pd.Series(
        {
            "Expected payoff": payoff.mean(),
            "Median payoff": payoff.median(),
            "P5 payoff": payoff.quantile(0.05),
            "P1 payoff": payoff.quantile(0.01),
            "Worst payoff": payoff.min(),
            "Probability of loss": (payoff < 0).mean(),
            "Expected shortfall 5%": tail.mean() if not tail.empty else np.nan,
        }
    )


def payoff_surface_bins(
    scenario_payoffs: pd.DataFrame,
    terminal_market_caps: pd.DataFrame,
    current_market_caps: pd.Series | dict[str, float],
    *,
    selected_ticker: str,
    competitor_ticker: str,
    x_bins: int = 12,
    y_bins: int = 12,
) -> pd.DataFrame:
    """Aggregate payoff into a 2D surface by selected and competitor cap ratios."""

    current_caps = pd.Series(current_market_caps, dtype=float)
    if selected_ticker not in terminal_market_caps.columns or competitor_ticker not in terminal_market_caps.columns:
        raise ValueError("selected_ticker and competitor_ticker must be in terminal_market_caps.")
    data = pd.DataFrame(
        {
            "Selected ratio": terminal_market_caps[selected_ticker].astype(float) / float(current_caps.loc[selected_ticker]),
            "Competitor ratio": terminal_market_caps[competitor_ticker].astype(float) / float(current_caps.loc[competitor_ticker]),
            "Total payoff": scenario_payoffs["Total payoff"].astype(float),
        }
    ).dropna()
    data["Selected bin"] = pd.qcut(data["Selected ratio"], q=min(x_bins, data["Selected ratio"].nunique()), duplicates="drop")
    data["Competitor bin"] = pd.qcut(data["Competitor ratio"], q=min(y_bins, data["Competitor ratio"].nunique()), duplicates="drop")
    grouped = data.groupby(["Selected bin", "Competitor bin"], observed=True)
    surface = grouped.agg(
        selected_ratio=("Selected ratio", "mean"),
        competitor_ratio=("Competitor ratio", "mean"),
        expected_payoff=("Total payoff", "mean"),
        scenario_count=("Total payoff", "size"),
    ).reset_index(drop=True)
    surface["scenario_probability"] = surface["scenario_count"] / len(data)
    surface["weighted_payoff_contribution"] = surface["expected_payoff"] * surface["scenario_probability"]
    return surface
