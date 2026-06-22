"""Conditional probability boundary calculations for ranking events.

Phase 2 asks where probabilities change, not how to hedge them. The functions in
this module sit on top of the Phase 1 probability engine and answer questions
such as: what selected-company market cap is needed for a target P(#1), or what
pairwise market-cap gap corresponds to a target probability of beating one
competitor.
"""

from __future__ import annotations

from dataclasses import dataclass
from statistics import NormalDist

import numpy as np
import pandas as pd

from model import run_probability_engine, validate_company_inputs


_STANDARD_NORMAL = NormalDist()


@dataclass(frozen=True)
class BoundaryResult:
    ticker: str
    target_probability: float
    boundary_market_cap: float
    current_market_cap: float
    absolute_gap: float
    relative_gap: float
    achieved_probability: float
    iterations: int


def selected_probability(results: pd.DataFrame, ticker: str) -> float:
    row = results.loc[results["Ticker"] == ticker]
    if row.empty:
        raise ValueError(f"Ticker {ticker} not found in results.")
    return float(row.iloc[0]["Model probability"])


def with_market_cap(company_inputs: pd.DataFrame, ticker: str, market_cap: float) -> pd.DataFrame:
    if market_cap <= 0:
        raise ValueError("market_cap must be positive.")
    adjusted = validate_company_inputs(company_inputs).copy()
    mask = adjusted["Ticker"] == ticker
    if not mask.any():
        raise ValueError(f"Ticker {ticker} not found in company inputs.")
    adjusted.loc[mask, "Current market cap"] = float(market_cap)
    return adjusted


def winner_probability_at_market_cap(
    company_inputs: pd.DataFrame,
    correlation_matrix: pd.DataFrame,
    *,
    ticker: str,
    market_cap: float,
    days_to_target: int,
    simulations: int,
    seed: int,
) -> float:
    adjusted = with_market_cap(company_inputs, ticker, market_cap)
    result = run_probability_engine(
        adjusted,
        correlation_matrix,
        days_to_target=days_to_target,
        simulations=simulations,
        seed=seed,
    )
    return selected_probability(result.results, ticker)


def find_market_cap_boundary_for_winner_probability(
    company_inputs: pd.DataFrame,
    correlation_matrix: pd.DataFrame,
    *,
    ticker: str,
    target_probability: float,
    days_to_target: int,
    simulations: int,
    seed: int,
    max_iterations: int = 24,
    tolerance: float = 0.0025,
) -> BoundaryResult:
    """Find selected ticker market cap needed to reach target P(#1).

    The search uses repeated Monte Carlo runs with the same seed. That keeps the
    random draws fixed across boundary evaluations, making the probability curve
    much smoother for bisection.
    """

    if not 0.0 < target_probability < 1.0:
        raise ValueError("target_probability must be between 0 and 1.")

    clean_inputs = validate_company_inputs(company_inputs)
    current_cap = float(clean_inputs.loc[clean_inputs["Ticker"] == ticker, "Current market cap"].iloc[0])

    low = current_cap * 0.05
    high = current_cap * 5.0
    low_probability = winner_probability_at_market_cap(
        clean_inputs,
        correlation_matrix,
        ticker=ticker,
        market_cap=low,
        days_to_target=days_to_target,
        simulations=simulations,
        seed=seed,
    )
    high_probability = winner_probability_at_market_cap(
        clean_inputs,
        correlation_matrix,
        ticker=ticker,
        market_cap=high,
        days_to_target=days_to_target,
        simulations=simulations,
        seed=seed,
    )

    expansion_count = 0
    while high_probability < target_probability and expansion_count < 8:
        high *= 2.0
        high_probability = winner_probability_at_market_cap(
            clean_inputs,
            correlation_matrix,
            ticker=ticker,
            market_cap=high,
            days_to_target=days_to_target,
            simulations=simulations,
            seed=seed,
        )
        expansion_count += 1

    if low_probability > target_probability:
        boundary = low
        achieved = low_probability
        iterations = 0
    elif high_probability < target_probability:
        boundary = high
        achieved = high_probability
        iterations = expansion_count
    else:
        boundary = high
        achieved = high_probability
        iterations = 0
        for iteration in range(max_iterations):
            midpoint = 0.5 * (low + high)
            probability = winner_probability_at_market_cap(
                clean_inputs,
                correlation_matrix,
                ticker=ticker,
                market_cap=midpoint,
                days_to_target=days_to_target,
                simulations=simulations,
                seed=seed,
            )
            iterations = iteration + 1 + expansion_count
            boundary = midpoint
            achieved = probability
            if abs(probability - target_probability) <= tolerance:
                break
            if probability < target_probability:
                low = midpoint
            else:
                high = midpoint

    absolute_gap = boundary - current_cap
    relative_gap = absolute_gap / current_cap
    return BoundaryResult(
        ticker=ticker,
        target_probability=target_probability,
        boundary_market_cap=boundary,
        current_market_cap=current_cap,
        absolute_gap=absolute_gap,
        relative_gap=relative_gap,
        achieved_probability=achieved,
        iterations=iterations,
    )


def winner_probability_curve(
    company_inputs: pd.DataFrame,
    correlation_matrix: pd.DataFrame,
    *,
    ticker: str,
    cap_multipliers: list[float],
    days_to_target: int,
    simulations: int,
    seed: int,
) -> pd.DataFrame:
    clean_inputs = validate_company_inputs(company_inputs)
    current_cap = float(clean_inputs.loc[clean_inputs["Ticker"] == ticker, "Current market cap"].iloc[0])
    rows = []
    for multiplier in cap_multipliers:
        market_cap = current_cap * float(multiplier)
        probability = winner_probability_at_market_cap(
            clean_inputs,
            correlation_matrix,
            ticker=ticker,
            market_cap=market_cap,
            days_to_target=days_to_target,
            simulations=simulations,
            seed=seed,
        )
        rows.append(
            {
                "Ticker": ticker,
                "Market cap multiplier": float(multiplier),
                "Market cap": market_cap,
                "P(#1)": probability,
                "Market-cap move": market_cap - current_cap,
            }
        )
    return pd.DataFrame(rows)


def pairwise_probability(
    selected_cap: float,
    competitor_cap: float,
    selected_iv: float,
    competitor_iv: float,
    correlation: float,
    days_to_target: int,
) -> float:
    """Analytic probability that selected terminal cap exceeds competitor cap."""

    horizon_years = days_to_target / 365.0
    relative_variance = selected_iv**2 + competitor_iv**2 - 2.0 * correlation * selected_iv * competitor_iv
    relative_vol = np.sqrt(max(relative_variance, 0.0) * horizon_years)
    if relative_vol <= 0:
        return float(selected_cap > competitor_cap)

    mean_log_ratio = np.log(selected_cap / competitor_cap) - 0.5 * (selected_iv**2 - competitor_iv**2) * horizon_years
    return float(_STANDARD_NORMAL.cdf(mean_log_ratio / relative_vol))


def pairwise_market_cap_boundary(
    selected_cap: float,
    competitor_cap: float,
    selected_iv: float,
    competitor_iv: float,
    correlation: float,
    days_to_target: int,
    target_probability: float,
) -> float:
    """Selected market cap needed for target P(selected > competitor)."""

    if not 0.0 < target_probability < 1.0:
        raise ValueError("target_probability must be between 0 and 1.")

    horizon_years = days_to_target / 365.0
    relative_variance = selected_iv**2 + competitor_iv**2 - 2.0 * correlation * selected_iv * competitor_iv
    relative_vol = np.sqrt(max(relative_variance, 0.0) * horizon_years)
    z_score = _STANDARD_NORMAL.inv_cdf(target_probability)
    required_log_ratio = z_score * relative_vol + 0.5 * (selected_iv**2 - competitor_iv**2) * horizon_years
    return float(competitor_cap * np.exp(required_log_ratio))


def pairwise_boundary_table(
    company_inputs: pd.DataFrame,
    correlation_matrix: pd.DataFrame,
    *,
    selected_ticker: str,
    target_probabilities: list[float],
    days_to_target: int,
) -> pd.DataFrame:
    clean_inputs = validate_company_inputs(company_inputs).set_index("Ticker")
    if selected_ticker not in clean_inputs.index:
        raise ValueError(f"Ticker {selected_ticker} not found in company inputs.")

    selected = clean_inputs.loc[selected_ticker]
    rows = []
    for competitor_ticker, competitor in clean_inputs.drop(index=selected_ticker).iterrows():
        rho = float(correlation_matrix.loc[selected_ticker, competitor_ticker])
        current_probability = pairwise_probability(
            float(selected["Current market cap"]),
            float(competitor["Current market cap"]),
            float(selected["Implied volatility"]),
            float(competitor["Implied volatility"]),
            rho,
            days_to_target,
        )
        for target_probability in target_probabilities:
            boundary = pairwise_market_cap_boundary(
                float(selected["Current market cap"]),
                float(competitor["Current market cap"]),
                float(selected["Implied volatility"]),
                float(competitor["Implied volatility"]),
                rho,
                days_to_target,
                target_probability,
            )
            rows.append(
                {
                    "Selected": selected_ticker,
                    "Competitor": competitor_ticker,
                    "Target pair probability": target_probability,
                    "Current pair probability": current_probability,
                    "Boundary selected market cap": boundary,
                    "Current selected market cap": float(selected["Current market cap"]),
                    "Competitor market cap": float(competitor["Current market cap"]),
                    "Boundary gap vs current": boundary - float(selected["Current market cap"]),
                    "Boundary gap vs competitor": boundary - float(competitor["Current market cap"]),
                    "Boundary move vs current": boundary / float(selected["Current market cap"]) - 1.0,
                    "Correlation": rho,
                    "Selected IV": float(selected["Implied volatility"]),
                    "Competitor IV": float(competitor["Implied volatility"]),
                }
            )
    return pd.DataFrame(rows)
