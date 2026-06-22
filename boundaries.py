"""Conditional probability boundary calculations for ranking events.

Phase 2 is centered on conditional probability curves from Phase 1 Monte Carlo
scenarios. For a selected ticker, we look at its simulated terminal market cap,
its simulated final rank, and whether it won the ranking event. That produces
boundary zones such as: above this terminal market-cap level, the selected ticker
wins at least 95% of similar scenarios.

Some older inverse helpers remain in this module as diagnostics, but the primary
Phase 2 method is scenario-based conditioning.
"""

from __future__ import annotations

from dataclasses import dataclass
from statistics import NormalDist

import numpy as np
import pandas as pd

from model import rank_descending, run_probability_engine, validate_company_inputs


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


def calculate_rank_matrix(simulated_market_caps: pd.DataFrame) -> pd.DataFrame:
    """Convert simulated terminal market caps to ranks where 1 is largest."""

    if simulated_market_caps.empty:
        raise ValueError("simulated_market_caps cannot be empty.")
    values = simulated_market_caps.to_numpy(dtype=float)
    ranks = rank_descending(values)
    return pd.DataFrame(ranks, index=simulated_market_caps.index, columns=simulated_market_caps.columns)


def calculate_conditional_win_curve(
    simulated_market_caps: pd.DataFrame,
    selected_ticker: str,
    *,
    ranks: pd.DataFrame | None = None,
    current_market_cap: float | None = None,
    n_bins: int = 25,
) -> pd.DataFrame:
    """Estimate P(win | selected terminal market cap is in a quantile bin).

    Each row is one quantile bin of the selected ticker's simulated terminal
    market cap. The curve is intentionally empirical: it uses the actual Phase 1
    Monte Carlo scenarios and therefore automatically reflects the full joint
    distribution, including competitors' simulated values.
    """

    if selected_ticker not in simulated_market_caps.columns:
        raise ValueError(f"{selected_ticker} is not in simulated market caps.")
    if n_bins < 3:
        raise ValueError("n_bins must be at least 3.")

    rank_matrix = calculate_rank_matrix(simulated_market_caps) if ranks is None else ranks
    if selected_ticker not in rank_matrix.columns:
        raise ValueError(f"{selected_ticker} is not in rank matrix.")

    selected_caps = simulated_market_caps[selected_ticker].astype(float)
    selected_ranks = rank_matrix[selected_ticker].astype(float)
    data = pd.DataFrame(
        {
            "Terminal market cap": selected_caps,
            "Rank": selected_ranks,
            "Won": selected_ranks == 1,
        }
    ).dropna()
    if data.empty:
        raise ValueError("No valid scenarios available for conditional curve.")

    unique_values = data["Terminal market cap"].nunique()
    effective_bins = int(min(n_bins, unique_values))
    if effective_bins < 3:
        raise ValueError("Not enough distinct terminal market-cap values for binning.")

    data["Bin"] = pd.qcut(data["Terminal market cap"], q=effective_bins, duplicates="drop")
    grouped = data.groupby("Bin", observed=True)
    curve = grouped.agg(
        bin_low=("Terminal market cap", "min"),
        bin_high=("Terminal market cap", "max"),
        average_market_cap=("Terminal market cap", "mean"),
        win_probability=("Won", "mean"),
        average_rank=("Rank", "mean"),
        scenario_count=("Won", "size"),
    ).reset_index(drop=True)
    curve["loss_probability"] = 1.0 - curve["win_probability"]
    curve["market_cap_midpoint"] = 0.5 * (curve["bin_low"] + curve["bin_high"])
    if current_market_cap is not None and current_market_cap > 0:
        curve["market_cap_to_current"] = curve["average_market_cap"] / float(current_market_cap)
        curve["bin_low_to_current"] = curve["bin_low"] / float(current_market_cap)
        curve["bin_high_to_current"] = curve["bin_high"] / float(current_market_cap)
    else:
        curve["market_cap_to_current"] = np.nan
        curve["bin_low_to_current"] = np.nan
        curve["bin_high_to_current"] = np.nan
    return curve


def find_probability_boundaries(
    conditional_curve: pd.DataFrame,
    confidence_levels: list[float],
    *,
    current_market_cap: float,
    ticker: str,
) -> pd.DataFrame:
    """Find lower loss and upper win boundaries from a conditional curve."""

    required = {"average_market_cap", "win_probability", "loss_probability"}
    missing = required - set(conditional_curve.columns)
    if missing:
        raise ValueError(f"conditional_curve missing columns: {sorted(missing)}")
    if current_market_cap <= 0:
        raise ValueError("current_market_cap must be positive.")

    curve = conditional_curve.sort_values("average_market_cap").reset_index(drop=True)
    rows = []
    for confidence in confidence_levels:
        if not 0.0 < confidence < 1.0:
            raise ValueError("confidence levels must be between 0 and 1.")

        win_candidates = curve[curve["win_probability"] >= confidence]
        loss_candidates = curve[curve["loss_probability"] >= confidence]

        upper_win_boundary = np.nan if win_candidates.empty else float(win_candidates.iloc[0]["average_market_cap"])
        lower_loss_boundary = np.nan if loss_candidates.empty else float(loss_candidates.iloc[-1]["average_market_cap"])

        rows.append(
            {
                "Ticker": ticker,
                "Confidence level": confidence,
                "Lower loss boundary": lower_loss_boundary,
                "Upper win boundary": upper_win_boundary,
                "Lower loss boundary / current": lower_loss_boundary / current_market_cap if np.isfinite(lower_loss_boundary) else np.nan,
                "Upper win boundary / current": upper_win_boundary / current_market_cap if np.isfinite(upper_win_boundary) else np.nan,
            }
        )
    return pd.DataFrame(rows)


def calculate_boundaries_for_all_tickers(
    simulated_market_caps: pd.DataFrame,
    current_market_caps: pd.Series | dict[str, float],
    confidence_levels: list[float],
    *,
    ranks: pd.DataFrame | None = None,
    n_bins: int = 25,
) -> pd.DataFrame:
    """Repeat conditional boundary calculation for every ticker."""

    current_caps = pd.Series(current_market_caps, dtype=float)
    rank_matrix = calculate_rank_matrix(simulated_market_caps) if ranks is None else ranks
    frames = []
    for ticker in simulated_market_caps.columns:
        curve = calculate_conditional_win_curve(
            simulated_market_caps,
            ticker,
            ranks=rank_matrix,
            current_market_cap=float(current_caps.loc[ticker]),
            n_bins=n_bins,
        )
        frames.append(
            find_probability_boundaries(
                curve,
                confidence_levels,
                current_market_cap=float(current_caps.loc[ticker]),
                ticker=ticker,
            )
        )
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


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
    """Diagnostic inverse helper: current cap shock needed for target P(#1)."""

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
