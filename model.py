"""Probability engine for largest market-cap ranking events.

Phase 1 is intentionally limited to probability estimation. It does not build
hedges, option payoff heatmaps, or portfolio optimization.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


DEFAULT_UNIVERSE = [
    {"Ticker": "NVDA", "Current market cap": 4_300_000_000_000, "Implied volatility": 0.42, "Polymarket YES price": 0.30},
    {"Ticker": "AAPL", "Current market cap": 3_100_000_000_000, "Implied volatility": 0.24, "Polymarket YES price": 0.16},
    {"Ticker": "MSFT", "Current market cap": 3_700_000_000_000, "Implied volatility": 0.25, "Polymarket YES price": 0.22},
    {"Ticker": "GOOGL", "Current market cap": 2_100_000_000_000, "Implied volatility": 0.28, "Polymarket YES price": 0.05},
    {"Ticker": "AMZN", "Current market cap": 2_300_000_000_000, "Implied volatility": 0.31, "Polymarket YES price": 0.06},
    {"Ticker": "META", "Current market cap": 1_800_000_000_000, "Implied volatility": 0.34, "Polymarket YES price": 0.03},
    {"Ticker": "AVGO", "Current market cap": 1_700_000_000_000, "Implied volatility": 0.38, "Polymarket YES price": 0.04},
    {"Ticker": "TSLA", "Current market cap": 1_100_000_000_000, "Implied volatility": 0.58, "Polymarket YES price": 0.03},
    {"Ticker": "BRK.B", "Current market cap": 1_000_000_000_000, "Implied volatility": 0.18, "Polymarket YES price": 0.01},
    {"Ticker": "LLY", "Current market cap": 850_000_000_000, "Implied volatility": 0.30, "Polymarket YES price": 0.01},
]

RESULT_COLUMNS = [
    "Ticker",
    "Current market cap",
    "Implied volatility",
    "Polymarket YES price",
    "Model probability",
    "Edge",
    "Expected value",
    "ROI",
    "Average rank",
    "Probability Top 2",
    "Probability Top 3",
]


@dataclass(frozen=True)
class SimulationResult:
    results: pd.DataFrame
    terminal_market_caps: pd.DataFrame
    ranks: pd.DataFrame
    rank_distribution: pd.DataFrame
    cleaned_correlation: pd.DataFrame
    warnings: tuple[str, ...]

    @property
    def most_undervalued(self) -> pd.Series:
        return self.results.sort_values("Edge", ascending=False).iloc[0]

    @property
    def most_overvalued(self) -> pd.Series:
        return self.results.sort_values("Edge", ascending=True).iloc[0]


def default_company_inputs() -> pd.DataFrame:
    return pd.DataFrame(DEFAULT_UNIVERSE)


def default_correlation_matrix(tickers: list[str]) -> pd.DataFrame:
    size = len(tickers)
    matrix = np.full((size, size), 0.45, dtype=float)
    np.fill_diagonal(matrix, 1.0)
    return pd.DataFrame(matrix, index=tickers, columns=tickers)


def run_probability_engine(
    company_inputs: pd.DataFrame,
    correlation_matrix: pd.DataFrame,
    *,
    days_to_target: int,
    simulations: int,
    seed: int,
) -> SimulationResult:
    """Run correlated lognormal market-cap simulations.

    Financial model:
        MC_T = MC_0 * exp((mu - 0.5 * sigma^2) * T + sigma * sqrt(T) * Z)

    ``mu * T`` is the total forward log carry taken from an optional
    "Forward / spot" column (put-call parity implied forward). When no forward
    ratios are supplied it is zero, so the baseline model reduces to the pure
    lognormal convexity adjustment. The model accepts current market cap,
    implied volatility, correlations, and (optionally) forward carry as inputs;
    it does not forecast an equity risk premium.
    """

    clean_inputs = validate_company_inputs(company_inputs)
    tickers = clean_inputs["Ticker"].tolist()
    cleaned_corr, corr_warnings = clean_correlation_matrix(correlation_matrix, tickers)
    cholesky, chol_warnings = cholesky_with_jitter(cleaned_corr.to_numpy(dtype=float))

    if days_to_target <= 0:
        raise ValueError("days_to_target must be positive.")
    if simulations <= 0:
        raise ValueError("simulations must be positive.")

    rng = np.random.default_rng(seed)
    horizon_years = days_to_target / 365.0

    independent_normals = rng.standard_normal((simulations, len(tickers)))
    correlated_normals = independent_normals @ cholesky.T

    market_caps_0 = clean_inputs["Current market cap"].to_numpy(dtype=float)
    volatilities = clean_inputs["Implied volatility"].to_numpy(dtype=float)

    log_carry = forward_log_carry(clean_inputs, tickers)
    drift = log_carry - 0.5 * np.square(volatilities) * horizon_years
    diffusion = volatilities * np.sqrt(horizon_years) * correlated_normals
    terminal_caps = market_caps_0 * np.exp(drift + diffusion)
    terminal_market_caps = pd.DataFrame(terminal_caps, columns=tickers)

    # Rank 1 means largest simulated market capitalization in that simulation.
    ranks_array = rank_descending(terminal_caps)
    ranks = pd.DataFrame(ranks_array, columns=tickers)

    model_probability = (ranks_array == 1).mean(axis=0)
    probability_top_2 = (ranks_array <= 2).mean(axis=0)
    probability_top_3 = (ranks_array <= 3).mean(axis=0)
    average_rank = ranks_array.mean(axis=0)

    yes_price = clean_inputs["Polymarket YES price"].to_numpy(dtype=float)
    edge = model_probability - yes_price

    # A YES share with price p and $1 payout has EV = P(win) - p.
    expected_value = edge
    roi = np.divide(edge, yes_price, out=np.full_like(edge, np.nan), where=yes_price > 0)

    results = pd.DataFrame(
        {
            "Ticker": tickers,
            "Current market cap": market_caps_0,
            "Implied volatility": volatilities,
            "Polymarket YES price": yes_price,
            "Model probability": model_probability,
            "Edge": edge,
            "Expected value": expected_value,
            "ROI": roi,
            "Average rank": average_rank,
            "Probability Top 2": probability_top_2,
            "Probability Top 3": probability_top_3,
        }
    ).sort_values("Edge", ascending=False, ignore_index=True)

    return SimulationResult(
        results=results[RESULT_COLUMNS],
        terminal_market_caps=terminal_market_caps,
        ranks=ranks,
        rank_distribution=build_rank_distribution(ranks_array, tickers),
        cleaned_correlation=cleaned_corr,
        warnings=tuple(corr_warnings + chol_warnings),
    )


def forward_log_carry(company_inputs: pd.DataFrame, tickers: list[str]) -> np.ndarray:
    """Total log forward carry over the horizon from an optional forward column.

    Reads a "Forward / spot" column (target-date forward divided by spot, e.g.
    from put-call parity). Returns zeros when the column is absent or invalid,
    keeping the baseline zero-drift behaviour unchanged. When present, the base
    engine centres each company on its forward: E[MC_T] = MC_0 * forward / spot.
    """

    if "Forward / spot" not in company_inputs.columns:
        return np.zeros(len(tickers))
    ratios = pd.to_numeric(company_inputs["Forward / spot"], errors="coerce").to_numpy(dtype=float)
    ratios = np.where(np.isfinite(ratios) & (ratios > 0.0), ratios, 1.0)
    return np.log(ratios)


def validate_company_inputs(company_inputs: pd.DataFrame) -> pd.DataFrame:
    required_columns = {
        "Ticker",
        "Current market cap",
        "Implied volatility",
        "Polymarket YES price",
    }
    missing = required_columns - set(company_inputs.columns)
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(sorted(missing))}.")

    clean = company_inputs.copy()
    clean["Ticker"] = clean["Ticker"].astype(str).str.strip()
    clean = clean[clean["Ticker"] != ""].reset_index(drop=True)

    if clean.empty:
        raise ValueError("At least one company is required.")
    if clean["Ticker"].duplicated().any():
        duplicates = clean.loc[clean["Ticker"].duplicated(), "Ticker"].tolist()
        raise ValueError(f"Duplicate tickers are not allowed: {duplicates}.")

    numeric_columns = ["Current market cap", "Implied volatility", "Polymarket YES price"]
    for column in numeric_columns:
        clean[column] = pd.to_numeric(clean[column], errors="coerce")
        if clean[column].isna().any():
            raise ValueError(f"{column} contains non-numeric values.")

    if (clean["Current market cap"] <= 0).any():
        raise ValueError("Current market cap must be positive for every company.")
    if (clean["Implied volatility"] <= 0).any():
        raise ValueError("Implied volatility must be positive for every company.")
    if ((clean["Polymarket YES price"] < 0) | (clean["Polymarket YES price"] > 1)).any():
        raise ValueError("Polymarket YES price must be between 0 and 1.")

    return clean


def clean_correlation_matrix(
    correlation_matrix: pd.DataFrame,
    tickers: list[str],
    *,
    tolerance: float = 1e-8,
) -> tuple[pd.DataFrame, list[str]]:
    """Validate and gently repair small numerical correlation matrix issues."""

    warnings: list[str] = []
    matrix = correlation_matrix.copy()

    if list(matrix.index) != tickers or list(matrix.columns) != tickers:
        matrix = matrix.reindex(index=tickers, columns=tickers)
        warnings.append("Correlation matrix was reindexed to match ticker order.")

    values = matrix.to_numpy(dtype=float)
    if values.shape != (len(tickers), len(tickers)):
        raise ValueError("Correlation matrix must be square and match the number of tickers.")
    if not np.isfinite(values).all():
        raise ValueError("Correlation matrix contains missing or non-finite values.")

    if np.max(np.abs(values - values.T)) > tolerance:
        values = 0.5 * (values + values.T)
        warnings.append("Correlation matrix was symmetrized.")

    if np.any(values < -1.0 - tolerance) or np.any(values > 1.0 + tolerance):
        raise ValueError("Correlation values must be between -1 and 1.")

    values = np.clip(values, -1.0, 1.0)
    if np.max(np.abs(np.diag(values) - 1.0)) > tolerance:
        warnings.append("Correlation matrix diagonal was reset to 1.")
    np.fill_diagonal(values, 1.0)

    return pd.DataFrame(values, index=tickers, columns=tickers), warnings


def cholesky_with_jitter(
    correlation_values: np.ndarray,
    *,
    max_attempts: int = 8,
    initial_jitter: float = 1e-10,
) -> tuple[np.ndarray, list[str]]:
    """Return Cholesky factor, adding small diagonal jitter if needed."""

    warnings: list[str] = []
    jitter = 0.0

    for attempt in range(max_attempts):
        try:
            adjusted = correlation_values.copy()
            if jitter > 0:
                adjusted = adjusted + np.eye(adjusted.shape[0]) * jitter
            cholesky = np.linalg.cholesky(adjusted)
            if jitter > 0:
                warnings.append(
                    f"Correlation matrix needed diagonal jitter of {jitter:.1e} for Cholesky."
                )
            return cholesky, warnings
        except np.linalg.LinAlgError:
            jitter = initial_jitter if attempt == 0 else jitter * 10.0

    raise ValueError(
        "Correlation matrix is not positive definite. Please lower extreme correlations "
        "or use a better-conditioned matrix."
    )


def rank_descending(values: np.ndarray) -> np.ndarray:
    """Convert simulated terminal values into ranks, where 1 is largest."""

    order = np.argsort(-values, axis=1)
    ranks = np.empty_like(order)
    row_index = np.arange(values.shape[0])[:, None]
    ranks[row_index, order] = np.arange(1, values.shape[1] + 1)
    return ranks


def build_rank_distribution(ranks_array: np.ndarray, tickers: list[str]) -> pd.DataFrame:
    rows = []
    max_rank = len(tickers)
    for ticker_index, ticker in enumerate(tickers):
        counts = np.bincount(ranks_array[:, ticker_index], minlength=max_rank + 1)[1:]
        probabilities = counts / ranks_array.shape[0]
        for rank, probability in enumerate(probabilities, start=1):
            rows.append({"Ticker": ticker, "Rank": rank, "Probability": probability})
    return pd.DataFrame(rows)
