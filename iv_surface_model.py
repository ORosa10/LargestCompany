"""Risk-neutral marginal distributions derived from manually calibrated IV smiles.

The July rehearsal uses one expiry that matches the Polymarket resolution date.
Each equity marginal is extracted from a smooth option-price curve, while the
cross-sectional dependence remains controlled by the selected correlation
matrix through a Gaussian copula.
"""

from __future__ import annotations

from math import exp, log, pi, sqrt

import numpy as np
import pandas as pd

from model import (
    SimulationResult,
    build_rank_distribution,
    cholesky_with_jitter,
    clean_correlation_matrix,
    rank_descending,
    validate_company_inputs,
)


SURFACE_AS_OF = "2026-07-04"
SURFACE_EXPIRY = "2026-07-31"

# Nodes read from OptionCharts screenshots. Only the OTM put wing below spot and
# OTM call wing above spot are retained; obvious stale-quote spikes are removed.
_SURFACE_NODES = {
    "NVDA": {
        "spot": 194.42,
        "nodes": [
            (100, 0.95, "Put"), (110, 0.87, "Put"), (120, 0.77, "Put"),
            (130, 0.67, "Put"), (140, 0.60, "Put"), (150, 0.56, "Put"),
            (160, 0.51, "Put"), (170, 0.46, "Put"), (180, 0.43, "Put"),
            (190, 0.40, "Put"), (195, 0.39, "ATM"), (200, 0.38, "Call"),
            (210, 0.37, "Call"), (220, 0.37, "Call"), (230, 0.39, "Call"),
            (240, 0.41, "Call"), (250, 0.44, "Call"), (260, 0.48, "Call"),
            (270, 0.51, "Call"), (280, 0.53, "Call"), (290, 0.56, "Call"),
            (300, 0.57, "Call"), (310, 0.59, "Call"), (320, 0.62, "Call"),
        ],
    },
    "GOOGL": {
        "spot": 359.08,
        "nodes": [
            (200, 1.10, "Put"), (220, 0.92, "Put"), (230, 0.75, "Put"),
            (240, 0.67, "Put"), (250, 0.61, "Put"), (260, 0.58, "Put"),
            (280, 0.49, "Put"), (300, 0.44, "Put"), (320, 0.42, "Put"),
            (340, 0.39, "Put"), (350, 0.38, "Put"), (360, 0.40, "ATM"),
            (380, 0.40, "Call"), (400, 0.40, "Call"), (420, 0.41, "Call"),
            (440, 0.42, "Call"), (460, 0.46, "Call"), (480, 0.53, "Call"),
            (500, 0.57, "Call"), (520, 0.63, "Call"), (530, 0.67, "Call"),
        ],
    },
    "AAPL": {
        "spot": 308.46,
        "nodes": [
            (180, 0.87, "Put"), (190, 0.70, "Put"), (200, 0.65, "Put"),
            (210, 0.56, "Put"), (220, 0.52, "Put"), (230, 0.48, "Put"),
            (240, 0.44, "Put"), (250, 0.41, "Put"), (260, 0.38, "Put"),
            (270, 0.35, "Put"), (280, 0.32, "Put"), (290, 0.30, "Put"),
            (300, 0.29, "Put"), (310, 0.285, "ATM"), (320, 0.28, "Call"),
            (330, 0.28, "Call"), (340, 0.27, "Call"), (350, 0.28, "Call"),
            (360, 0.29, "Call"), (370, 0.30, "Call"), (380, 0.32, "Call"),
            (390, 0.35, "Call"), (400, 0.40, "Call"),
        ],
    },
}


def default_surface_nodes() -> pd.DataFrame:
    rows = []
    for ticker, definition in _SURFACE_NODES.items():
        spot = float(definition["spot"])
        for strike, iv, wing in definition["nodes"]:
            rows.append(
                {
                    "Ticker": ticker,
                    "Observed spot": spot,
                    "Strike": float(strike),
                    "Moneyness": float(strike) / spot,
                    "IV": float(iv),
                    "Wing": wing,
                    "As of": SURFACE_AS_OF,
                    "Expiry": SURFACE_EXPIRY,
                }
            )
    return pd.DataFrame(rows)


def apply_surface_atm_ivs(company_inputs: pd.DataFrame) -> pd.DataFrame:
    updated = company_inputs.copy()
    atm = {"NVDA": 0.39, "AAPL": 0.285, "GOOGL": 0.40}
    mapped = updated["Ticker"].map(atm)
    updated.loc[mapped.notna(), "Implied volatility"] = mapped[mapped.notna()]
    return updated


def normal_cdf_approx(values: np.ndarray) -> np.ndarray:
    """Fast vectorized standard-normal CDF approximation."""
    z = np.asarray(values, dtype=float)
    absolute = np.abs(z)
    t = 1.0 / (1.0 + 0.2316419 * absolute)
    polynomial = t * (
        0.319381530
        + t * (-0.356563782 + t * (1.781477937 + t * (-1.821255978 + t * 1.330274429)))
    )
    tail = np.exp(-0.5 * absolute * absolute) / sqrt(2.0 * pi) * polynomial
    cdf_positive = 1.0 - tail
    return np.where(z >= 0.0, cdf_positive, 1.0 - cdf_positive)


def _interpolated_iv(ticker_nodes: pd.DataFrame, strike_ratios: np.ndarray) -> np.ndarray:
    nodes = ticker_nodes.sort_values("Moneyness")
    x = np.log(nodes["Moneyness"].to_numpy(dtype=float))
    y = nodes["IV"].to_numpy(dtype=float)
    return np.clip(np.interp(np.log(strike_ratios), x, y, left=y[0], right=y[-1]), 0.01, 4.0)


def build_surface_marginal(
    ticker_nodes: pd.DataFrame,
    *,
    forward_ratio: float,
    horizon_years: float,
    risk_free_rate: float,
    grid_size: int = 5000,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Build a monotone risk-neutral CDF for terminal stock-price ratio S_T/S_0."""
    if horizon_years <= 0:
        raise ValueError("horizon_years must be positive.")
    if forward_ratio <= 0:
        raise ValueError("forward_ratio must be positive.")

    observed = ticker_nodes["Moneyness"].to_numpy(dtype=float)
    lower = max(0.02, float(observed.min()) * 0.45)
    upper = max(3.0, float(observed.max()) * 1.75)
    strike_ratio = np.geomspace(lower, upper, int(grid_size))
    iv = _interpolated_iv(ticker_nodes, strike_ratio)
    discount = exp(-float(risk_free_rate) * horizon_years)
    root_t = sqrt(horizon_years)
    d1 = (np.log(forward_ratio / strike_ratio) + 0.5 * iv * iv * horizon_years) / (iv * root_t)
    d2 = d1 - iv * root_t
    call_price = discount * (
        forward_ratio * normal_cdf_approx(d1) - strike_ratio * normal_cdf_approx(d2)
    )

    # Breeden-Litzenberger first derivative: dC/dK = -D * Q(S_T > K).
    raw_cdf = 1.0 + np.gradient(call_price, strike_ratio) / discount
    projected = np.maximum.accumulate(np.clip(raw_cdf, 0.0, 1.0))
    span = float(projected[-1] - projected[0])
    if span <= 1e-8:
        raise ValueError("IV smile did not produce a usable terminal CDF.")
    projected = (projected - projected[0]) / span
    projected[0] = 0.0
    projected[-1] = 1.0
    unique_cdf, unique_indices = np.unique(projected, return_index=True)
    unique_strikes = strike_ratio[unique_indices]
    diagnostics = {
        "min_moneyness": float(observed.min()),
        "max_moneyness": float(observed.max()),
        "atm_iv": float(_interpolated_iv(ticker_nodes, np.array([1.0]))[0]),
        "projection_adjustment": float(np.mean(np.abs(projected - np.clip(raw_cdf, 0.0, 1.0)))),
    }
    return unique_cdf, unique_strikes, diagnostics


def sample_surface_marginal(
    uniforms: np.ndarray,
    ticker_nodes: pd.DataFrame,
    *,
    forward_ratio: float,
    horizon_years: float,
    risk_free_rate: float,
) -> tuple[np.ndarray, dict]:
    cdf, terminal_ratios, diagnostics = build_surface_marginal(
        ticker_nodes,
        forward_ratio=forward_ratio,
        horizon_years=horizon_years,
        risk_free_rate=risk_free_rate,
    )
    samples = np.interp(np.clip(uniforms, 1e-8, 1.0 - 1e-8), cdf, terminal_ratios)
    # Numerical CDF projection can move the first moment slightly. Re-anchor to
    # the option-implied forward so only the distribution shape changes.
    raw_mean = float(samples.mean())
    if raw_mean <= 0:
        raise ValueError("Surface marginal generated a non-positive mean.")
    samples *= forward_ratio / raw_mean
    diagnostics["sample_mean_before_reanchor"] = raw_mean
    diagnostics["sample_mean_after_reanchor"] = float(samples.mean())
    return samples, diagnostics


def run_surface_probability_engine(
    company_inputs: pd.DataFrame,
    correlation_matrix: pd.DataFrame,
    *,
    days_to_target: int,
    simulations: int,
    seed: int,
    surface_nodes: pd.DataFrame | None = None,
    risk_free_rate: float = 0.04,
) -> tuple[SimulationResult, pd.DataFrame]:
    """Run Gaussian-copula simulation with surface marginals where available."""
    if days_to_target <= 0 or simulations <= 0:
        raise ValueError("days_to_target and simulations must be positive.")
    clean_inputs = validate_company_inputs(company_inputs)
    tickers = clean_inputs["Ticker"].tolist()
    cleaned_corr, corr_warnings = clean_correlation_matrix(correlation_matrix, tickers)
    cholesky, chol_warnings = cholesky_with_jitter(cleaned_corr.to_numpy(dtype=float))
    rng = np.random.default_rng(int(seed))
    correlated_normals = rng.standard_normal((int(simulations), len(tickers))) @ cholesky.T
    uniforms = normal_cdf_approx(correlated_normals)
    years = days_to_target / 365.0

    caps_0 = clean_inputs["Current market cap"].to_numpy(dtype=float)
    ivs = clean_inputs["Implied volatility"].to_numpy(dtype=float)
    forward_ratios = (
        clean_inputs["Forward / spot"].to_numpy(dtype=float)
        if "Forward / spot" in clean_inputs.columns
        else np.ones(len(tickers), dtype=float)
    )
    nodes = default_surface_nodes() if surface_nodes is None else surface_nodes.copy()
    terminal_ratios = np.empty_like(correlated_normals)
    diagnostic_rows = []

    for index, ticker in enumerate(tickers):
        ticker_nodes = nodes[nodes["Ticker"] == ticker]
        if ticker_nodes.empty:
            terminal_ratios[:, index] = forward_ratios[index] * np.exp(
                -0.5 * ivs[index] ** 2 * years + ivs[index] * sqrt(years) * correlated_normals[:, index]
            )
            diagnostic_rows.append(
                {
                    "Ticker": ticker,
                    "Marginal model": "ATM lognormal fallback",
                    "ATM IV": ivs[index],
                    "Forward / spot": forward_ratios[index],
                    "Projection adjustment": np.nan,
                }
            )
        else:
            samples, diagnostics = sample_surface_marginal(
                uniforms[:, index],
                ticker_nodes,
                forward_ratio=float(forward_ratios[index]),
                horizon_years=years,
                risk_free_rate=float(risk_free_rate),
            )
            terminal_ratios[:, index] = samples
            diagnostic_rows.append(
                {
                    "Ticker": ticker,
                    "Marginal model": "IV surface risk-neutral CDF",
                    "ATM IV": diagnostics["atm_iv"],
                    "Forward / spot": forward_ratios[index],
                    "Projection adjustment": diagnostics["projection_adjustment"],
                }
            )

    terminal_caps = caps_0 * terminal_ratios
    ranks_array = rank_descending(terminal_caps)
    yes_price = clean_inputs["Polymarket YES price"].to_numpy(dtype=float)
    model_probability = (ranks_array == 1).mean(axis=0)
    results = pd.DataFrame(
        {
            "Ticker": tickers,
            "Current market cap": caps_0,
            "Implied volatility": ivs,
            "Polymarket YES price": yes_price,
            "Model probability": model_probability,
            "Edge": model_probability - yes_price,
            "Expected value": model_probability - yes_price,
            "ROI": np.divide(model_probability - yes_price, yes_price, out=np.full_like(yes_price, np.nan), where=yes_price > 0),
            "Average rank": ranks_array.mean(axis=0),
            "Probability Top 2": (ranks_array <= 2).mean(axis=0),
            "Probability Top 3": (ranks_array <= 3).mean(axis=0),
        }
    ).sort_values("Edge", ascending=False, ignore_index=True)
    result = SimulationResult(
        results=results,
        terminal_market_caps=pd.DataFrame(terminal_caps, columns=tickers),
        ranks=pd.DataFrame(ranks_array, columns=tickers),
        rank_distribution=build_rank_distribution(ranks_array, tickers),
        cleaned_correlation=cleaned_corr,
        warnings=tuple(corr_warnings + chol_warnings),
    )
    return result, pd.DataFrame(diagnostic_rows)
