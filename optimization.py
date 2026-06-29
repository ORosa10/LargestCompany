"""Phase 5 option portfolio optimization engine.

Searches a flexible call/put strike library with transparent greedy forward
selection and coordinate refinement. Positive quantities are long; negative
quantities are short during optimization.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from option_construction import black_scholes_price


OBJECTIVES = [
    "Maximum expected payoff",
    "Risk-adjusted payoff",
    "Tail-aware payoff",
    "Minimum SD with baseline EV floor",
]


@dataclass
class OptimizationResult:
    quantities: np.ndarray
    selected_legs: pd.DataFrame
    optimized_payoffs: np.ndarray
    baseline_metrics: pd.Series
    optimized_metrics: pd.Series
    objective_score: float
    iterations: int


def build_candidate_option_universe(
    *, ticker: str, spot: float, volatility: float, time_to_expiry: float,
    risk_free_rate: float, strike_multipliers: list[float] | np.ndarray,
    include_calls: bool = True, include_puts: bool = True,
) -> pd.DataFrame:
    """Create long-option basis instruments across a strike grid."""
    if spot <= 0 or volatility <= 0 or time_to_expiry <= 0:
        raise ValueError("spot, volatility, and time_to_expiry must be positive.")
    if not include_calls and not include_puts:
        raise ValueError("At least one option type must be enabled.")

    rows = []
    option_types = (["Call"] if include_calls else []) + (["Put"] if include_puts else [])
    for multiplier in sorted({float(value) for value in strike_multipliers if float(value) > 0}):
        strike = spot * multiplier
        for option_type in option_types:
            premium = black_scholes_price(
                spot=spot, strike=strike, time_to_expiry=time_to_expiry,
                volatility=volatility, risk_free_rate=risk_free_rate,
                option_type=option_type,
            )
            rows.append({
                "Instrument": f"{ticker} {option_type} {multiplier:.0%}",
                "Ticker": ticker,
                "Option type": option_type,
                "Position": "Long",
                "Strike": strike,
                "Strike / spot": multiplier,
                "Spot": spot,
                "Model IV": volatility,
                "Risk-free rate": risk_free_rate,
                "Time to expiry": time_to_expiry,
                "Theoretical premium": premium,
                "Quantity": 0.0,
            })
    if not rows:
        raise ValueError("The strike grid produced no candidates.")
    return pd.DataFrame(rows)


def long_option_payoff_matrix(
    terminal_prices: np.ndarray | pd.Series,
    candidates: pd.DataFrame,
    *, contract_multiplier: float = 100.0,
    include_premiums: bool = True,
) -> np.ndarray:
    """Return scenario payoff for one long unit of every candidate."""
    prices = np.asarray(terminal_prices, dtype=float).reshape(-1, 1)
    strikes = candidates["Strike"].to_numpy(dtype=float).reshape(1, -1)
    calls = candidates["Option type"].astype(str).eq("Call").to_numpy().reshape(1, -1)
    payoff = np.where(calls, np.maximum(prices - strikes, 0.0), np.maximum(strikes - prices, 0.0))
    if include_premiums:
        payoff = payoff - candidates["Theoretical premium"].to_numpy(dtype=float).reshape(1, -1)
    return payoff * float(contract_multiplier)


def payoff_metrics(payoffs: np.ndarray | pd.Series, *, shortfall_probability: float = 0.05) -> pd.Series:
    values = np.asarray(payoffs, dtype=float)
    threshold = np.quantile(values, shortfall_probability)
    tail = values[values <= threshold]
    return pd.Series({
        "Expected payoff": values.mean(),
        "Payoff standard deviation": values.std(ddof=0),
        "Median payoff": np.median(values),
        "Probability of loss": (values < 0).mean(),
        "Expected shortfall 5%": tail.mean() if len(tail) else np.nan,
        "Worst payoff": values.min(),
    })


def objective_score(
    payoffs: np.ndarray, *, objective: str, risk_aversion: float,
    tail_weight: float, minimum_expected_payoff: float,
) -> float:
    metrics = payoff_metrics(payoffs)
    mean = float(metrics["Expected payoff"])
    sd = float(metrics["Payoff standard deviation"])
    expected_shortfall = float(metrics["Expected shortfall 5%"])
    if objective == "Maximum expected payoff":
        return mean
    if objective == "Risk-adjusted payoff":
        return mean - float(risk_aversion) * sd
    if objective == "Tail-aware payoff":
        return mean + float(tail_weight) * expected_shortfall
    if objective == "Minimum SD with baseline EV floor":
        return -sd if mean >= minimum_expected_payoff else -np.inf
    raise ValueError(f"Unknown objective: {objective}")


def selected_quantities_to_legs(candidates: pd.DataFrame, quantities: np.ndarray, *, tolerance: float = 1e-12) -> pd.DataFrame:
    mask = np.abs(quantities) > tolerance
    selected = candidates.loc[mask].copy()
    if selected.empty:
        return candidates.iloc[0:0].copy()
    signed = quantities[mask]
    selected["Position"] = np.where(signed >= 0.0, "Long", "Short")
    selected["Quantity"] = np.abs(signed)
    selected["Instrument"] = selected.apply(
        lambda row: f"{row['Position']} {row['Ticker']} {row['Option type']} {row['Strike']:.2f}", axis=1
    )
    return selected.reset_index(drop=True)


def optimize_option_portfolio(
    polymarket_payoffs: np.ndarray | pd.Series,
    option_payoff_matrix: np.ndarray,
    candidates: pd.DataFrame,
    *, quantity_min: float = -0.25, quantity_max: float = 0.25,
    quantity_step: float = 0.025, max_legs: int = 4,
    max_total_absolute_quantity: float = 0.50,
    objective: str = "Risk-adjusted payoff", risk_aversion: float = 0.25,
    tail_weight: float = 0.10, minimum_expected_payoff: float | None = None,
    optimization_scenarios: int = 20_000, seed: int = 42,
) -> OptimizationResult:
    """Select strikes and signed quantities, then refine active quantities."""
    base = np.asarray(polymarket_payoffs, dtype=float)
    matrix = np.asarray(option_payoff_matrix, dtype=float)
    if matrix.ndim != 2 or matrix.shape != (len(base), len(candidates)):
        raise ValueError("Option payoff matrix dimensions do not match inputs.")
    if quantity_step <= 0 or quantity_min > quantity_max:
        raise ValueError("Invalid quantity grid.")
    if objective not in OBJECTIVES:
        raise ValueError(f"objective must be one of {OBJECTIVES}.")

    baseline_metrics = payoff_metrics(base)
    ev_floor = float(baseline_metrics["Expected payoff"] if minimum_expected_payoff is None else minimum_expected_payoff)
    rng = np.random.default_rng(seed)
    sample_size = min(int(optimization_scenarios), len(base))
    sample_index = rng.choice(len(base), size=sample_size, replace=False) if sample_size < len(base) else np.arange(len(base))
    sample_base = base[sample_index]
    sample_matrix = matrix[sample_index]

    grid = np.arange(quantity_min, quantity_max + quantity_step * 0.5, quantity_step)
    grid = np.unique(np.append(grid, 0.0))
    quantities = np.zeros(matrix.shape[1], dtype=float)
    active = []
    current = sample_base.copy()
    current_score = objective_score(
        current, objective=objective, risk_aversion=risk_aversion,
        tail_weight=tail_weight, minimum_expected_payoff=ev_floor,
    )
    iterations = 0

    for _ in range(max(int(max_legs), 0)):
        best = None
        for leg_index in range(matrix.shape[1]):
            if leg_index in active:
                continue
            for quantity in grid:
                if abs(quantity) < 1e-12:
                    continue
                if np.abs(quantities).sum() + abs(quantity) > max_total_absolute_quantity + 1e-12:
                    continue
                trial = current + sample_matrix[:, leg_index] * quantity
                score = objective_score(
                    trial, objective=objective, risk_aversion=risk_aversion,
                    tail_weight=tail_weight, minimum_expected_payoff=ev_floor,
                )
                if best is None or score > best[0]:
                    best = (score, leg_index, quantity)
        if best is None or best[0] <= current_score + 1e-10:
            break
        current_score, leg_index, quantity = best
        quantities[leg_index] = quantity
        active.append(leg_index)
        current = current + sample_matrix[:, leg_index] * quantity
        iterations += 1

    for _ in range(3):
        improved = False
        for leg_index in active:
            without_leg = current - sample_matrix[:, leg_index] * quantities[leg_index]
            other_abs = np.abs(quantities).sum() - abs(quantities[leg_index])
            best_score, best_quantity = current_score, quantities[leg_index]
            for quantity in grid:
                if other_abs + abs(quantity) > max_total_absolute_quantity + 1e-12:
                    continue
                trial = without_leg + sample_matrix[:, leg_index] * quantity
                score = objective_score(
                    trial, objective=objective, risk_aversion=risk_aversion,
                    tail_weight=tail_weight, minimum_expected_payoff=ev_floor,
                )
                if score > best_score + 1e-10:
                    best_score, best_quantity = score, quantity
            if best_quantity != quantities[leg_index]:
                quantities[leg_index] = best_quantity
                current = without_leg + sample_matrix[:, leg_index] * best_quantity
                current_score = best_score
                improved = True
                iterations += 1
        if not improved:
            break

    optimized_payoffs = base + matrix @ quantities
    return OptimizationResult(
        quantities=quantities,
        selected_legs=selected_quantities_to_legs(candidates, quantities),
        optimized_payoffs=optimized_payoffs,
        baseline_metrics=baseline_metrics,
        optimized_metrics=payoff_metrics(optimized_payoffs),
        objective_score=float(current_score),
        iterations=iterations,
    )
