"""Phase 7: risk assessment and sensitivity engine.

Phase 7 changes no probability or payoff model. It systematically stresses the
inputs of the existing engine and quantifies how fragile the outputs are, so an
edge can be classified as a robust signal or an artifact of assumptions *before*
it is traded. Every simulation reuses :func:`model.run_probability_engine` and
the Phase 4 payoff surface, so Phase 7 measures exactly the numbers the rest of
the app produces.

Implemented stresses (handoff tests 1, 2, 3, 5):

1. Monte Carlo error / multi-seed reruns. Means and probabilities converge
   quickly, but tail metrics (expected shortfall, worst case) rest on few
   scenarios and converge slowly. We rerun across seeds and report the
   dispersion of tail metrics, not just the mean, as ``X +/- Y``.
2. Tail-dependence stress (copula family). Swap only the dependence family
   (Gaussian copula vs Student-t copula df=5, shared chi-square shock) while
   keeping marginals fixed, and compare tail metrics and edge.
3. Gap vs randomness decomposition. Scale every implied volatility by ``k`` and,
   separately, widen/compress the market-cap gaps at fixed volatility, then watch
   how far P(#1) moves. A large move under IV scaling means randomness-dominated
   (IV is the critical input); a large move under gap scaling means
   gap-dominated.
5. Model-uncertainty / robustness. Report the range of P(#1) and edge across a
   grid of plausible models (correlation variants x shock models). An edge that
   survives the grid is tradeable; one that depends on the model choice is not.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from model import run_probability_engine
from payoff_surface import calculate_scenario_payoffs, payoff_summary


# ---------------------------------------------------------------------------
# Portfolio description
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class PortfolioSpec:
    """The real portfolio whose tail metrics Phase 7 stresses.

    Mirrors the inputs Phase 4 already consumes: a set of option legs (with
    quantities, strikes, and theoretical premiums), the current market caps and
    spot prices needed to translate terminal caps into stock prices, and the
    Polymarket position behind the trade.
    """

    option_legs: pd.DataFrame
    current_market_caps: pd.Series
    spot_prices: pd.Series
    selected_ticker: str
    polymarket_side: str = "NO"
    polymarket_entry_price: float = 0.0
    polymarket_quantity: float = 0.0
    contract_multiplier: float = 100.0
    include_option_premiums: bool = True

    def has_position(self) -> bool:
        legs = self.option_legs
        option_active = "Quantity" in legs.columns and bool(
            (pd.to_numeric(legs["Quantity"], errors="coerce").fillna(0.0) != 0.0).any()
        )
        return option_active or float(self.polymarket_quantity) != 0.0


TAIL_METRIC_KEYS = {
    "Expected payoff": "Expected payoff",
    "Expected shortfall 5%": "Expected shortfall 5%",
    "P1 payoff": "P1 payoff",
    "Worst payoff": "Worst payoff",
    "Probability of loss": "Probability of loss",
}


# ---------------------------------------------------------------------------
# Small shared helpers
# ---------------------------------------------------------------------------
def _selected_probability_and_edge(result, selected_ticker: str) -> tuple[float, float]:
    row = result.results.loc[result.results["Ticker"].astype(str) == str(selected_ticker)]
    if row.empty:
        raise ValueError(f"{selected_ticker} is not in the simulation universe.")
    return float(row["Model probability"].iloc[0]), float(row["Edge"].iloc[0])


def portfolio_scenarios(result, portfolio: PortfolioSpec) -> pd.DataFrame:
    """Per-scenario portfolio payoff for a saved simulation, via the Phase 4 engine.

    Shared by Phase 7 tail metrics and Phase 8 risk management so both read the
    exact same payoff distribution.
    """

    legs = portfolio.option_legs.copy()
    if "Quantity" not in legs.columns:
        legs["Quantity"] = 0.0
    active = legs[pd.to_numeric(legs["Quantity"], errors="coerce").fillna(0.0) != 0.0]
    required = [portfolio.selected_ticker]
    if not active.empty and "Ticker" in active.columns:
        required.extend(active["Ticker"].astype(str).tolist())
    required = list(dict.fromkeys(required))

    missing = [ticker for ticker in required if ticker not in result.terminal_market_caps.columns]
    if missing:
        raise ValueError("Missing terminal market-cap scenarios for: " + ", ".join(missing) + ".")

    return calculate_scenario_payoffs(
        result.terminal_market_caps[required],
        result.ranks,
        portfolio.current_market_caps,
        portfolio.spot_prices,
        legs,
        selected_ticker=portfolio.selected_ticker,
        polymarket_side=portfolio.polymarket_side,
        polymarket_entry_price=float(portfolio.polymarket_entry_price),
        polymarket_quantity=float(portfolio.polymarket_quantity),
        contract_multiplier=float(portfolio.contract_multiplier),
        include_option_premiums=bool(portfolio.include_option_premiums),
    )


def _portfolio_tail_summary(result, portfolio: PortfolioSpec, *, shortfall_probability: float = 0.05) -> pd.Series:
    scenario = portfolio_scenarios(result, portfolio)
    return payoff_summary(scenario, shortfall_probability=shortfall_probability)


def constant_correlation(tickers: list[str], rho: float) -> pd.DataFrame:
    """Build a constant off-diagonal correlation matrix for stress variants."""

    size = len(tickers)
    matrix = np.full((size, size), float(rho), dtype=float)
    np.fill_diagonal(matrix, 1.0)
    return pd.DataFrame(matrix, index=tickers, columns=tickers)


# ---------------------------------------------------------------------------
# Test 1 - Monte Carlo error / multi-seed reruns
# ---------------------------------------------------------------------------
def multi_seed_dispersion(
    company_inputs: pd.DataFrame,
    correlation_matrix: pd.DataFrame,
    portfolio: PortfolioSpec | None = None,
    *,
    days_to_target: int,
    simulations: int,
    seeds,
    shock_model: str = "Normal shocks",
    shortfall_probability: float = 0.05,
) -> pd.DataFrame:
    """Rerun the engine over several seeds; return one metric row per seed."""

    seeds = [int(seed) for seed in seeds]
    if not seeds:
        raise ValueError("At least one seed is required.")
    selected_ticker = portfolio.selected_ticker if portfolio is not None else company_inputs["Ticker"].iloc[0]
    rows = []
    for seed in seeds:
        result = run_probability_engine(
            company_inputs,
            correlation_matrix,
            days_to_target=days_to_target,
            simulations=simulations,
            seed=seed,
            shock_model=shock_model,
        )
        prob, edge = _selected_probability_and_edge(result, selected_ticker)
        metrics = {"Seed": seed, "P(#1) selected": prob, "Edge selected": edge}
        if portfolio is not None and portfolio.has_position():
            summary = _portfolio_tail_summary(result, portfolio, shortfall_probability=shortfall_probability)
            for label, key in TAIL_METRIC_KEYS.items():
                metrics[label] = float(summary[key])
        rows.append(metrics)
    return pd.DataFrame(rows)


def dispersion_summary(per_seed: pd.DataFrame) -> pd.DataFrame:
    """Collapse per-seed metrics into mean, cross-seed std, and range.

    The cross-seed standard deviation is the Monte Carlo error to report as the
    ``+/- Y`` around each metric. Relative dispersion (std / |mean|) flags which
    metrics are unstable at the chosen simulation count.
    """

    metric_columns = [column for column in per_seed.columns if column != "Seed"]
    rows = []
    for column in metric_columns:
        values = per_seed[column].astype(float)
        mean = float(values.mean())
        std = float(values.std(ddof=1)) if len(values) > 1 else 0.0
        rows.append(
            {
                "Metric": column,
                "Mean": mean,
                "MC error (std)": std,
                "Relative dispersion": (std / abs(mean)) if mean != 0 else np.nan,
                "Min": float(values.min()),
                "Max": float(values.max()),
                "Seeds": int(len(values)),
            }
        )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Test 2 - Tail-dependence stress (copula family)
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class CopulaStressResult:
    comparison: pd.DataFrame
    baseline_per_seed: pd.DataFrame
    stress_per_seed: pd.DataFrame
    baseline_model: str
    stress_model: str


def copula_tail_stress(
    company_inputs: pd.DataFrame,
    correlation_matrix: pd.DataFrame,
    portfolio: PortfolioSpec | None = None,
    *,
    days_to_target: int,
    simulations: int,
    seeds,
    baseline_model: str = "Normal shocks",
    stress_model: str = "Student-t copula df=5",
    shortfall_probability: float = 0.05,
) -> CopulaStressResult:
    """Compare tail metrics under a Gaussian copula vs a Student-t copula.

    Same seeds, same simulations, same marginals - only the dependence family
    changes, so any difference is pure tail dependence.
    """

    baseline = multi_seed_dispersion(
        company_inputs, correlation_matrix, portfolio,
        days_to_target=days_to_target, simulations=simulations, seeds=seeds,
        shock_model=baseline_model, shortfall_probability=shortfall_probability,
    )
    stress = multi_seed_dispersion(
        company_inputs, correlation_matrix, portfolio,
        days_to_target=days_to_target, simulations=simulations, seeds=seeds,
        shock_model=stress_model, shortfall_probability=shortfall_probability,
    )
    metric_columns = [column for column in baseline.columns if column != "Seed"]
    base_mean = baseline[metric_columns].astype(float).mean()
    stress_mean = stress[metric_columns].astype(float).mean()
    comparison = pd.DataFrame(
        {
            "Metric": metric_columns,
            baseline_model: base_mean.to_numpy(),
            stress_model: stress_mean.to_numpy(),
        }
    )
    comparison["Change"] = comparison[stress_model] - comparison[baseline_model]
    denom = comparison[baseline_model].abs().replace(0.0, np.nan)
    comparison["Change %"] = comparison["Change"] / denom
    return CopulaStressResult(
        comparison=comparison,
        baseline_per_seed=baseline,
        stress_per_seed=stress,
        baseline_model=baseline_model,
        stress_model=stress_model,
    )


# ---------------------------------------------------------------------------
# Test 3 - Gap vs randomness decomposition
# ---------------------------------------------------------------------------
def _probability_by_ticker(result) -> pd.Series:
    return result.results.set_index("Ticker")["Model probability"].astype(float)


def iv_scaling_scan(
    company_inputs: pd.DataFrame,
    correlation_matrix: pd.DataFrame,
    *,
    selected_ticker: str,
    days_to_target: int,
    simulations: int,
    seed: int,
    factors,
    shock_model: str = "Normal shocks",
) -> pd.DataFrame:
    """Scale every implied volatility by ``k`` and track P(#1).

    A large swing in P(#1) means the outcome is randomness-dominated and the IV
    level is the critical lever.
    """

    base = company_inputs.copy()
    base_iv = base["Implied volatility"].astype(float)
    tickers = base["Ticker"].astype(str).tolist()
    rows = []
    for factor in factors:
        scaled = base.copy()
        scaled["Implied volatility"] = base_iv * float(factor)
        result = run_probability_engine(
            scaled, correlation_matrix,
            days_to_target=days_to_target, simulations=simulations, seed=int(seed),
            shock_model=shock_model,
        )
        probs = _probability_by_ticker(result)
        row = {"IV scale": float(factor), "P(#1) selected": float(probs.get(selected_ticker, np.nan))}
        for ticker in tickers:
            row[f"P(#1) {ticker}"] = float(probs.get(ticker, np.nan))
        rows.append(row)
    return pd.DataFrame(rows)


def gap_scaling_scan(
    company_inputs: pd.DataFrame,
    correlation_matrix: pd.DataFrame,
    *,
    selected_ticker: str,
    days_to_target: int,
    simulations: int,
    seed: int,
    factors,
    shock_model: str = "Normal shocks",
) -> pd.DataFrame:
    """Widen or compress the market-cap gaps at fixed volatility, track P(#1).

    Caps are re-scaled around the largest name: ``cap_i -> ref * (cap_i / ref)**g``.
    ``g = 1`` leaves caps unchanged, ``g > 1`` widens the gaps (the leader pulls
    ahead), ``g < 1`` compresses them toward the top. A large swing here means
    the outcome is gap-dominated (structural).
    """

    base = company_inputs.copy()
    caps = base["Current market cap"].astype(float).to_numpy()
    reference = float(caps.max())
    tickers = base["Ticker"].astype(str).tolist()
    rows = []
    for factor in factors:
        scaled = base.copy()
        scaled["Current market cap"] = reference * np.power(caps / reference, float(factor))
        result = run_probability_engine(
            scaled, correlation_matrix,
            days_to_target=days_to_target, simulations=simulations, seed=int(seed),
            shock_model=shock_model,
        )
        probs = _probability_by_ticker(result)
        row = {"Gap scale": float(factor), "P(#1) selected": float(probs.get(selected_ticker, np.nan))}
        for ticker in tickers:
            row[f"P(#1) {ticker}"] = float(probs.get(ticker, np.nan))
        rows.append(row)
    return pd.DataFrame(rows)


def scan_sensitivity(scan: pd.DataFrame, value_column: str = "P(#1) selected") -> dict:
    """Range (max - min) of a scanned probability, used to rank the levers."""

    values = scan[value_column].astype(float)
    return {
        "min": float(values.min()),
        "max": float(values.max()),
        "range": float(values.max() - values.min()),
    }


def gap_vs_randomness(iv_scan: pd.DataFrame, gap_scan: pd.DataFrame, *, value_column: str = "P(#1) selected") -> pd.DataFrame:
    """Side-by-side lever ranking: which scan moves P(#1) more."""

    iv = scan_sensitivity(iv_scan, value_column)
    gap = scan_sensitivity(gap_scan, value_column)
    verdict = "randomness-dominated (IV lever)" if iv["range"] >= gap["range"] else "gap-dominated (structural)"
    return pd.DataFrame(
        [
            {"Lever": "IV scaling", "P(#1) min": iv["min"], "P(#1) max": iv["max"], "P(#1) range": iv["range"]},
            {"Lever": "Gap scaling", "P(#1) min": gap["min"], "P(#1) max": gap["max"], "P(#1) range": gap["range"]},
        ]
    ).assign(Verdict=verdict)


# ---------------------------------------------------------------------------
# Test 5 - Model-uncertainty / robustness
# ---------------------------------------------------------------------------
def model_robustness(
    company_inputs: pd.DataFrame,
    correlation_variants: dict[str, pd.DataFrame],
    *,
    selected_ticker: str,
    days_to_target: int,
    simulations: int,
    seed: int,
    shock_models,
) -> pd.DataFrame:
    """Grid over correlation variants x shock models; P(#1) and edge per cell."""

    if not correlation_variants:
        raise ValueError("At least one correlation variant is required.")
    shock_models = list(shock_models)
    if not shock_models:
        raise ValueError("At least one shock model is required.")
    rows = []
    for corr_name, corr in correlation_variants.items():
        for shock_model in shock_models:
            result = run_probability_engine(
                company_inputs, corr,
                days_to_target=days_to_target, simulations=simulations, seed=int(seed),
                shock_model=shock_model,
            )
            prob, edge = _selected_probability_and_edge(result, selected_ticker)
            rows.append(
                {
                    "Correlation": corr_name,
                    "Shock model": shock_model,
                    "P(#1) selected": prob,
                    "Edge selected": edge,
                }
            )
    return pd.DataFrame(rows)


def robustness_summary(grid: pd.DataFrame) -> pd.Series:
    """Collapse the robustness grid into a tradeability verdict."""

    prob = grid["P(#1) selected"].astype(float)
    edge = grid["Edge selected"].astype(float)
    edge_sign_consistent = bool((edge > 0).all() or (edge < 0).all())
    return pd.Series(
        {
            "P(#1) min": float(prob.min()),
            "P(#1) max": float(prob.max()),
            "P(#1) spread": float(prob.max() - prob.min()),
            "Edge min": float(edge.min()),
            "Edge max": float(edge.max()),
            "Edge spread": float(edge.max() - edge.min()),
            "Edge sign consistent": edge_sign_consistent,
            "Models": int(len(grid)),
        }
    )
