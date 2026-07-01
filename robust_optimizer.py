"""Robust payoff-floor optimizer and diagnostics for Phase 5."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from optimization import payoff_metrics, selected_quantities_to_legs


@dataclass
class RobustOptimizationResult:
    quantities: np.ndarray
    selected_legs: pd.DataFrame
    payoffs: np.ndarray
    profile: pd.DataFrame
    score: float
    iterations: int


def price_bin_profile(
    terminal_prices: np.ndarray,
    payoffs: np.ndarray,
    *,
    bins: int = 20,
) -> pd.DataFrame:
    frame = pd.DataFrame(
        {
            "Terminal price": np.asarray(terminal_prices, dtype=float),
            "Payoff": np.asarray(payoffs, dtype=float),
        }
    )
    frame["Price bin"] = pd.qcut(frame["Terminal price"], q=bins, duplicates="drop")
    rows = []
    total = len(frame)
    for interval, group in frame.groupby("Price bin", observed=True):
        values = group["Payoff"].to_numpy(dtype=float)
        probability = len(group) / total
        rows.append(
            {
                "Price bin": str(interval),
                "Price midpoint": float(group["Terminal price"].mean()),
                "Price lower": float(interval.left),
                "Price upper": float(interval.right),
                "Scenario probability": probability,
                "Expected payoff": float(values.mean()),
                "Payoff SD": float(values.std(ddof=0)),
                "Payoff P1": float(np.quantile(values, 0.01)),
                "Payoff P5": float(np.quantile(values, 0.05)),
                "Contribution to EV": probability * float(values.mean()),
            }
        )
    return pd.DataFrame(rows)


def robust_metrics(payoffs: np.ndarray) -> pd.Series:
    values = np.asarray(payoffs, dtype=float)
    q01 = float(np.quantile(values, 0.01))
    q05 = float(np.quantile(values, 0.05))
    return pd.Series(
        {
            "Expected payoff": float(values.mean()),
            "Payoff SD": float(values.std(ddof=0)),
            "VaR 5% payoff": q05,
            "VaR 1% payoff": q01,
            "ES 5%": float(values[values <= q05].mean()),
            "ES 1%": float(values[values <= q01].mean()),
            "P(loss)": float((values < 0).mean()),
            "Worst payoff": float(values.min()),
        }
    )


def _profile_flatness(terminal_prices: np.ndarray, payoffs: np.ndarray, bins: int) -> float:
    order = np.argsort(terminal_prices)
    chunks = np.array_split(order, bins)
    means = np.array([payoffs[index].mean() for index in chunks if len(index)], dtype=float)
    return float(means.std(ddof=0))


def _score(
    terminal_prices: np.ndarray,
    payoffs: np.ndarray,
    *,
    minimum_ev: float,
    flatness_penalty: float,
    bins: int,
) -> float:
    mean = float(payoffs.mean())
    if mean < minimum_ev:
        return -np.inf
    payoff_p1 = float(np.quantile(payoffs, 0.01))
    payoff_p5 = float(np.quantile(payoffs, 0.05))
    flatness = _profile_flatness(terminal_prices, payoffs, bins)
    return 0.65 * payoff_p5 + 0.35 * payoff_p1 - flatness_penalty * flatness


def _duplicate_family(
    candidates: pd.DataFrame,
    quantities: np.ndarray,
    index: int,
    quantity: float,
) -> bool:
    if abs(quantity) < 1e-12:
        return False
    candidate = candidates.iloc[index]
    for active in np.flatnonzero(np.abs(quantities) > 1e-12):
        row = candidates.iloc[active]
        if (
            str(row["Ticker"]) == str(candidate["Ticker"])
            and str(row["Option type"]) == str(candidate["Option type"])
            and np.sign(quantities[active]) == np.sign(quantity)
        ):
            return True
    return False


def optimize_payoff_floor(
    base_payoff: np.ndarray,
    option_payoff_matrix: np.ndarray,
    candidates: pd.DataFrame,
    terminal_prices: np.ndarray,
    *,
    quantity_min: float,
    quantity_max: float,
    quantity_step: float,
    max_legs: int,
    max_total_quantity: float,
    minimum_ev: float,
    flatness_penalty: float,
    profile_bins: int,
    optimization_scenarios: int,
    seed: int,
) -> RobustOptimizationResult:
    base = np.asarray(base_payoff, dtype=float)
    matrix = np.asarray(option_payoff_matrix, dtype=float)
    prices = np.asarray(terminal_prices, dtype=float)
    rng = np.random.default_rng(seed)
    sample_size = min(int(optimization_scenarios), len(base))
    sample_index = rng.choice(len(base), sample_size, replace=False) if sample_size < len(base) else np.arange(len(base))
    sample_base = base[sample_index]
    sample_matrix = matrix[sample_index]
    sample_prices = prices[sample_index]

    grid = np.arange(quantity_min, quantity_max + 0.5 * quantity_step, quantity_step)
    grid = np.unique(np.append(grid, 0.0))
    quantities = np.zeros(matrix.shape[1], dtype=float)
    active: list[int] = []
    current = sample_base.copy()
    current_score = _score(sample_prices, current, minimum_ev=minimum_ev, flatness_penalty=flatness_penalty, bins=profile_bins)
    iterations = 0

    for _ in range(max(int(max_legs), 0)):
        best = None
        for leg_index in range(matrix.shape[1]):
            if leg_index in active:
                continue
            for quantity in grid:
                if abs(quantity) < 1e-12:
                    continue
                if np.abs(quantities).sum() + abs(quantity) > max_total_quantity + 1e-12:
                    continue
                if _duplicate_family(candidates, quantities, leg_index, quantity):
                    continue
                trial = current + sample_matrix[:, leg_index] * quantity
                score = _score(sample_prices, trial, minimum_ev=minimum_ev, flatness_penalty=flatness_penalty, bins=profile_bins)
                if best is None or score > best[0]:
                    best = (score, leg_index, quantity)
        if best is None or best[0] <= current_score + 1e-10:
            break
        current_score, leg_index, quantity = best
        quantities[leg_index] = quantity
        active.append(leg_index)
        current += sample_matrix[:, leg_index] * quantity
        iterations += 1

    full_payoff = base + matrix @ quantities
    return RobustOptimizationResult(
        quantities=quantities,
        selected_legs=selected_quantities_to_legs(candidates, quantities),
        payoffs=full_payoff,
        profile=price_bin_profile(prices, full_payoff, bins=profile_bins),
        score=float(current_score),
        iterations=iterations,
    )


def robust_metrics_table(base_payoff: np.ndarray, result: RobustOptimizationResult) -> pd.DataFrame:
    rows = []
    for name, payoff in [("Polymarket only", base_payoff), ("Optimizer 2", result.payoffs)]:
        metrics = robust_metrics(np.asarray(payoff, dtype=float))
        rows.append(
            {
                "Portfolio": name,
                "Expected payoff": f"${metrics['Expected payoff']:,.2f}",
                "Payoff SD": f"${metrics['Payoff SD']:,.2f}",
                "VaR 5% payoff": f"${metrics['VaR 5% payoff']:,.2f}",
                "VaR 1% payoff": f"${metrics['VaR 1% payoff']:,.2f}",
                "ES 5%": f"${metrics['ES 5%']:,.2f}",
                "ES 1%": f"${metrics['ES 1%']:,.2f}",
                "P(loss)": f"{metrics['P(loss)']:.2%}",
                "Worst payoff": f"${metrics['Worst payoff']:,.2f}",
            }
        )
    return pd.DataFrame(rows)


def payoff_profile_figure(profile: pd.DataFrame) -> go.Figure:
    x = profile["Price midpoint"]
    figure = go.Figure()
    figure.add_trace(go.Scatter(x=x, y=profile["Expected payoff"], name="Expected payoff", mode="lines+markers", line=dict(width=3)))
    figure.add_trace(go.Scatter(x=x, y=profile["Payoff P5"], name="Payoff P5", mode="lines", line=dict(dash="dash")))
    figure.add_trace(go.Scatter(x=x, y=profile["Payoff P1"], name="Payoff P1", mode="lines", line=dict(dash="dot")))
    figure.add_hline(y=0, line_dash="dash", line_color="black")
    figure.update_layout(title="Conditional payoff profile by terminal price", xaxis_title="Terminal stock price (today = 100)", yaxis_title="Payoff", height=480, legend=dict(orientation="h"))
    return figure


def scenario_density_figure(profile: pd.DataFrame) -> go.Figure:
    figure = go.Figure(go.Bar(x=profile["Price midpoint"], y=profile["Scenario probability"], name="Scenario probability"))
    figure.update_layout(title="Scenario probability by the same terminal-price bins", xaxis_title="Terminal stock price (today = 100)", yaxis_title="Probability", yaxis_tickformat=".1%", height=300)
    return figure


def render_robust_optimizer(
    *,
    base_payoff: np.ndarray,
    option_payoff_matrix: np.ndarray | None,
    candidates: pd.DataFrame | None,
    terminal_prices: np.ndarray,
    quantity_min: float,
    quantity_max: float,
    quantity_step: float,
    max_legs: int,
    max_total_quantity: float,
    default_minimum_ev: float,
    optimization_scenarios: int,
    seed: int,
) -> None:
    st.subheader("Optimizer 2: payoff floor and flatness")
    st.caption("This optimizer raises the 1%/5% payoff floor and penalizes hills in expected payoff across terminal-price bins. It does not replace the classic optimizer.")
    if candidates is None or option_payoff_matrix is None:
        st.info("Configure a valid candidate universe in the classic Optimizer controls first.")
        return

    controls = st.columns(3)
    minimum_ev = controls[0].number_input("Minimum expected payoff", value=float(max(default_minimum_ev, 0.0)), step=1.0)
    flatness_penalty = controls[1].number_input("Profile flatness penalty", min_value=0.0, value=0.35, step=0.05)
    profile_bins = controls[2].number_input("Terminal-price bins", min_value=10, max_value=50, value=20, step=5)

    if st.button("Update Optimizer 2", type="primary"):
        with st.spinner("Searching for a flatter payoff profile on stored scenarios..."):
            try:
                result = optimize_payoff_floor(
                    base_payoff,
                    option_payoff_matrix,
                    candidates,
                    terminal_prices,
                    quantity_min=quantity_min,
                    quantity_max=quantity_max,
                    quantity_step=quantity_step,
                    max_legs=max_legs,
                    max_total_quantity=max_total_quantity,
                    minimum_ev=float(minimum_ev),
                    flatness_penalty=float(flatness_penalty),
                    profile_bins=int(profile_bins),
                    optimization_scenarios=optimization_scenarios,
                    seed=seed,
                )
                st.session_state.phase5_robust_optimization = result
                st.session_state.phase5_robust_error = None
            except Exception as exc:
                st.session_state.phase5_robust_error = str(exc)

    if st.session_state.get("phase5_robust_error"):
        st.error(st.session_state.phase5_robust_error)
    result = st.session_state.get("phase5_robust_optimization")
    if result is None:
        st.info("Run Optimizer 2 to create the robust comparison.")
        return

    st.dataframe(robust_metrics_table(base_payoff, result), use_container_width=True, hide_index=True)
    st.subheader("Selected option legs")
    st.dataframe(result.selected_legs, use_container_width=True, hide_index=True)
    st.plotly_chart(payoff_profile_figure(result.profile), use_container_width=True, key="optimizer2_profile")
    st.plotly_chart(scenario_density_figure(result.profile), use_container_width=True, key="optimizer2_density")
    with st.expander("Show terminal-price bin statistics"):
        st.dataframe(result.profile, use_container_width=True, hide_index=True)
