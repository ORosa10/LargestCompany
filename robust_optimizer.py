"""Transparent payoff-floor optimizer and diagnostics for Phase 5."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

from optimization import selected_quantities_to_legs


@dataclass
class RobustOptimizationResult:
    quantities: np.ndarray
    selected_legs: pd.DataFrame
    payoffs: np.ndarray
    profile: pd.DataFrame
    worst_bin_p5: float
    profile_flatness: float
    expected_payoff: float
    iterations: int


def price_bin_profile(
    terminal_prices: np.ndarray,
    payoffs: np.ndarray,
    *,
    bin_width: float = 5.0,
) -> pd.DataFrame:
    """Summarize payoff in fixed terminal-price intervals; today is 100."""
    prices = np.asarray(terminal_prices, dtype=float)
    values = np.asarray(payoffs, dtype=float)
    core_low = np.floor(np.quantile(prices, 0.01) / bin_width) * bin_width
    core_high = np.ceil(np.quantile(prices, 0.99) / bin_width) * bin_width
    finite_edges = np.arange(core_low, core_high + bin_width * 0.5, bin_width)
    edges = np.concatenate(([-np.inf], finite_edges, [np.inf]))
    frame = pd.DataFrame({"Terminal price": prices, "Payoff": values})
    frame["Price bin"] = pd.cut(frame["Terminal price"], bins=edges, include_lowest=True)
    rows = []
    total = len(frame)
    for interval, group in frame.groupby("Price bin", observed=True):
        bin_values = group["Payoff"].to_numpy(dtype=float)
        lower = float(group["Terminal price"].min()) if not np.isfinite(interval.left) else float(interval.left)
        upper = float(group["Terminal price"].max()) if not np.isfinite(interval.right) else float(interval.right)
        if not np.isfinite(interval.left):
            label = f"<{interval.right:.0f}%"
        elif not np.isfinite(interval.right):
            label = f">={interval.left:.0f}%"
        else:
            label = f"{interval.left:.0f}-{interval.right:.0f}%"
        probability = len(group) / total
        expected = float(bin_values.mean())
        rows.append(
            {
                "Price bin": label,
                "Price midpoint": float(group["Terminal price"].mean()),
                "Price lower": lower,
                "Price upper": upper,
                "Scenario probability": probability,
                "Expected payoff": expected,
                "Payoff SD": float(bin_values.std(ddof=0)),
                "Payoff P1": float(np.quantile(bin_values, 0.01)),
                "Payoff P5": float(np.quantile(bin_values, 0.05)),
                "Contribution to EV": probability * expected,
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


def _equal_count_bin_statistics(
    terminal_prices: np.ndarray,
    payoffs: np.ndarray,
    bins: int,
) -> tuple[float, float]:
    """Return worst conditional P5 and SD of conditional means."""
    order = np.argsort(terminal_prices)
    chunks = [index for index in np.array_split(order, bins) if len(index)]
    conditional_p5 = np.array([np.quantile(payoffs[index], 0.05) for index in chunks])
    conditional_means = np.array([payoffs[index].mean() for index in chunks])
    return float(conditional_p5.min()), float(conditional_means.std(ddof=0))


def _objective_components(
    terminal_prices: np.ndarray,
    payoffs: np.ndarray,
    *,
    minimum_ev: float,
    bins: int,
) -> tuple[float, float, float]:
    expected_payoff = float(payoffs.mean())
    if expected_payoff < minimum_ev:
        return -np.inf, np.inf, expected_payoff
    worst_bin_p5, flatness = _equal_count_bin_statistics(terminal_prices, payoffs, bins)
    return worst_bin_p5, flatness, expected_payoff


def _is_better(candidate: tuple[float, float, float], current: tuple[float, float, float]) -> bool:
    """Lexicographic comparison: maximize floor, then minimize flatness."""
    candidate_floor, candidate_flatness, _ = candidate
    current_floor, current_flatness, _ = current
    if candidate_floor > current_floor + 1e-10:
        return True
    return abs(candidate_floor - current_floor) <= 1e-10 and candidate_flatness < current_flatness - 1e-10


def _duplicate_family(candidates: pd.DataFrame, quantities: np.ndarray, index: int, quantity: float) -> bool:
    if abs(quantity) < 1e-12:
        return False
    candidate = candidates.iloc[index]
    for active in np.flatnonzero(np.abs(quantities) > 1e-12):
        row = candidates.iloc[active]
        if str(row["Ticker"]) == str(candidate["Ticker"]) and str(row["Option type"]) == str(candidate["Option type"]) and np.sign(quantities[active]) == np.sign(quantity):
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
    profile_bins: int,
    price_bin_width: float,
    optimization_scenarios: int,
    seed: int,
) -> RobustOptimizationResult:
    base = np.asarray(base_payoff, dtype=float)
    matrix = np.asarray(option_payoff_matrix, dtype=float)
    prices = np.asarray(terminal_prices, dtype=float)
    rng = np.random.default_rng(seed)
    sample_size = min(int(optimization_scenarios), len(base))
    sample_index = rng.choice(len(base), sample_size, replace=False) if sample_size < len(base) else np.arange(len(base))
    sample_base, sample_matrix, sample_prices = base[sample_index], matrix[sample_index], prices[sample_index]
    grid = np.unique(np.append(np.arange(quantity_min, quantity_max + 0.5 * quantity_step, quantity_step), 0.0))
    quantities = np.zeros(matrix.shape[1], dtype=float)
    active: list[int] = []
    current = sample_base.copy()
    current_components = _objective_components(sample_prices, current, minimum_ev=minimum_ev, bins=profile_bins)
    iterations = 0

    for _ in range(max(int(max_legs), 0)):
        best = None
        for leg_index in range(matrix.shape[1]):
            if leg_index in active:
                continue
            for quantity in grid:
                if abs(quantity) < 1e-12 or np.abs(quantities).sum() + abs(quantity) > max_total_quantity + 1e-12:
                    continue
                if _duplicate_family(candidates, quantities, leg_index, quantity):
                    continue
                trial = current + sample_matrix[:, leg_index] * quantity
                components = _objective_components(sample_prices, trial, minimum_ev=minimum_ev, bins=profile_bins)
                if best is None or _is_better(components, best[0]):
                    best = (components, leg_index, quantity)
        if best is None or not _is_better(best[0], current_components):
            break
        current_components, leg_index, quantity = best
        quantities[leg_index] = quantity
        active.append(leg_index)
        current += sample_matrix[:, leg_index] * quantity
        iterations += 1

    full_payoff = base + matrix @ quantities
    full_floor, full_flatness = _equal_count_bin_statistics(prices, full_payoff, profile_bins)
    return RobustOptimizationResult(
        quantities=quantities,
        selected_legs=selected_quantities_to_legs(candidates, quantities),
        payoffs=full_payoff,
        profile=price_bin_profile(prices, full_payoff, bin_width=price_bin_width),
        worst_bin_p5=full_floor,
        profile_flatness=full_flatness,
        expected_payoff=float(full_payoff.mean()),
        iterations=iterations,
    )


def robust_metrics_table(base_payoff: np.ndarray, result: RobustOptimizationResult) -> pd.DataFrame:
    rows = []
    for name, payoff in [("Polymarket only", base_payoff), ("Optimizer 2", result.payoffs)]:
        metrics = robust_metrics(np.asarray(payoff, dtype=float))
        rows.append({"Portfolio": name, "Expected payoff": f"${metrics['Expected payoff']:,.2f}", "Payoff SD": f"${metrics['Payoff SD']:,.2f}", "VaR 5% payoff": f"${metrics['VaR 5% payoff']:,.2f}", "VaR 1% payoff": f"${metrics['VaR 1% payoff']:,.2f}", "ES 5%": f"${metrics['ES 5%']:,.2f}", "ES 1%": f"${metrics['ES 1%']:,.2f}", "P(loss)": f"{metrics['P(loss)']:.2%}", "Worst payoff": f"${metrics['Worst payoff']:,.2f}"})
    return pd.DataFrame(rows)


def objective_audit_table(base_payoff: np.ndarray, terminal_prices: np.ndarray, result: RobustOptimizationResult, bins: int, minimum_ev: float) -> pd.DataFrame:
    base_floor, base_flatness = _equal_count_bin_statistics(terminal_prices, np.asarray(base_payoff, dtype=float), bins)
    return pd.DataFrame(
        [
            {"Portfolio": "Polymarket only", "EV floor constraint": f"${minimum_ev:,.2f}", "Expected payoff": f"${np.mean(base_payoff):,.2f}", "Worst conditional P5": f"${base_floor:,.2f}", "Profile flatness": f"${base_flatness:,.2f}"},
            {"Portfolio": "Optimizer 2", "EV floor constraint": f"${minimum_ev:,.2f}", "Expected payoff": f"${result.expected_payoff:,.2f}", "Worst conditional P5": f"${result.worst_bin_p5:,.2f}", "Profile flatness": f"${result.profile_flatness:,.2f}"},
        ]
    )


def aligned_profile_figure(base_profile: pd.DataFrame, optimized_profile: pd.DataFrame) -> go.Figure:
    labels = optimized_profile["Price bin"]
    probability_text = optimized_profile["Scenario probability"].map(lambda value: f"{value:.1%}")
    colors = np.where(optimized_profile["Expected payoff"] >= 0, "#16a34a", "#dc2626")
    figure = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.08, row_heights=[0.72, 0.28])
    figure.add_trace(go.Scatter(x=labels, y=base_profile["Expected payoff"], name="Polymarket-only expected payoff", mode="lines", line=dict(color="#94a3b8", dash="dash", width=2)), row=1, col=1)
    figure.add_trace(go.Bar(x=labels, y=optimized_profile["Expected payoff"], name="Optimizer 2 expected payoff", marker_color=colors), row=1, col=1)
    figure.add_trace(go.Scatter(x=labels, y=optimized_profile["Payoff P5"], name="Optimizer 2 P5", mode="lines+markers", line=dict(color="#f59e0b", dash="dash")), row=1, col=1)
    figure.add_trace(go.Scatter(x=labels, y=optimized_profile["Payoff P1"], name="Optimizer 2 P1", mode="lines+markers", line=dict(color="#dc2626", dash="dot")), row=1, col=1)
    figure.add_trace(go.Bar(x=labels, y=optimized_profile["Scenario probability"], text=probability_text, textposition="outside", name="Scenario probability", marker_color="#60a5fa"), row=2, col=1)
    figure.add_hline(y=0, line_dash="dash", line_color="black", row=1, col=1)
    figure.update_yaxes(title_text="Payoff", row=1, col=1)
    figure.update_yaxes(title_text="Probability", tickformat=".1%", row=2, col=1)
    figure.update_xaxes(title_text="Terminal stock price / current price", tickangle=-45, row=2, col=1)
    figure.update_layout(title="Payoff and scenario probability aligned by terminal-price bin", height=760, barmode="overlay", legend=dict(orientation="h", yanchor="bottom", y=1.02), margin=dict(l=50, r=30, t=100, b=100))
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
    st.subheader("Optimizer 2: conditional payoff floor")
    st.markdown("**Objective (lexicographic):** (1) require `Expected payoff >= minimum EV`; (2) maximize the worst conditional P5 across equal-count terminal-price bins; (3) if tied, minimize SD of conditional mean payoffs across those bins.")
    if candidates is None or option_payoff_matrix is None:
        st.info("Configure a valid candidate universe in the classic Optimizer controls first.")
        return
    controls = st.columns(3)
    minimum_ev = controls[0].number_input("Minimum expected payoff", value=float(max(default_minimum_ev, 0.0)), step=1.0)
    profile_bins = controls[1].number_input("Optimization bins (equal scenario count)", min_value=10, max_value=50, value=20, step=5, help="More bins detect finer local problems but add Monte Carlo noise.")
    price_bin_width = controls[2].number_input("Displayed price-bin width (%)", min_value=2.5, max_value=20.0, value=5.0, step=2.5, help="Display only; does not change the optimization objective.")

    if st.button("Update Optimizer 2", type="primary"):
        with st.spinner("Maximizing the worst conditional P5 on stored scenarios..."):
            try:
                result = optimize_payoff_floor(base_payoff, option_payoff_matrix, candidates, terminal_prices, quantity_min=quantity_min, quantity_max=quantity_max, quantity_step=quantity_step, max_legs=max_legs, max_total_quantity=max_total_quantity, minimum_ev=float(minimum_ev), profile_bins=int(profile_bins), price_bin_width=float(price_bin_width), optimization_scenarios=optimization_scenarios, seed=seed)
                st.session_state.phase5_robust_optimization = result
                st.session_state.phase5_robust_error = None
            except Exception as exc:
                st.session_state.phase5_robust_error = str(exc)
    if st.session_state.get("phase5_robust_error"):
        st.error(st.session_state.phase5_robust_error)
    result = st.session_state.get("phase5_robust_optimization")
    if result is None:
        st.info("Run Optimizer 2 to create the conditional-floor comparison.")
        return

    base_profile = price_bin_profile(np.asarray(terminal_prices, dtype=float), np.asarray(base_payoff, dtype=float), bin_width=float(price_bin_width))
    st.subheader("Objective audit")
    st.dataframe(objective_audit_table(base_payoff, np.asarray(terminal_prices, dtype=float), result, int(profile_bins), float(minimum_ev)), use_container_width=True, hide_index=True)
    st.subheader("Global payoff metrics")
    st.dataframe(robust_metrics_table(base_payoff, result), use_container_width=True, hide_index=True)
    st.subheader("Selected option legs")
    st.dataframe(result.selected_legs, use_container_width=True, hide_index=True)
    st.plotly_chart(aligned_profile_figure(base_profile, result.profile), use_container_width=True, key="optimizer2_aligned_profile")
    with st.expander("Show terminal-price bin statistics"):
        display = result.profile.copy()
        display["Scenario probability"] = display["Scenario probability"].map(lambda value: f"{value:.2%}")
        st.dataframe(display, use_container_width=True, hide_index=True)
