"""Transparent payoff-floor optimizer and diagnostics for Phase 5."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

from optimization import selected_quantities_to_legs


FLOOR_METRICS = ["Conditional P5", "Conditional mean"]


@dataclass
class RobustOptimizationResult:
    quantities: np.ndarray
    selected_legs: pd.DataFrame
    payoffs: np.ndarray
    profile: pd.DataFrame
    objective_metric: str
    objective_floor: float
    worst_bin_p5: float
    worst_bin_mean: float
    profile_flatness: float
    expected_payoff: float
    expected_shortfall_5: float
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
    frame["Price bin"] = pd.cut(
        frame["Terminal price"], bins=edges, include_lowest=True
    )
    rows = []
    total = len(frame)
    for interval, group in frame.groupby("Price bin", observed=True):
        bin_values = group["Payoff"].to_numpy(dtype=float)
        lower = (
            float(group["Terminal price"].min())
            if not np.isfinite(interval.left)
            else float(interval.left)
        )
        upper = (
            float(group["Terminal price"].max())
            if not np.isfinite(interval.right)
            else float(interval.right)
        )
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


def _expected_shortfall_5(payoffs: np.ndarray) -> float:
    values = np.asarray(payoffs, dtype=float)
    threshold = float(np.quantile(values, 0.05))
    return float(values[values <= threshold].mean())


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
) -> dict[str, float]:
    """Return conditional floors and dispersion using equally likely bins."""
    order = np.argsort(terminal_prices)
    chunks = [index for index in np.array_split(order, bins) if len(index)]
    conditional_p5 = np.array(
        [np.quantile(payoffs[index], 0.05) for index in chunks]
    )
    conditional_means = np.array([payoffs[index].mean() for index in chunks])
    return {
        "Worst conditional P5": float(conditional_p5.min()),
        "Worst conditional mean": float(conditional_means.min()),
        "Profile flatness": float(conditional_means.std(ddof=0)),
    }


def _objective_components(
    terminal_prices: np.ndarray,
    payoffs: np.ndarray,
    *,
    minimum_ev: float,
    minimum_es5: float,
    bins: int,
    floor_metric: str,
) -> tuple[float, float, float, float]:
    expected_payoff = float(payoffs.mean())
    expected_shortfall = _expected_shortfall_5(payoffs)
    if expected_payoff < minimum_ev or expected_shortfall < minimum_es5:
        return -np.inf, np.inf, expected_payoff, expected_shortfall
    statistics = _equal_count_bin_statistics(terminal_prices, payoffs, bins)
    objective_floor = float(statistics[f"Worst {floor_metric.lower()}"])
    return (
        objective_floor,
        statistics["Profile flatness"],
        expected_payoff,
        expected_shortfall,
    )


def _is_better(
    candidate: tuple[float, float, float, float],
    current: tuple[float, float, float, float],
) -> bool:
    """Lexicographic comparison: maximize floor, then minimize flatness."""
    candidate_floor, candidate_flatness, _, _ = candidate
    current_floor, current_flatness, _, _ = current
    if candidate_floor > current_floor + 1e-10:
        return True
    return (
        abs(candidate_floor - current_floor) <= 1e-10
        and candidate_flatness < current_flatness - 1e-10
    )


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
    minimum_es5: float,
    floor_metric: str,
    profile_bins: int,
    price_bin_width: float,
    optimization_scenarios: int,
    seed: int,
) -> RobustOptimizationResult:
    """Greedily select legs, then refine all active quantities."""
    if floor_metric not in FLOOR_METRICS:
        raise ValueError(f"floor_metric must be one of {FLOOR_METRICS}.")
    base = np.asarray(base_payoff, dtype=float)
    matrix = np.asarray(option_payoff_matrix, dtype=float)
    prices = np.asarray(terminal_prices, dtype=float)
    rng = np.random.default_rng(seed)
    sample_size = min(int(optimization_scenarios), len(base))
    sample_index = (
        rng.choice(len(base), sample_size, replace=False)
        if sample_size < len(base)
        else np.arange(len(base))
    )
    sample_base = base[sample_index]
    sample_matrix = matrix[sample_index]
    sample_prices = prices[sample_index]
    grid = np.unique(
        np.append(
            np.arange(
                quantity_min,
                quantity_max + 0.5 * quantity_step,
                quantity_step,
            ),
            0.0,
        )
    )
    quantities = np.zeros(matrix.shape[1], dtype=float)
    active: list[int] = []
    current = sample_base.copy()
    current_components = _objective_components(
        sample_prices,
        current,
        minimum_ev=minimum_ev,
        minimum_es5=minimum_es5,
        bins=profile_bins,
        floor_metric=floor_metric,
    )
    iterations = 0

    for _ in range(max(int(max_legs), 0)):
        best = None
        for leg_index in range(matrix.shape[1]):
            if leg_index in active:
                continue
            for quantity in grid:
                if (
                    abs(quantity) < 1e-12
                    or np.abs(quantities).sum() + abs(quantity)
                    > max_total_quantity + 1e-12
                ):
                    continue
                if _duplicate_family(
                    candidates, quantities, leg_index, quantity
                ):
                    continue
                trial = current + sample_matrix[:, leg_index] * quantity
                components = _objective_components(
                    sample_prices,
                    trial,
                    minimum_ev=minimum_ev,
                    minimum_es5=minimum_es5,
                    bins=profile_bins,
                    floor_metric=floor_metric,
                )
                if best is None or _is_better(components, best[0]):
                    best = (components, leg_index, quantity)
        if best is None or not _is_better(best[0], current_components):
            break
        current_components, leg_index, quantity = best
        quantities[leg_index] = quantity
        active.append(leg_index)
        current += sample_matrix[:, leg_index] * quantity
        iterations += 1

    # Selection finds useful strikes. Coordinate refinement then searches the
    # complete quantity grid again for every active leg while holding the rest.
    for _ in range(4):
        improved = False
        for leg_index in active:
            old_quantity = quantities[leg_index]
            without_leg = current - sample_matrix[:, leg_index] * old_quantity
            other_quantity = np.abs(quantities).sum() - abs(old_quantity)
            best_components = current_components
            best_quantity = old_quantity
            for quantity in grid:
                if other_quantity + abs(quantity) > max_total_quantity + 1e-12:
                    continue
                trial = without_leg + sample_matrix[:, leg_index] * quantity
                components = _objective_components(
                    sample_prices,
                    trial,
                    minimum_ev=minimum_ev,
                    minimum_es5=minimum_es5,
                    bins=profile_bins,
                    floor_metric=floor_metric,
                )
                if _is_better(components, best_components):
                    best_components = components
                    best_quantity = quantity
            if abs(best_quantity - old_quantity) > 1e-12:
                quantities[leg_index] = best_quantity
                current = without_leg + sample_matrix[:, leg_index] * best_quantity
                current_components = best_components
                iterations += 1
                improved = True
        if not improved:
            break

    full_payoff = base + matrix @ quantities
    statistics = _equal_count_bin_statistics(prices, full_payoff, profile_bins)
    objective_floor = float(statistics[f"Worst {floor_metric.lower()}"])
    return RobustOptimizationResult(
        quantities=quantities,
        selected_legs=selected_quantities_to_legs(candidates, quantities),
        payoffs=full_payoff,
        profile=price_bin_profile(
            prices, full_payoff, bin_width=price_bin_width
        ),
        objective_metric=floor_metric,
        objective_floor=objective_floor,
        worst_bin_p5=statistics["Worst conditional P5"],
        worst_bin_mean=statistics["Worst conditional mean"],
        profile_flatness=statistics["Profile flatness"],
        expected_payoff=float(full_payoff.mean()),
        expected_shortfall_5=_expected_shortfall_5(full_payoff),
        iterations=iterations,
    )


def robust_metrics_table(
    base_payoff: np.ndarray,
    result: RobustOptimizationResult,
) -> pd.DataFrame:
    rows = []
    for name, payoff in [
        ("Polymarket only", base_payoff),
        ("Optimizer 2", result.payoffs),
    ]:
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


def objective_audit_table(
    base_payoff: np.ndarray,
    terminal_prices: np.ndarray,
    result: RobustOptimizationResult,
    bins: int,
    minimum_ev: float,
    minimum_es5: float,
) -> pd.DataFrame:
    base_statistics = _equal_count_bin_statistics(
        terminal_prices, np.asarray(base_payoff, dtype=float), bins
    )
    rows = []
    for name, payoff, statistics in [
        ("Polymarket only", np.asarray(base_payoff), base_statistics),
        (
            "Optimizer 2",
            result.payoffs,
            {
                "Worst conditional P5": result.worst_bin_p5,
                "Worst conditional mean": result.worst_bin_mean,
                "Profile flatness": result.profile_flatness,
            },
        ),
    ]:
        rows.append(
            {
                "Portfolio": name,
                "Primary objective": result.objective_metric,
                "EV floor": f"${minimum_ev:,.2f}",
                "ES 5% floor": f"${minimum_es5:,.2f}",
                "Expected payoff": f"${np.mean(payoff):,.2f}",
                "ES 5%": f"${_expected_shortfall_5(payoff):,.2f}",
                "Worst conditional mean": (
                    f"${statistics['Worst conditional mean']:,.2f}"
                ),
                "Worst conditional P5": (
                    f"${statistics['Worst conditional P5']:,.2f}"
                ),
                "Profile flatness": f"${statistics['Profile flatness']:,.2f}",
            }
        )
    return pd.DataFrame(rows)


def aligned_profile_figure(
    base_profile: pd.DataFrame,
    optimized_profile: pd.DataFrame,
) -> go.Figure:
    labels = optimized_profile["Price bin"]
    probability_text = optimized_profile["Scenario probability"].map(
        lambda value: f"{value:.1%}"
    )
    colors = np.where(
        optimized_profile["Expected payoff"] >= 0, "#16a34a", "#dc2626"
    )
    figure = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.08,
        row_heights=[0.72, 0.28],
    )
    figure.add_trace(
        go.Scatter(
            x=labels,
            y=base_profile["Expected payoff"],
            name="Polymarket-only expected payoff",
            mode="lines",
            line=dict(color="#94a3b8", dash="dash", width=2),
        ),
        row=1,
        col=1,
    )
    figure.add_trace(
        go.Bar(
            x=labels,
            y=optimized_profile["Expected payoff"],
            name="Optimizer 2 expected payoff",
            marker_color=colors,
        ),
        row=1,
        col=1,
    )
    figure.add_trace(
        go.Scatter(
            x=labels,
            y=optimized_profile["Payoff P5"],
            name="Optimizer 2 P5",
            mode="lines+markers",
            line=dict(color="#f59e0b", dash="dash"),
        ),
        row=1,
        col=1,
    )
    figure.add_trace(
        go.Scatter(
            x=labels,
            y=optimized_profile["Payoff P1"],
            name="Optimizer 2 P1",
            mode="lines+markers",
            line=dict(color="#dc2626", dash="dot"),
        ),
        row=1,
        col=1,
    )
    figure.add_trace(
        go.Bar(
            x=labels,
            y=optimized_profile["Scenario probability"],
            text=probability_text,
            textposition="outside",
            name="Scenario probability",
            marker_color="#60a5fa",
        ),
        row=2,
        col=1,
    )
    figure.add_hline(
        y=0, line_dash="dash", line_color="black", row=1, col=1
    )
    figure.update_yaxes(title_text="Payoff", row=1, col=1)
    figure.update_yaxes(
        title_text="Probability", tickformat=".1%", row=2, col=1
    )
    figure.update_xaxes(
        title_text="Terminal stock price / current price",
        tickangle=-45,
        row=2,
        col=1,
    )
    figure.update_layout(
        title="Payoff and scenario probability aligned by terminal-price bin",
        height=760,
        barmode="overlay",
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        margin=dict(l=50, r=30, t=100, b=100),
    )
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
    st.markdown(
        "**Objective (lexicographic):** (1) require the selected EV and ES 5% "
        "floors; (2) maximize the selected worst conditional payoff metric "
        "across equal-count terminal-price bins; (3) if tied, minimize the SD "
        "of conditional mean payoffs."
    )
    if candidates is None or option_payoff_matrix is None:
        st.info(
            "Configure a valid candidate universe in the classic Optimizer "
            "controls first."
        )
        return

    baseline_metrics = robust_metrics(np.asarray(base_payoff, dtype=float))
    controls = st.columns(5)
    floor_metric = controls[0].selectbox(
        "Primary floor metric",
        FLOOR_METRICS,
        help=(
            "Conditional mean lifts the visible average-payoff valley. "
            "Conditional P5 prioritizes stress outcomes inside each bin."
        ),
    )
    minimum_ev = controls[1].number_input(
        "Minimum expected payoff",
        value=float(max(default_minimum_ev, 0.0)),
        step=1.0,
    )
    minimum_es5 = controls[2].number_input(
        "Minimum ES 5%",
        value=float(baseline_metrics["ES 5%"]),
        step=1.0,
        help=(
            "Optimizer rejects portfolios with a more negative ES 5%. "
            "Lower this value deliberately if some tail deterioration is acceptable."
        ),
    )
    profile_bins = controls[3].number_input(
        "Optimization bins",
        min_value=10,
        max_value=50,
        value=20,
        step=5,
        help=(
            "Equal-count bins: 20 means each optimization bin contains about "
            "5% of scenarios."
        ),
    )
    price_bin_width = controls[4].number_input(
        "Displayed bin width (%)",
        min_value=2.5,
        max_value=20.0,
        value=5.0,
        step=2.5,
        help="Display only; does not change the optimization objective.",
    )

    if st.button("Update Optimizer 2", type="primary"):
        with st.spinner("Optimizing the conditional payoff floor..."):
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
                    minimum_es5=float(minimum_es5),
                    floor_metric=floor_metric,
                    profile_bins=int(profile_bins),
                    price_bin_width=float(price_bin_width),
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
        st.info("Run Optimizer 2 to create the conditional-floor comparison.")
        return

    base_profile = price_bin_profile(
        np.asarray(terminal_prices, dtype=float),
        np.asarray(base_payoff, dtype=float),
        bin_width=float(price_bin_width),
    )
    st.caption(
        f"Selected objective: {result.objective_metric}. "
        f"Quantity refinement completed in {result.iterations} accepted steps."
    )
    st.subheader("Objective audit")
    st.dataframe(
        objective_audit_table(
            base_payoff,
            np.asarray(terminal_prices, dtype=float),
            result,
            int(profile_bins),
            float(minimum_ev),
            float(minimum_es5),
        ),
        use_container_width=True,
        hide_index=True,
    )
    st.subheader("Global payoff metrics")
    st.dataframe(
        robust_metrics_table(base_payoff, result),
        use_container_width=True,
        hide_index=True,
    )
    st.subheader("Selected option legs")
    st.dataframe(
        result.selected_legs, use_container_width=True, hide_index=True
    )
    st.plotly_chart(
        aligned_profile_figure(base_profile, result.profile),
        use_container_width=True,
        key="optimizer2_aligned_profile",
    )
    with st.expander("Show terminal-price bin statistics"):
        display = result.profile.copy()
        display["Scenario probability"] = display["Scenario probability"].map(
            lambda value: f"{value:.2%}"
        )
        st.dataframe(display, use_container_width=True, hide_index=True)
