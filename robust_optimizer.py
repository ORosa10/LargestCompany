"""Transparent conditional-payoff optimizer and diagnostics for Phase 5."""

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


def price_bin_profile(terminal_prices, payoffs, *, bin_width=5.0) -> pd.DataFrame:
    """Summarize payoff in fixed terminal-price intervals; today is 100."""
    prices = np.asarray(terminal_prices, dtype=float)
    values = np.asarray(payoffs, dtype=float)
    low = np.floor(np.quantile(prices, 0.01) / bin_width) * bin_width
    high = np.ceil(np.quantile(prices, 0.99) / bin_width) * bin_width
    finite_edges = np.arange(low, high + bin_width * 0.5, bin_width)
    edges = np.concatenate(([-np.inf], finite_edges, [np.inf]))
    frame = pd.DataFrame({"Terminal price": prices, "Payoff": values})
    frame["Price bin"] = pd.cut(frame["Terminal price"], edges, include_lowest=True)
    rows = []
    for interval, group in frame.groupby("Price bin", observed=True):
        bin_values = group["Payoff"].to_numpy(dtype=float)
        if not np.isfinite(interval.left):
            label = f"<{interval.right:.0f}%"
        elif not np.isfinite(interval.right):
            label = f">={interval.left:.0f}%"
        else:
            label = f"{interval.left:.0f}-{interval.right:.0f}%"
        probability = len(group) / len(frame)
        mean = float(bin_values.mean())
        rows.append({
            "Price bin": label,
            "Price midpoint": float(group["Terminal price"].mean()),
            "Scenario probability": probability,
            "Expected payoff": mean,
            "Payoff SD": float(bin_values.std(ddof=0)),
            "Payoff P1": float(np.quantile(bin_values, 0.01)),
            "Payoff P5": float(np.quantile(bin_values, 0.05)),
            "Contribution to EV": probability * mean,
        })
    return pd.DataFrame(rows)


def _expected_shortfall_5(payoffs) -> float:
    values = np.asarray(payoffs, dtype=float)
    threshold = np.quantile(values, 0.05)
    return float(values[values <= threshold].mean())


def robust_metrics(payoffs) -> pd.Series:
    values = np.asarray(payoffs, dtype=float)
    q01, q05 = np.quantile(values, [0.01, 0.05])
    return pd.Series({
        "Expected payoff": values.mean(),
        "Payoff SD": values.std(ddof=0),
        "VaR 5% payoff": q05,
        "VaR 1% payoff": q01,
        "ES 5%": values[values <= q05].mean(),
        "ES 1%": values[values <= q01].mean(),
        "P(loss)": (values < 0).mean(),
        "Worst payoff": values.min(),
    })


def _equal_count_bin_statistics(terminal_prices, payoffs, bins) -> dict[str, float]:
    order = np.argsort(terminal_prices)
    chunks = [chunk for chunk in np.array_split(order, bins) if len(chunk)]
    p5 = np.array([np.quantile(payoffs[chunk], 0.05) for chunk in chunks])
    means = np.array([payoffs[chunk].mean() for chunk in chunks])
    return {
        "Worst conditional P5": float(p5.min()),
        "Worst conditional mean": float(means.min()),
        "Profile flatness": float(means.std(ddof=0)),
    }


def _floor_key(floor_metric: str) -> str:
    return "Worst conditional P5" if floor_metric == "Conditional P5" else "Worst conditional mean"


def _objective_components(
    terminal_prices, payoffs, *, minimum_ev, minimum_es5, bins, floor_metric
):
    mean = float(np.mean(payoffs))
    es5 = _expected_shortfall_5(payoffs)
    if mean < minimum_ev or es5 < minimum_es5:
        return -np.inf, np.inf, mean, es5
    stats = _equal_count_bin_statistics(terminal_prices, payoffs, bins)
    return stats[_floor_key(floor_metric)], stats["Profile flatness"], mean, es5


def _is_better(candidate, current) -> bool:
    candidate_floor, candidate_flatness = candidate[:2]
    current_floor, current_flatness = current[:2]
    return candidate_floor > current_floor + 1e-10 or (
        abs(candidate_floor - current_floor) <= 1e-10
        and candidate_flatness < current_flatness - 1e-10
    )


def _duplicate_family(candidates, quantities, index, quantity) -> bool:
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
    base_payoff, option_payoff_matrix, candidates, terminal_prices, *,
    quantity_min, quantity_max, quantity_step, max_legs, max_total_quantity,
    minimum_ev, minimum_es5, floor_metric, profile_bins, price_bin_width,
    optimization_scenarios, seed,
) -> RobustOptimizationResult:
    """Select useful strikes, then refine active quantities on the full grid."""
    if floor_metric not in FLOOR_METRICS:
        raise ValueError(f"floor_metric must be one of {FLOOR_METRICS}.")
    base = np.asarray(base_payoff, dtype=float)
    matrix = np.asarray(option_payoff_matrix, dtype=float)
    prices = np.asarray(terminal_prices, dtype=float)
    rng = np.random.default_rng(seed)
    sample_size = min(int(optimization_scenarios), len(base))
    index = rng.choice(len(base), sample_size, replace=False) if sample_size < len(base) else np.arange(len(base))
    sample_base, sample_matrix, sample_prices = base[index], matrix[index], prices[index]
    grid = np.unique(np.append(np.arange(quantity_min, quantity_max + 0.5 * quantity_step, quantity_step), 0.0))
    quantities = np.zeros(matrix.shape[1])
    active = []
    current = sample_base.copy()
    kwargs = dict(minimum_ev=minimum_ev, minimum_es5=minimum_es5, bins=profile_bins, floor_metric=floor_metric)
    current_score = _objective_components(sample_prices, current, **kwargs)
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
                score = _objective_components(sample_prices, trial, **kwargs)
                if best is None or _is_better(score, best[0]):
                    best = score, leg_index, quantity
        if best is None or not _is_better(best[0], current_score):
            break
        current_score, leg_index, quantity = best
        quantities[leg_index] = quantity
        active.append(leg_index)
        current += sample_matrix[:, leg_index] * quantity
        iterations += 1

    for _ in range(4):
        improved = False
        for leg_index in active:
            old = quantities[leg_index]
            without = current - sample_matrix[:, leg_index] * old
            other_total = np.abs(quantities).sum() - abs(old)
            best_score, best_quantity = current_score, old
            for quantity in grid:
                if other_total + abs(quantity) > max_total_quantity + 1e-12:
                    continue
                trial = without + sample_matrix[:, leg_index] * quantity
                score = _objective_components(sample_prices, trial, **kwargs)
                if _is_better(score, best_score):
                    best_score, best_quantity = score, quantity
            if abs(best_quantity - old) > 1e-12:
                quantities[leg_index] = best_quantity
                current = without + sample_matrix[:, leg_index] * best_quantity
                current_score = best_score
                iterations += 1
                improved = True
        if not improved:
            break

    full_payoff = base + matrix @ quantities
    stats = _equal_count_bin_statistics(prices, full_payoff, profile_bins)
    return RobustOptimizationResult(
        quantities=quantities,
        selected_legs=selected_quantities_to_legs(candidates, quantities),
        payoffs=full_payoff,
        profile=price_bin_profile(prices, full_payoff, bin_width=price_bin_width),
        objective_metric=floor_metric,
        objective_floor=stats[_floor_key(floor_metric)],
        worst_bin_p5=stats["Worst conditional P5"],
        worst_bin_mean=stats["Worst conditional mean"],
        profile_flatness=stats["Profile flatness"],
        expected_payoff=float(full_payoff.mean()),
        expected_shortfall_5=_expected_shortfall_5(full_payoff),
        iterations=iterations,
    )


def robust_metrics_table(base_payoff, result) -> pd.DataFrame:
    rows = []
    for name, payoff in [("Polymarket only", base_payoff), ("Optimizer 2", result.payoffs)]:
        m = robust_metrics(payoff)
        rows.append({
            "Portfolio": name,
            "Expected payoff": f"${m['Expected payoff']:,.2f}",
            "Payoff SD": f"${m['Payoff SD']:,.2f}",
            "VaR 5% payoff": f"${m['VaR 5% payoff']:,.2f}",
            "VaR 1% payoff": f"${m['VaR 1% payoff']:,.2f}",
            "ES 5%": f"${m['ES 5%']:,.2f}",
            "ES 1%": f"${m['ES 1%']:,.2f}",
            "P(loss)": f"{m['P(loss)']:.2%}",
            "Worst payoff": f"${m['Worst payoff']:,.2f}",
        })
    return pd.DataFrame(rows)


def objective_audit_table(base_payoff, terminal_prices, result, bins, minimum_ev, minimum_es5):
    base_stats = _equal_count_bin_statistics(terminal_prices, np.asarray(base_payoff), bins)
    rows = []
    for name, payoff, stats in [
        ("Polymarket only", np.asarray(base_payoff), base_stats),
        ("Optimizer 2", result.payoffs, {
            "Worst conditional P5": result.worst_bin_p5,
            "Worst conditional mean": result.worst_bin_mean,
            "Profile flatness": result.profile_flatness,
        }),
    ]:
        rows.append({
            "Portfolio": name,
            "Primary objective": result.objective_metric,
            "EV floor": f"${minimum_ev:,.2f}",
            "ES 5% floor": f"${minimum_es5:,.2f}",
            "Expected payoff": f"${np.mean(payoff):,.2f}",
            "ES 5%": f"${_expected_shortfall_5(payoff):,.2f}",
            "Worst conditional mean": f"${stats['Worst conditional mean']:,.2f}",
            "Worst conditional P5": f"${stats['Worst conditional P5']:,.2f}",
            "Profile flatness": f"${stats['Profile flatness']:,.2f}",
        })
    return pd.DataFrame(rows)


def aligned_profile_figure(base_profile, optimized_profile) -> go.Figure:
    labels = optimized_profile["Price bin"]
    colors = np.where(optimized_profile["Expected payoff"] >= 0, "#16a34a", "#dc2626")
    figure = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.08, row_heights=[0.72, 0.28])
    figure.add_trace(go.Scatter(x=labels, y=base_profile["Expected payoff"], name="Polymarket-only mean", mode="lines", line=dict(color="#94a3b8", dash="dash", width=2)), row=1, col=1)
    figure.add_trace(go.Bar(x=labels, y=optimized_profile["Expected payoff"], name="Optimizer 2 mean", marker_color=colors), row=1, col=1)
    figure.add_trace(go.Scatter(x=labels, y=optimized_profile["Payoff P5"], name="Optimizer 2 P5", mode="lines+markers", line=dict(color="#f59e0b", dash="dash")), row=1, col=1)
    figure.add_trace(go.Scatter(x=labels, y=optimized_profile["Payoff P1"], name="Optimizer 2 P1", mode="lines+markers", line=dict(color="#dc2626", dash="dot")), row=1, col=1)
    figure.add_trace(go.Bar(x=labels, y=optimized_profile["Scenario probability"], text=optimized_profile["Scenario probability"].map(lambda x: f"{x:.1%}"), textposition="outside", name="Scenario probability", marker_color="#60a5fa"), row=2, col=1)
    figure.add_hline(y=0, line_dash="dash", line_color="black", row=1, col=1)
    figure.update_yaxes(title_text="Payoff", row=1, col=1)
    figure.update_yaxes(title_text="Probability", tickformat=".1%", row=2, col=1)
    figure.update_xaxes(title_text="Terminal stock price / current price", tickangle=-45, row=2, col=1)
    figure.update_layout(title="Payoff and scenario probability aligned by terminal-price bin", height=760, legend=dict(orientation="h", yanchor="bottom", y=1.02), margin=dict(l=50, r=30, t=100, b=100))
    return figure


def render_robust_optimizer(
    *, base_payoff, option_payoff_matrix, candidates, terminal_prices,
    quantity_min, quantity_max, quantity_step, max_legs, max_total_quantity,
    default_minimum_ev, optimization_scenarios, seed,
) -> None:
    st.subheader("Optimizer 2: conditional payoff floor")
    st.markdown(
        "**Objective:** enforce EV and ES 5% floors, maximize the selected worst "
        "conditional metric across equal-count price bins, then minimize the "
        "dispersion of conditional means."
    )
    if candidates is None or option_payoff_matrix is None:
        st.info("Configure a valid candidate universe in the classic Optimizer controls first.")
        return

    baseline = robust_metrics(base_payoff)
    controls = st.columns(5)
    floor_metric = controls[0].selectbox(
        "Primary floor metric", FLOOR_METRICS,
        help="Conditional mean lifts the visible valley. Conditional P5 prioritizes stress outcomes within every bin.",
    )
    minimum_ev = controls[1].number_input("Minimum expected payoff", value=float(max(default_minimum_ev, 0.0)), step=1.0)
    minimum_es5 = controls[2].number_input(
        "Minimum ES 5%", value=float(baseline["ES 5%"]), step=1.0,
        help="Lower deliberately if some tail deterioration is acceptable.",
    )
    profile_bins = controls[3].number_input(
        "Optimization bins", 10, 50, 20, 5,
        help="Equal-count bins: 20 means roughly 5% of scenarios per bin.",
    )
    price_bin_width = controls[4].number_input(
        "Displayed bin width (%)", 2.5, 20.0, 5.0, 2.5,
        help="Display only; it does not change the objective.",
    )

    if st.button("Update Optimizer 2", type="primary"):
        with st.spinner("Optimizing the conditional payoff floor..."):
            try:
                st.session_state.phase5_robust_optimization = optimize_payoff_floor(
                    base_payoff, option_payoff_matrix, candidates, terminal_prices,
                    quantity_min=quantity_min, quantity_max=quantity_max,
                    quantity_step=quantity_step, max_legs=max_legs,
                    max_total_quantity=max_total_quantity,
                    minimum_ev=float(minimum_ev), minimum_es5=float(minimum_es5),
                    floor_metric=floor_metric, profile_bins=int(profile_bins),
                    price_bin_width=float(price_bin_width),
                    optimization_scenarios=optimization_scenarios, seed=seed,
                )
                st.session_state.phase5_robust_error = None
            except Exception as exc:
                st.session_state.phase5_robust_error = str(exc)
    if st.session_state.get("phase5_robust_error"):
        st.error(st.session_state.phase5_robust_error)
    result = st.session_state.get("phase5_robust_optimization")
    if result is None:
        st.info("Run Optimizer 2 to create the conditional-floor comparison.")
        return

    base_profile = price_bin_profile(terminal_prices, base_payoff, bin_width=float(price_bin_width))
    st.caption(f"Objective: {result.objective_metric}. Quantity refinement used {result.iterations} accepted steps.")
    st.subheader("Objective audit")
    st.dataframe(objective_audit_table(base_payoff, np.asarray(terminal_prices), result, int(profile_bins), float(minimum_ev), float(minimum_es5)), use_container_width=True, hide_index=True)
    st.subheader("Global payoff metrics")
    st.dataframe(robust_metrics_table(base_payoff, result), use_container_width=True, hide_index=True)
    st.subheader("Selected option legs")
    st.dataframe(result.selected_legs, use_container_width=True, hide_index=True)
    st.plotly_chart(aligned_profile_figure(base_profile, result.profile), use_container_width=True, key="optimizer2_aligned_profile")
    with st.expander("Show terminal-price bin statistics"):
        display = result.profile.copy()
        display["Scenario probability"] = display["Scenario probability"].map(lambda x: f"{x:.2%}")
        st.dataframe(display, use_container_width=True, hide_index=True)
