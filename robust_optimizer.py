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
    transaction_cost: float
    iterations: int


def price_bin_profile(terminal_prices, payoffs, *, bin_width=5.0):
    prices, values = np.asarray(terminal_prices, float), np.asarray(payoffs, float)
    low = np.floor(np.quantile(prices, .01) / bin_width) * bin_width
    high = np.ceil(np.quantile(prices, .99) / bin_width) * bin_width
    edges = np.concatenate(([-np.inf], np.arange(low, high + .5 * bin_width, bin_width), [np.inf]))
    frame = pd.DataFrame({"Terminal price": prices, "Payoff": values})
    frame["Price bin"] = pd.cut(frame["Terminal price"], edges, include_lowest=True)
    rows = []
    for interval, group in frame.groupby("Price bin", observed=True):
        x = group["Payoff"].to_numpy(float)
        label = f"<{interval.right:.0f}%" if not np.isfinite(interval.left) else (f">={interval.left:.0f}%" if not np.isfinite(interval.right) else f"{interval.left:.0f}-{interval.right:.0f}%")
        probability, mean = len(group) / len(frame), float(x.mean())
        rows.append({"Price bin": label, "Price midpoint": group["Terminal price"].mean(), "Scenario probability": probability, "Expected payoff": mean, "Payoff SD": x.std(), "Payoff P1": np.quantile(x, .01), "Payoff P5": np.quantile(x, .05), "Contribution to EV": probability * mean})
    return pd.DataFrame(rows)


def _expected_shortfall_5(payoffs):
    x = np.asarray(payoffs, float)
    q = np.quantile(x, .05)
    return float(x[x <= q].mean())


def robust_metrics(payoffs):
    x = np.asarray(payoffs, float)
    q01, q05 = np.quantile(x, [.01, .05])
    return pd.Series({"Expected payoff": x.mean(), "Payoff SD": x.std(), "VaR 5% payoff": q05, "VaR 1% payoff": q01, "ES 5%": x[x <= q05].mean(), "ES 1%": x[x <= q01].mean(), "P(loss)": (x < 0).mean(), "Worst payoff": x.min()})


def _equal_count_bin_statistics(prices, payoffs, bins):
    chunks = [chunk for chunk in np.array_split(np.argsort(prices), bins) if len(chunk)]
    p5 = np.array([np.quantile(payoffs[chunk], .05) for chunk in chunks])
    means = np.array([payoffs[chunk].mean() for chunk in chunks])
    return {"Worst conditional P5": float(p5.min()), "Worst conditional mean": float(means.min()), "Profile flatness": float(means.std())}


def _floor_key(metric):
    return "Worst conditional P5" if metric == "Conditional P5" else "Worst conditional mean"


def _components(prices, payoffs, *, minimum_ev, minimum_es5, bins, floor_metric):
    mean, es5 = float(np.mean(payoffs)), _expected_shortfall_5(payoffs)
    if mean < minimum_ev or es5 < minimum_es5:
        return -np.inf, np.inf, mean, es5
    stats = _equal_count_bin_statistics(prices, payoffs, bins)
    return stats[_floor_key(floor_metric)], stats["Profile flatness"], mean, es5


def _better(candidate, current):
    return candidate[0] > current[0] + 1e-10 or (abs(candidate[0] - current[0]) <= 1e-10 and candidate[1] < current[1] - 1e-10)


def _duplicate_family(candidates, quantities, index, quantity):
    if abs(quantity) < 1e-12:
        return False
    candidate = candidates.iloc[index]
    for active in np.flatnonzero(np.abs(quantities) > 1e-12):
        row = candidates.iloc[active]
        if str(row["Ticker"]) == str(candidate["Ticker"]) and str(row["Option type"]) == str(candidate["Option type"]) and np.sign(quantities[active]) == np.sign(quantity):
            return True
    return False


def _leg_effect(matrix, costs, index, quantity):
    """Scenario payoff net of a direction-independent execution cost."""
    return matrix[:, index] * quantity - costs[index] * abs(quantity)


def optimize_payoff_floor(
    base_payoff, option_payoff_matrix, candidates, terminal_prices, *, quantity_min,
    quantity_max, quantity_step, max_legs, max_total_quantity, minimum_ev,
    minimum_es5, floor_metric, profile_bins, price_bin_width,
    optimization_scenarios, seed, transaction_cost_rate=.01,
    contract_multiplier=1.0,
):
    if floor_metric not in FLOOR_METRICS:
        raise ValueError(f"floor_metric must be one of {FLOOR_METRICS}.")
    if not 0 <= transaction_cost_rate <= 1:
        raise ValueError("transaction_cost_rate must be between 0 and 1.")
    base, matrix, prices = np.asarray(base_payoff, float), np.asarray(option_payoff_matrix, float), np.asarray(terminal_prices, float)
    costs = candidates["Theoretical premium"].to_numpy(float) * float(contract_multiplier) * float(transaction_cost_rate)
    rng = np.random.default_rng(seed)
    size = min(int(optimization_scenarios), len(base))
    index = rng.choice(len(base), size, replace=False) if size < len(base) else np.arange(len(base))
    sample_base, sample_matrix, sample_prices = base[index], matrix[index], prices[index]
    grid = np.unique(np.append(np.arange(quantity_min, quantity_max + .5 * quantity_step, quantity_step), 0.0))
    quantities, active, current = np.zeros(matrix.shape[1]), [], sample_base.copy()
    kwargs = dict(minimum_ev=minimum_ev, minimum_es5=minimum_es5, bins=profile_bins, floor_metric=floor_metric)
    score, iterations = _components(sample_prices, current, **kwargs), 0

    for _ in range(min(max(int(max_legs), 0), 5)):
        best = None
        for leg in range(matrix.shape[1]):
            if leg in active:
                continue
            for quantity in grid:
                if abs(quantity) < 1e-12 or np.abs(quantities).sum() + abs(quantity) > max_total_quantity + 1e-12 or _duplicate_family(candidates, quantities, leg, quantity):
                    continue
                trial = current + _leg_effect(sample_matrix, costs, leg, quantity)
                trial_score = _components(sample_prices, trial, **kwargs)
                if best is None or _better(trial_score, best[0]):
                    best = trial_score, leg, quantity
        if best is None or not _better(best[0], score):
            break
        score, leg, quantity = best
        quantities[leg] = quantity
        active.append(leg)
        current += _leg_effect(sample_matrix, costs, leg, quantity)
        iterations += 1

    for _ in range(4):
        improved = False
        for leg in active:
            old = quantities[leg]
            without = current - _leg_effect(sample_matrix, costs, leg, old)
            other_total = np.abs(quantities).sum() - abs(old)
            best_score, best_quantity = score, old
            for quantity in grid:
                if other_total + abs(quantity) > max_total_quantity + 1e-12:
                    continue
                trial = without + _leg_effect(sample_matrix, costs, leg, quantity)
                trial_score = _components(sample_prices, trial, **kwargs)
                if _better(trial_score, best_score):
                    best_score, best_quantity = trial_score, quantity
            if abs(best_quantity - old) > 1e-12:
                quantities[leg], current, score = best_quantity, without + _leg_effect(sample_matrix, costs, leg, best_quantity), best_score
                iterations, improved = iterations + 1, True
        if not improved:
            break

    total_tc = float(costs @ np.abs(quantities))
    full_payoff = base + matrix @ quantities - total_tc
    stats = _equal_count_bin_statistics(prices, full_payoff, profile_bins)
    selected = selected_quantities_to_legs(candidates, quantities)
    if not selected.empty:
        selected["Execution cost estimate"] = selected["Quantity"].to_numpy(float) * selected["Theoretical premium"].to_numpy(float) * float(contract_multiplier) * float(transaction_cost_rate)
    return RobustOptimizationResult(quantities, selected, full_payoff, price_bin_profile(prices, full_payoff, bin_width=price_bin_width), floor_metric, stats[_floor_key(floor_metric)], stats["Worst conditional P5"], stats["Worst conditional mean"], stats["Profile flatness"], float(full_payoff.mean()), _expected_shortfall_5(full_payoff), total_tc, iterations)


def robust_metrics_table(base, result):
    rows = []
    for name, payoff in [("Polymarket only", base), ("Optimizer 2", result.payoffs)]:
        m = robust_metrics(payoff)
        rows.append({"Portfolio": name, "Expected payoff": f"${m['Expected payoff']:,.2f}", "Payoff SD": f"${m['Payoff SD']:,.2f}", "VaR 5% payoff": f"${m['VaR 5% payoff']:,.2f}", "VaR 1% payoff": f"${m['VaR 1% payoff']:,.2f}", "ES 5%": f"${m['ES 5%']:,.2f}", "ES 1%": f"${m['ES 1%']:,.2f}", "P(loss)": f"{m['P(loss)']:.2%}", "Worst payoff": f"${m['Worst payoff']:,.2f}"})
    return pd.DataFrame(rows)


def audit_table(base, prices, result, bins, minimum_ev, minimum_es5):
    base_stats = _equal_count_bin_statistics(prices, np.asarray(base), bins)
    rows = []
    for name, payoff, stats, tc in [("Polymarket only", np.asarray(base), base_stats, 0.0), ("Optimizer 2", result.payoffs, {"Worst conditional P5": result.worst_bin_p5, "Worst conditional mean": result.worst_bin_mean, "Profile flatness": result.profile_flatness}, result.transaction_cost)]:
        rows.append({"Portfolio": name, "Primary objective": result.objective_metric, "EV floor": f"${minimum_ev:,.2f}", "ES 5% floor": f"${minimum_es5:,.2f}", "Expected payoff": f"${np.mean(payoff):,.2f}", "ES 5%": f"${_expected_shortfall_5(payoff):,.2f}", "Worst conditional mean": f"${stats['Worst conditional mean']:,.2f}", "Worst conditional P5": f"${stats['Worst conditional P5']:,.2f}", "Profile flatness": f"${stats['Profile flatness']:,.2f}", "Execution cost": f"${tc:,.2f}"})
    return pd.DataFrame(rows)


def aligned_profile_figure(base_profile, profile):
    labels = profile["Price bin"]
    colors = np.where(profile["Expected payoff"] >= 0, "#16a34a", "#dc2626")
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=.08, row_heights=[.72, .28])
    fig.add_trace(go.Scatter(x=labels, y=base_profile["Expected payoff"], name="Polymarket-only mean", mode="lines", line=dict(color="#94a3b8", dash="dash")), row=1, col=1)
    fig.add_trace(go.Bar(x=labels, y=profile["Expected payoff"], name="Optimizer 2 mean", marker_color=colors), row=1, col=1)
    fig.add_trace(go.Scatter(x=labels, y=profile["Payoff P5"], name="Optimizer 2 P5", mode="lines+markers", line=dict(color="#f59e0b", dash="dash")), row=1, col=1)
    fig.add_trace(go.Scatter(x=labels, y=profile["Payoff P1"], name="Optimizer 2 P1", mode="lines+markers", line=dict(color="#dc2626", dash="dot")), row=1, col=1)
    fig.add_trace(go.Bar(x=labels, y=profile["Scenario probability"], text=profile["Scenario probability"].map(lambda x: f"{x:.1%}"), textposition="outside", name="Scenario probability", marker_color="#60a5fa"), row=2, col=1)
    fig.add_hline(y=0, line_dash="dash", line_color="black", row=1, col=1)
    fig.update_yaxes(title_text="Payoff", row=1, col=1)
    fig.update_yaxes(title_text="Probability", tickformat=".1%", row=2, col=1)
    fig.update_xaxes(title_text="Terminal stock price / current price", tickangle=-45, row=2, col=1)
    fig.update_layout(title="Payoff and scenario probability aligned by terminal-price bin", height=760, legend=dict(orientation="h", yanchor="bottom", y=1.02))
    return fig


def render_robust_optimizer(*, base_payoff, option_payoff_matrix, candidates, terminal_prices, quantity_min, quantity_max, quantity_step, max_legs, max_total_quantity, default_minimum_ev, optimization_scenarios, seed):
    st.subheader("Optimizer 2: conditional payoff floor")
    st.markdown("**Objective:** enforce EV and ES 5% floors, maximize the selected worst conditional metric, then minimize dispersion. Execution cost is always deducted and active legs are capped at five.")
    if candidates is None or option_payoff_matrix is None:
        st.info("Configure a valid candidate universe in the classic Optimizer controls first.")
        return
    baseline = robust_metrics(base_payoff)
    baseline_ev = float(baseline["Expected payoff"])
    baseline_es5 = float(baseline["ES 5%"])
    ev_slack = max(1.0, 0.05 * abs(baseline_ev))
    es5_slack = max(1.0, 0.05 * abs(baseline_es5))
    default_ev_floor = baseline_ev - ev_slack
    default_es5_floor = baseline_es5 - es5_slack

    row1 = st.columns(5)
    metric = row1[0].selectbox("Primary floor metric", FLOOR_METRICS)
    minimum_ev = row1[1].number_input(
        "Minimum expected payoff", value=float(default_ev_floor), step=1.0,
        key="optimizer2_minimum_ev_v2",
        help="Defaults slightly below the Polymarket-only EV so the optimizer can exchange a small amount of EV for a flatter payoff profile.",
    )
    minimum_es5 = row1[2].number_input(
        "Minimum ES 5%", value=float(default_es5_floor), step=1.0,
        key="optimizer2_minimum_es5_v2",
        help="Defaults slightly below the Polymarket-only ES 5% to leave room for a feasible hedge.",
    )
    bins = row1[3].number_input("Optimization bins", 10, 50, 20, 5)
    width = row1[4].number_input("Displayed bin width (%)", 2.5, 20.0, 5.0, 2.5)
    row2 = st.columns(3)
    tc_rate = row2[0].number_input("Execution cost (% of option premium)", 0.0, 20.0, 1.0, .25) / 100
    optimizer_legs = row2[1].number_input("Maximum active option legs", 1, 5, min(max(int(max_legs), 1), 5), 1)
    row2[2].metric("Hard leg cap", "5")
    st.caption(
        f"Polymarket-only reference: EV ${baseline_ev:,.2f}, ES 5% ${baseline_es5:,.2f}. "
        "The default floors allow a small controlled trade-off instead of forcing an exact no-trade solution."
    )

    if st.button("Update Optimizer 2", type="primary"):
        try:
            st.session_state.phase5_robust_optimization = optimize_payoff_floor(base_payoff, option_payoff_matrix, candidates, terminal_prices, quantity_min=quantity_min, quantity_max=quantity_max, quantity_step=quantity_step, max_legs=int(optimizer_legs), max_total_quantity=max_total_quantity, minimum_ev=float(minimum_ev), minimum_es5=float(minimum_es5), floor_metric=metric, profile_bins=int(bins), price_bin_width=float(width), optimization_scenarios=optimization_scenarios, seed=seed, transaction_cost_rate=float(tc_rate))
            st.session_state.phase5_robust_error = None
        except Exception as exc:
            st.session_state.phase5_robust_error = str(exc)
    if st.session_state.get("phase5_robust_error"):
        st.error(st.session_state.phase5_robust_error)
    result = st.session_state.get("phase5_robust_optimization")
    if result is None:
        st.info("Run Optimizer 2 to create the comparison.")
        return
    base_profile = price_bin_profile(terminal_prices, base_payoff, bin_width=float(width))
    st.caption(f"{result.objective_metric}; {len(result.selected_legs)} active legs; ${result.transaction_cost:,.2f} execution cost; {result.iterations} accepted search steps.")
    if result.selected_legs.empty:
        st.warning(
            "No candidate leg improved the conditional objective while satisfying both floors. "
            "This is a valid no-trade result. To test a wider trade-off, lower Minimum expected payoff or Minimum ES 5%, increase Maximum total absolute quantity, or reduce Quantity step."
        )
    st.subheader("Objective audit")
    st.dataframe(audit_table(base_payoff, np.asarray(terminal_prices), result, int(bins), float(minimum_ev), float(minimum_es5)), use_container_width=True, hide_index=True)
    st.subheader("Global payoff metrics")
    st.dataframe(robust_metrics_table(base_payoff, result), use_container_width=True, hide_index=True)
    st.subheader("Selected option legs")
    st.dataframe(result.selected_legs, use_container_width=True, hide_index=True)
    st.plotly_chart(aligned_profile_figure(base_profile, result.profile), use_container_width=True, key="optimizer2_aligned_profile")
    with st.expander("Show terminal-price bin statistics"):
        display = result.profile.copy()
        display["Scenario probability"] = display["Scenario probability"].map(lambda x: f"{x:.2%}")
        st.dataframe(display, use_container_width=True, hide_index=True)
