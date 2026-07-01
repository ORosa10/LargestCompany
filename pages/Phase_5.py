from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from boundaries import calculate_conditional_win_curve
from interactive_portfolio import confidence_at_strike, default_interactive_rows, optimized_legs_to_interactive_rows, render_interactive_leg_editor
from manual_portfolio import manual_option_payoffs_and_analytics, resolve_manual_option_legs
from optimization import OBJECTIVES, build_candidate_option_universe, long_option_payoff_matrix, optimize_option_portfolio, payoff_metrics
from option_sensitivity import calculate_boundary_quantity_sensitivity, render_boundary_quantity_sensitivity
from payoff_surface import polymarket_payoff, selected_payoff_profile_bins, terminal_stock_prices, winner_from_ranks
from phase4_ui import display_profile, dollars, payoff_by_bin_figure, payoff_profile_figure, pct
from robust_optimizer import render_robust_optimizer
from simulation_store import load_simulation_snapshot

NORMALIZED_SPOT = 100.0

st.set_page_config(page_title="Phase 5", layout="wide")
st.title("Phase 5")
st.caption("Interactive Option Portfolio Dashboard. Reuse stored Monte Carlo paths and edit strike or conditional boundary in one reciprocal portfolio table.")


def available_snapshot() -> dict | None:
    if st.session_state.get("phase4_result") is not None and st.session_state.get("phase4_inputs_used") is not None:
        metadata = {}
        phase4_legs = st.session_state.get("phase4_option_legs")
        if phase4_legs is not None and not phase4_legs.empty and "Time to expiry" in phase4_legs.columns:
            metadata["days_to_target"] = int(round(float(phase4_legs["Time to expiry"].iloc[0]) * 365.0))
        return {"result": st.session_state.phase4_result, "simulation_inputs": st.session_state.phase4_inputs_used, "run_metadata": metadata, "source": "Phase 4 session snapshot"}
    if st.session_state.get("last_result") is not None and st.session_state.get("last_simulation_inputs") is not None:
        return {"result": st.session_state.last_result, "simulation_inputs": st.session_state.last_simulation_inputs, "run_metadata": st.session_state.get("last_run") or {}, "source": "Phase 1 session snapshot"}
    return load_simulation_snapshot()


def metrics_comparison(baseline: pd.Series, portfolio: pd.Series, portfolio_name: str) -> pd.DataFrame:
    rows = []
    for label, metrics in [("Polymarket only", baseline), (portfolio_name, portfolio)]:
        expected = float(metrics["Expected payoff"])
        sd = float(metrics["Payoff standard deviation"])
        rows.append({
            "Portfolio": label,
            "Expected payoff": dollars(expected),
            "Payoff SD": dollars(sd),
            "EV / SD": f"{expected / sd:.3f}" if sd > 0 else "n/a",
            "Median payoff": dollars(float(metrics["Median payoff"])),
            "P(loss)": pct(float(metrics["Probability of loss"])),
            "Expected shortfall 5%": dollars(float(metrics["Expected shortfall 5%"])),
            "Worst payoff": dollars(float(metrics["Worst payoff"])),
        })
    return pd.DataFrame(rows)


def display_option_chain(chain: pd.DataFrame) -> pd.DataFrame:
    display = chain.copy()
    for column in ["Strike", "Spot", "Theoretical premium"]:
        display[column] = display[column].map(dollars)
    for column in ["Strike / spot", "Model IV"]:
        display[column] = display[column].map(pct)
    order = ["Ticker", "Option type", "Strike", "Strike / spot", "Boundary used", "Spot", "Model IV", "Theoretical premium"]
    return display[[column for column in order if column in display.columns]]


def display_legs(legs: pd.DataFrame) -> pd.DataFrame:
    display = legs.copy()
    for column in ["Strike", "Spot", "Theoretical premium"]:
        display[column] = display[column].map(dollars)
    for column in ["Strike / spot", "Model IV"]:
        display[column] = display[column].map(pct)
    order = ["Instrument", "Ticker", "Option type", "Position", "Quantity", "Strike", "Strike / spot", "Strike source", "Boundary used", "Theoretical premium", "Model IV"]
    return display[[column for column in order if column in display.columns]]


def display_leg_analytics(analytics: pd.DataFrame) -> pd.DataFrame:
    display = analytics.copy()
    for column in ["Expected option payoff", "Option payoff SD", "Expected shortfall 5%", "Worst option payoff", "Initial premium cashflow"]:
        display[column] = display[column].map(dollars)
    display["P(option loss)"] = display["P(option loss)"].map(pct)
    return display


def distribution_figure(baseline: np.ndarray, portfolio: np.ndarray, name: str) -> go.Figure:
    fig = go.Figure()
    fig.add_histogram(x=baseline, name="Polymarket only", opacity=0.55, nbinsx=80, histnorm="probability")
    fig.add_histogram(x=portfolio, name=name, opacity=0.55, nbinsx=80, histnorm="probability")
    fig.update_layout(title="Payoff distribution comparison", xaxis_title="Terminal payoff", yaxis_title="Scenario probability", barmode="overlay", yaxis_tickformat=".1%", legend=dict(orientation="h", yanchor="bottom", y=1.02))
    return fig


def make_profile(total_payoff: np.ndarray, option_payoff: np.ndarray, base_payoff: np.ndarray, selected_ticker: str) -> pd.DataFrame:
    scenario = pd.DataFrame({"Winner": winners, "Selected terminal market cap": result.terminal_market_caps[selected_ticker], "Selected terminal stock price": normalized_terminal_prices[selected_ticker], "Polymarket payoff": base_payoff, "Option payoff": option_payoff, "Total payoff": total_payoff})
    return selected_payoff_profile_bins(scenario, result.terminal_market_caps, current_caps, selected_ticker=selected_ticker, bins=20)


def clear_interactive_widget_state() -> None:
    for key in list(st.session_state.keys()):
        if str(key).startswith("leg_"):
            del st.session_state[key]


def add_boundary_labels(chain: pd.DataFrame, ticker: str) -> pd.DataFrame:
    labelled = chain.copy()
    labels = []
    for _, leg in labelled.iterrows():
        boundary_type = "Win boundary" if str(leg["Option type"]) == "Call" else "Loss boundary"
        confidence = confidence_at_strike(curves[ticker], float(leg["Strike"]), boundary_type=boundary_type, normalized_spot=NORMALIZED_SPOT)
        labels.append(f"{confidence:.1%} {boundary_type.lower()}")
    labelled["Boundary used"] = labels
    return labelled


snapshot = available_snapshot()
if snapshot is None:
    st.error("No Monte Carlo snapshot is available. Run Phase 1 once, then return here.")
    st.stop()

result = snapshot["result"]
simulation_inputs = snapshot["simulation_inputs"].copy()
run_metadata = snapshot.get("run_metadata") or {}
tickers = result.terminal_market_caps.columns.astype(str).tolist()
days_to_target = int(run_metadata.get("days_to_target", 365))
time_to_expiry = max(days_to_target, 1) / 365.0
current_caps = simulation_inputs.set_index("Ticker")["Current market cap"].astype(float)
normalized_spots = pd.Series(NORMALIZED_SPOT, index=tickers, dtype=float)
normalized_terminal_prices = terminal_stock_prices(result.terminal_market_caps, current_caps, normalized_spots)
winners = winner_from_ranks(result.ranks)
input_by_ticker = simulation_inputs.set_index("Ticker")

snapshot_key = (snapshot.get("source"), run_metadata.get("target_date"), run_metadata.get("seed"), len(result.terminal_market_caps), tuple(tickers))
if st.session_state.get("phase5_curve_snapshot_key") != snapshot_key:
    st.session_state.phase5_conditional_curves = {ticker: calculate_conditional_win_curve(result.terminal_market_caps, ticker, ranks=result.ranks, current_market_cap=float(current_caps.loc[ticker]), n_bins=40) for ticker in tickers}
    st.session_state.phase5_curve_snapshot_key = snapshot_key
curves = st.session_state.phase5_conditional_curves

st.success(f"Using {snapshot.get('source', 'saved simulation')} | target {run_metadata.get('target_date', 'saved horizon')} | {days_to_target} days | {len(result.terminal_market_caps):,} stored paths | normalized spot = 100")
st.caption("A strike of 80 means 80% of today's price. Editing the option table never reruns Monte Carlo.")

with st.sidebar:
    st.header("Common payoff settings")
    contract_multiplier = st.number_input("Payoff multiplier", min_value=0.01, value=1.0, step=0.25, help="Kept at 1 while prices are normalized to 100.")
    include_premiums = st.checkbox("Include theoretical premiums", value=True)
    risk_free_rate = st.number_input("Risk-free rate", min_value=0.0, max_value=0.20, value=0.04, step=0.005, format="%.3f")
    st.header("Optimizer candidate universe")
    st.caption("Strike bounds are estimated from stored terminal-price scenarios, not entered manually.")
    strike_grid_points = st.number_input("Data-driven strike grid points", min_value=7, max_value=61, value=25, step=2)
    lower_strike_quantile = st.number_input("Lower terminal-price quantile", min_value=0.001, max_value=0.25, value=0.01, step=0.005, format="%.3f")
    upper_strike_quantile = st.number_input("Upper terminal-price quantile", min_value=0.75, max_value=0.999, value=0.99, step=0.005, format="%.3f")
    include_calls = st.checkbox("Include calls", value=True)
    include_puts = st.checkbox("Include puts", value=True)
    allow_long = st.checkbox("Allow long positions", value=True)
    allow_short = st.checkbox("Allow short positions", value=True)
    st.header("Portfolio search")
    objective = st.selectbox("Objective", OBJECTIVES, index=1)
    max_legs = st.number_input("Maximum optimizer legs", min_value=0, max_value=10, value=4, step=1)
    max_quantity_per_leg = st.number_input("Maximum absolute quantity per leg", min_value=0.0, value=0.50, step=0.025, format="%.3f")
    quantity_step = st.number_input("Quantity grid step", min_value=0.001, value=0.025, step=0.005, format="%.3f")
    max_total_quantity = st.number_input("Maximum total absolute quantity", min_value=0.0, value=0.50, step=0.05, format="%.2f")
    optimization_scenarios = st.number_input("Stored paths used during search", min_value=min(500, len(result.terminal_market_caps)), max_value=len(result.terminal_market_caps), value=min(20_000, len(result.terminal_market_caps)), step=min(500, len(result.terminal_market_caps)))
    risk_aversion = st.number_input("SD penalty lambda", min_value=0.0, value=0.25, step=0.05)
    tail_weight = st.number_input("Expected-shortfall weight", min_value=0.0, value=0.10, step=0.05)

header_left, header_middle, header_right = st.columns(3)
with header_left:
    selected_ticker = st.selectbox("Selected Polymarket ticker", tickers, index=0)
with header_middle:
    polymarket_side = st.radio("Polymarket side", ["YES", "NO"], horizontal=True)
with header_right:
    yes_price = float(input_by_ticker.loc[selected_ticker, "Polymarket YES price"])
    default_entry = yes_price if polymarket_side == "YES" else 1.0 - yes_price
    entry_price = st.number_input(f"{polymarket_side} entry price", min_value=0.0, max_value=1.0, value=default_entry, step=0.01, key=f"phase5_entry_{selected_ticker}_{polymarket_side}")
polymarket_quantity = st.number_input("Polymarket shares", min_value=0.0, value=100.0, step=10.0)
base_payoff = polymarket_payoff(winners, selected_ticker=selected_ticker, side=polymarket_side, entry_price=float(entry_price), quantity=float(polymarket_quantity)).to_numpy(dtype=float)
baseline_metrics = payoff_metrics(base_payoff)

builder_tab, optimizer_tab, optimizer2_tab, chain_tab, payoff_tab, methodology_tab = st.tabs(["Manual Portfolio", "Optimizer", "Optimizer 2", "Option Chain", "Payoff Distribution", "Methodology"])

with builder_tab:
    st.subheader("Interactive option portfolio")
    st.write("For each row choose whether you define the leg by boundary confidence or by strike. The other field locks and is calculated automatically.")
    default_iv = float(input_by_ticker.loc[selected_ticker, "Implied volatility"])
    action_left, action_right, _ = st.columns([1, 1, 3])
    if action_left.button("Reset portfolio"):
        st.session_state.phase5_interactive_rows = default_interactive_rows(selected_ticker, default_iv)
        clear_interactive_widget_state()
        st.rerun()
    optimized_for_load = st.session_state.get("phase5_optimization")
    if action_right.button("Load optimized portfolio", disabled=optimized_for_load is None):
        st.session_state.phase5_interactive_rows = optimized_legs_to_interactive_rows(optimized_for_load.selected_legs, default_iv)
        clear_interactive_widget_state()
        st.rerun()
    interactive_inputs = render_interactive_leg_editor(tickers=tickers, curves=curves, default_ticker=selected_ticker, default_iv=default_iv, iv_by_ticker=input_by_ticker["Implied volatility"].astype(float), normalized_spot=NORMALIZED_SPOT)
    try:
        resolved_legs = resolve_manual_option_legs(interactive_inputs, pd.DataFrame(), time_to_expiry=time_to_expiry, risk_free_rate=float(risk_free_rate), normalized_spot=NORMALIZED_SPOT)
        active_metadata = interactive_inputs[interactive_inputs["Active"]].reset_index(drop=True)
        if not resolved_legs.empty:
            resolved_legs["Strike source"] = active_metadata["Definition mode"].to_numpy()
            resolved_legs["Boundary used"] = active_metadata.apply(lambda row: f"{row['Implied confidence (%)']:.1f}% {row['Boundary type']}" if row["Definition mode"] == "Strike" else f"{row['Boundary confidence (%)']:.1f}% {row['Boundary type']}", axis=1).to_numpy()
        option_payoff, leg_analytics = manual_option_payoffs_and_analytics(resolved_legs, normalized_terminal_prices, contract_multiplier=float(contract_multiplier), include_premiums=bool(include_premiums))
        total_payoff = base_payoff + option_payoff
        manual_metrics = payoff_metrics(total_payoff)
        manual_profile = make_profile(total_payoff, option_payoff, base_payoff, selected_ticker)
        st.session_state.phase5_manual_total_payoff = total_payoff
        st.session_state.phase5_manual_profile = manual_profile
        st.session_state.phase5_manual_legs = resolved_legs
        st.subheader("Polymarket-only versus manual portfolio")
        st.dataframe(metrics_comparison(baseline_metrics, manual_metrics, "Manual portfolio"), use_container_width=True, hide_index=True)
        st.subheader("Resolved portfolio")
        st.dataframe(display_legs(resolved_legs), use_container_width=True, hide_index=True)
        st.subheader("Standalone leg analytics")
        st.dataframe(display_leg_analytics(leg_analytics), use_container_width=True, hide_index=True)
        sensitivity = calculate_boundary_quantity_sensitivity(base_payoff, normalized_terminal_prices[selected_ticker].to_numpy(dtype=float), curves[selected_ticker], polymarket_side=polymarket_side, volatility=default_iv, time_to_expiry=time_to_expiry, risk_free_rate=float(risk_free_rate), include_premiums=bool(include_premiums), contract_multiplier=float(contract_multiplier), normalized_spot=NORMALIZED_SPOT)
        render_boundary_quantity_sensitivity(sensitivity, polymarket_side=polymarket_side)
    except Exception as exc:
        st.error(str(exc))

with optimizer_tab:
    st.subheader("Flexible option optimizer")
    st.caption("Quantities are searched on the displayed grid. Only one same-direction leg per ticker and option type is allowed; opposite-direction vertical spreads remain available.")
    option_underlyings = st.multiselect("Option underlyings", tickers, default=[selected_ticker])
    pricing_iv_source = input_by_ticker["Implied volatility"].reindex(option_underlyings)
    pricing_iv_table = pd.DataFrame({"Ticker": option_underlyings, "Option pricing IV": pricing_iv_source.to_numpy(dtype=float)})
    edited_pricing_ivs = st.data_editor(pricing_iv_table, use_container_width=True, hide_index=True, column_config={"Ticker": st.column_config.TextColumn(disabled=True), "Option pricing IV": st.column_config.NumberColumn(min_value=0.0001, max_value=5.0, step=0.01, format="%.2f")})
    validation_error = None
    if not option_underlyings:
        validation_error = "Select at least one option underlying."
    elif not include_calls and not include_puts:
        validation_error = "Enable calls, puts, or both."
    elif not allow_long and not allow_short:
        validation_error = "Enable long positions, short positions, or both."
    elif lower_strike_quantile >= upper_strike_quantile:
        validation_error = "Lower strike quantile must be below upper strike quantile."
    candidates = payoff_matrix = None
    if validation_error:
        st.error(validation_error)
    else:
        iv_lookup = edited_pricing_ivs.set_index("Ticker")["Option pricing IV"].astype(float)
        candidate_tables, payoff_matrices = [], []
        for ticker in option_underlyings:
            terminal_values = normalized_terminal_prices[ticker].to_numpy(dtype=float)
            lower_strike = float(np.quantile(terminal_values, float(lower_strike_quantile)))
            upper_strike = float(np.quantile(terminal_values, float(upper_strike_quantile)))
            strike_multipliers = np.unique(np.append(np.linspace(lower_strike / NORMALIZED_SPOT, upper_strike / NORMALIZED_SPOT, int(strike_grid_points)), 1.0))
            chain = build_candidate_option_universe(ticker=ticker, spot=NORMALIZED_SPOT, volatility=float(iv_lookup.loc[ticker]), time_to_expiry=time_to_expiry, risk_free_rate=float(risk_free_rate), strike_multipliers=strike_multipliers, include_calls=bool(include_calls), include_puts=bool(include_puts))
            chain = add_boundary_labels(chain, ticker)
            candidate_tables.append(chain)
            payoff_matrices.append(long_option_payoff_matrix(normalized_terminal_prices[ticker], chain, contract_multiplier=float(contract_multiplier), include_premiums=bool(include_premiums)))
        candidates = pd.concat(candidate_tables, ignore_index=True)
        payoff_matrix = np.concatenate(payoff_matrices, axis=1)
        st.session_state.phase5_live_candidates = candidates
    auto_optimize = st.checkbox("Auto-update optimizer", value=False)
    update_optimizer = st.button("Update optimized portfolio", type="primary")
    if (auto_optimize or update_optimizer) and validation_error is None:
        with st.spinner("Optimizing quantities and strikes on stored paths; Monte Carlo is not rerunning..."):
            try:
                optimized = optimize_option_portfolio(base_payoff, payoff_matrix, candidates, quantity_min=-float(max_quantity_per_leg) if allow_short else 0.0, quantity_max=float(max_quantity_per_leg) if allow_long else 0.0, quantity_step=float(quantity_step), max_legs=int(max_legs), max_total_absolute_quantity=float(max_total_quantity), objective=objective, risk_aversion=float(risk_aversion), tail_weight=float(tail_weight), optimization_scenarios=int(optimization_scenarios), seed=int(run_metadata.get("seed", 42)))
                optimized_option_payoff = optimized.optimized_payoffs - base_payoff
                st.session_state.phase5_optimization = optimized
                st.session_state.phase5_base_payoff = base_payoff
                st.session_state.phase5_profile = make_profile(optimized.optimized_payoffs, optimized_option_payoff, base_payoff, selected_ticker)
                st.session_state.phase5_selected_ticker = selected_ticker
                st.session_state.phase5_error = None
            except Exception as exc:
                st.session_state.phase5_error = str(exc)
    if st.session_state.get("phase5_error"):
        st.error(st.session_state.phase5_error)
    optimized = st.session_state.get("phase5_optimization")
    if optimized is None:
        st.info("Update the optimizer once or enable auto-update. Monte Carlo will not rerun.")
    else:
        st.dataframe(metrics_comparison(optimized.baseline_metrics, optimized.optimized_metrics, "Optimized portfolio"), use_container_width=True, hide_index=True)
        st.dataframe(display_legs(optimized.selected_legs), use_container_width=True, hide_index=True)
        if st.button("Load this result into Manual Portfolio"):
            st.session_state.phase5_interactive_rows = optimized_legs_to_interactive_rows(optimized.selected_legs, default_iv)
            clear_interactive_widget_state()
            st.rerun()

with optimizer2_tab:
    render_robust_optimizer(
        base_payoff=base_payoff,
        option_payoff_matrix=payoff_matrix,
        candidates=candidates,
        terminal_prices=normalized_terminal_prices[selected_ticker].to_numpy(dtype=float),
        quantity_min=-float(max_quantity_per_leg) if allow_short else 0.0,
        quantity_max=float(max_quantity_per_leg) if allow_long else 0.0,
        quantity_step=float(quantity_step),
        max_legs=int(max_legs),
        max_total_quantity=float(max_total_quantity),
        default_minimum_ev=float(baseline_metrics["Expected payoff"]),
        optimization_scenarios=int(optimization_scenarios),
        seed=int(run_metadata.get("seed", 42)),
    )

with chain_tab:
    chain = st.session_state.get("phase5_live_candidates")
    if chain is None:
        st.info("Open the Optimizer tab and select option underlyings.")
    else:
        st.subheader("Normalized theoretical option chain")
        filter_left, filter_right = st.columns(2)
        chain_tickers = filter_left.multiselect("Filter tickers", sorted(chain["Ticker"].unique()), default=sorted(chain["Ticker"].unique()))
        chain_types = filter_right.multiselect("Filter option types", ["Call", "Put"], default=["Call", "Put"])
        st.dataframe(display_option_chain(chain[chain["Ticker"].isin(chain_tickers) & chain["Option type"].isin(chain_types)]), use_container_width=True, hide_index=True)

with payoff_tab:
    available_views = []
    if st.session_state.get("phase5_manual_total_payoff") is not None:
        available_views.append("Manual portfolio")
    if st.session_state.get("phase5_optimization") is not None:
        available_views.append("Optimized portfolio")
    if not available_views:
        st.info("Build a manual or optimized portfolio first.")
    else:
        view = st.radio("Payoff view", available_views, horizontal=True)
        if view == "Manual portfolio":
            portfolio_payoff, profile = st.session_state.phase5_manual_total_payoff, st.session_state.phase5_manual_profile
        else:
            optimized = st.session_state.phase5_optimization
            portfolio_payoff, profile = optimized.optimized_payoffs, st.session_state.phase5_profile
        st.plotly_chart(distribution_figure(base_payoff, portfolio_payoff, view), use_container_width=True)
        st.plotly_chart(payoff_profile_figure(profile, selected_ticker), use_container_width=True)
        st.plotly_chart(payoff_by_bin_figure(profile, selected_ticker), use_container_width=True)
        st.dataframe(display_profile(profile), use_container_width=True, hide_index=True)

with methodology_tab:
    st.subheader("Reciprocal strike-boundary editor")
    st.markdown("""
Each option row has one `Define by` control:

- **Boundary:** confidence is editable; strike is locked and calculated from the Phase 2 conditional curve.
- **Strike:** strike is editable; confidence is locked and interpolated from the same conditional curve.

The classic optimizer searches expected payoff, SD, or expected shortfall objectives. Optimizer 2 is separate: it raises the 1% and 5% payoff floors while penalizing variation in expected payoff across terminal-price bins, subject to a minimum expected-payoff constraint.

The strike range is data-driven from stored terminal-price scenarios. `EV / SD` is a simple payoff-efficiency diagnostic, not a Sharpe ratio. All calculations reuse stored Phase 1/4 scenarios.
    """)
