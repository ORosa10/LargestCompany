from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from boundaries import calculate_boundaries_for_all_tickers
from manual_portfolio import (
    BOUNDARY_CONFIDENCES,
    OPTION_TYPES,
    POSITIONS,
    STRIKE_SOURCES,
    default_manual_portfolio,
    manual_option_payoffs_and_analytics,
    resolve_manual_option_legs,
)
from optimization import (
    OBJECTIVES,
    build_candidate_option_universe,
    long_option_payoff_matrix,
    optimize_option_portfolio,
    payoff_metrics,
)
from payoff_surface import polymarket_payoff, selected_payoff_profile_bins, terminal_stock_prices, winner_from_ranks
from phase4_ui import display_profile, dollars, payoff_by_bin_figure, payoff_profile_figure, pct
from simulation_store import load_simulation_snapshot


NORMALIZED_SPOT = 100.0

st.set_page_config(page_title="Phase 5", layout="wide")
st.title("Phase 5")
st.caption("Interactive Option Portfolio Dashboard. Reuse stored Monte Carlo paths, edit option legs directly, and monitor EV, SD, and tail risk without rerunning simulations.")


def available_snapshot() -> dict | None:
    if st.session_state.get("phase4_result") is not None and st.session_state.get("phase4_inputs_used") is not None:
        metadata = {}
        phase4_legs = st.session_state.get("phase4_option_legs")
        if phase4_legs is not None and not phase4_legs.empty and "Time to expiry" in phase4_legs.columns:
            metadata["days_to_target"] = int(round(float(phase4_legs["Time to expiry"].iloc[0]) * 365.0))
        return {
            "result": st.session_state.phase4_result,
            "simulation_inputs": st.session_state.phase4_inputs_used,
            "run_metadata": metadata,
            "source": "Phase 4 session snapshot",
        }
    if st.session_state.get("last_result") is not None and st.session_state.get("last_simulation_inputs") is not None:
        return {
            "result": st.session_state.last_result,
            "simulation_inputs": st.session_state.last_simulation_inputs,
            "run_metadata": st.session_state.get("last_run") or {},
            "source": "Phase 1 session snapshot",
        }
    return load_simulation_snapshot()


def metrics_comparison(baseline: pd.Series, portfolio: pd.Series, portfolio_name: str) -> pd.DataFrame:
    rows = []
    for label, metrics in [("Polymarket only", baseline), (portfolio_name, portfolio)]:
        rows.append({
            "Portfolio": label,
            "Expected payoff": dollars(float(metrics["Expected payoff"])),
            "Payoff SD": dollars(float(metrics["Payoff standard deviation"])),
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
    return display[["Ticker", "Option type", "Strike", "Strike / spot", "Spot", "Model IV", "Theoretical premium"]]


def display_legs(legs: pd.DataFrame) -> pd.DataFrame:
    display = legs.copy()
    for column in ["Strike", "Spot", "Theoretical premium"]:
        display[column] = display[column].map(dollars)
    for column in ["Strike / spot", "Model IV"]:
        display[column] = display[column].map(pct)
    order = [
        "Instrument", "Ticker", "Option type", "Position", "Quantity", "Strike", "Strike / spot",
        "Strike source", "Boundary used", "Theoretical premium", "Model IV",
    ]
    return display[[column for column in order if column in display.columns]]


def display_leg_analytics(analytics: pd.DataFrame) -> pd.DataFrame:
    display = analytics.copy()
    for column in [
        "Expected option payoff", "Option payoff SD", "Expected shortfall 5%",
        "Worst option payoff", "Initial premium cashflow",
    ]:
        display[column] = display[column].map(dollars)
    display["P(option loss)"] = display["P(option loss)"].map(pct)
    return display


def distribution_figure(baseline: np.ndarray, portfolio: np.ndarray, name: str) -> go.Figure:
    fig = go.Figure()
    fig.add_histogram(x=baseline, name="Polymarket only", opacity=0.55, nbinsx=80, histnorm="probability")
    fig.add_histogram(x=portfolio, name=name, opacity=0.55, nbinsx=80, histnorm="probability")
    fig.update_layout(
        title="Payoff distribution comparison",
        xaxis_title="Terminal payoff",
        yaxis_title="Scenario probability",
        barmode="overlay",
        yaxis_tickformat=".1%",
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
    )
    return fig


def optimized_legs_to_editor(legs: pd.DataFrame, fallback_ticker: str, fallback_iv: float) -> pd.DataFrame:
    if legs is None or legs.empty:
        return default_manual_portfolio(fallback_ticker, fallback_iv)
    rows = []
    for _, leg in legs.iterrows():
        rows.append({
            "Active": True,
            "Ticker": str(leg["Ticker"]),
            "Option type": str(leg["Option type"]),
            "Position": str(leg["Position"]),
            "Quantity": float(leg["Quantity"]),
            "Strike source": "Manual strike",
            "Boundary confidence (%)": 80,
            "Manual strike": float(leg["Strike"]),
            "Pricing IV": float(leg.get("Model IV", fallback_iv)),
        })
    return pd.DataFrame(rows)


def make_profile(
    total_payoff: np.ndarray,
    option_payoff: np.ndarray,
    base_payoff: np.ndarray,
    selected_ticker: str,
) -> pd.DataFrame:
    scenario = pd.DataFrame({
        "Winner": winners,
        "Selected terminal market cap": result.terminal_market_caps[selected_ticker],
        "Selected terminal stock price": normalized_terminal_prices[selected_ticker],
        "Polymarket payoff": base_payoff,
        "Option payoff": option_payoff,
        "Total payoff": total_payoff,
    })
    return selected_payoff_profile_bins(
        scenario,
        result.terminal_market_caps,
        current_caps,
        selected_ticker=selected_ticker,
        bins=20,
    )


snapshot = available_snapshot()
if snapshot is None:
    st.error("No Monte Carlo snapshot is available. Run Phase 1 once, then return here. The saved snapshot will survive future Streamlit restarts.")
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

snapshot_key = (
    snapshot.get("source"),
    run_metadata.get("target_date"),
    run_metadata.get("seed"),
    len(result.terminal_market_caps),
    tuple(tickers),
)
if st.session_state.get("phase5_boundary_snapshot_key") != snapshot_key:
    st.session_state.phase5_boundaries = calculate_boundaries_for_all_tickers(
        result.terminal_market_caps,
        current_caps,
        [value / 100.0 for value in BOUNDARY_CONFIDENCES],
        ranks=result.ranks,
        n_bins=30,
    )
    st.session_state.phase5_boundary_snapshot_key = snapshot_key
boundaries = st.session_state.phase5_boundaries

st.success(
    f"Using {snapshot.get('source', 'saved simulation')} | target {run_metadata.get('target_date', 'saved horizon')} | "
    f"{days_to_target} days | {len(result.terminal_market_caps):,} stored paths | normalized option spot = 100"
)
st.caption("All Phase 5 option prices are normalized: today = 100. A strike of 80 means 80% of today's stock price. No new Monte Carlo simulation runs on this page.")

with st.sidebar:
    st.header("Common payoff settings")
    contract_multiplier = st.number_input(
        "Payoff multiplier",
        min_value=0.01,
        value=1.0,
        step=0.25,
        help="Kept at 1 by default while prices are normalized to 100. Real listed-option contract multipliers can be introduced later.",
    )
    include_premiums = st.checkbox("Include theoretical premiums", value=True)
    risk_free_rate = st.number_input("Risk-free rate", min_value=0.0, max_value=0.20, value=0.04, step=0.005, format="%.3f")

    st.header("Optimizer strike universe")
    strike_min_pct = st.number_input("Minimum strike", min_value=10.0, max_value=300.0, value=50.0, step=5.0)
    strike_max_pct = st.number_input("Maximum strike", min_value=10.0, max_value=500.0, value=200.0, step=5.0)
    strike_step_pct = st.number_input("Strike step", min_value=1.0, max_value=50.0, value=10.0, step=1.0)
    include_calls = st.checkbox("Include calls", value=True)
    include_puts = st.checkbox("Include puts", value=True)
    allow_long = st.checkbox("Allow long positions", value=True)
    allow_short = st.checkbox("Allow short positions", value=True)

    st.header("Portfolio search")
    objective = st.selectbox("Objective", OBJECTIVES, index=1)
    max_legs = st.number_input("Maximum optimizer legs", min_value=0, max_value=10, value=4, step=1)
    max_quantity_per_leg = st.number_input("Maximum absolute quantity per leg", min_value=0.0, value=0.25, step=0.025, format="%.3f")
    quantity_step = st.number_input("Quantity grid step", min_value=0.001, value=0.025, step=0.005, format="%.3f")
    max_total_quantity = st.number_input("Maximum total absolute quantity", min_value=0.0, value=0.50, step=0.05, format="%.2f")
    optimization_scenarios = st.number_input(
        "Stored paths used during search",
        min_value=min(500, len(result.terminal_market_caps)),
        max_value=len(result.terminal_market_caps),
        value=min(20_000, len(result.terminal_market_caps)),
        step=min(500, len(result.terminal_market_caps)),
    )
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
    entry_price = st.number_input(
        f"{polymarket_side} entry price",
        min_value=0.0,
        max_value=1.0,
        value=default_entry,
        step=0.01,
        key=f"phase5_entry_{selected_ticker}_{polymarket_side}",
    )
polymarket_quantity = st.number_input("Polymarket shares", min_value=0.0, value=100.0, step=10.0)
base_payoff = polymarket_payoff(
    winners,
    selected_ticker=selected_ticker,
    side=polymarket_side,
    entry_price=float(entry_price),
    quantity=float(polymarket_quantity),
).to_numpy(dtype=float)
baseline_metrics = payoff_metrics(base_payoff)

builder_tab, optimizer_tab, chain_tab, payoff_tab, methodology_tab = st.tabs(
    ["Manual Portfolio", "Optimizer", "Option Chain", "Payoff Distribution", "Methodology"]
)

with builder_tab:
    st.subheader("Interactive option portfolio")
    st.write("Every active row is one option leg. Edit the table directly; portfolio metrics update immediately on the stored scenarios.")

    default_iv = float(input_by_ticker.loc[selected_ticker, "Implied volatility"])
    if "phase5_manual_editor" not in st.session_state:
        st.session_state.phase5_manual_editor = default_manual_portfolio(selected_ticker, default_iv)

    action_left, action_right, _ = st.columns([1, 1, 3])
    if action_left.button("Reset portfolio"):
        st.session_state.phase5_manual_editor = default_manual_portfolio(selected_ticker, default_iv)
        st.rerun()
    optimized_for_load = st.session_state.get("phase5_optimization")
    if action_right.button("Load optimized portfolio", disabled=optimized_for_load is None):
        st.session_state.phase5_manual_editor = optimized_legs_to_editor(
            optimized_for_load.selected_legs,
            selected_ticker,
            default_iv,
        )
        st.rerun()

    edited_manual = st.data_editor(
        st.session_state.phase5_manual_editor,
        num_rows="dynamic",
        use_container_width=True,
        hide_index=True,
        column_config={
            "Active": st.column_config.CheckboxColumn(),
            "Ticker": st.column_config.SelectboxColumn(options=tickers, required=True),
            "Option type": st.column_config.SelectboxColumn(options=OPTION_TYPES, required=True),
            "Position": st.column_config.SelectboxColumn(options=POSITIONS, required=True),
            "Quantity": st.column_config.NumberColumn(min_value=0.0, step=0.025, format="%.3f"),
            "Strike source": st.column_config.SelectboxColumn(options=STRIKE_SOURCES, required=True),
            "Boundary confidence (%)": st.column_config.SelectboxColumn(options=BOUNDARY_CONFIDENCES),
            "Manual strike": st.column_config.NumberColumn(min_value=0.01, step=5.0, format="%.2f"),
            "Pricing IV": st.column_config.NumberColumn(min_value=0.0001, max_value=5.0, step=0.01, format="%.2f"),
        },
        key="phase5_manual_portfolio_editor_widget",
    )
    st.session_state.phase5_manual_editor = edited_manual

    try:
        resolved_legs = resolve_manual_option_legs(
            edited_manual,
            boundaries,
            time_to_expiry=time_to_expiry,
            risk_free_rate=float(risk_free_rate),
            normalized_spot=NORMALIZED_SPOT,
        )
        option_payoff, leg_analytics = manual_option_payoffs_and_analytics(
            resolved_legs,
            normalized_terminal_prices,
            contract_multiplier=float(contract_multiplier),
            include_premiums=bool(include_premiums),
        )
        total_payoff = base_payoff + option_payoff
        manual_metrics = payoff_metrics(total_payoff)
        manual_profile = make_profile(total_payoff, option_payoff, base_payoff, selected_ticker)

        st.session_state.phase5_manual_total_payoff = total_payoff
        st.session_state.phase5_manual_profile = manual_profile
        st.session_state.phase5_manual_legs = resolved_legs

        st.subheader("Polymarket-only versus manual portfolio")
        st.dataframe(metrics_comparison(baseline_metrics, manual_metrics, "Manual portfolio"), use_container_width=True, hide_index=True)

        st.subheader("Resolved option legs")
        st.dataframe(display_legs(resolved_legs), use_container_width=True, hide_index=True)

        st.subheader("Standalone leg analytics")
        st.caption("Each row is evaluated alone on the stored paths, including its quantity and premium. Portfolio interaction is captured in the comparison table above.")
        st.dataframe(display_leg_analytics(leg_analytics), use_container_width=True, hide_index=True)
    except Exception as exc:
        st.error(str(exc))

with optimizer_tab:
    st.subheader("Flexible option optimizer")
    option_underlyings = st.multiselect("Option underlyings", tickers, default=[selected_ticker])
    pricing_iv_source = input_by_ticker["Implied volatility"].reindex(option_underlyings)
    pricing_iv_table = pd.DataFrame({"Ticker": option_underlyings, "Option pricing IV": pricing_iv_source.to_numpy(dtype=float)})
    edited_pricing_ivs = st.data_editor(
        pricing_iv_table,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Ticker": st.column_config.TextColumn(disabled=True),
            "Option pricing IV": st.column_config.NumberColumn(min_value=0.0001, max_value=5.0, step=0.01, format="%.2f"),
        },
    )
    st.caption("Pricing IV changes theoretical premiums only. Stored Monte Carlo paths remain unchanged.")

    validation_error = None
    if not option_underlyings:
        validation_error = "Select at least one option underlying."
    elif not include_calls and not include_puts:
        validation_error = "Enable calls, puts, or both."
    elif not allow_long and not allow_short:
        validation_error = "Enable long positions, short positions, or both."
    elif strike_min_pct >= strike_max_pct:
        validation_error = "Minimum strike must be below maximum strike."

    candidates = None
    payoff_matrix = None
    if validation_error:
        st.error(validation_error)
    else:
        strike_multipliers = np.arange(
            strike_min_pct / 100.0,
            strike_max_pct / 100.0 + strike_step_pct / 200.0,
            strike_step_pct / 100.0,
        )
        iv_lookup = edited_pricing_ivs.set_index("Ticker")["Option pricing IV"].astype(float)
        candidate_tables = []
        payoff_matrices = []
        for ticker in option_underlyings:
            chain = build_candidate_option_universe(
                ticker=ticker,
                spot=NORMALIZED_SPOT,
                volatility=float(iv_lookup.loc[ticker]),
                time_to_expiry=time_to_expiry,
                risk_free_rate=float(risk_free_rate),
                strike_multipliers=strike_multipliers,
                include_calls=bool(include_calls),
                include_puts=bool(include_puts),
            )
            candidate_tables.append(chain)
            payoff_matrices.append(
                long_option_payoff_matrix(
                    normalized_terminal_prices[ticker],
                    chain,
                    contract_multiplier=float(contract_multiplier),
                    include_premiums=bool(include_premiums),
                )
            )
        candidates = pd.concat(candidate_tables, ignore_index=True)
        payoff_matrix = np.concatenate(payoff_matrices, axis=1)
        st.session_state.phase5_live_candidates = candidates

    auto_optimize = st.checkbox("Auto-update optimizer", value=False)
    update_optimizer = st.button("Update optimized portfolio", type="primary")
    if (auto_optimize or update_optimizer) and validation_error is None:
        with st.spinner("Optimizing on stored paths; Monte Carlo is not rerunning..."):
            try:
                optimized = optimize_option_portfolio(
                    base_payoff,
                    payoff_matrix,
                    candidates,
                    quantity_min=-float(max_quantity_per_leg) if allow_short else 0.0,
                    quantity_max=float(max_quantity_per_leg) if allow_long else 0.0,
                    quantity_step=float(quantity_step),
                    max_legs=int(max_legs),
                    max_total_absolute_quantity=float(max_total_quantity),
                    objective=objective,
                    risk_aversion=float(risk_aversion),
                    tail_weight=float(tail_weight),
                    optimization_scenarios=int(optimization_scenarios),
                    seed=int(run_metadata.get("seed", 42)),
                )
                optimized_option_payoff = optimized.optimized_payoffs - base_payoff
                optimized_profile = make_profile(optimized.optimized_payoffs, optimized_option_payoff, base_payoff, selected_ticker)
                st.session_state.phase5_optimization = optimized
                st.session_state.phase5_base_payoff = base_payoff
                st.session_state.phase5_profile = optimized_profile
                st.session_state.phase5_selected_ticker = selected_ticker
                st.session_state.phase5_error = None
            except Exception as exc:
                st.session_state.phase5_error = str(exc)

    if st.session_state.get("phase5_error"):
        st.error(st.session_state.phase5_error)

    optimized = st.session_state.get("phase5_optimization")
    if optimized is None:
        st.info("Update the optimizer once or enable auto-update. This never reruns Monte Carlo.")
    else:
        st.dataframe(metrics_comparison(optimized.baseline_metrics, optimized.optimized_metrics, "Optimized portfolio"), use_container_width=True, hide_index=True)
        if optimized.selected_legs.empty:
            st.info("No improving option leg was found under these settings.")
        else:
            st.dataframe(display_legs(optimized.selected_legs), use_container_width=True, hide_index=True)
            if st.button("Load this result into Manual Portfolio"):
                st.session_state.phase5_manual_editor = optimized_legs_to_editor(optimized.selected_legs, selected_ticker, default_iv)
                st.rerun()

with chain_tab:
    chain = st.session_state.get("phase5_live_candidates")
    if chain is None:
        st.info("Open the Optimizer tab and select option underlyings.")
    else:
        st.subheader("Normalized theoretical option chain")
        st.caption("Spot is fixed at 100. Strikes therefore equal percentages of today's price. Premiums use editable fixed IV, maturity, and risk-free rate.")
        filter_left, filter_right = st.columns(2)
        chain_tickers = filter_left.multiselect("Filter tickers", sorted(chain["Ticker"].unique()), default=sorted(chain["Ticker"].unique()))
        chain_types = filter_right.multiselect("Filter option types", OPTION_TYPES, default=OPTION_TYPES)
        filtered_chain = chain[chain["Ticker"].isin(chain_tickers) & chain["Option type"].isin(chain_types)]
        st.dataframe(display_option_chain(filtered_chain), use_container_width=True, hide_index=True)

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
            portfolio_payoff = st.session_state.phase5_manual_total_payoff
            profile = st.session_state.phase5_manual_profile
        else:
            optimized = st.session_state.phase5_optimization
            portfolio_payoff = optimized.optimized_payoffs
            profile = st.session_state.phase5_profile
        st.plotly_chart(distribution_figure(base_payoff, portfolio_payoff, view), use_container_width=True)
        st.plotly_chart(payoff_profile_figure(profile, selected_ticker), use_container_width=True)
        st.plotly_chart(payoff_by_bin_figure(profile, selected_ticker), use_container_width=True)
        st.dataframe(display_profile(profile), use_container_width=True, hide_index=True)

with methodology_tab:
    st.subheader("How Phase 5 works")
    st.markdown(
        """
Phase 5 reuses stored Phase 1/4 scenarios. It never reruns the market-cap simulation when an option row changes.

### Normalized option prices

Every stock starts at `100`. Terminal option-underlying prices are:

```text
Normalized terminal price = 100 * terminal market cap / current market cap
```

A strike of 80 therefore means 80% of today's stock price. This keeps the research focused on relative moves rather than live executable option prices.

### Manual Portfolio

- Add or delete rows directly.
- Choose ticker, call/put, long/short, and quantity.
- Set a manual normalized strike or use an 80/90/95/99% Phase 2 win/loss boundary.
- Pricing IV determines the theoretical premium for that leg.
- Every edit immediately updates portfolio EV, SD, loss probability, expected shortfall, and standalone leg analytics.

### Optimizer

The optimizer searches the normalized option chain and can produce long puts, naked short calls, collars, capped collars, or other combinations. Its result can be loaded into the manual builder and modified freely.
        """
    )
