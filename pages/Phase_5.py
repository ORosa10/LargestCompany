from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from market_data import fetch_spot_prices
from optimization import OBJECTIVES, build_candidate_option_universe, long_option_payoff_matrix, optimize_option_portfolio
from payoff_surface import polymarket_payoff, selected_payoff_profile_bins, terminal_stock_prices, winner_from_ranks
from phase4_ui import display_profile, dollars, payoff_by_bin_figure, payoff_profile_figure, pct
from simulation_store import load_simulation_snapshot


st.set_page_config(page_title="Phase 5", layout="wide")
st.title("Phase 5")
st.caption("Option Chain & Optimization Dashboard. Reuse completed Monte Carlo scenarios and explore option structures without rerunning the probability engine.")


@st.cache_data(show_spinner=False, ttl=15 * 60)
def load_spots(tickers: tuple[str, ...]) -> pd.DataFrame:
    return fetch_spot_prices(list(tickers))


def available_snapshot() -> dict | None:
    """Prefer the newest in-session phase result, then the persisted Phase 1 snapshot."""
    if st.session_state.get("phase4_result") is not None and st.session_state.get("phase4_inputs_used") is not None:
        metadata = {}
        legs = st.session_state.get("phase4_option_legs")
        if legs is not None and not legs.empty and "Time to expiry" in legs.columns:
            metadata["days_to_target"] = int(round(float(legs["Time to expiry"].iloc[0]) * 365.0))
        return {
            "result": st.session_state.phase4_result,
            "simulation_inputs": st.session_state.phase4_inputs_used,
            "run_metadata": metadata,
            "source": "Phase 4 session snapshot",
            "spots": st.session_state.get("phase4_spots"),
        }
    if st.session_state.get("last_result") is not None and st.session_state.get("last_simulation_inputs") is not None:
        return {
            "result": st.session_state.last_result,
            "simulation_inputs": st.session_state.last_simulation_inputs,
            "run_metadata": st.session_state.get("last_run") or {},
            "source": "Phase 1 session snapshot",
            "spots": None,
        }
    return load_simulation_snapshot()


def metrics_comparison(baseline: pd.Series, optimized: pd.Series) -> pd.DataFrame:
    rows = []
    for label, metrics in [("Polymarket only", baseline), ("Optimized portfolio", optimized)]:
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
    display["Strike"] = display["Strike"].map(dollars)
    display["Strike / spot"] = display["Strike / spot"].map(pct)
    display["Spot"] = display["Spot"].map(dollars)
    display["Model IV"] = display["Model IV"].map(pct)
    display["Theoretical premium"] = display["Theoretical premium"].map(dollars)
    return display[["Ticker", "Option type", "Strike", "Strike / spot", "Spot", "Model IV", "Theoretical premium"]]


def display_optimized_legs(legs: pd.DataFrame) -> pd.DataFrame:
    display = legs.copy()
    for column in ["Strike", "Spot", "Theoretical premium"]:
        display[column] = display[column].map(dollars)
    for column in ["Strike / spot", "Model IV"]:
        display[column] = display[column].map(pct)
    order = ["Instrument", "Ticker", "Option type", "Position", "Quantity", "Strike", "Strike / spot", "Theoretical premium", "Model IV"]
    return display[[column for column in order if column in display.columns]]


def distribution_figure(baseline: np.ndarray, optimized: np.ndarray) -> go.Figure:
    fig = go.Figure()
    fig.add_histogram(x=baseline, name="Polymarket only", opacity=0.55, nbinsx=80, histnorm="probability")
    fig.add_histogram(x=optimized, name="Optimized portfolio", opacity=0.55, nbinsx=80, histnorm="probability")
    fig.update_layout(
        title="Payoff distribution comparison",
        xaxis_title="Terminal payoff",
        yaxis_title="Scenario probability",
        barmode="overlay",
        yaxis_tickformat=".1%",
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
    )
    return fig


snapshot = available_snapshot()
if snapshot is None:
    st.error("No Monte Carlo snapshot is available. Open Phase 1, run the baseline once, then return here. Future Streamlit restarts will restore that saved snapshot automatically.")
    st.stop()

result = snapshot["result"]
simulation_inputs = snapshot["simulation_inputs"].copy()
run_metadata = snapshot.get("run_metadata") or {}
tickers = result.terminal_market_caps.columns.astype(str).tolist()
days_to_target = int(run_metadata.get("days_to_target", 365))
time_to_expiry = max(days_to_target, 1) / 365.0

spots_table = snapshot.get("spots")
if spots_table is None:
    spots_table = load_spots(tuple(tickers))
spot_series = spots_table.set_index("ticker")["spot_price"].astype(float)
current_caps = simulation_inputs.set_index("Ticker")["Current market cap"].astype(float)
terminal_prices = terminal_stock_prices(result.terminal_market_caps, current_caps, spot_series)
winners = winner_from_ranks(result.ranks)

st.success(
    f"Using {snapshot.get('source', 'saved simulation')} | target {run_metadata.get('target_date', 'saved horizon')} | "
    f"{days_to_target} days | {len(result.terminal_market_caps):,} stored paths | seed {run_metadata.get('seed', 'saved')}"
)
st.caption("Changing option settings below does not rerun Monte Carlo. It reprices the option library and evaluates payoffs on these stored paths.")

with st.sidebar:
    st.header("Option-chain monitor")
    strike_min_pct = st.number_input("Minimum strike (% of spot)", min_value=10.0, max_value=300.0, value=50.0, step=5.0)
    strike_max_pct = st.number_input("Maximum strike (% of spot)", min_value=10.0, max_value=500.0, value=200.0, step=5.0)
    strike_step_pct = st.number_input("Strike step (percentage points)", min_value=1.0, max_value=50.0, value=10.0, step=1.0)
    include_calls = st.checkbox("Include calls", value=True)
    include_puts = st.checkbox("Include puts", value=True)
    allow_long = st.checkbox("Allow long positions", value=True)
    allow_short = st.checkbox("Allow short positions", value=True)
    contract_multiplier = st.number_input("Shares per option contract", min_value=1.0, value=100.0, step=1.0)
    include_premiums = st.checkbox("Include theoretical premiums", value=True)
    risk_free_rate = st.number_input("Risk-free rate", min_value=0.0, max_value=0.20, value=0.04, step=0.005, format="%.3f")

    st.header("Portfolio search")
    objective = st.selectbox("Objective", OBJECTIVES, index=1)
    max_legs = st.number_input("Maximum active option legs", min_value=0, max_value=10, value=4, step=1)
    max_quantity_per_leg = st.number_input("Maximum absolute quantity per leg", min_value=0.0, value=0.25, step=0.025, format="%.3f")
    quantity_step = st.number_input("Quantity grid step", min_value=0.001, value=0.025, step=0.005, format="%.3f")
    max_total_quantity = st.number_input("Maximum total absolute quantity", min_value=0.0, value=0.50, step=0.05, format="%.2f")
    optimization_scenarios = st.number_input(
        "Stored paths used during search",
        min_value=2_000,
        max_value=max(2_000, len(result.terminal_market_caps)),
        value=min(20_000, len(result.terminal_market_caps)),
        step=2_000,
    )
    risk_aversion = st.number_input("SD penalty lambda", min_value=0.0, value=0.25, step=0.05)
    tail_weight = st.number_input("Expected-shortfall weight", min_value=0.0, value=0.10, step=0.05)
    auto_optimize = st.checkbox("Auto-update optimizer", value=False, help="When enabled, changing any control immediately reruns only the option optimizer, never Monte Carlo.")
    update_button = st.button("Update optimized portfolio", type="primary", use_container_width=True)


results_tab, chain_tab, payoff_tab, methodology_tab = st.tabs(
    ["Portfolio Dashboard", "Option Chain Monitor", "Payoff Distribution", "Methodology"]
)

with results_tab:
    left, right = st.columns(2)
    with left:
        selected_ticker = st.selectbox("Selected Polymarket ticker", tickers, index=0)
        polymarket_side = st.segmented_control("Polymarket side", ["YES", "NO"], default="YES")
        yes_price = float(simulation_inputs.set_index("Ticker").loc[selected_ticker, "Polymarket YES price"])
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
    with right:
        option_underlyings = st.multiselect("Option underlyings", tickers, default=[selected_ticker])
        pricing_iv_source = simulation_inputs.set_index("Ticker")["Implied volatility"].reindex(option_underlyings)
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
        st.caption("These IVs reprice the option chain only. They do not alter the stored Monte Carlo paths.")

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
    base_payoff = polymarket_payoff(
        winners,
        selected_ticker=selected_ticker,
        side=polymarket_side,
        entry_price=float(entry_price),
        quantity=float(polymarket_quantity),
    ).to_numpy(dtype=float)

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
                spot=float(spot_series.loc[ticker]),
                volatility=float(iv_lookup.loc[ticker]),
                time_to_expiry=float(time_to_expiry),
                risk_free_rate=float(risk_free_rate),
                strike_multipliers=strike_multipliers,
                include_calls=bool(include_calls),
                include_puts=bool(include_puts),
            )
            candidate_tables.append(chain)
            payoff_matrices.append(
                long_option_payoff_matrix(
                    terminal_prices[ticker],
                    chain,
                    contract_multiplier=float(contract_multiplier),
                    include_premiums=bool(include_premiums),
                )
            )
        candidates = pd.concat(candidate_tables, ignore_index=True)
        payoff_matrix = np.concatenate(payoff_matrices, axis=1)
        st.session_state.phase5_live_candidates = candidates

    should_optimize = (update_button or auto_optimize) and validation_error is None
    if should_optimize:
        with st.spinner("Optimizing option legs on stored Monte Carlo paths..."):
            try:
                quantity_min = -float(max_quantity_per_leg) if allow_short else 0.0
                quantity_max = float(max_quantity_per_leg) if allow_long else 0.0
                optimized = optimize_option_portfolio(
                    base_payoff,
                    payoff_matrix,
                    candidates,
                    quantity_min=quantity_min,
                    quantity_max=quantity_max,
                    quantity_step=float(quantity_step),
                    max_legs=int(max_legs),
                    max_total_absolute_quantity=float(max_total_quantity),
                    objective=objective,
                    risk_aversion=float(risk_aversion),
                    tail_weight=float(tail_weight),
                    optimization_scenarios=int(optimization_scenarios),
                    seed=int(run_metadata.get("seed", 42)),
                )
                scenario = pd.DataFrame({
                    "Winner": winners,
                    "Selected terminal market cap": result.terminal_market_caps[selected_ticker],
                    "Selected terminal stock price": terminal_prices[selected_ticker],
                    "Polymarket payoff": base_payoff,
                    "Option payoff": optimized.optimized_payoffs - base_payoff,
                    "Total payoff": optimized.optimized_payoffs,
                })
                profile = selected_payoff_profile_bins(
                    scenario,
                    result.terminal_market_caps,
                    current_caps,
                    selected_ticker=selected_ticker,
                    bins=20,
                )
                st.session_state.phase5_optimization = optimized
                st.session_state.phase5_base_payoff = base_payoff
                st.session_state.phase5_profile = profile
                st.session_state.phase5_selected_ticker = selected_ticker
                st.session_state.phase5_settings = {
                    "objective": objective,
                    "selected_ticker": selected_ticker,
                    "side": polymarket_side,
                    "underlyings": option_underlyings,
                }
                st.session_state.phase5_error = None
            except Exception as exc:
                st.session_state.phase5_error = str(exc)

    if st.session_state.get("phase5_error"):
        st.error(st.session_state.phase5_error)

    optimized = st.session_state.get("phase5_optimization")
    if optimized is None:
        st.info("The option chain is live. Click Update optimized portfolio once, or enable Auto-update optimizer. Monte Carlo will not rerun.")
    else:
        settings = st.session_state.get("phase5_settings") or {}
        st.subheader("Current optimized portfolio")
        st.caption(f"Result settings: {settings.get('selected_ticker')} {settings.get('side')} | {settings.get('objective')} | underlyings: {', '.join(settings.get('underlyings', []))}")
        st.dataframe(metrics_comparison(optimized.baseline_metrics, optimized.optimized_metrics), use_container_width=True, hide_index=True)
        if optimized.selected_legs.empty:
            st.info("No improving option leg was found under these settings and constraints.")
        else:
            st.dataframe(display_optimized_legs(optimized.selected_legs), use_container_width=True, hide_index=True)

with chain_tab:
    chain = st.session_state.get("phase5_live_candidates")
    if chain is None:
        st.info("Select option underlyings on the Portfolio Dashboard.")
    else:
        st.subheader("Live theoretical option chain")
        st.caption("Generated from the current spot, editable fixed IV, maturity, risk-free rate, and strike grid. This table updates without Monte Carlo.")
        st.dataframe(display_option_chain(chain), use_container_width=True, hide_index=True)

with payoff_tab:
    optimized = st.session_state.get("phase5_optimization")
    profile = st.session_state.get("phase5_profile")
    if optimized is None or profile is None:
        st.info("Update the optimized portfolio first.")
    else:
        st.plotly_chart(distribution_figure(st.session_state.phase5_base_payoff, optimized.optimized_payoffs), use_container_width=True)
        selected = st.session_state.phase5_selected_ticker
        st.plotly_chart(payoff_profile_figure(profile, selected), use_container_width=True)
        st.plotly_chart(payoff_by_bin_figure(profile, selected), use_container_width=True)
        st.dataframe(display_profile(profile), use_container_width=True, hide_index=True)

with methodology_tab:
    st.subheader("Dashboard architecture")
    st.markdown(
        """
Phase 5 does not run a new market-cap simulation. It consumes the latest available snapshot from Phase 4, Phase 1 session state, or the persisted Phase 1 snapshot on disk.

Fast loop:

1. Reuse stored terminal market caps, ranks, and winners.
2. Fetch/cache current spot prices.
3. Generate a theoretical option chain from editable pricing IVs and the strike grid.
4. Calculate every option's payoff on the stored paths.
5. Search strikes and quantities under the selected objective.
6. Update EV, SD, loss probability, expected shortfall, and the weighted payoff profile.

Changing option IV affects theoretical premiums only. It intentionally does not rewrite the probability distribution. To change the underlying probability model, rerun Phase 1 and the saved snapshot will be replaced.
        """
    )
