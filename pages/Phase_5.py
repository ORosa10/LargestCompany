from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from correlations import ewma_correlation, fetch_adjusted_close, rolling_correlation, smooth_vol_adjusted_correlation
from market_data import apply_market_caps, fetch_market_caps, fetch_spot_prices
from model import default_company_inputs, run_probability_engine
from optimization import (
    OBJECTIVES,
    build_candidate_option_universe,
    long_option_payoff_matrix,
    optimize_option_portfolio,
)
from payoff_surface import polymarket_payoff, selected_payoff_profile_bins, terminal_stock_prices, winner_from_ranks
from phase4_ui import display_profile, dollars, payoff_by_bin_figure, payoff_profile_figure, pct


st.set_page_config(page_title="Phase 5", layout="wide")
st.title("Phase 5")
st.caption("Optimization Engine. Search option strikes, positions, and quantities against the Monte Carlo payoff distribution.")

CORRELATION_METHODS = ["EWMA historical correlation", "Vol-adjusted smooth correlation", "Rolling historical correlation"]


@st.cache_data(show_spinner=False, ttl=15 * 60)
def load_market_caps(tickers: tuple[str, ...]) -> pd.DataFrame:
    return fetch_market_caps(list(tickers))


@st.cache_data(show_spinner=False, ttl=15 * 60)
def load_spots(tickers: tuple[str, ...]) -> pd.DataFrame:
    return fetch_spot_prices(list(tickers))


@st.cache_data(show_spinner=False, ttl=60 * 60)
def load_prices(tickers: tuple[str, ...], period: str) -> pd.DataFrame:
    return fetch_adjusted_close(list(tickers), period=period)


def build_correlation(method: str, prices: pd.DataFrame, inputs: pd.DataFrame, ewma_lambda: float, lookback: int) -> pd.DataFrame:
    if method == "EWMA historical correlation":
        return ewma_correlation(prices, ewma_lambda)
    if method == "Rolling historical correlation":
        return rolling_correlation(prices, lookback)
    ivs = inputs.set_index("Ticker")["Implied volatility"].astype(float)
    corr, _ = smooth_vol_adjusted_correlation(
        prices,
        ivs,
        vol_window=63,
        low_quantile=0.40,
        high_quantile=0.60,
        min_observations=30,
    )
    return corr


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


with st.sidebar:
    st.header("Phase 5 controls")
    target_date = st.date_input("Target date / maturity", value=date.today() + timedelta(days=365), min_value=date.today() + timedelta(days=1))
    days_to_target = max((target_date - date.today()).days, 1)
    time_to_expiry = days_to_target / 365.0
    st.caption(f"Horizon: {days_to_target} days ({time_to_expiry:.2f} years)")
    simulations = st.number_input("Monte Carlo simulations", min_value=5_000, max_value=500_000, value=50_000, step=5_000)
    seed = st.number_input("Random seed", min_value=0, value=42, step=1)

    st.header("Candidate option library")
    strike_min_pct = st.number_input("Minimum strike (% of spot)", min_value=10.0, max_value=300.0, value=50.0, step=5.0)
    strike_max_pct = st.number_input("Maximum strike (% of spot)", min_value=10.0, max_value=500.0, value=200.0, step=5.0)
    strike_step_pct = st.number_input("Strike grid step (percentage points)", min_value=1.0, max_value=50.0, value=10.0, step=1.0)
    include_calls = st.checkbox("Include calls", value=True)
    include_puts = st.checkbox("Include puts", value=True)
    allow_long = st.checkbox("Allow long positions", value=True)
    allow_short = st.checkbox("Allow short positions", value=True)
    contract_multiplier = st.number_input("Shares per option contract", min_value=1.0, value=100.0, step=1.0)
    include_premiums = st.checkbox("Include theoretical premiums", value=True)
    risk_free_rate = st.number_input("Risk-free rate", min_value=0.0, max_value=0.20, value=0.04, step=0.005, format="%.3f")

    st.header("Optimization")
    objective = st.selectbox("Objective", OBJECTIVES, index=1)
    max_legs = st.number_input("Maximum active option legs", min_value=0, max_value=10, value=4, step=1)
    max_quantity_per_leg = st.number_input("Maximum absolute quantity per leg", min_value=0.0, value=0.25, step=0.025, format="%.3f")
    quantity_step = st.number_input("Quantity grid step", min_value=0.001, value=0.025, step=0.005, format="%.3f")
    max_total_quantity = st.number_input("Maximum total absolute quantity", min_value=0.0, value=0.50, step=0.05, format="%.2f")
    optimization_scenarios = st.number_input("Scenarios used during search", min_value=2_000, max_value=100_000, value=20_000, step=2_000)
    risk_aversion = st.number_input("SD penalty lambda", min_value=0.0, value=0.25, step=0.05)
    tail_weight = st.number_input("Expected-shortfall weight", min_value=0.0, value=0.10, step=0.05)

    st.header("Correlation")
    correlation_method = st.selectbox("Correlation method", CORRELATION_METHODS, index=0)
    history_period = st.selectbox("Yahoo price history", ["1y", "3y", "5y", "10y"], index=2)
    ewma_lambda = st.selectbox("EWMA lambda", [0.94, 0.97], index=1)
    rolling_lookback = st.selectbox("Rolling lookback days", [63, 126, 252, 504, 756], index=2)

    run_button = st.button("Run optimization", type="primary", use_container_width=True)


results_tab, payoff_tab, library_tab, methodology_tab = st.tabs(["Optimization Results", "Payoff Distribution", "Candidate Library", "Methodology"])

with results_tab:
    st.subheader("Inputs")
    if "phase5_company_inputs" not in st.session_state:
        st.session_state.phase5_company_inputs = default_company_inputs()

    stored = st.session_state.phase5_company_inputs.copy()
    edited = st.data_editor(
        stored[["Ticker", "Implied volatility", "Polymarket YES price"]],
        use_container_width=True,
        hide_index=True,
        num_rows="dynamic",
        column_config={
            "Ticker": st.column_config.TextColumn(required=True),
            "Implied volatility": st.column_config.NumberColumn("Manual IV", min_value=0.0001, max_value=5.0, step=0.01),
            "Polymarket YES price": st.column_config.NumberColumn("Manual Polymarket YES", min_value=0.0, max_value=1.0, step=0.01),
        },
    )
    fallback_caps = stored.set_index("Ticker")["Current market cap"].to_dict()
    company_inputs = edited.copy()
    company_inputs["Current market cap"] = company_inputs["Ticker"].map(fallback_caps).fillna(1_000_000_000_000.0)
    company_inputs = company_inputs[["Ticker", "Current market cap", "Implied volatility", "Polymarket YES price"]]
    st.session_state.phase5_company_inputs = company_inputs

    tickers = company_inputs["Ticker"].astype(str).str.strip().tolist()
    selected_ticker = st.selectbox("Selected Polymarket ticker", tickers, index=0)
    option_underlyings = st.multiselect("Option underlyings", tickers, default=[selected_ticker])
    polymarket_side = st.selectbox("Polymarket side", ["YES", "NO"], index=0)
    yes_price = float(company_inputs.loc[company_inputs["Ticker"] == selected_ticker, "Polymarket YES price"].iloc[0])
    default_entry = yes_price if polymarket_side == "YES" else 1.0 - yes_price
    entry_price = st.number_input(
        f"Polymarket {polymarket_side} entry price",
        min_value=0.0,
        max_value=1.0,
        value=default_entry,
        step=0.01,
        key=f"phase5_entry_{selected_ticker}_{polymarket_side}",
    )
    polymarket_quantity = st.number_input("Polymarket shares", min_value=0.0, value=100.0, step=10.0)

    if run_button:
        if not option_underlyings:
            st.error("Select at least one option underlying.")
        elif not include_calls and not include_puts:
            st.error("Enable calls, puts, or both.")
        elif not allow_long and not allow_short:
            st.error("Enable long positions, short positions, or both.")
        elif strike_min_pct >= strike_max_pct:
            st.error("Minimum strike must be below maximum strike.")
        else:
            with st.spinner("Building candidate options and optimizing the scenario payoff..."):
                try:
                    market_caps = load_market_caps(tuple(tickers))
                    spots = load_spots(tuple(tickers))
                    simulation_inputs = apply_market_caps(company_inputs, market_caps)
                    prices = load_prices(tuple(tickers), history_period)
                    corr = build_correlation(correlation_method, prices, simulation_inputs, float(ewma_lambda), int(rolling_lookback))
                    probability_result = run_probability_engine(
                        simulation_inputs,
                        corr,
                        days_to_target=int(days_to_target),
                        simulations=int(simulations),
                        seed=int(seed),
                    )
                    current_caps = simulation_inputs.set_index("Ticker")["Current market cap"]
                    spot_series = spots.set_index("ticker")["spot_price"]
                    terminal_prices = terminal_stock_prices(probability_result.terminal_market_caps, current_caps, spot_series)
                    winners = winner_from_ranks(probability_result.ranks)
                    base_payoff = polymarket_payoff(
                        winners,
                        selected_ticker=selected_ticker,
                        side=polymarket_side,
                        entry_price=float(entry_price),
                        quantity=float(polymarket_quantity),
                    ).to_numpy(dtype=float)

                    strike_multipliers = np.arange(
                        strike_min_pct / 100.0,
                        strike_max_pct / 100.0 + strike_step_pct / 200.0,
                        strike_step_pct / 100.0,
                    )
                    candidate_tables = []
                    payoff_matrices = []
                    iv_series = simulation_inputs.set_index("Ticker")["Implied volatility"]
                    for ticker in option_underlyings:
                        candidate_table = build_candidate_option_universe(
                            ticker=ticker,
                            spot=float(spot_series.loc[ticker]),
                            volatility=float(iv_series.loc[ticker]),
                            time_to_expiry=float(time_to_expiry),
                            risk_free_rate=float(risk_free_rate),
                            strike_multipliers=strike_multipliers,
                            include_calls=bool(include_calls),
                            include_puts=bool(include_puts),
                        )
                        candidate_tables.append(candidate_table)
                        payoff_matrices.append(
                            long_option_payoff_matrix(
                                terminal_prices[ticker],
                                candidate_table,
                                contract_multiplier=float(contract_multiplier),
                                include_premiums=bool(include_premiums),
                            )
                        )
                    candidates = pd.concat(candidate_tables, ignore_index=True)
                    payoff_matrix = np.concatenate(payoff_matrices, axis=1)

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
                        seed=int(seed),
                    )

                    scenario = pd.DataFrame({
                        "Winner": winners,
                        "Selected terminal market cap": probability_result.terminal_market_caps[selected_ticker],
                        "Selected terminal stock price": terminal_prices[selected_ticker],
                        "Polymarket payoff": base_payoff,
                        "Option payoff": optimized.optimized_payoffs - base_payoff,
                        "Total payoff": optimized.optimized_payoffs,
                    })
                    profile = selected_payoff_profile_bins(
                        scenario,
                        probability_result.terminal_market_caps,
                        current_caps,
                        selected_ticker=selected_ticker,
                        bins=20,
                    )

                    st.session_state.phase5_optimization = optimized
                    st.session_state.phase5_candidates = candidates
                    st.session_state.phase5_base_payoff = base_payoff
                    st.session_state.phase5_scenario = scenario
                    st.session_state.phase5_profile = profile
                    st.session_state.phase5_selected_ticker = selected_ticker
                    st.session_state.phase5_error = None
                except Exception as exc:
                    st.session_state.phase5_error = str(exc)

    if st.session_state.get("phase5_error"):
        st.error(st.session_state.phase5_error)

    optimized = st.session_state.get("phase5_optimization")
    if optimized is None:
        st.info("Run the optimization to generate a portfolio.")
    else:
        st.subheader("Baseline versus optimized portfolio")
        st.dataframe(metrics_comparison(optimized.baseline_metrics, optimized.optimized_metrics), use_container_width=True, hide_index=True)
        st.caption(f"Optimization iterations: {optimized.iterations}. Search objective score: {optimized.objective_score:,.4f}.")

        st.subheader("Selected option structure")
        if optimized.selected_legs.empty:
            st.info("The optimizer did not find an improving option leg under the selected objective and constraints.")
        else:
            st.dataframe(display_optimized_legs(optimized.selected_legs), use_container_width=True, hide_index=True)

with payoff_tab:
    optimized = st.session_state.get("phase5_optimization")
    profile = st.session_state.get("phase5_profile")
    if optimized is None or profile is None:
        st.info("Run the optimization first.")
    else:
        st.plotly_chart(distribution_figure(st.session_state.phase5_base_payoff, optimized.optimized_payoffs), use_container_width=True)
        selected = st.session_state.phase5_selected_ticker
        st.plotly_chart(payoff_profile_figure(profile, selected), use_container_width=True)
        st.plotly_chart(payoff_by_bin_figure(profile, selected), use_container_width=True)
        st.dataframe(display_profile(profile), use_container_width=True, hide_index=True)

with library_tab:
    candidates = st.session_state.get("phase5_candidates")
    if candidates is None:
        st.info("Run the optimization first.")
    else:
        display = candidates.copy()
        display["Strike"] = display["Strike"].map(dollars)
        display["Strike / spot"] = display["Strike / spot"].map(pct)
        display["Theoretical premium"] = display["Theoretical premium"].map(dollars)
        st.dataframe(display, use_container_width=True, hide_index=True)

with methodology_tab:
    st.subheader("Methodology")
    st.markdown(
        """
Phase 5 searches a flexible vanilla-option library rather than assuming one fixed hedge template.

1. Create call and put candidates for every selected underlying and strike-grid point.
2. Price each option with the same simplified fixed-IV Black-Scholes model used in Phase 3.
3. Calculate every candidate's payoff in the Phase 1 Monte Carlo scenarios.
4. Use signed quantities: positive is long, negative is short.
5. Greedily add the best improving strike/quantity pair until the maximum number of legs is reached.
6. Refine quantities for the selected strikes with coordinate search.
7. Recalculate metrics on the full scenario set.

Available objectives:

- **Maximum expected payoff:** maximizes mean scenario payoff.
- **Risk-adjusted payoff:** maximizes `EV - lambda * SD`.
- **Tail-aware payoff:** maximizes `EV + weight * Expected Shortfall 5%`.
- **Minimum SD with baseline EV floor:** minimizes SD without accepting expected payoff below the Polymarket-only baseline.

The algorithm can construct long puts, naked short calls, collars, capped collars, or other multi-leg combinations. These are model outputs under simplified premiums and are not trade recommendations. Phase 6 will stress-test their robustness.
        """
    )
