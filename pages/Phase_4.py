from __future__ import annotations

from datetime import date, timedelta

import pandas as pd
import plotly.express as px
import streamlit as st

from boundaries import calculate_boundaries_for_all_tickers
from correlations import ewma_correlation, fetch_adjusted_close, rolling_correlation, smooth_vol_adjusted_correlation
from market_data import apply_market_caps, fetch_market_caps, fetch_spot_prices
from model import default_company_inputs, run_probability_engine
from option_construction import attach_theoretical_premiums, construct_candidate_option_structure
from payoff_surface import calculate_scenario_payoffs, payoff_summary, payoff_surface_bins


st.set_page_config(page_title="Phase 4", layout="wide")
st.title("Phase 4")
st.caption("Payoff Surface Engine. This phase combines Polymarket payoff and candidate option legs across Monte Carlo scenarios. It does not optimize hedge ratios.")

CORRELATION_METHODS = [
    "EWMA historical correlation",
    "Vol-adjusted smooth correlation",
    "Rolling historical correlation",
]
CONFIDENCE_LEVELS = [0.80, 0.90, 0.95, 0.99]
CONSTRUCTION_MODE_LABELS = {
    "Selected-only hedge": "selected_only",
    "Selected + single competitor diagnostic": "single_competitor",
    "Selected + full universe competitors": "full_universe",
}


@st.cache_data(show_spinner=False, ttl=15 * 60)
def load_yahoo_market_caps(tickers: tuple[str, ...]) -> pd.DataFrame:
    return fetch_market_caps(list(tickers))


@st.cache_data(show_spinner=False, ttl=15 * 60)
def load_spot_prices(tickers: tuple[str, ...]) -> pd.DataFrame:
    return fetch_spot_prices(list(tickers))


@st.cache_data(show_spinner=False, ttl=60 * 60)
def load_adjusted_close(tickers: tuple[str, ...], period: str) -> pd.DataFrame:
    return fetch_adjusted_close(list(tickers), period=period)


def pct(value: float) -> str:
    return "" if pd.isna(value) else f"{value:.2%}"


def dollars(value: float) -> str:
    return "" if pd.isna(value) else f"${value:,.2f}"


def dollars_trillions(value: float) -> str:
    return "" if pd.isna(value) else f"${value / 1e12:,.2f}T"


def years(days: int) -> float:
    return max(int(days), 1) / 365.0


def build_correlation_matrix(
    method: str,
    prices: pd.DataFrame,
    simulation_inputs: pd.DataFrame,
    ewma_lambda: float,
    rolling_lookback: int,
    smooth_low_quantile: float,
    smooth_high_quantile: float,
) -> pd.DataFrame:
    if method == "EWMA historical correlation":
        return ewma_correlation(prices, ewma_lambda)
    if method == "Rolling historical correlation":
        return rolling_correlation(prices, rolling_lookback)
    if method == "Vol-adjusted smooth correlation":
        current_ivs = simulation_inputs.set_index("Ticker")["Implied volatility"].astype(float)
        corr, _ = smooth_vol_adjusted_correlation(
            prices,
            current_ivs,
            vol_window=63,
            low_quantile=smooth_low_quantile,
            high_quantile=smooth_high_quantile,
            min_observations=30,
        )
        return corr
    raise ValueError(f"Unknown correlation method: {method}")


def display_option_legs(table: pd.DataFrame) -> pd.DataFrame:
    display = table.copy()
    for column in ["Strike", "Spot", "Theoretical premium"]:
        if column in display.columns:
            display[column] = display[column].map(dollars)
    for column in ["Boundary market cap"]:
        if column in display.columns:
            display[column] = display[column].map(dollars_trillions)
    for column in ["Boundary / current cap", "Model IV", "Risk-free rate"]:
        if column in display.columns:
            display[column] = display[column].map(pct)
    if "Time to expiry" in display.columns:
        display["Time to expiry"] = display["Time to expiry"].map(lambda value: f"{value:.2f}y")
    return display


def display_summary(summary: pd.Series) -> pd.DataFrame:
    display = summary.to_frame("Value").reset_index().rename(columns={"index": "Metric"})
    display["Value"] = display["Value"].map(lambda value: pct(value) if "Probability" in str(display.loc[display["Value"] == value, "Metric"].iloc[0]) else dollars(value))
    return display


def display_scenarios(table: pd.DataFrame) -> pd.DataFrame:
    display = table.copy()
    for column in ["Selected terminal market cap"]:
        display[column] = display[column].map(dollars_trillions)
    for column in ["Polymarket payoff", "Option payoff", "Total payoff"]:
        display[column] = display[column].map(dollars)
    return display


def display_surface(table: pd.DataFrame) -> pd.DataFrame:
    display = table.copy()
    display["selected_ratio"] = display["selected_ratio"].map(pct)
    display["competitor_ratio"] = display["competitor_ratio"].map(pct)
    for column in ["expected_payoff", "weighted_payoff_contribution"]:
        display[column] = display[column].map(dollars)
    display["scenario_probability"] = display["scenario_probability"].map(pct)
    return display.rename(
        columns={
            "selected_ratio": "Selected cap / current",
            "competitor_ratio": "Competitor cap / current",
            "expected_payoff": "Expected payoff in bin",
            "scenario_probability": "Scenario probability",
            "weighted_payoff_contribution": "Contribution to expected payoff",
            "scenario_count": "Scenario count",
        }
    )


def heatmap_figure(surface: pd.DataFrame, selected_ticker: str, competitor_ticker: str):
    pivot = surface.pivot_table(
        index="competitor_ratio",
        columns="selected_ratio",
        values="expected_payoff",
        aggfunc="mean",
    ).sort_index(ascending=False)
    fig = px.imshow(
        pivot,
        aspect="auto",
        color_continuous_scale="RdYlGn",
        labels={"x": f"{selected_ticker} terminal cap / current", "y": f"{competitor_ticker} terminal cap / current", "color": "Avg payoff"},
        title=f"Payoff surface: {selected_ticker} vs {competitor_ticker}",
    )
    fig.update_xaxes(tickformat=".0%")
    fig.update_yaxes(tickformat=".0%")
    return fig


with st.sidebar:
    st.header("Phase 4 controls")
    target_date = st.date_input("Target date / maturity", value=date.today() + timedelta(days=365), min_value=date.today() + timedelta(days=1))
    days_to_target = max((target_date - date.today()).days, 1)
    st.caption(f"Horizon: {days_to_target} days ({days_to_target / 365:.2f} years)")

    simulations = st.number_input("Monte Carlo simulations", min_value=2_000, max_value=1_000_000, value=100_000, step=10_000)
    seed = st.number_input("Random seed", min_value=0, value=42, step=1)
    boundary_bins = st.slider("Phase 2 market-cap quantile bins", min_value=10, max_value=100, value=30, step=5)
    surface_bins = st.slider("Payoff surface bins", min_value=4, max_value=25, value=12, step=1)
    confidence_level = st.selectbox("Boundary confidence level", CONFIDENCE_LEVELS, index=3, format_func=lambda value: f"{value:.0%}")

    st.header("Polymarket position")
    polymarket_side = st.selectbox("Side", ["YES", "NO"], index=0)
    polymarket_quantity = st.number_input("Polymarket shares", min_value=0.0, value=100.0, step=10.0)

    st.header("Option construction")
    construction_mode_label = st.selectbox("Option construction mode", list(CONSTRUCTION_MODE_LABELS), index=0)
    construction_mode = CONSTRUCTION_MODE_LABELS[construction_mode_label]
    include_competitor_short_puts = st.checkbox("Include competitor short puts", value=construction_mode == "single_competitor")
    risk_free_rate = st.number_input("Risk-free rate", min_value=0.0, max_value=0.20, value=0.04, step=0.005, format="%.3f")
    contract_multiplier = st.number_input("Option contract multiplier", min_value=1.0, value=100.0, step=1.0)
    include_option_premiums = st.checkbox("Include theoretical option premiums", value=True)

    st.header("Correlation")
    correlation_method = st.selectbox("Correlation method", CORRELATION_METHODS, index=0)
    price_history_period = st.selectbox("Yahoo price history", ["1y", "3y", "5y", "10y"], index=2)
    ewma_lambda = st.selectbox("EWMA lambda", [0.94, 0.97], index=1)
    rolling_lookback = st.selectbox("Rolling lookback days", [63, 126, 252, 504, 756], index=2)
    smooth_low_quantile = st.selectbox("Smooth low-vol bucket", [0.30, 0.40, 0.50], index=1, format_func=lambda value: f"{value:.0%}")
    smooth_high_quantile = st.selectbox("Smooth high-vol bucket", [0.50, 0.60, 0.70], index=1, format_func=lambda value: f"{value:.0%}")

    run_button = st.button("Build payoff surface", type="primary", use_container_width=True)

summary_tab, surface_tab, scenarios_tab, methodology_tab = st.tabs(["Payoff Summary", "Payoff Surface", "Scenario Table", "Methodology"])

with summary_tab:
    st.subheader("Inputs")
    if "phase4_company_inputs" not in st.session_state:
        st.session_state.phase4_company_inputs = default_company_inputs()

    stored_inputs = st.session_state.phase4_company_inputs.copy()
    manual_inputs = stored_inputs[["Ticker", "Implied volatility", "Polymarket YES price"]]
    edited_manual_inputs = st.data_editor(
        manual_inputs,
        num_rows="dynamic",
        use_container_width=True,
        hide_index=True,
        column_config={
            "Ticker": st.column_config.TextColumn(required=True),
            "Implied volatility": st.column_config.NumberColumn("Manual IV", min_value=0.0001, max_value=5.0, step=0.01),
            "Polymarket YES price": st.column_config.NumberColumn("Manual Polymarket YES", min_value=0.0, max_value=1.0, step=0.01),
        },
    )
    fallback_caps = stored_inputs.set_index("Ticker")["Current market cap"].to_dict()
    company_inputs = edited_manual_inputs.copy()
    company_inputs["Current market cap"] = company_inputs["Ticker"].map(fallback_caps).fillna(1_000_000_000_000.0)
    company_inputs = company_inputs[["Ticker", "Current market cap", "Implied volatility", "Polymarket YES price"]]
    st.session_state.phase4_company_inputs = company_inputs

    tickers = [ticker for ticker in company_inputs["Ticker"].astype(str).str.strip().tolist() if ticker]
    selected_ticker = st.selectbox("Selected Polymarket ticker", tickers, index=0)
    competitor_options = [ticker for ticker in tickers if ticker != selected_ticker]
    surface_competitor_ticker = st.selectbox("Surface competitor axis", competitor_options, index=0)

    default_entry = float(company_inputs.loc[company_inputs["Ticker"] == selected_ticker, "Polymarket YES price"].iloc[0])
    polymarket_entry_price = st.number_input("Polymarket entry price", min_value=0.0, max_value=1.0, value=default_entry, step=0.01, format="%.2f")

    if run_button:
        with st.spinner("Running scenarios, constructing option candidates, and calculating payoff surface..."):
            try:
                market_caps = load_yahoo_market_caps(tuple(tickers))
                spots = load_spot_prices(tuple(tickers))
                simulation_inputs = apply_market_caps(company_inputs, market_caps)
                prices = load_adjusted_close(tuple(tickers), price_history_period)
                corr = build_correlation_matrix(
                    correlation_method,
                    prices,
                    simulation_inputs,
                    float(ewma_lambda),
                    int(rolling_lookback),
                    float(smooth_low_quantile),
                    float(smooth_high_quantile),
                )
                result = run_probability_engine(
                    simulation_inputs,
                    corr,
                    days_to_target=int(days_to_target),
                    simulations=int(simulations),
                    seed=int(seed),
                )
                current_caps = simulation_inputs.set_index("Ticker")["Current market cap"]
                boundaries = calculate_boundaries_for_all_tickers(
                    result.terminal_market_caps,
                    current_caps,
                    [float(confidence_level)],
                    ranks=result.ranks,
                    n_bins=int(boundary_bins),
                )
                spot_series = spots.set_index("ticker")["spot_price"]
                competitor_ticker = surface_competitor_ticker if construction_mode == "single_competitor" else None
                structure = construct_candidate_option_structure(
                    boundaries,
                    result.results,
                    current_caps,
                    spot_series,
                    selected_ticker=selected_ticker,
                    competitor_ticker=competitor_ticker,
                    confidence_level=float(confidence_level),
                    construction_mode=construction_mode,
                    include_competitor_short_puts=bool(include_competitor_short_puts),
                )
                valued_structure = attach_theoretical_premiums(
                    structure,
                    simulation_inputs.set_index("Ticker")["Implied volatility"],
                    time_to_expiry=years(int(days_to_target)),
                    risk_free_rate=float(risk_free_rate),
                )
                valued_structure["Quantity"] = 0.0
                st.session_state.phase4_inputs_used = simulation_inputs
                st.session_state.phase4_result = result
                st.session_state.phase4_spots = spots
                st.session_state.phase4_boundaries = boundaries
                st.session_state.phase4_option_legs = valued_structure
                st.session_state.phase4_selected_ticker = selected_ticker
                st.session_state.phase4_surface_competitor = surface_competitor_ticker
                st.session_state.phase4_polymarket_entry = float(polymarket_entry_price)
                st.session_state.phase4_error = None
            except Exception as exc:
                st.session_state.phase4_error = str(exc)

    if st.session_state.get("phase4_error"):
        st.error(st.session_state.phase4_error)

    option_legs = st.session_state.get("phase4_option_legs")
    result = st.session_state.get("phase4_result")
    simulation_inputs = st.session_state.get("phase4_inputs_used")
    spots = st.session_state.get("phase4_spots")

    if option_legs is None or result is None or simulation_inputs is None or spots is None:
        st.info("Build the payoff surface to generate scenario-level payoff outputs.")
    else:
        st.subheader("Candidate option legs and quantities")
        editable_legs = option_legs.copy()
        edited_legs = st.data_editor(
            editable_legs,
            use_container_width=True,
            hide_index=True,
            column_config={"Quantity": st.column_config.NumberColumn("Quantity", step=1.0)},
            disabled=[column for column in editable_legs.columns if column != "Quantity"],
        )
        st.session_state.phase4_option_legs = edited_legs

        selected = st.session_state.phase4_selected_ticker
        surface_competitor = st.session_state.phase4_surface_competitor
        current_caps = simulation_inputs.set_index("Ticker")["Current market cap"]
        spot_series = spots.set_index("ticker")["spot_price"]
        scenario = calculate_scenario_payoffs(
            result.terminal_market_caps,
            result.ranks,
            current_caps,
            spot_series,
            edited_legs,
            selected_ticker=selected,
            polymarket_side=polymarket_side,
            polymarket_entry_price=float(polymarket_entry_price),
            polymarket_quantity=float(polymarket_quantity),
            contract_multiplier=float(contract_multiplier),
            include_option_premiums=bool(include_option_premiums),
        )
        surface = payoff_surface_bins(
            scenario,
            result.terminal_market_caps,
            current_caps,
            selected_ticker=selected,
            competitor_ticker=surface_competitor,
            x_bins=int(surface_bins),
            y_bins=int(surface_bins),
        )
        st.session_state.phase4_scenario_payoffs = scenario
        st.session_state.phase4_surface = surface

        summary = payoff_summary(scenario)
        cols = st.columns(5)
        cols[0].metric("Expected payoff", dollars(float(summary["Expected payoff"])))
        cols[1].metric("Median payoff", dollars(float(summary["Median payoff"])))
        cols[2].metric("P(loss)", pct(float(summary["Probability of loss"])))
        cols[3].metric("Expected shortfall 5%", dollars(float(summary["Expected shortfall 5%"])))
        cols[4].metric("Worst payoff", dollars(float(summary["Worst payoff"])))

        st.subheader("Payoff components")
        component_summary = scenario[["Polymarket payoff", "Option payoff", "Total payoff"]].mean().to_frame("Expected payoff").reset_index().rename(columns={"index": "Component"})
        component_summary["Expected payoff"] = component_summary["Expected payoff"].map(dollars)
        st.dataframe(component_summary, use_container_width=True, hide_index=True)

        with st.expander("Option legs used"):
            st.dataframe(display_option_legs(edited_legs), use_container_width=True, hide_index=True)

with surface_tab:
    surface = st.session_state.get("phase4_surface")
    result = st.session_state.get("phase4_result")
    if surface is None or result is None:
        st.info("Build the payoff surface first.")
    else:
        selected = st.session_state.phase4_selected_ticker
        surface_competitor = st.session_state.phase4_surface_competitor
        st.plotly_chart(heatmap_figure(surface, selected, surface_competitor), use_container_width=True)
        st.subheader("Probability-weighted payoff bins")
        st.write("Each row is a two-dimensional scenario bin. Scenario probability times average payoff gives that bin's contribution to total expected payoff.")
        st.dataframe(display_surface(surface), use_container_width=True, hide_index=True)

with scenarios_tab:
    scenario = st.session_state.get("phase4_scenario_payoffs")
    if scenario is None:
        st.info("Build the payoff surface first.")
    else:
        st.subheader("Scenario-level payoff sample")
        st.dataframe(display_scenarios(scenario.head(500)), use_container_width=True, hide_index=True)
        st.caption("Showing first 500 simulated scenarios only.")

with methodology_tab:
    st.subheader("Methodology")
    st.markdown(
        """
Phase 4 evaluates payoff, but does not optimize anything.

Workflow:

- Run the same Monte Carlo scenario engine as Phase 1.
- Recalculate Phase 2 boundaries for the selected confidence level.
- Construct Phase 3 candidate option legs.
- Let the user enter option quantities manually.
- Calculate Polymarket payoff in each scenario.
- Convert terminal market caps into terminal stock prices and calculate option payoff in each scenario.
- Add Polymarket payoff and option payoff into total scenario payoff.
- Aggregate the scenario distribution into expected payoff, loss probability, tail loss, and payoff surface bins.

Polymarket payoff:

```text
YES payoff = 1 - entry price if selected ticker wins, otherwise -entry price
NO payoff  = 1 - entry price if selected ticker loses, otherwise -entry price
```

Option payoff uses the theoretical premiums from Phase 3 when enabled. Quantities are manual because optimization belongs to Phase 5.
        """
    )
