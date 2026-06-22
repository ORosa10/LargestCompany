from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from boundaries import (
    find_market_cap_boundary_for_winner_probability,
    pairwise_boundary_table,
    winner_probability_curve,
)
from correlations import ewma_correlation, fetch_adjusted_close, rolling_correlation, smooth_vol_adjusted_correlation
from market_data import apply_market_caps, fetch_market_caps
from model import default_company_inputs, run_probability_engine


st.set_page_config(page_title="Phase 2", layout="wide")
st.title("Phase 2")
st.caption("Conditional probability boundaries. This phase extends the Phase 1 probability engine without adding hedging or payoff optimization yet.")

CORRELATION_METHODS = [
    "EWMA historical correlation",
    "Vol-adjusted smooth correlation",
    "Rolling historical correlation",
]

WINNER_TARGETS = [0.40, 0.50, 0.60, 0.70, 0.80, 0.90]
PAIR_TARGETS = [0.50, 0.60, 0.70, 0.80, 0.90]


@st.cache_data(show_spinner=False, ttl=15 * 60)
def load_yahoo_market_caps(tickers: tuple[str, ...]) -> pd.DataFrame:
    return fetch_market_caps(list(tickers))


@st.cache_data(show_spinner=False, ttl=60 * 60)
def load_adjusted_close(tickers: tuple[str, ...], period: str) -> pd.DataFrame:
    return fetch_adjusted_close(list(tickers), period=period)


def pct(value: float) -> str:
    return "" if pd.isna(value) else f"{value:.2%}"


def dollars_trillions(value: float) -> str:
    return f"${value / 1e12:,.2f}T"


def display_boundary_table(table: pd.DataFrame) -> pd.DataFrame:
    display = table.copy()
    for column in ["Boundary market cap", "Current market cap", "Gap vs current"]:
        display[column] = display[column].map(dollars_trillions)
    for column in ["Target P(#1)", "Achieved P(#1)", "Move vs current"]:
        display[column] = display[column].map(pct)
    return display


def display_pairwise_table(table: pd.DataFrame) -> pd.DataFrame:
    display = table.copy()
    for column in [
        "Boundary selected market cap",
        "Current selected market cap",
        "Competitor market cap",
        "Boundary gap vs current",
        "Boundary gap vs competitor",
    ]:
        display[column] = display[column].map(dollars_trillions)
    for column in [
        "Target pair probability",
        "Current pair probability",
        "Boundary move vs current",
        "Correlation",
        "Selected IV",
        "Competitor IV",
    ]:
        display[column] = display[column].map(pct)
    return display


def display_market_caps(market_caps: pd.DataFrame) -> pd.DataFrame:
    display = market_caps.copy().rename(columns={"ticker": "Ticker", "yahoo_ticker": "Yahoo ticker", "market_cap": "Market cap", "source": "Source"})
    display["Market cap"] = display["Market cap"].map(dollars_trillions)
    return display[["Ticker", "Yahoo ticker", "Market cap", "Source"]]


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


with st.sidebar:
    st.header("Phase 2 controls")
    target_date = st.date_input("Target date / maturity", value=date.today() + timedelta(days=365), min_value=date.today() + timedelta(days=1))
    days_to_target = max((target_date - date.today()).days, 1)
    st.caption(f"Horizon: {days_to_target} days ({days_to_target / 365:.2f} years)")

    simulations = st.number_input("Monte Carlo simulations", min_value=2_000, max_value=300_000, value=30_000, step=5_000)
    seed = st.number_input("Random seed", min_value=0, value=42, step=1)

    st.header("Correlation")
    correlation_method = st.selectbox("Correlation method", CORRELATION_METHODS, index=0)
    price_history_period = st.selectbox("Yahoo price history", ["1y", "3y", "5y", "10y"], index=2)
    ewma_lambda = st.selectbox("EWMA lambda", [0.94, 0.97], index=1)
    rolling_lookback = st.selectbox("Rolling lookback days", [63, 126, 252, 504, 756], index=2)
    smooth_low_quantile = st.selectbox("Smooth low-vol bucket", [0.30, 0.40, 0.50], index=1, format_func=lambda value: f"{value:.0%}")
    smooth_high_quantile = st.selectbox("Smooth high-vol bucket", [0.50, 0.60, 0.70], index=1, format_func=lambda value: f"{value:.0%}")

    st.header("Boundary targets")
    selected_winner_targets = st.multiselect("P(#1) targets", WINNER_TARGETS, default=[0.50, 0.60, 0.70, 0.80], format_func=lambda value: f"{value:.0%}")
    selected_pair_targets = st.multiselect("Pairwise targets", PAIR_TARGETS, default=[0.50, 0.60, 0.70], format_func=lambda value: f"{value:.0%}")

    run_button = st.button("Run Phase 2 analysis", type="primary", use_container_width=True)

conditional_tab, roadmap_tab = st.tabs(["Conditional Boundaries", "Phase 2 Roadmap"])

with conditional_tab:
    st.subheader("Conditional Boundaries")
    st.write("This tab asks where probabilities change: required market-cap levels, pairwise breakpoints, and P(#1) curves. Market caps are refreshed from Yahoo Finance; IV and Polymarket prices remain manual assumptions.")

    if "boundary_company_inputs" not in st.session_state:
        st.session_state.boundary_company_inputs = default_company_inputs()

    stored_inputs = st.session_state.boundary_company_inputs.copy()
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
    st.session_state.boundary_company_inputs = company_inputs

    if run_button:
        with st.spinner("Calculating conditional probability boundaries..."):
            try:
                tickers = [ticker for ticker in company_inputs["Ticker"].astype(str).str.strip().tolist() if ticker]
                market_caps = load_yahoo_market_caps(tuple(tickers))
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
                baseline = run_probability_engine(
                    simulation_inputs,
                    corr,
                    days_to_target=int(days_to_target),
                    simulations=int(simulations),
                    seed=int(seed),
                )
                st.session_state.boundary_inputs_used = simulation_inputs
                st.session_state.boundary_market_caps = market_caps
                st.session_state.boundary_corr = corr
                st.session_state.boundary_baseline = baseline.results
                st.session_state.boundary_error = None
            except Exception as exc:
                st.session_state.boundary_error = str(exc)

    if st.session_state.get("boundary_error"):
        st.error(st.session_state.boundary_error)

    simulation_inputs = st.session_state.get("boundary_inputs_used")
    corr = st.session_state.get("boundary_corr")
    baseline = st.session_state.get("boundary_baseline")
    market_caps = st.session_state.get("boundary_market_caps")

    if simulation_inputs is None or corr is None or baseline is None:
        st.info("Run the Phase 2 analysis to calculate conditional probability levels.")
    else:
        tickers = simulation_inputs["Ticker"].tolist()
        st.subheader("Baseline probabilities")
        baseline_display = baseline.copy().rename(columns={"Model probability": "P(#1)", "Probability Top 2": "Top 2", "Probability Top 3": "Top 3"})
        baseline_display["Current market cap"] = baseline_display["Current market cap"].map(dollars_trillions)
        for column in ["Implied volatility", "Polymarket YES price", "P(#1)", "Edge", "Top 2", "Top 3"]:
            baseline_display[column] = baseline_display[column].map(pct)
        baseline_display["Average rank"] = baseline_display["Average rank"].map(lambda value: f"{value:.2f}")
        st.dataframe(
            baseline_display[["Ticker", "Current market cap", "Implied volatility", "Polymarket YES price", "P(#1)", "Edge", "Average rank", "Top 2", "Top 3"]],
            use_container_width=True,
            hide_index=True,
        )

        selected_ticker = st.selectbox("Ticker for boundary detail", tickers, index=0)

        st.subheader(f"Winner probability boundaries: {selected_ticker}")
        boundary_rows = []
        for target_probability in selected_winner_targets:
            boundary = find_market_cap_boundary_for_winner_probability(
                simulation_inputs,
                corr,
                ticker=selected_ticker,
                target_probability=float(target_probability),
                days_to_target=int(days_to_target),
                simulations=int(simulations),
                seed=int(seed),
            )
            boundary_rows.append(
                {
                    "Ticker": boundary.ticker,
                    "Target P(#1)": boundary.target_probability,
                    "Achieved P(#1)": boundary.achieved_probability,
                    "Boundary market cap": boundary.boundary_market_cap,
                    "Current market cap": boundary.current_market_cap,
                    "Gap vs current": boundary.absolute_gap,
                    "Move vs current": boundary.relative_gap,
                    "Iterations": boundary.iterations,
                }
            )
        boundary_table = pd.DataFrame(boundary_rows)
        st.dataframe(display_boundary_table(boundary_table), use_container_width=True, hide_index=True)

        current_cap = float(simulation_inputs.loc[simulation_inputs["Ticker"] == selected_ticker, "Current market cap"].iloc[0])
        cap_multipliers = np.round(np.linspace(0.75, 1.35, 25), 4).tolist()
        curve = winner_probability_curve(
            simulation_inputs,
            corr,
            ticker=selected_ticker,
            cap_multipliers=cap_multipliers,
            days_to_target=int(days_to_target),
            simulations=int(simulations),
            seed=int(seed),
        )
        curve["Market cap ($T)"] = curve["Market cap"] / 1e12
        fig = px.line(
            curve,
            x="Market cap ($T)",
            y="P(#1)",
            markers=True,
            title=f"{selected_ticker}: P(#1) as market cap changes",
        )
        fig.add_vline(x=current_cap / 1e12, line_dash="dash", annotation_text="current")
        st.plotly_chart(fig, use_container_width=True)

        st.subheader(f"Pairwise boundaries: {selected_ticker} vs competitors")
        pairwise = pairwise_boundary_table(
            simulation_inputs,
            corr,
            selected_ticker=selected_ticker,
            target_probabilities=[float(value) for value in selected_pair_targets],
            days_to_target=int(days_to_target),
        )
        st.dataframe(display_pairwise_table(pairwise), use_container_width=True, hide_index=True)

        pair_heatmap = pairwise.pivot(index="Competitor", columns="Target pair probability", values="Boundary move vs current")
        st.plotly_chart(
            px.imshow(
                pair_heatmap,
                text_auto=".1%",
                color_continuous_scale="RdYlGn",
                title=f"{selected_ticker}: required move vs current for pairwise target probabilities",
            ),
            use_container_width=True,
        )

        with st.expander("Yahoo market caps used"):
            st.dataframe(display_market_caps(market_caps), use_container_width=True, hide_index=True)

        with st.expander("Correlation matrix used"):
            st.dataframe(corr.style.format("{:.2%}"), use_container_width=True)

        st.subheader("Methodology")
        st.write(
            "Winner boundaries are solved by repeatedly rerunning the Phase 1 Monte Carlo engine while changing only the selected company's current market cap. "
            "Pairwise boundaries use the analytic lognormal ratio formula for P(selected terminal cap > competitor terminal cap). "
            "This is still probability analysis only; no hedge construction or payoff optimization is included in Phase 2 yet."
        )

with roadmap_tab:
    st.subheader("Phase 2 Roadmap")
    st.write("Phase 2 should stay focused on probability boundaries, not hedging. New Phase 2 modules should be added here as tabs instead of separate sidebar pages.")
    st.markdown(
        """
- Conditional winner boundaries: implemented.
- Pairwise probability boundaries: implemented.
- Boundary sensitivity to IV and correlation assumptions: next candidate.
- Scenario export / comparison table: next candidate.
- Hedging and payoff surfaces: defer to Phase 3 and Phase 4.
        """
    )
