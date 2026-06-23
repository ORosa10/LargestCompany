from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from boundaries import (
    calculate_boundaries_for_all_tickers,
    calculate_conditional_win_curve,
    find_market_cap_boundary_for_winner_probability,
    find_probability_boundaries,
)
from correlations import ewma_correlation, fetch_adjusted_close, rolling_correlation, smooth_vol_adjusted_correlation
from market_data import apply_market_caps, fetch_market_caps
from model import default_company_inputs, run_probability_engine


st.set_page_config(page_title="Phase 2", layout="wide")
st.title("Phase 2")
st.caption("Conditional probability boundaries from Phase 1 Monte Carlo scenarios. No hedging or payoff optimization yet.")

CORRELATION_METHODS = [
    "EWMA historical correlation",
    "Vol-adjusted smooth correlation",
    "Rolling historical correlation",
]

CONFIDENCE_LEVELS = [0.80, 0.90, 0.95, 0.99]


@st.cache_data(show_spinner=False, ttl=15 * 60)
def load_yahoo_market_caps(tickers: tuple[str, ...]) -> pd.DataFrame:
    return fetch_market_caps(list(tickers))


@st.cache_data(show_spinner=False, ttl=60 * 60)
def load_adjusted_close(tickers: tuple[str, ...], period: str) -> pd.DataFrame:
    return fetch_adjusted_close(list(tickers), period=period)


def pct(value: float) -> str:
    return "" if pd.isna(value) else f"{value:.2%}"


def dollars_trillions(value: float) -> str:
    return "" if pd.isna(value) else f"${value / 1e12:,.2f}T"


def display_market_caps(market_caps: pd.DataFrame) -> pd.DataFrame:
    display = market_caps.copy().rename(columns={"ticker": "Ticker", "yahoo_ticker": "Yahoo ticker", "market_cap": "Market cap", "source": "Source"})
    display["Market cap"] = display["Market cap"].map(dollars_trillions)
    return display[["Ticker", "Yahoo ticker", "Market cap", "Source"]]


def display_boundary_table(table: pd.DataFrame) -> pd.DataFrame:
    display = table.copy()
    for column in ["Lower loss boundary", "Upper win boundary"]:
        display[column] = display[column].map(dollars_trillions)
    for column in ["Confidence level", "Lower loss boundary / current", "Upper win boundary / current"]:
        display[column] = display[column].map(pct)
    return display


def display_curve_table(table: pd.DataFrame) -> pd.DataFrame:
    display = table.copy().rename(
        columns={
            "bin_low": "Bin low",
            "bin_high": "Bin high",
            "average_market_cap": "Average terminal market cap",
            "market_cap_to_current": "Average cap / current",
            "win_probability": "Win probability",
            "loss_probability": "Loss probability",
            "average_rank": "Average rank",
            "scenario_count": "Scenario count",
        }
    )
    for column in ["Bin low", "Bin high", "Average terminal market cap"]:
        display[column] = display[column].map(dollars_trillions)
    for column in ["Average cap / current", "Win probability", "Loss probability"]:
        display[column] = display[column].map(pct)
    display["Average rank"] = display["Average rank"].map(lambda value: f"{value:.2f}")
    return display[["Bin low", "Bin high", "Average terminal market cap", "Average cap / current", "Win probability", "Loss probability", "Average rank", "Scenario count"]]


def display_distribution_table(table: pd.DataFrame) -> pd.DataFrame:
    display = table.copy().rename(
        columns={
            "bin_label": "Terminal cap / current bin",
            "scenario_probability": "Scenario probability",
            "win_probability": "Conditional win probability",
            "win_contribution": "Contribution to total P(#1)",
            "average_rank": "Average rank",
            "scenario_count": "Scenario count",
        }
    )
    for column in ["Scenario probability", "Conditional win probability", "Contribution to total P(#1)"]:
        display[column] = display[column].map(pct)
    display["Average rank"] = display["Average rank"].map(lambda value: f"{value:.2f}")
    return display[["Terminal cap / current bin", "Scenario probability", "Conditional win probability", "Contribution to total P(#1)", "Average rank", "Scenario count"]]


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


def selected_result_row(results: pd.DataFrame, ticker: str) -> pd.Series:
    row = results.loc[results["Ticker"] == ticker]
    if row.empty:
        raise ValueError(f"Ticker {ticker} is not in results.")
    return row.iloc[0]


def calculate_probability_weighted_bins(
    terminal_market_caps: pd.DataFrame,
    ranks: pd.DataFrame,
    selected_ticker: str,
    current_market_cap: float,
    bin_width: float,
) -> pd.DataFrame:
    terminal_ratio = terminal_market_caps[selected_ticker].astype(float) / float(current_market_cap)
    selected_ranks = ranks[selected_ticker].astype(float)
    data = pd.DataFrame({"terminal_ratio": terminal_ratio, "rank": selected_ranks})
    data["won"] = data["rank"] == 1
    data = data.dropna()
    if data.empty:
        return pd.DataFrame()

    low_edge = np.floor(data["terminal_ratio"].min() / bin_width) * bin_width
    high_edge = np.ceil(data["terminal_ratio"].max() / bin_width) * bin_width
    edges = np.arange(low_edge, high_edge + bin_width, bin_width)
    if len(edges) < 2:
        edges = np.array([low_edge, low_edge + bin_width])

    data["bin"] = pd.cut(data["terminal_ratio"], bins=edges, include_lowest=True, right=False)
    grouped = data.groupby("bin", observed=True)
    total_count = len(data)
    table = grouped.agg(
        bin_low=("terminal_ratio", "min"),
        bin_high=("terminal_ratio", "max"),
        average_ratio=("terminal_ratio", "mean"),
        win_probability=("won", "mean"),
        average_rank=("rank", "mean"),
        scenario_count=("won", "size"),
    ).reset_index(drop=True)
    table["scenario_probability"] = table["scenario_count"] / total_count
    table["win_contribution"] = table["scenario_probability"] * table["win_probability"]
    table["bin_label"] = table.apply(lambda row: f"{row['bin_low']:.0%} to {row['bin_high']:.0%}", axis=1)
    return table


def conditional_probability_figure(curve: pd.DataFrame, boundaries: pd.DataFrame, confidence_levels: list[float], selected_ticker: str) -> go.Figure:
    fig = px.line(
        curve,
        x="market_cap_to_current",
        y="win_probability",
        markers=True,
        title=f"{selected_ticker}: P(win | terminal market cap level)",
        labels={"market_cap_to_current": "Selected terminal market cap / current market cap", "win_probability": "Conditional probability of finishing #1"},
    )
    for confidence in confidence_levels:
        fig.add_hline(y=confidence, line_dash="dot", annotation_text=f"{confidence:.0%}", annotation_position="right")
    for _, row in boundaries.iterrows():
        if pd.notna(row["Upper win boundary / current"]):
            fig.add_vline(
                x=float(row["Upper win boundary / current"]),
                line_dash="dash",
                annotation_text=f"win {row['Confidence level']:.0%}",
                annotation_position="top",
            )
        if pd.notna(row["Lower loss boundary / current"]):
            fig.add_vline(
                x=float(row["Lower loss boundary / current"]),
                line_dash="dash",
                annotation_text=f"loss {row['Confidence level']:.0%}",
                annotation_position="bottom",
            )
    fig.update_yaxes(tickformat=".0%", range=[0, 1])
    fig.update_xaxes(tickformat=".0%")
    return fig


def probability_weighted_figure(distribution: pd.DataFrame, selected_ticker: str) -> go.Figure:
    fig = go.Figure()
    fig.add_bar(
        x=distribution["bin_label"],
        y=distribution["scenario_probability"],
        name="Scenario probability",
        marker_color="#7aa6ff",
        yaxis="y",
    )
    fig.add_scatter(
        x=distribution["bin_label"],
        y=distribution["win_probability"],
        name="Conditional win probability",
        mode="lines+markers",
        yaxis="y2",
        line=dict(color="#1f3a8a", width=3),
    )
    fig.add_bar(
        x=distribution["bin_label"],
        y=distribution["win_contribution"],
        name="Contribution to P(#1)",
        marker_color="#22c55e",
        opacity=0.55,
        yaxis="y",
    )
    fig.update_layout(
        title=f"{selected_ticker}: probability-weighted terminal market-cap bins",
        xaxis_title="Selected terminal market cap / current market cap",
        yaxis=dict(title="Scenario probability / contribution", tickformat=".0%"),
        yaxis2=dict(title="Conditional win probability", tickformat=".0%", overlaying="y", side="right", range=[0, 1]),
        barmode="group",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
    )
    return fig


def rank_figure(curve: pd.DataFrame, selected_ticker: str) -> go.Figure:
    fig = px.line(
        curve,
        x="market_cap_to_current",
        y="average_rank",
        markers=True,
        title=f"{selected_ticker}: average final rank by terminal market-cap bin",
        labels={"market_cap_to_current": "Selected terminal market cap / current market cap", "average_rank": "Average final rank"},
    )
    fig.update_xaxes(tickformat=".0%")
    fig.update_yaxes(autorange="reversed")
    return fig


with st.sidebar:
    st.header("Phase 2 controls")
    target_date = st.date_input("Target date / maturity", value=date.today() + timedelta(days=365), min_value=date.today() + timedelta(days=1))
    days_to_target = max((target_date - date.today()).days, 1)
    st.caption(f"Horizon: {days_to_target} days ({days_to_target / 365:.2f} years)")

    simulations = st.number_input("Monte Carlo simulations", min_value=2_000, max_value=1_000_000, value=100_000, step=10_000)
    seed = st.number_input("Random seed", min_value=0, value=42, step=1)
    n_bins = st.slider("Market-cap quantile bins", min_value=10, max_value=100, value=30, step=5)
    distribution_bin_width = st.selectbox("Distribution bin width", [0.05, 0.10, 0.20], index=1, format_func=lambda value: f"{value:.0%} points")

    st.header("Correlation")
    correlation_method = st.selectbox("Correlation method", CORRELATION_METHODS, index=0)
    price_history_period = st.selectbox("Yahoo price history", ["1y", "3y", "5y", "10y"], index=2)
    ewma_lambda = st.selectbox("EWMA lambda", [0.94, 0.97], index=1)
    rolling_lookback = st.selectbox("Rolling lookback days", [63, 126, 252, 504, 756], index=2)
    smooth_low_quantile = st.selectbox("Smooth low-vol bucket", [0.30, 0.40, 0.50], index=1, format_func=lambda value: f"{value:.0%}")
    smooth_high_quantile = st.selectbox("Smooth high-vol bucket", [0.50, 0.60, 0.70], index=1, format_func=lambda value: f"{value:.0%}")

    st.header("Confidence levels")
    selected_confidence_levels = st.multiselect("Boundary confidence levels", CONFIDENCE_LEVELS, default=CONFIDENCE_LEVELS, format_func=lambda value: f"{value:.0%}")

    run_button = st.button("Run Phase 2 analysis", type="primary", use_container_width=True)

conditional_tab, inverse_tab, roadmap_tab = st.tabs(["Conditional Boundaries", "Inverse Check", "Phase 2 Roadmap"])

with conditional_tab:
    st.subheader("Conditional Boundaries")
    st.write("This tab uses the actual Phase 1 Monte Carlo scenarios. For a selected ticker, it bins that ticker's simulated terminal market cap and estimates P(win), P(loss), and average rank inside each bin.")

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
        with st.spinner("Running Phase 1 scenarios and calculating conditional boundaries..."):
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
                result = run_probability_engine(
                    simulation_inputs,
                    corr,
                    days_to_target=int(days_to_target),
                    simulations=int(simulations),
                    seed=int(seed),
                )
                current_caps = simulation_inputs.set_index("Ticker")["Current market cap"]
                all_boundaries = calculate_boundaries_for_all_tickers(
                    result.terminal_market_caps,
                    current_caps,
                    [float(value) for value in selected_confidence_levels],
                    ranks=result.ranks,
                    n_bins=int(n_bins),
                )
                st.session_state.boundary_inputs_used = simulation_inputs
                st.session_state.boundary_market_caps = market_caps
                st.session_state.boundary_corr = corr
                st.session_state.boundary_result = result
                st.session_state.boundary_all_boundaries = all_boundaries
                st.session_state.boundary_error = None
            except Exception as exc:
                st.session_state.boundary_error = str(exc)

    if st.session_state.get("boundary_error"):
        st.error(st.session_state.boundary_error)

    simulation_inputs = st.session_state.get("boundary_inputs_used")
    result = st.session_state.get("boundary_result")
    all_boundaries = st.session_state.get("boundary_all_boundaries")
    market_caps = st.session_state.get("boundary_market_caps")
    corr = st.session_state.get("boundary_corr")

    if simulation_inputs is None or result is None or all_boundaries is None:
        st.info("Run the Phase 2 analysis to calculate conditional probability boundaries.")
    else:
        tickers = simulation_inputs["Ticker"].tolist()
        selected_ticker = st.selectbox("Selected ticker", tickers, index=0)
        current_cap = float(simulation_inputs.set_index("Ticker").loc[selected_ticker, "Current market cap"])
        selected_row = selected_result_row(result.results, selected_ticker)

        summary_cols = st.columns(4)
        summary_cols[0].metric("Selected ticker", selected_ticker)
        summary_cols[1].metric("Unconditional P(#1)", pct(float(selected_row["Model probability"])))
        summary_cols[2].metric("Current market cap", dollars_trillions(current_cap))
        summary_cols[3].metric("Bins", f"{int(n_bins)}")

        curve = calculate_conditional_win_curve(
            result.terminal_market_caps,
            selected_ticker,
            ranks=result.ranks,
            current_market_cap=current_cap,
            n_bins=int(n_bins),
        )
        boundaries = find_probability_boundaries(
            curve,
            [float(value) for value in selected_confidence_levels],
            current_market_cap=current_cap,
            ticker=selected_ticker,
        )
        distribution = calculate_probability_weighted_bins(
            result.terminal_market_caps,
            result.ranks,
            selected_ticker,
            current_cap,
            float(distribution_bin_width),
        )

        st.subheader("Boundary table")
        st.dataframe(display_boundary_table(boundaries), use_container_width=True, hide_index=True)

        st.subheader("Conditional probability chart")
        st.plotly_chart(conditional_probability_figure(curve, boundaries, [float(value) for value in selected_confidence_levels], selected_ticker), use_container_width=True)

        st.subheader("Probability-weighted scenario distribution")
        st.write("Bars show how likely each terminal market-cap zone is. The line shows conditional win probability in that zone. Green bars show how much each zone contributes to total P(#1).")
        st.plotly_chart(probability_weighted_figure(distribution, selected_ticker), use_container_width=True)
        with st.expander("Probability-weighted bin table"):
            st.dataframe(display_distribution_table(distribution), use_container_width=True, hide_index=True)

        st.subheader("Rank chart")
        st.plotly_chart(rank_figure(curve, selected_ticker), use_container_width=True)

        st.subheader("Scenario table")
        st.dataframe(display_curve_table(curve), use_container_width=True, hide_index=True)

        with st.expander("All tickers boundary summary"):
            st.dataframe(display_boundary_table(all_boundaries), use_container_width=True, hide_index=True)

        with st.expander("Baseline ranking probabilities"):
            baseline_display = result.results.copy().rename(columns={"Model probability": "P(#1)", "Probability Top 2": "Top 2", "Probability Top 3": "Top 3"})
            baseline_display["Current market cap"] = baseline_display["Current market cap"].map(dollars_trillions)
            for column in ["Implied volatility", "Polymarket YES price", "P(#1)", "Edge", "Top 2", "Top 3"]:
                baseline_display[column] = baseline_display[column].map(pct)
            baseline_display["Average rank"] = baseline_display["Average rank"].map(lambda value: f"{value:.2f}")
            st.dataframe(
                baseline_display[["Ticker", "Current market cap", "Implied volatility", "Polymarket YES price", "P(#1)", "Edge", "Average rank", "Top 2", "Top 3"]],
                use_container_width=True,
                hide_index=True,
            )

        with st.expander("Yahoo market caps used"):
            st.dataframe(display_market_caps(market_caps), use_container_width=True, hide_index=True)

        with st.expander("Correlation matrix used"):
            st.dataframe(corr.style.format("{:.2%}"), use_container_width=True)

with inverse_tab:
    st.subheader("Inverse Check")
    st.write("This is diagnostic only. It asks what current market cap would make the full Monte Carlo P(#1) equal a selected probability, such as the manual Polymarket price. It is not the primary Phase 2 boundary method.")

    simulation_inputs = st.session_state.get("boundary_inputs_used")
    corr = st.session_state.get("boundary_corr")
    result = st.session_state.get("boundary_result")

    if simulation_inputs is None or corr is None or result is None:
        st.info("Run the Conditional Boundaries tab first.")
    else:
        tickers = simulation_inputs["Ticker"].tolist()
        inverse_ticker = st.selectbox("Ticker for inverse check", tickers, index=0, key="phase2_inverse_ticker")
        row = selected_result_row(result.results, inverse_ticker)
        inverse_target = st.number_input("Target P(#1)", min_value=0.01, max_value=0.99, value=float(row["Polymarket YES price"]), step=0.01, format="%.2f")
        if st.button("Run inverse check", type="secondary"):
            boundary = find_market_cap_boundary_for_winner_probability(
                simulation_inputs,
                corr,
                ticker=inverse_ticker,
                target_probability=float(inverse_target),
                days_to_target=int(days_to_target),
                simulations=int(simulations),
                seed=int(seed),
            )
            cols = st.columns(4)
            cols[0].metric("Target P(#1)", pct(float(inverse_target)))
            cols[1].metric("Implied current cap", dollars_trillions(boundary.boundary_market_cap))
            cols[2].metric("Move vs current", pct(boundary.relative_gap))
            cols[3].metric("Achieved P(#1)", pct(boundary.achieved_probability))

with roadmap_tab:
    st.subheader("Phase 2 Roadmap")
    st.write("Phase 2 should stay focused on conditional probability boundaries from simulated scenarios. New Phase 2 modules should be added here as tabs instead of separate sidebar pages.")
    st.markdown(
        """
- Conditional win/loss boundaries from simulated terminal market-cap bins: implemented.
- Conditional probability chart with confidence levels and boundary markers: implemented.
- Probability-weighted terminal market-cap bins: implemented.
- Rank chart by selected terminal market-cap bin: implemented.
- Scenario/bin table: implemented.
- All-ticker boundary summary: implemented.
- Inverse Polymarket-style check: diagnostic only.
- Hedging and payoff surfaces: defer to Phase 3 and Phase 4.
        """
    )
