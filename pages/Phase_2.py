from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from boundaries import (
    calculate_boundaries_for_all_tickers,
    calculate_conditional_win_curve,
    find_probability_boundaries,
)
from simulation_store import load_simulation_snapshot


st.set_page_config(page_title="Phase 2", layout="wide")
st.title("Phase 2: Conditional Probability Boundaries")
st.caption("Phase 2 analyzes the exact Monte Carlo paths saved by Phase 1. It does not reload market data, change IV, rebuild correlations, or run another simulation.")

CONFIDENCE_LEVELS = [0.80, 0.90, 0.95, 0.99]


def pct(value: float) -> str:
    return "" if pd.isna(value) else f"{value:.2%}"


def dollars_trillions(value: float) -> str:
    return "" if pd.isna(value) else f"${value / 1e12:,.2f}T"


def selected_result_row(results: pd.DataFrame, ticker: str) -> pd.Series:
    row = results.loc[results["Ticker"] == ticker]
    if row.empty:
        raise ValueError(f"Ticker {ticker} is not in the Phase 1 result.")
    return row.iloc[0]


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


def calculate_probability_weighted_bins(
    terminal_market_caps: pd.DataFrame,
    ranks: pd.DataFrame,
    selected_ticker: str,
    current_market_cap: float,
    bin_width: float,
) -> pd.DataFrame:
    data = pd.DataFrame(
        {
            "terminal_ratio": terminal_market_caps[selected_ticker].astype(float) / float(current_market_cap),
            "rank": ranks[selected_ticker].astype(float),
        }
    ).dropna()
    data["won"] = data["rank"] == 1
    if data.empty:
        return pd.DataFrame()

    low_edge = np.floor(data["terminal_ratio"].min() / bin_width) * bin_width
    high_edge = np.ceil(data["terminal_ratio"].max() / bin_width) * bin_width
    edges = np.arange(low_edge, high_edge + bin_width, bin_width)
    if len(edges) < 2:
        edges = np.array([low_edge, low_edge + bin_width])

    data["bin"] = pd.cut(data["terminal_ratio"], bins=edges, include_lowest=True, right=False)
    table = data.groupby("bin", observed=True).agg(
        bin_low=("terminal_ratio", "min"),
        bin_high=("terminal_ratio", "max"),
        win_probability=("won", "mean"),
        average_rank=("rank", "mean"),
        scenario_count=("won", "size"),
    ).reset_index(drop=True)
    table["scenario_probability"] = table["scenario_count"] / len(data)
    table["win_contribution"] = table["scenario_probability"] * table["win_probability"]
    table["bin_label"] = table.apply(lambda row: f"{row['bin_low']:.0%} to {row['bin_high']:.0%}", axis=1)
    return table


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


def conditional_probability_figure(
    curve: pd.DataFrame,
    boundaries: pd.DataFrame,
    confidence_levels: list[float],
    selected_ticker: str,
) -> go.Figure:
    fig = px.line(
        curve,
        x="market_cap_to_current",
        y="win_probability",
        markers=True,
        title=f"{selected_ticker}: P(win | terminal market cap level)",
        labels={
            "market_cap_to_current": "Terminal market cap / current market cap",
            "win_probability": "Conditional probability of finishing #1",
        },
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
    fig.add_bar(x=distribution["bin_label"], y=distribution["scenario_probability"], name="Scenario probability", marker_color="#7aa6ff", yaxis="y")
    fig.add_scatter(x=distribution["bin_label"], y=distribution["win_probability"], name="Conditional win probability", mode="lines+markers", yaxis="y2", line=dict(color="#1f3a8a", width=3))
    fig.add_bar(x=distribution["bin_label"], y=distribution["win_contribution"], name="Contribution to P(#1)", marker_color="#22c55e", opacity=0.55, yaxis="y")
    fig.update_layout(
        title=f"{selected_ticker}: probability-weighted terminal market-cap bins",
        xaxis_title="Terminal market cap / current market cap",
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
        labels={"market_cap_to_current": "Terminal market cap / current market cap", "average_rank": "Average final rank"},
    )
    fig.update_xaxes(tickformat=".0%")
    fig.update_yaxes(autorange="reversed")
    return fig


snapshot = load_simulation_snapshot()
if snapshot is None:
    st.error("No saved Phase 1 simulation was found. Open Phase 1 and click Run / refresh first.")
    st.stop()

result = snapshot["result"]
simulation_inputs = snapshot["simulation_inputs"].copy()
run = snapshot.get("run_metadata") or {}
source = snapshot.get("source", "Phase 1")
tickers = simulation_inputs["Ticker"].astype(str).tolist()
current_caps = simulation_inputs.set_index("Ticker")["Current market cap"].astype(float)

with st.sidebar:
    st.header("Phase 2 controls")
    selected_ticker = st.selectbox("Selected ticker", tickers, index=tickers.index("NVDA") if "NVDA" in tickers else 0)
    selected_confidence_levels = st.multiselect(
        "Boundary confidence levels",
        CONFIDENCE_LEVELS,
        default=CONFIDENCE_LEVELS,
        format_func=lambda value: f"{value:.0%}",
    )
    n_bins = st.slider("Conditional-curve quantile bins", min_value=10, max_value=100, value=30, step=5)
    distribution_bin_width = st.selectbox(
        "Probability-distribution bin width",
        [0.05, 0.10, 0.20],
        index=1,
        format_func=lambda value: f"{value:.0%} points",
    )

if not selected_confidence_levels:
    st.warning("Select at least one confidence level.")
    st.stop()

st.success(
    f"Using saved Phase 1 snapshot | target {run.get('target_date', 'n/a')} | "
    f"{run.get('days_to_target', 'n/a')} days | {run.get('simulations', len(result.terminal_market_caps)):,} paths | "
    f"seed {run.get('seed', 'n/a')}"
)
st.caption(f"Locked Phase 1 model: {source}. Change market data, IV surface, correlation, date, or seed in Phase 1 and rerun it there.")

confidence_levels = [float(value) for value in selected_confidence_levels]
current_cap = float(current_caps.loc[selected_ticker])
selected_row = selected_result_row(result.results, selected_ticker)
curve = calculate_conditional_win_curve(
    result.terminal_market_caps,
    selected_ticker,
    ranks=result.ranks,
    current_market_cap=current_cap,
    n_bins=int(n_bins),
)
boundaries = find_probability_boundaries(
    curve,
    confidence_levels,
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
all_boundaries = calculate_boundaries_for_all_tickers(
    result.terminal_market_caps,
    current_caps,
    confidence_levels,
    ranks=result.ranks,
    n_bins=int(n_bins),
)

st.session_state.boundary_inputs_used = simulation_inputs
st.session_state.boundary_result = result
st.session_state.boundary_all_boundaries = all_boundaries
st.session_state.boundary_snapshot_metadata = run

summary_cols = st.columns(4)
summary_cols[0].metric("Selected ticker", selected_ticker)
summary_cols[1].metric("Unconditional P(#1)", pct(float(selected_row["Model probability"])))
summary_cols[2].metric("Current market cap", dollars_trillions(current_cap))
summary_cols[3].metric("Conditional bins", f"{int(n_bins)}")

st.subheader("Boundary table")
st.dataframe(display_boundary_table(boundaries), use_container_width=True, hide_index=True)

st.subheader("Conditional probability chart")
st.plotly_chart(
    conditional_probability_figure(curve, boundaries, confidence_levels, selected_ticker),
    use_container_width=True,
    key="phase2_conditional_probability",
)

st.subheader("Probability-weighted scenario distribution")
st.write("Blue bars show how likely each terminal market-cap zone is. The line shows conditional win probability in that zone. Green bars show each zone's contribution to total P(#1).")
st.plotly_chart(
    probability_weighted_figure(distribution, selected_ticker),
    use_container_width=True,
    key="phase2_weighted_distribution",
)
with st.expander("Probability-weighted bin table"):
    st.dataframe(display_distribution_table(distribution), use_container_width=True, hide_index=True)

st.subheader("Rank chart")
st.plotly_chart(rank_figure(curve, selected_ticker), use_container_width=True, key="phase2_rank_chart")

with st.expander("Conditional-curve scenario table"):
    st.dataframe(display_curve_table(curve), use_container_width=True, hide_index=True)

with st.expander("All-ticker boundary summary"):
    st.dataframe(display_boundary_table(all_boundaries), use_container_width=True, hide_index=True)

with st.expander("Locked Phase 1 inputs used"):
    display_inputs = simulation_inputs.copy()
    display_inputs["Current market cap"] = display_inputs["Current market cap"].map(dollars_trillions)
    if "Implied volatility" in display_inputs.columns:
        display_inputs["Implied volatility"] = display_inputs["Implied volatility"].map(pct)
    if "Polymarket YES price" in display_inputs.columns:
        display_inputs["Polymarket YES price"] = display_inputs["Polymarket YES price"].map(pct)
    if "Forward / spot" in display_inputs.columns:
        display_inputs["Forward / spot"] = display_inputs["Forward / spot"].map(pct)
    st.dataframe(display_inputs, use_container_width=True, hide_index=True)
