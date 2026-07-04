from __future__ import annotations

import pandas as pd
import plotly.express as px
import streamlit as st

from market_data import fetch_spot_prices
from option_construction import attach_theoretical_premiums, construct_candidate_option_structure, payoff_grid_for_leg
from simulation_store import load_simulation_snapshot, save_phase_artifact


st.set_page_config(page_title="Phase 3", layout="wide")
st.title("Phase 3: Option Construction")
st.caption("Phase 3 converts the saved Phase 2 probability boundaries into candidate option legs. It does not rerun Phase 1 or optimize quantities.")

CONSTRUCTION_MODE_LABELS = {
    "Selected-only hedge": "selected_only",
    "Selected + single competitor diagnostic": "single_competitor",
    "Selected + full universe competitors": "full_universe",
}


@st.cache_data(show_spinner=False, ttl=15 * 60)
def load_spot_prices(tickers: tuple[str, ...]) -> pd.DataFrame:
    return fetch_spot_prices(list(tickers))


def pct(value: float) -> str:
    return "" if pd.isna(value) else f"{value:.2%}"


def pct_or_not_reached(value: float) -> str:
    return "Not reached" if pd.isna(value) else f"{value:.2%}"


def dollars(value: float) -> str:
    return "" if pd.isna(value) else f"${value:,.2f}"


def dollars_trillions(value: float) -> str:
    return "" if pd.isna(value) else f"${value / 1e12:,.2f}T"


def dollars_or_not_reached(value: float) -> str:
    return "Not reached" if pd.isna(value) else dollars_trillions(value)


def years(days: int) -> float:
    return max(int(days), 1) / 365.0


def theoretical_cashflow(structure: pd.DataFrame) -> float:
    signs = structure["Premium direction"].map({"Credit": 1.0, "Debit": -1.0}).fillna(0.0)
    return float((structure["Theoretical premium"].astype(float) * signs).sum())


def display_structure(table: pd.DataFrame) -> pd.DataFrame:
    display = table.copy()
    display["Strike"] = display["Strike"].map(dollars)
    display["Boundary market cap"] = display["Boundary market cap"].map(dollars_trillions)
    display["Boundary / current cap"] = display["Boundary / current cap"].map(pct)
    display["Spot"] = display["Spot"].map(dollars)
    for column in ["Model IV", "Risk-free rate"]:
        if column in display.columns:
            display[column] = display[column].map(pct)
    if "Time to expiry" in display.columns:
        display["Time to expiry"] = display["Time to expiry"].map(lambda value: "" if pd.isna(value) else f"{value:.2f}y")
    if "Theoretical premium" in display.columns:
        display["Theoretical premium"] = display["Theoretical premium"].map(dollars)
    return display


def display_boundary_table(boundaries: pd.DataFrame) -> pd.DataFrame:
    display = boundaries.copy()
    display["Lower loss boundary"] = display["Lower loss boundary"].map(dollars_or_not_reached)
    display["Upper win boundary"] = display["Upper win boundary"].map(dollars_or_not_reached)
    display["Confidence level"] = display["Confidence level"].map(pct)
    display["Lower loss boundary / current"] = display["Lower loss boundary / current"].map(pct_or_not_reached)
    display["Upper win boundary / current"] = display["Upper win boundary / current"].map(pct_or_not_reached)
    return display


def display_spots(spots: pd.DataFrame) -> pd.DataFrame:
    display = spots.copy().rename(columns={"ticker": "Ticker", "yahoo_ticker": "Yahoo ticker", "spot_price": "Spot", "source": "Source"})
    display["Spot"] = display["Spot"].map(dollars)
    return display[["Ticker", "Yahoo ticker", "Spot", "Source"]]


snapshot = load_simulation_snapshot()
boundaries = st.session_state.get("boundary_all_boundaries")
if snapshot is None:
    st.error("No saved Phase 1 simulation was found. Run Phase 1 first.")
    st.stop()
if boundaries is None or boundaries.empty:
    st.error("No Phase 2 boundaries are available in this session. Open Phase 2, choose confidence levels, and return here.")
    st.stop()

result = snapshot["result"]
simulation_inputs = snapshot["simulation_inputs"].copy()
run = snapshot.get("run_metadata") or {}
source = snapshot.get("source", "Phase 1")
tickers = simulation_inputs["Ticker"].astype(str).tolist()
available_confidence_levels = sorted(boundaries["Confidence level"].dropna().astype(float).unique().tolist())

with st.sidebar:
    st.header("Phase 3 controls")
    selected_ticker = st.selectbox("Selected Polymarket ticker", tickers, index=tickers.index("NVDA") if "NVDA" in tickers else 0)
    confidence_level = st.selectbox(
        "Phase 2 boundary confidence",
        available_confidence_levels,
        index=len(available_confidence_levels) - 1,
        format_func=lambda value: f"{value:.0%}",
    )
    construction_mode_label = st.selectbox("Option construction mode", list(CONSTRUCTION_MODE_LABELS), index=0)
    construction_mode = CONSTRUCTION_MODE_LABELS[construction_mode_label]
    include_competitor_short_puts = st.checkbox(
        "Include competitor short puts",
        value=construction_mode == "single_competitor",
        help="Optional income legs rather than pure protection.",
    )
    risk_free_rate = st.number_input("Risk-free rate for theoretical premiums", min_value=0.0, max_value=0.20, value=0.04, step=0.005, format="%.3f")

st.success(
    f"Using Phase 2 boundaries from the saved Phase 1 snapshot | target {run.get('target_date', 'n/a')} | "
    f"{run.get('simulations', len(result.terminal_market_caps)):,} paths"
)
st.caption(f"Locked probability model: {source}. Phase 3 only selects boundary confidence and construction structure.")

competitor_ticker = None
competitor_options = [ticker for ticker in tickers if ticker != selected_ticker]
if construction_mode == "single_competitor":
    auto_competitor = result.results[result.results["Ticker"] != selected_ticker].sort_values("Model probability", ascending=False).iloc[0]["Ticker"]
    competitor_ticker = st.selectbox(
        "Competitor ticker",
        competitor_options,
        index=competitor_options.index(auto_competitor) if auto_competitor in competitor_options else 0,
    )

spots = load_spot_prices(tuple(tickers))
current_caps = simulation_inputs.set_index("Ticker")["Current market cap"].astype(float)
spot_series = spots.set_index("ticker")["spot_price"].astype(float)
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
    simulation_inputs.set_index("Ticker")["Implied volatility"].astype(float),
    time_to_expiry=years(int(run.get("days_to_target", 1))),
    risk_free_rate=float(risk_free_rate),
)
st.session_state.phase3_structure = valued_structure
st.session_state.phase3_result = result
st.session_state.phase3_inputs_used = simulation_inputs
st.session_state.phase3_spots = spots
save_phase_artifact(
    "phase3",
    {
        "structure": valued_structure,
        "selected_ticker": selected_ticker,
        "confidence_level": float(confidence_level),
        "construction_mode": construction_mode,
        "risk_free_rate": float(risk_free_rate),
        "run_metadata": run,
        "source": source,
    },
)

construction_tab, payoff_tab, methodology_tab = st.tabs(["Boundary Strikes", "Standalone Payoffs", "Methodology"])

with construction_tab:
    debit_count = int((valued_structure["Premium direction"] == "Debit").sum())
    credit_count = int((valued_structure["Premium direction"] == "Credit").sum())
    net_cashflow = theoretical_cashflow(valued_structure)
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Selected ticker", selected_ticker)
    col2.metric("Mode", construction_mode_label)
    col3.metric("Option legs", f"{len(valued_structure)}")
    col4.metric("Net theoretical premium", dollars(net_cashflow), help="Positive means net credit; negative means net debit. Per share before contract multipliers.")
    st.caption(f"Premium mix: {credit_count} credit leg(s), {debit_count} debit leg(s). Quantities are not optimized in Phase 3.")

    st.subheader("Suggested option structure")
    st.dataframe(display_structure(valued_structure), use_container_width=True, hide_index=True)

    with st.expander("Phase 2 boundaries used"):
        selected_boundaries = boundaries[boundaries["Confidence level"].astype(float).round(8) == round(float(confidence_level), 8)]
        st.dataframe(display_boundary_table(selected_boundaries), use_container_width=True, hide_index=True)
    with st.expander("Current spot prices used for strike conversion"):
        st.dataframe(display_spots(spots), use_container_width=True, hide_index=True)

with payoff_tab:
    st.subheader("Standalone option payoff functions")
    include_premium = st.toggle("Include theoretical premium", value=True)
    selected_instrument = st.selectbox("Instrument", valued_structure["Instrument"].tolist())
    leg = valued_structure[valued_structure["Instrument"] == selected_instrument].iloc[0]
    payoff = payoff_grid_for_leg(leg, premium=None if include_premium else 0.0)
    fig = px.line(payoff, x="Terminal price", y="Payoff", title=f"Standalone payoff: {selected_instrument}")
    fig.add_hline(y=0.0, line_dash="dot")
    fig.add_vline(x=float(leg["Strike"]), line_dash="dash", annotation_text="strike")
    fig.add_vline(x=float(leg["Spot"]), line_dash="dot", annotation_text="spot")
    st.plotly_chart(fig, use_container_width=True, key="phase3_standalone_payoff")

with methodology_tab:
    st.write("Phase 3 consumes Phase 2 boundaries and converts market-cap levels to stock-price strikes:")
    st.code("strike = current stock price * boundary market cap / current market cap")
    st.write("It may fetch current stock spot prices because strike conversion is new information for this phase. It does not change Phase 1 market caps, IV surfaces, correlations, dates, paths, or ranks.")
    st.write("Phase 4 will consume the saved candidate structure and combine it with the same Phase 1 scenario payoffs.")
