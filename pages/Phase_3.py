from __future__ import annotations

import pandas as pd
import plotly.express as px
import streamlit as st

from iv_surface_model import SURFACE_EXPIRY
from market_data import fetch_spot_prices
from option_construction import construct_candidate_option_structure, payoff_grid_for_leg
from option_valuation import attach_market_consistent_premiums
from phase2_artifacts import boundaries_from_curves
from simulation_store import load_phase_artifact, load_simulation_snapshot, save_phase_artifact


st.set_page_config(page_title="Phase 3", layout="wide")
st.title("Phase 3: Option Construction")
st.caption("Phase 3 queries the complete Phase 2 conditional curves and converts probability levels into candidate option strikes. It does not rerun Phase 1 or optimize quantities.")

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
    if structure.empty:
        return 0.0
    signs = structure["Premium direction"].map({"Credit": 1.0, "Debit": -1.0}).fillna(0.0)
    return float((structure["Theoretical premium"].astype(float) * signs).sum())


def display_structure(table: pd.DataFrame) -> pd.DataFrame:
    display = table.copy()
    if display.empty:
        return display
    display["Strike"] = display["Strike"].map(dollars)
    display["Boundary market cap"] = display["Boundary market cap"].map(dollars_trillions)
    display["Boundary / current cap"] = display["Boundary / current cap"].map(pct)
    display["Spot"] = display["Spot"].map(dollars)
    for column in ["Model IV", "Risk-free rate", "Implied dividend yield"]:
        if column in display.columns:
            display[column] = display[column].map(pct)
    if "Forward / spot" in display.columns:
        display["Forward / spot"] = display["Forward / spot"].map(lambda value: "" if pd.isna(value) else f"{value:.5f}")
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
phase2_artifact = load_phase_artifact("phase2")
if snapshot is None:
    st.error("No saved Phase 1 simulation was found. Run Phase 1 first.")
    st.stop()
if phase2_artifact is None or not phase2_artifact.get("curves"):
    st.error("No complete Phase 2 curves were found. Open Phase 2 once after the latest Phase 1 run.")
    st.stop()

result = snapshot["result"]
simulation_inputs = snapshot["simulation_inputs"].copy()
run = snapshot.get("run_metadata") or {}
source = snapshot.get("source", "Phase 1")
curves = phase2_artifact["curves"]
tickers = simulation_inputs["Ticker"].astype(str).tolist()
current_caps = simulation_inputs.set_index("Ticker")["Current market cap"].astype(float)
event_tickers = simulation_inputs.loc[simulation_inputs["Polymarket YES price"].astype(float) > 0, "Ticker"].astype(str).tolist()
if not event_tickers:
    event_tickers = tickers

with st.sidebar:
    st.header("Phase 3 controls")
    selected_ticker = st.selectbox("Selected Polymarket ticker", event_tickers, index=event_tickers.index("NVDA") if "NVDA" in event_tickers else 0)
    confidence_level = st.slider(
        "Boundary confidence query",
        min_value=0.50,
        max_value=0.99,
        value=0.80,
        step=0.01,
        format="%.2f",
        help="Queries the complete Phase 2 curve. It is not limited to the marker levels displayed in Phase 2.",
    )
    construction_mode_label = st.selectbox("Option construction mode", list(CONSTRUCTION_MODE_LABELS), index=0)
    construction_mode = CONSTRUCTION_MODE_LABELS[construction_mode_label]
    include_competitor_short_puts = st.checkbox("Include competitor short puts", value=construction_mode == "single_competitor")
    risk_free_rate = st.number_input("Risk-free rate for theoretical premiums", min_value=0.0, max_value=0.20, value=0.04, step=0.005, format="%.3f")

boundaries = boundaries_from_curves(curves, current_caps, [float(confidence_level)])
st.success(
    f"Using complete Phase 2 curves | target {run.get('target_date', 'n/a')} | "
    f"{run.get('simulations', len(result.terminal_market_caps)):,} Phase 1 paths"
)
st.caption(f"Locked probability model: {source}. Phase 3 only queries a probability level and constructs option building blocks.")

competitor_ticker = None
competitor_options = [ticker for ticker in tickers if ticker != selected_ticker]
if construction_mode == "single_competitor":
    auto_competitor = result.results[result.results["Ticker"] != selected_ticker].sort_values("Model probability", ascending=False).iloc[0]["Ticker"]
    competitor_ticker = st.selectbox("Competitor ticker", competitor_options, index=competitor_options.index(auto_competitor) if auto_competitor in competitor_options else 0)

spots = load_spot_prices(tuple(tickers))
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
use_surface_pricing = (
    str(run.get("target_date", "")) == SURFACE_EXPIRY
    and "surface" in str(source).lower()
)
forward_ratios = (
    simulation_inputs.set_index("Ticker")["Forward / spot"].astype(float)
    if "Forward / spot" in simulation_inputs.columns
    else None
)
valued_structure = attach_market_consistent_premiums(
    structure,
    simulation_inputs.set_index("Ticker")["Implied volatility"].astype(float),
    forward_ratios=forward_ratios,
    time_to_expiry=years(int(run.get("days_to_target", 1))),
    risk_free_rate=float(risk_free_rate),
    use_surface=use_surface_pricing,
) if not structure.empty else structure.copy()

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
        "use_surface_pricing": bool(use_surface_pricing),
        "run_metadata": run,
        "source": source,
    },
)

construction_tab, payoff_tab, methodology_tab = st.tabs(["Boundary Strikes", "Standalone Payoffs", "Methodology"])

with construction_tab:
    st.subheader("Queried Phase 2 boundaries")
    st.dataframe(display_boundary_table(boundaries[boundaries["Ticker"].isin([selected_ticker] + ([competitor_ticker] if competitor_ticker else []))]), use_container_width=True, hide_index=True)

    if valued_structure.empty:
        st.warning("No option leg could be constructed at this probability level because the required empirical boundary was not reached. Try a lower confidence query or a finer Phase 2 curve.")
    else:
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
        if use_surface_pricing:
            st.caption(f"Pricing IV is interpolated separately for every strike from the calibrated {SURFACE_EXPIRY} smile. Carry comes from the saved Phase 1 forward ratio.")
        else:
            st.warning(f"The saved Phase 1 run does not match the calibrated {SURFACE_EXPIRY} surface. Option premiums therefore use the saved ticker ATM IV fallback; they are not surface-priced.")

    with st.expander("Current spot prices used for strike conversion"):
        st.dataframe(display_spots(spots), use_container_width=True, hide_index=True)

with payoff_tab:
    st.subheader("Standalone option payoff functions")
    if valued_structure.empty:
        st.info("No valid option legs exist for the current boundary query.")
    else:
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
    st.write("Phase 3 consumes complete Phase 2 curves and converts any queried probability level to option strikes:")
    st.code("strike = current stock price * boundary market cap / current market cap")
    st.write("Each option leg is then valued at its own strike-specific smile IV. Phase 1 Forward / spot is converted to an implied continuous dividend yield so option pricing and the simulated risk-neutral marginal share the same carry.")
    st.write("Current stock spot is new information needed for strike conversion. Phase 3 does not change Phase 1 market caps, IV surfaces, correlations, dates, paths, or ranks.")
    st.write("Phase 4 should consume this saved structure and combine it with the same Phase 1 scenario payoffs.")
