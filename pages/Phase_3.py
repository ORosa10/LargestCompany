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
st.caption("Query Phase 2 probability boundaries and translate them into candidate option building blocks. Phase 3 does not rerun Monte Carlo or optimize quantities.")

MODE_LABELS = {
    "Selected ticker only": "selected_only",
    "Selected + strongest relevant competitor": "single_competitor",
    "Selected + all relevant Phase 1 outcomes": "relevant_universe",
    "Selected + complete simulation universe (diagnostic)": "simulation_universe",
}


@st.cache_data(show_spinner=False, ttl=15 * 60)
def load_spots(tickers: tuple[str, ...]) -> pd.DataFrame:
    return fetch_spot_prices(list(tickers))


def pct(value: float) -> str:
    return "Not reached" if pd.isna(value) else f"{value:.2%}"


def dollars(value: float) -> str:
    return "" if pd.isna(value) else f"${value:,.2f}"


def trillions(value: float) -> str:
    return "Not reached" if pd.isna(value) else f"${value / 1e12:,.2f}T"


def display_boundaries(table: pd.DataFrame) -> pd.DataFrame:
    display = table.copy()
    for column in ["Lower loss boundary", "Upper win boundary"]:
        display[column] = display[column].map(trillions)
    for column in ["Confidence level", "Lower loss boundary / current", "Upper win boundary / current"]:
        display[column] = display[column].map(pct)
    return display


def display_structure(table: pd.DataFrame) -> pd.DataFrame:
    display = table.copy()
    if display.empty:
        return display
    for column in ["Strike", "Spot", "Theoretical premium"]:
        if column in display.columns:
            display[column] = display[column].map(dollars)
    if "Boundary market cap" in display.columns:
        display["Boundary market cap"] = display["Boundary market cap"].map(trillions)
    for column in ["Boundary / current cap", "Model IV", "Risk-free rate", "Implied dividend yield"]:
        if column in display.columns:
            display[column] = display[column].map(pct)
    if "Forward / spot" in display.columns:
        display["Forward / spot"] = display["Forward / spot"].map(lambda value: f"{value:.5f}")
    if "Time to expiry" in display.columns:
        display["Time to expiry"] = display["Time to expiry"].map(lambda value: f"{value:.3f}y")
    return display


def matching_metadata(left: dict, right: dict) -> bool:
    return all(left.get(key) == right.get(key) for key in ["target_date", "days_to_target", "simulations", "seed"])


snapshot = load_simulation_snapshot()
phase2 = load_phase_artifact("phase2")
if snapshot is None:
    st.error("No saved Phase 1 simulation was found. Run Phase 1 first.")
    st.stop()
if phase2 is None or not phase2.get("curves"):
    st.error("No complete Phase 2 curves were found. Open Phase 2 after the latest Phase 1 run.")
    st.stop()

result = snapshot["result"]
inputs = snapshot["simulation_inputs"].copy()
run = snapshot.get("run_metadata") or {}
source = snapshot.get("source", "Phase 1")
if not matching_metadata(run, phase2.get("run_metadata") or {}):
    st.error("The saved Phase 2 curves belong to a different Phase 1 run. Open Phase 2 once and return here.")
    st.stop()

curves = phase2["curves"]
tickers = inputs["Ticker"].astype(str).tolist()
current_caps = inputs.set_index("Ticker")["Current market cap"].astype(float)
relevant_tickers = inputs.loc[inputs["Polymarket YES price"].astype(float) > 0, "Ticker"].astype(str).tolist()
if not relevant_tickers:
    relevant_tickers = tickers.copy()

with st.sidebar:
    st.header("Phase 3 controls")
    selected_ticker = st.selectbox(
        "Selected Polymarket ticker",
        relevant_tickers,
        index=relevant_tickers.index("NVDA") if "NVDA" in relevant_tickers else 0,
    )
    confidence = st.slider(
        "Boundary confidence query",
        0.50,
        0.99,
        0.80,
        0.01,
        format="%.2f",
        help="Queries the complete saved Phase 2 curve; it is not limited to Phase 2 chart markers.",
    )
    mode_label = st.selectbox("Option construction mode", list(MODE_LABELS), index=0)
    mode = MODE_LABELS[mode_label]
    include_short_puts = st.checkbox("Include competitor short puts", value=mode == "single_competitor")
    risk_free_rate = st.number_input("Risk-free rate for theoretical premiums", 0.0, 0.20, 0.04, 0.005, format="%.3f")

probabilities = result.results.set_index("Ticker")["Model probability"].astype(float)
relevant_competitors = [ticker for ticker in relevant_tickers if ticker != selected_ticker]
all_competitors = [ticker for ticker in tickers if ticker != selected_ticker]
competitor_ticker = None
if mode == "single_competitor":
    choices = relevant_competitors or all_competitors
    default_competitor = max(choices, key=lambda ticker: float(probabilities.get(ticker, 0.0)))
    competitor_ticker = st.selectbox("Relevant competitor", choices, index=choices.index(default_competitor))

if mode == "relevant_universe":
    construction_results = result.results[result.results["Ticker"].isin(relevant_tickers)].copy()
    engine_mode = "full_universe"
elif mode == "simulation_universe":
    construction_results = result.results.copy()
    engine_mode = "full_universe"
else:
    construction_results = result.results.copy()
    engine_mode = mode

boundaries = boundaries_from_curves(curves, current_caps, [float(confidence)])
spots = load_spots(tuple(tickers))
spot_series = spots.set_index("ticker")["spot_price"].astype(float)
use_surface = str(run.get("target_date", "")) == SURFACE_EXPIRY and "surface" in str(source).lower()
forward_ratios = inputs.set_index("Ticker")["Forward / spot"].astype(float) if "Forward / spot" in inputs.columns else None
fallback_ivs = inputs.set_index("Ticker")["Implied volatility"].astype(float)
time_to_expiry = max(int(run.get("days_to_target", 1)), 1) / 365.0


def strongest_competitor(candidate: str) -> str | None:
    choices = [ticker for ticker in relevant_tickers if ticker != candidate]
    if not choices:
        choices = [ticker for ticker in tickers if ticker != candidate]
    return max(choices, key=lambda ticker: float(probabilities.get(ticker, 0.0))) if choices else None


def build_structure(candidate: str, competitor: str | None = None) -> pd.DataFrame:
    raw = construct_candidate_option_structure(
        boundaries,
        construction_results,
        current_caps,
        spot_series,
        selected_ticker=candidate,
        competitor_ticker=competitor,
        confidence_level=float(confidence),
        construction_mode=engine_mode,
        include_competitor_short_puts=bool(include_short_puts),
    )
    if raw.empty:
        return raw.copy()
    return attach_market_consistent_premiums(
        raw,
        fallback_ivs,
        forward_ratios=forward_ratios,
        time_to_expiry=time_to_expiry,
        risk_free_rate=float(risk_free_rate),
        use_surface=use_surface,
    )


selected_competitor = competitor_ticker if mode == "single_competitor" else None
valued = build_structure(selected_ticker, selected_competitor)
structures_by_ticker: dict[str, pd.DataFrame] = {}
for candidate in relevant_tickers:
    candidate_competitor = None
    if mode == "single_competitor":
        candidate_competitor = selected_competitor if candidate == selected_ticker else strongest_competitor(candidate)
    structures_by_ticker[candidate] = valued.copy() if candidate == selected_ticker else build_structure(candidate, candidate_competitor)

save_phase_artifact(
    "phase3",
    {
        "structure": valued,
        "structures_by_ticker": structures_by_ticker,
        "selected_ticker": selected_ticker,
        "confidence_level": float(confidence),
        "construction_mode": mode,
        "construction_mode_label": mode_label,
        "relevant_tickers": relevant_tickers,
        "risk_free_rate": float(risk_free_rate),
        "use_surface_pricing": bool(use_surface),
        "run_metadata": run,
        "source": source,
    },
)
st.session_state.phase3_structure = valued
st.session_state.phase3_result = result
st.session_state.phase3_inputs_used = inputs
st.session_state.phase3_spots = spots

st.success(
    f"Using Phase 2 curves | target {run.get('target_date', 'n/a')} | "
    f"{run.get('simulations', len(result.terminal_market_caps)):,} Phase 1 paths"
)
st.caption(
    "Relevant hedge outcomes: " + ", ".join(relevant_tickers) + ". "
    "All simulation tickers still remain inside Phase 1 probabilities and ranks."
)

construction_tab, payoff_tab, methodology_tab = st.tabs(["Boundary Strikes", "Standalone Payoffs", "Methodology"])

with construction_tab:
    used_tickers = valued["Ticker"].astype(str).unique().tolist() if not valued.empty else [selected_ticker]
    st.subheader("Queried Phase 2 boundaries")
    st.dataframe(display_boundaries(boundaries[boundaries["Ticker"].isin(used_tickers)]), width="stretch", hide_index=True)
    if valued.empty:
        st.warning("The required empirical boundary was not reached. Try a lower confidence query.")
    else:
        signs = valued["Premium direction"].map({"Credit": 1.0, "Debit": -1.0}).fillna(0.0)
        net_premium = float((valued["Theoretical premium"].astype(float) * signs).sum())
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Selected ticker", selected_ticker)
        c2.metric("Mode", mode_label)
        c3.metric("Option legs", len(valued))
        c4.metric("Net theoretical premium", dollars(net_premium))
        st.subheader("Suggested option structure")
        st.dataframe(display_structure(valued), width="stretch", hide_index=True)
        if use_surface:
            st.caption(f"Every leg uses its own strike-specific IV from the calibrated {SURFACE_EXPIRY} smile and Phase 1 forward carry.")
        else:
            st.warning(f"This run does not match the calibrated {SURFACE_EXPIRY} surface; premiums use Phase 1 ATM fallback IV.")

with payoff_tab:
    if valued.empty:
        st.info("No valid option legs for this query.")
    else:
        include_premium = st.toggle("Include theoretical premium", value=True)
        instrument = st.selectbox("Instrument", valued["Instrument"].tolist())
        leg = valued[valued["Instrument"] == instrument].iloc[0]
        payoff = payoff_grid_for_leg(leg, premium=None if include_premium else 0.0)
        fig = px.line(payoff, x="Terminal price", y="Payoff", title=f"Standalone payoff: {instrument}")
        fig.add_hline(y=0.0, line_dash="dot")
        fig.add_vline(x=float(leg["Strike"]), line_dash="dash", annotation_text="strike")
        fig.add_vline(x=float(leg["Spot"]), line_dash="dot", annotation_text="spot")
        st.plotly_chart(fig, width="stretch", key="phase3_standalone_payoff")

with methodology_tab:
    st.markdown("""
**Simulation universe** and **hedge universe** are deliberately separate.

- Every Phase 1 ticker remains in the joint simulation, correlation matrix, rankings, and winner probabilities.
- Relevant Phase 1 outcomes are the tickers with a positive Polymarket price in the saved Phase 1 input.
- Phase 3 saves an equivalent candidate structure for every relevant outcome so downstream phases can switch the Polymarket ticker without rerunning Monte Carlo.
- Complete simulation-universe construction is retained as a diagnostic for residual competitor risk.

Boundary conversion:

```text
option strike = current stock price * boundary market cap / current market cap
```

Each leg is priced with its own strike-specific smile IV and the Phase 1 forward carry. Quantities remain a later-phase decision.
    """)
