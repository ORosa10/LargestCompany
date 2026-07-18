"""Phase 8: risk management and position sizing workspace.

Phase 8 reads the saved Phase 1 scenarios and the saved Phase 4 portfolio and
turns them into the money view: capital deployed, maximum loss, return on
capital and on capital at risk, probability of profit and of ruin, breakeven
levels, and Kelly-based position sizing for a bankroll. It changes no model;
payoffs come straight from the payoff surface. It reads the real Phase 6
execution candidate when saved, and falls back to the Phase 4 theory portfolio.
See ``phase8.py``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from phase7 import PHASE6_ARTIFACT, PortfolioSpec, load_saved_portfolio, portfolio_scenarios
from phase8 import (
    breakeven_bands,
    budget_scaling,
    capital_metrics,
    capital_return_table,
    kelly_sizing,
    return_distribution,
    risk_metrics,
    value_at_risk,
)
from simulation_store import load_phase_artifact, load_simulation_snapshot

st.set_page_config(page_title="Phase 8", layout="wide")
st.title("Phase 8: Risk management & position sizing")
st.caption(
    "Phase 8 turns the payoff distribution and real execution costs into the "
    "decision metrics you size against. It reads the real Phase 6 executed portfolio "
    "when available, so every number is net of real fills and the Polymarket entry cost."
)


def dollars(value: float) -> str:
    return "n/a" if pd.isna(value) else f"${value:,.0f}"


def pct(value: float) -> str:
    return "n/a" if pd.isna(value) else f"{value:.2%}"


def ratio(value: float) -> str:
    return "n/a" if pd.isna(value) else f"{value:.2f}x"


snapshot = load_simulation_snapshot()
if snapshot is None:
    st.error("No saved Phase 1 simulation was found. Run Phase 1 first.")
    st.stop()

result = snapshot["result"]
simulation_inputs = snapshot["simulation_inputs"].copy()
current_caps = simulation_inputs.set_index("Ticker")["Current market cap"].astype(float)
default_ticker = str(simulation_inputs["Ticker"].iloc[0])
phase6 = load_phase_artifact(PHASE6_ARTIFACT)
phase4 = load_phase_artifact("phase4")
portfolio, portfolio_source = load_saved_portfolio(current_caps, default_ticker, phase6=phase6, phase4=phase4)
if portfolio is None:
    st.error(
        "No saved portfolio was found. Save a Phase 6 execution candidate (preferred) "
        "or open Phase 4 to save a theory portfolio, then return here."
    )
    st.stop()
selected_ticker = portfolio.selected_ticker

st.info(
    f"Portfolio: {portfolio_source} - {portfolio.polymarket_side} {selected_ticker} + option legs | "
    f"{len(result.terminal_market_caps):,} scenarios."
)

with st.sidebar:
    st.header("Phase 8 controls")
    shortfall_probability = st.slider("Expected-shortfall tail", 0.01, 0.10, 0.05, 0.01)
    ruin_fraction = st.slider("Ruin threshold (share of capital at risk lost)", 0.5, 1.0, 0.9, 0.05)
    st.divider()
    budget = st.number_input("Capital budget for sizing ($)", min_value=0.0, value=50_000.0, step=1_000.0)
    budget_basis = st.radio("Budget caps", ["capital-at-risk", "cash"], index=0,
                            help="capital-at-risk caps the maximum loss at the budget; cash deploys the budget as up-front cash.")
    kelly_choice = st.select_slider("Kelly fraction to recommend", options=[1.0, 0.5, 0.25], value=0.5)

metrics = risk_metrics(result, portfolio, shortfall_probability=float(shortfall_probability), ruin_fraction=float(ruin_fraction))
capital = capital_metrics(portfolio)

capital_tab, roc_tab, profit_tab, sizing_tab, methodology = st.tabs(
    ["Capital & max loss", "RoC / RoCaR", "Profit, breakeven & ruin", "Position sizing", "Methodology"]
)

with capital_tab:
    st.subheader("Capital needed and risk-based losses")
    var5 = value_at_risk(result, portfolio, 0.05)
    var1 = value_at_risk(result, portfolio, 0.01)
    cols = st.columns(4)
    cols[0].metric("Net cash outlay", dollars(metrics["Net cash outlay"]))
    cols[1].metric("VaR 5% (95% worst loss)", dollars(var5))
    cols[2].metric("VaR 1% (99% worst loss)", dollars(var1))
    cols[3].metric("Max loss (worst case)", dollars(metrics["Max loss (capital at risk)"]))
    st.caption(
        "Capital you must reserve depends on how conservative you are: the cash you "
        "actually pay, the VaR at 95%/99% (the loss you would not exceed at that "
        "confidence), or the absolute worst simulated loss. Polymarket shares are "
        "fully collateralized; long option legs are a debit, short legs a credit."
    )

    st.subheader("Return on capital by capital basis")
    st.caption("Expected profit divided by each capital-at-risk definition, from least to most conservative.")
    cap_table = capital_return_table(result, portfolio)
    cap_display = cap_table.copy()
    cap_display["Capital needed ($)"] = cap_display["Capital needed ($)"].map(lambda v: dollars(v))
    cap_display["Return on capital"] = cap_display["Return on capital"].map(lambda v: pct(v))
    st.dataframe(cap_display, width="stretch", hide_index=True)

    with st.expander("Capital components"):
        st.dataframe(capital.rename("Amount ($)").reset_index().rename(columns={"index": "Component"}), width="stretch", hide_index=True)

with roc_tab:
    st.subheader("Return on capital")
    cols = st.columns(4)
    cols[0].metric("Expected profit", dollars(metrics["Expected profit"]))
    cols[1].metric("Return on capital", pct(metrics["Return on capital"]))
    cols[2].metric("Return on capital-at-risk", pct(metrics["Return on capital-at-risk"]))
    cols[3].metric("Expected shortfall", dollars(metrics["Expected shortfall"]))
    st.caption(
        "Return on capital divides expected profit by the up-front cash. Return "
        "on capital-at-risk divides it by the maximum loss - the cleanest measure "
        "for a bounded-loss structure and always well defined."
    )
    st.dataframe(metrics.rename("Value").reset_index().rename(columns={"index": "Metric"}), width="stretch", hide_index=True)

with profit_tab:
    st.subheader("Profit odds, breakeven, and ruin")
    cols = st.columns(3)
    cols[0].metric("Probability of profit", pct(metrics["Probability of profit"]))
    cols[1].metric("Probability of loss", pct(metrics["Probability of loss"]))
    ruin_key = [k for k in metrics.index if k.startswith("Risk of ruin")][0]
    cols[2].metric(ruin_key, pct(metrics[ruin_key]))

    dist = return_distribution(result, portfolio)
    st.plotly_chart(
        px.bar(dist, x="Percentile", y="Profit", title="Profit distribution by percentile"),
        width="stretch",
    )
    st.dataframe(dist, width="stretch", hide_index=True)

    st.subheader(f"Breakeven levels on {selected_ticker}")
    bands = breakeven_bands(result, portfolio)
    if bands.empty:
        st.write("No sign change in expected payoff across the selected-ticker range (single-sided profile).")
    else:
        st.dataframe(bands, width="stretch", hide_index=True)

with sizing_tab:
    st.subheader("Budget scaling")
    st.caption("The structure is fixed, so scaling to a budget scales every payoff by the same factor.")
    try:
        scaled = budget_scaling(metrics, float(budget), basis=str(budget_basis))
        cols = st.columns(4)
        cols[0].metric("Scale factor", ratio(scaled["Scale factor"]))
        cols[1].metric("Scaled expected profit", dollars(scaled["Scaled expected profit"]))
        cols[2].metric("Scaled max loss", dollars(scaled["Scaled max loss"]))
        cols[3].metric("Scaled cash outlay", dollars(scaled["Scaled cash outlay"]))
    except ValueError as exc:
        st.warning(str(exc))

    st.subheader("Kelly sizing")
    st.caption(
        "Kelly maximizes long-run log growth. The fraction is of bankroll put at "
        "risk (worst scenario = a full unit lost). Most traders use a fractional "
        "Kelly to cut variance."
    )
    kelly = kelly_sizing(result, portfolio)
    full = float(kelly["Full Kelly fraction"])
    chosen_fraction = full * float(kelly_choice)
    cols = st.columns(4)
    cols[0].metric("Full Kelly fraction", pct(full))
    cols[1].metric(f"{kelly_choice:g}x Kelly fraction", pct(chosen_fraction))
    cols[2].metric("Capital at risk (1x)", dollars(kelly["Capital at risk (1x portfolio)"]))
    cols[3].metric("Recommended stake", dollars(chosen_fraction * float(budget)))
    if full == 0.0:
        st.warning("Expected return is not positive on these scenarios: Kelly recommends no bet.")
    st.dataframe(kelly.rename("Value").reset_index().rename(columns={"index": "Metric"}), width="stretch", hide_index=True)

with methodology:
    st.markdown(
        """
Phase 8 is the money view over the Phase 4 payoff distribution. It changes no
model.

**Sign convention.** Phase 4 payoffs are already net of option premiums and the
Polymarket entry cost, so "Total payoff" is net P&L, the worst scenario is the
maximum loss, and the mean is expected net profit.

**Capital.** Net cash outlay = Polymarket cost (shares x entry) + net option
debit (long premiums minus short credits). Gross premium paid ignores credits as
the most conservative "money in".

**Returns.** Return on capital = expected profit / cash outlay. Return on
capital-at-risk = expected profit / maximum loss; for a bounded-loss structure
this is the cleanest and always-defined measure.

**Odds and ruin.** Probability of profit and of loss come straight from the
scenario payoffs. Risk of ruin is the probability of losing at least the chosen
share of the capital at risk. Breakeven levels are the selected-ticker terminal
levels where the expected-payoff profile crosses zero.

**Sizing.** Budget scaling scales the fixed structure linearly to a capital or
risk budget. Kelly sizing maximizes E[log(1 + f x return)] over the scenarios,
with the return expressed per unit of capital at risk (worst case = -1), and
reports full and fractional Kelly fractions.
        """
    )
