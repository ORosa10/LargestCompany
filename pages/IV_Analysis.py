from __future__ import annotations

from datetime import date, timedelta

import pandas as pd
import plotly.express as px
import streamlit as st

from correlations import ewma_correlation, fetch_adjusted_close
from iv_surfaces import apply_iv_estimates, estimate_atm_ivs
from market_data import apply_market_caps, fetch_market_caps
from model import default_company_inputs, default_correlation_matrix, run_probability_engine


st.set_page_config(page_title="IV Analysis", layout="wide")
st.title("IV Analysis")
st.caption("Sensitivity view: isolate how ranking probabilities react to implied-volatility assumptions.")

IV_SHOCKS = [level / 100 for level in range(-20, 25, 5)]


@st.cache_data(show_spinner=False, ttl=60 * 60)
def load_adjusted_close(tickers: tuple[str, ...], period: str) -> pd.DataFrame:
    return fetch_adjusted_close(list(tickers), period=period)


@st.cache_data(show_spinner=False, ttl=30 * 60)
def load_yahoo_atm_ivs(tickers: tuple[str, ...], target_date_iso: str) -> pd.DataFrame:
    return estimate_atm_ivs(list(tickers), date.fromisoformat(target_date_iso))


@st.cache_data(show_spinner=False, ttl=15 * 60)
def load_yahoo_market_caps(tickers: tuple[str, ...]) -> pd.DataFrame:
    return fetch_market_caps(list(tickers))


def pct(value: float) -> str:
    return "" if pd.isna(value) else f"{value:.2%}"


def shock_label(value: float) -> str:
    return f"{value * 100:+.0f} vol pts"


def prepare_inputs(company_inputs: pd.DataFrame, market_cap_source: str, iv_source: str, target_date_value: date) -> pd.DataFrame:
    tickers = [ticker for ticker in company_inputs["Ticker"].astype(str).str.strip().tolist() if ticker]
    simulation_inputs = company_inputs.copy()
    if market_cap_source == "Yahoo Finance current market cap":
        simulation_inputs = apply_market_caps(simulation_inputs, load_yahoo_market_caps(tuple(tickers)))
    if iv_source == "Yahoo option-chain near-ATM IV":
        simulation_inputs = apply_iv_estimates(simulation_inputs, load_yahoo_atm_ivs(tuple(tickers), target_date_value.isoformat()))
    return simulation_inputs


def select_correlation_matrix(inputs: pd.DataFrame, method: str, period: str, ewma_lambda: float) -> tuple[pd.DataFrame, str]:
    tickers = inputs["Ticker"].astype(str).tolist()
    if method == "Manual/default correlation matrix":
        return default_correlation_matrix(tickers), "Manual/default correlation matrix"
    prices = load_adjusted_close(tuple(tickers), period)
    return ewma_correlation(prices, ewma_lambda), f"EWMA historical correlation, lambda={ewma_lambda}, Yahoo Finance {period}"


def apply_iv_shock(inputs: pd.DataFrame, shock: float, shocked_tickers: list[str] | None) -> pd.DataFrame:
    shocked = inputs.copy()
    if shocked_tickers is None:
        mask = pd.Series(True, index=shocked.index)
    else:
        mask = shocked["Ticker"].astype(str).isin(shocked_tickers)
    shocked.loc[mask, "Implied volatility"] = (shocked.loc[mask, "Implied volatility"].astype(float) + shock).clip(lower=0.0001)
    return shocked


def run_iv_grid(inputs: pd.DataFrame, corr: pd.DataFrame, days_to_target: int, simulations: int, seed: int, mode: str, shocked_tickers: list[str] | None) -> pd.DataFrame:
    rows = []
    for shock in IV_SHOCKS:
        result = run_probability_engine(apply_iv_shock(inputs, shock, shocked_tickers), corr, days_to_target=days_to_target, simulations=simulations, seed=seed)
        for _, row in result.results.iterrows():
            rows.append(
                {
                    "Mode": mode,
                    "Shock": shock,
                    "Shocked tickers": "All" if shocked_tickers is None else ", ".join(shocked_tickers),
                    "Ticker": row["Ticker"],
                    "Model probability": row["Model probability"],
                    "Average rank": row["Average rank"],
                    "Top 2": row["Probability Top 2"],
                    "Top 3": row["Probability Top 3"],
                }
            )
    return pd.DataFrame(rows)


def display_table(table: pd.DataFrame) -> pd.DataFrame:
    display = table.copy()
    display["Shock"] = display["Shock"].map(shock_label)
    for column in ["Model probability", "Top 2", "Top 3"]:
        display[column] = display[column].map(pct)
    display["Average rank"] = display["Average rank"].map(lambda value: f"{value:.2f}")
    return display


with st.sidebar:
    st.header("IV analysis controls")
    target_date = st.date_input("Target date / maturity", value=date.today() + timedelta(days=365), min_value=date.today() + timedelta(days=1))
    days_to_target = max((target_date - date.today()).days, 1)
    st.caption(f"Horizon: {days_to_target} days ({days_to_target / 365:.2f} years)")
    simulations = st.number_input("Monte Carlo simulations per shock", min_value=1_000, max_value=500_000, value=100_000, step=10_000)
    seed = st.number_input("Random seed", min_value=0, value=42, step=1)
    market_cap_source = st.selectbox("Market cap source", ["Yahoo Finance current market cap", "Manual market cap inputs"], index=0)
    iv_source = st.selectbox("Base IV source", ["Manual IV inputs", "Yahoo option-chain near-ATM IV"], index=0)
    correlation_method = st.selectbox("Correlation method", ["EWMA historical correlation", "Manual/default correlation matrix"], index=0)
    price_history_period = st.selectbox("Yahoo Finance price history", ["1y", "3y", "5y", "10y"], index=2)
    ewma_lambda = st.selectbox("EWMA lambda", [0.94, 0.97], index=1)

if "iv_analysis_company_inputs" not in st.session_state:
    st.session_state.iv_analysis_company_inputs = default_company_inputs()

company_inputs = st.data_editor(
    st.session_state.iv_analysis_company_inputs,
    num_rows="dynamic",
    use_container_width=True,
    hide_index=True,
)
st.session_state.iv_analysis_company_inputs = company_inputs

base_tickers = [ticker for ticker in company_inputs["Ticker"].astype(str).str.strip().tolist() if ticker]
if not base_tickers:
    st.warning("Add at least one ticker.")
    st.stop()

col1, col2, col3 = st.columns(3)
with col1:
    single_ticker = st.selectbox("Single-name shock ticker", base_tickers, index=0)
with col2:
    pair_ticker_1 = st.selectbox("Pair shock ticker 1", base_tickers, index=0)
with col3:
    pair_ticker_2 = st.selectbox("Pair shock ticker 2", base_tickers, index=1 if len(base_tickers) > 1 else 0)

if st.button("Run IV sensitivity", type="primary"):
    with st.spinner("Running IV sensitivity shocks..."):
        try:
            base_inputs = prepare_inputs(company_inputs, market_cap_source, iv_source, target_date)
            corr, corr_label = select_correlation_matrix(base_inputs, correlation_method, price_history_period, float(ewma_lambda))
            st.session_state.iv_global = run_iv_grid(base_inputs, corr, int(days_to_target), int(simulations), int(seed), "Global", None)
            st.session_state.iv_single = run_iv_grid(base_inputs, corr, int(days_to_target), int(simulations), int(seed), "Single-name", [single_ticker])
            st.session_state.iv_pair = run_iv_grid(base_inputs, corr, int(days_to_target), int(simulations), int(seed), "Pair", [pair_ticker_1, pair_ticker_2])
            st.session_state.iv_corr_label = corr_label
            st.session_state.iv_error = None
        except Exception as exc:
            st.session_state.iv_error = str(exc)

if st.session_state.get("iv_error"):
    st.error(st.session_state.iv_error)

global_table = st.session_state.get("iv_global")
single_table = st.session_state.get("iv_single")
pair_table = st.session_state.get("iv_pair")

if global_table is None:
    st.info("Run IV sensitivity to calculate global, single-name, and pair IV shock tables.")
    st.stop()

st.caption(f"Correlation assumption: {st.session_state.get('iv_corr_label')}")

for title, table in [("Global IV shock", global_table), ("Single-name IV shock", single_table), ("Pair IV shock", pair_table)]:
    if table is None or table.empty:
        continue
    st.subheader(title)
    pivot = table.pivot(index="Shock", columns="Ticker", values="Model probability")
    pivot.index = [shock_label(value) for value in pivot.index]
    st.dataframe(pivot.map(pct), use_container_width=True)
    selected = st.selectbox(f"Ticker chart: {title}", table["Ticker"].unique().tolist(), key=f"chart_{title}")
    slice_ = table[table["Ticker"] == selected].sort_values("Shock")
    left, right = st.columns(2)
    with left:
        st.plotly_chart(px.line(slice_, x="Shock", y="Model probability", markers=True, title=f"{selected}: P(#1) vs IV shock"), use_container_width=True, key=f"prob_{title}")
    with right:
        st.plotly_chart(px.line(slice_, x="Shock", y="Average rank", markers=True, title=f"{selected}: average rank vs IV shock"), use_container_width=True, key=f"rank_{title}")

with st.expander("Full IV sensitivity tables"):
    st.write("Global IV shock")
    st.dataframe(display_table(global_table), use_container_width=True, hide_index=True)
    st.write("Single-name IV shock")
    st.dataframe(display_table(single_table), use_container_width=True, hide_index=True)
    st.write("Pair IV shock")
    st.dataframe(display_table(pair_table), use_container_width=True, hide_index=True)

st.subheader("Interpretation")
st.write("If the probability barely moves across IV shocks, the current market-cap gap dominates. If it moves a lot, the conclusion is sensitive to volatility assumptions and should be treated as less stable.")
