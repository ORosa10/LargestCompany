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


def dollars_trillions(value: float) -> str:
    return f"${value / 1e12:,.2f}T"


def shock_label(value: float) -> str:
    return f"{value * 100:+.0f} percentage points"


def display_market_caps(market_caps: pd.DataFrame | None) -> pd.DataFrame | None:
    if market_caps is None or market_caps.empty:
        return None
    display = market_caps.copy().rename(columns={"ticker": "Ticker", "yahoo_ticker": "Yahoo ticker", "market_cap": "Market cap", "source": "Source"})
    display["Market cap"] = display["Market cap"].map(dollars_trillions)
    return display[["Ticker", "Yahoo ticker", "Market cap", "Source"]]


def display_base_inputs(inputs: pd.DataFrame) -> pd.DataFrame:
    display = inputs.copy()
    display["Current market cap"] = display["Current market cap"].map(dollars_trillions)
    display["Implied volatility"] = display["Implied volatility"].map(pct)
    display["Polymarket YES price"] = display["Polymarket YES price"].map(pct)
    return display


def prepare_inputs(company_inputs: pd.DataFrame, market_cap_source: str, iv_source: str, target_date_value: date) -> tuple[pd.DataFrame, pd.DataFrame | None]:
    tickers = [ticker for ticker in company_inputs["Ticker"].astype(str).str.strip().tolist() if ticker]
    simulation_inputs = company_inputs.copy()
    market_caps = None
    if market_cap_source == "Yahoo Finance current market cap":
        market_caps = load_yahoo_market_caps(tuple(tickers))
        simulation_inputs = apply_market_caps(simulation_inputs, market_caps)
    if iv_source == "Yahoo option-chain near-ATM IV":
        simulation_inputs = apply_iv_estimates(simulation_inputs, load_yahoo_atm_ivs(tuple(tickers), target_date_value.isoformat()))
    return simulation_inputs, market_caps


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


def apply_pair_iv_shocks(inputs: pd.DataFrame, ticker_x: str, shock_x: float, ticker_y: str, shock_y: float) -> pd.DataFrame:
    shocked = inputs.copy()
    mask_x = shocked["Ticker"].astype(str) == ticker_x
    shocked.loc[mask_x, "Implied volatility"] = (shocked.loc[mask_x, "Implied volatility"].astype(float) + shock_x).clip(lower=0.0001)
    mask_y = shocked["Ticker"].astype(str) == ticker_y
    shocked.loc[mask_y, "Implied volatility"] = (shocked.loc[mask_y, "Implied volatility"].astype(float) + shock_y).clip(lower=0.0001)
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


def run_pair_surface(inputs: pd.DataFrame, corr: pd.DataFrame, days_to_target: int, simulations: int, seed: int, ticker_x: str, ticker_y: str, output_ticker: str) -> pd.DataFrame:
    rows = []
    for shock_y in IV_SHOCKS:
        for shock_x in IV_SHOCKS:
            shocked = apply_pair_iv_shocks(inputs, ticker_x, shock_x, ticker_y, shock_y)
            result = run_probability_engine(shocked, corr, days_to_target=days_to_target, simulations=simulations, seed=seed)
            row = result.results.set_index("Ticker").loc[output_ticker]
            rows.append(
                {
                    "X ticker": ticker_x,
                    "Y ticker": ticker_y,
                    "Output ticker": output_ticker,
                    "X shock": shock_x,
                    "Y shock": shock_y,
                    "X shock label": shock_label(shock_x),
                    "Y shock label": shock_label(shock_y),
                    "Model probability": row["Model probability"],
                    "Average rank": row["Average rank"],
                    "Top 2": row["Probability Top 2"],
                    "Top 3": row["Probability Top 3"],
                }
            )
    return pd.DataFrame(rows)


def display_table(table: pd.DataFrame) -> pd.DataFrame:
    display = table.copy()
    if "Shock" in display.columns:
        display["Shock"] = display["Shock"].map(shock_label)
    for column in ["Model probability", "Top 2", "Top 3"]:
        if column in display.columns:
            display[column] = display[column].map(pct)
    if "Average rank" in display.columns:
        display["Average rank"] = display["Average rank"].map(lambda value: f"{value:.2f}")
    return display


with st.sidebar:
    st.header("IV analysis controls")
    target_date = st.date_input("Target date / maturity", value=date.today() + timedelta(days=365), min_value=date.today() + timedelta(days=1))
    days_to_target = max((target_date - date.today()).days, 1)
    st.caption(f"Horizon: {days_to_target} days ({days_to_target / 365:.2f} years)")
    simulations = st.number_input("1D simulations per shock", min_value=1_000, max_value=500_000, value=100_000, step=10_000)
    pair_surface_simulations = st.number_input("2D pair-grid simulations per cell", min_value=1_000, max_value=200_000, value=25_000, step=5_000)
    seed = st.number_input("Random seed", min_value=0, value=42, step=1)
    market_cap_source = st.selectbox("Market cap source", ["Yahoo Finance current market cap", "Manual market cap inputs"], index=0)
    iv_source = st.selectbox("Base IV source", ["Manual IV inputs", "Yahoo option-chain near-ATM IV"], index=0)
    correlation_method = st.selectbox("Correlation method", ["EWMA historical correlation", "Manual/default correlation matrix"], index=0)
    price_history_period = st.selectbox("Yahoo Finance price history", ["1y", "3y", "5y", "10y"], index=2)
    ewma_lambda = st.selectbox("EWMA lambda", [0.94, 0.97], index=1)

if "iv_analysis_company_inputs" not in st.session_state:
    st.session_state.iv_analysis_company_inputs = default_company_inputs()

st.subheader("Editable base inputs")
st.write("Market caps in this table are manual placeholders unless Yahoo Finance current market cap is selected. After running, the app shows the actual base inputs used.")
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

default_x = base_tickers.index("NVDA") if "NVDA" in base_tickers else 0
default_y = base_tickers.index("GOOGL") if "GOOGL" in base_tickers else min(1, len(base_tickers) - 1)

col1, col2, col3, col4 = st.columns(4)
with col1:
    single_ticker = st.selectbox("Single-name shock ticker", base_tickers, index=default_x)
with col2:
    pair_ticker_x = st.selectbox("Pair grid X ticker", base_tickers, index=default_x)
with col3:
    pair_ticker_y = st.selectbox("Pair grid Y ticker", base_tickers, index=default_y)
with col4:
    output_ticker = st.selectbox("Pair grid output ticker", base_tickers, index=default_x)

if st.button("Run IV sensitivity", type="primary"):
    with st.spinner("Running IV sensitivity shocks..."):
        try:
            base_inputs, market_caps = prepare_inputs(company_inputs, market_cap_source, iv_source, target_date)
            corr, corr_label = select_correlation_matrix(base_inputs, correlation_method, price_history_period, float(ewma_lambda))
            st.session_state.iv_base_inputs = base_inputs
            st.session_state.iv_market_caps = market_caps
            st.session_state.iv_global = run_iv_grid(base_inputs, corr, int(days_to_target), int(simulations), int(seed), "Global", None)
            st.session_state.iv_single = run_iv_grid(base_inputs, corr, int(days_to_target), int(simulations), int(seed), "Single-name", [single_ticker])
            st.session_state.iv_pair_line = run_iv_grid(base_inputs, corr, int(days_to_target), int(simulations), int(seed), "Pair one-dimensional", [pair_ticker_x, pair_ticker_y])
            st.session_state.iv_pair_surface = run_pair_surface(base_inputs, corr, int(days_to_target), int(pair_surface_simulations), int(seed), pair_ticker_x, pair_ticker_y, output_ticker)
            st.session_state.iv_corr_label = corr_label
            st.session_state.iv_sources = {"Market cap source": market_cap_source, "IV source": iv_source}
            st.session_state.iv_error = None
        except Exception as exc:
            st.session_state.iv_error = str(exc)

if st.session_state.get("iv_error"):
    st.error(st.session_state.iv_error)

global_table = st.session_state.get("iv_global")
single_table = st.session_state.get("iv_single")
pair_line_table = st.session_state.get("iv_pair_line")
pair_surface = st.session_state.get("iv_pair_surface")

if global_table is None:
    st.info("Run IV sensitivity to calculate global, single-name, and pair IV shock tables.")
    st.stop()

sources = st.session_state.get("iv_sources", {})
st.caption(f"Market cap source: {sources.get('Market cap source')}")
st.caption(f"IV source: {sources.get('IV source')}")
st.caption(f"Correlation assumption: {st.session_state.get('iv_corr_label')}")

base_inputs_used = st.session_state.get("iv_base_inputs")
if base_inputs_used is not None:
    with st.expander("Base inputs actually used in this IV analysis", expanded=True):
        st.dataframe(display_base_inputs(base_inputs_used), use_container_width=True, hide_index=True)

market_caps_used = display_market_caps(st.session_state.get("iv_market_caps"))
if market_caps_used is not None:
    with st.expander("Yahoo market caps used"):
        st.dataframe(market_caps_used, use_container_width=True, hide_index=True)

for title, table in [("Global IV shock", global_table), ("Single-name IV shock", single_table), ("Pair one-dimensional IV shock", pair_line_table)]:
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

if pair_surface is not None and not pair_surface.empty:
    st.subheader("Pair IV shock surface")
    meta = pair_surface.iloc[0]
    st.write(f"X axis shocks **{meta['X ticker']}** IV. Y axis shocks **{meta['Y ticker']}** IV. Cell values show **{meta['Output ticker']}** model probability.")
    surface_pivot = pair_surface.pivot(index="Y shock label", columns="X shock label", values="Model probability")
    st.dataframe(surface_pivot.map(pct), use_container_width=True)
    heatmap = px.imshow(
        surface_pivot.astype(float),
        text_auto=".1%",
        color_continuous_scale="RdYlGn",
        title=f"{meta['Output ticker']} P(#1): {meta['X ticker']} IV shock vs {meta['Y ticker']} IV shock",
        aspect="auto",
    )
    heatmap.update_layout(xaxis_title=f"{meta['X ticker']} IV shock", yaxis_title=f"{meta['Y ticker']} IV shock")
    st.plotly_chart(heatmap, use_container_width=True, key="pair_iv_surface_heatmap")

with st.expander("Full IV sensitivity tables"):
    st.write("Global IV shock")
    st.dataframe(display_table(global_table), use_container_width=True, hide_index=True)
    st.write("Single-name IV shock")
    st.dataframe(display_table(single_table), use_container_width=True, hide_index=True)
    st.write("Pair one-dimensional IV shock")
    st.dataframe(display_table(pair_line_table), use_container_width=True, hide_index=True)
    if pair_surface is not None:
        st.write("Pair two-dimensional IV surface")
        st.dataframe(display_table(pair_surface), use_container_width=True, hide_index=True)

st.subheader("Interpretation")
st.write("If the probability barely moves across IV shocks, the current market-cap gap dominates. If it moves a lot, the conclusion is sensitive to volatility assumptions and should be treated as less stable.")
