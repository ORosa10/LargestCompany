from __future__ import annotations

from datetime import date, timedelta

import pandas as pd
import plotly.express as px
import streamlit as st

from correlations import ewma_correlation, fetch_adjusted_close
from market_data import apply_market_caps, fetch_market_caps
from model import default_company_inputs, run_probability_engine


st.set_page_config(page_title="IV Analysis", layout="wide")
st.title("IV Analysis")
st.caption("Sensitivity view: isolate how ranking probabilities react to manual IV assumptions. Market caps use Yahoo Finance; correlations use historical stock prices.")

IV_SHOCKS = [level / 100 for level in range(-20, 25, 5)]


@st.cache_data(show_spinner=False, ttl=60 * 60)
def load_adjusted_close(tickers: tuple[str, ...], period: str) -> pd.DataFrame:
    return fetch_adjusted_close(list(tickers), period=period)


@st.cache_data(show_spinner=False, ttl=15 * 60)
def load_yahoo_market_caps(tickers: tuple[str, ...]) -> pd.DataFrame:
    return fetch_market_caps(list(tickers))


def pct(value: float) -> str:
    return "" if pd.isna(value) else f"{value:.2%}"


def dollars_trillions(value: float) -> str:
    return f"${value / 1e12:,.2f}T"


def shock_label(value: float) -> str:
    return f"{value * 100:+.0f} pp"


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


def prepare_inputs(company_inputs: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame | None]:
    tickers = [ticker for ticker in company_inputs["Ticker"].astype(str).str.strip().tolist() if ticker]
    market_caps = load_yahoo_market_caps(tuple(tickers))
    return apply_market_caps(company_inputs.copy(), market_caps), market_caps


def select_correlation_matrix(inputs: pd.DataFrame, period: str, ewma_lambda: float) -> tuple[pd.DataFrame, str]:
    tickers = inputs["Ticker"].astype(str).tolist()
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
                    "X shock": shock_x,
                    "Y shock": shock_y,
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
    price_history_period = st.selectbox("Yahoo price history for correlation", ["1y", "3y", "5y", "10y"], index=2)
    ewma_lambda = st.selectbox("EWMA lambda", [0.94, 0.97], index=1)
    st.caption("Market caps: Yahoo Finance. Correlation: EWMA from historical adjusted close prices. IV and Polymarket prices: manual.")

if "iv_analysis_company_inputs" not in st.session_state:
    st.session_state.iv_analysis_company_inputs = default_company_inputs()

st.subheader("Manual IV and Polymarket inputs")
st.write("Only IV and Polymarket prices are edited here. Market caps are pulled from Yahoo Finance when the analysis runs.")
previous_inputs = st.session_state.iv_analysis_company_inputs.copy()
editable_inputs = previous_inputs[["Ticker", "Implied volatility", "Polymarket YES price"]].copy()
edited_inputs = st.data_editor(
    editable_inputs,
    num_rows="dynamic",
    use_container_width=True,
    hide_index=True,
    column_config={
        "Ticker": st.column_config.TextColumn(required=True),
        "Implied volatility": st.column_config.NumberColumn("Manual IV", min_value=0.0001, max_value=5.0, step=0.01),
        "Polymarket YES price": st.column_config.NumberColumn("Manual Polymarket YES price", min_value=0.0, max_value=1.0, step=0.01),
    },
)
cap_fallback = pd.concat([previous_inputs[["Ticker", "Current market cap"]], default_company_inputs()[["Ticker", "Current market cap"]]]).drop_duplicates("Ticker", keep="first")
company_inputs = edited_inputs.merge(cap_fallback, on="Ticker", how="left")
company_inputs["Current market cap"] = company_inputs["Current market cap"].fillna(default_company_inputs()["Current market cap"].median())
company_inputs = company_inputs[["Ticker", "Current market cap", "Implied volatility", "Polymarket YES price"]]
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
            base_inputs, market_caps = prepare_inputs(company_inputs)
            corr, corr_label = select_correlation_matrix(base_inputs, price_history_period, float(ewma_lambda))
            st.session_state.iv_base_inputs = base_inputs
            st.session_state.iv_market_caps = market_caps
            st.session_state.iv_global = run_iv_grid(base_inputs, corr, int(days_to_target), int(simulations), int(seed), "Global", None)
            st.session_state.iv_single = run_iv_grid(base_inputs, corr, int(days_to_target), int(simulations), int(seed), "Single-name", [single_ticker])
            st.session_state.iv_pair_line = run_iv_grid(base_inputs, corr, int(days_to_target), int(simulations), int(seed), "Pair one-dimensional", [pair_ticker_x, pair_ticker_y])
            st.session_state.iv_pair_surface = run_pair_surface(base_inputs, corr, int(days_to_target), int(pair_surface_simulations), int(seed), pair_ticker_x, pair_ticker_y, output_ticker)
            st.session_state.iv_surface_meta = {"X ticker": pair_ticker_x, "Y ticker": pair_ticker_y, "Output ticker": output_ticker}
            st.session_state.iv_corr_label = corr_label
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

st.caption("Market cap source: Yahoo Finance current market cap")
st.caption("IV source: Manual IV inputs")
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
    pivot = table.pivot(index="Shock", columns="Ticker", values="Model probability").reindex(index=IV_SHOCKS)
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
    meta = st.session_state.get("iv_surface_meta", {})
    x_ticker = meta.get("X ticker", "X")
    y_ticker = meta.get("Y ticker", "Y")
    out_ticker = meta.get("Output ticker", "Output")
    st.write(f"X axis shocks **{x_ticker}** IV. Y axis shocks **{y_ticker}** IV. Cell values show **{out_ticker}** model probability.")
    surface_numeric = pair_surface.pivot(index="Y shock", columns="X shock", values="Model probability").reindex(index=IV_SHOCKS, columns=IV_SHOCKS)
    surface_display = surface_numeric.copy()
    surface_display.index = [shock_label(value) for value in surface_display.index]
    surface_display.columns = [shock_label(value) for value in surface_display.columns]
    st.dataframe(surface_display.map(pct), use_container_width=True)
    heatmap = px.imshow(
        surface_numeric,
        x=[shock_label(value) for value in surface_numeric.columns],
        y=[shock_label(value) for value in surface_numeric.index],
        text_auto=".1%",
        color_continuous_scale="RdYlGn",
        title=f"{out_ticker} P(#1): {x_ticker} IV shock vs {y_ticker} IV shock",
        aspect="auto",
    )
    heatmap.update_layout(xaxis_title=f"{x_ticker} IV shock", yaxis_title=f"{y_ticker} IV shock")
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
