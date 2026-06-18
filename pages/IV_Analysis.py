from __future__ import annotations

from datetime import date, timedelta

import pandas as pd
import plotly.express as px
import streamlit as st

from correlations import (
    ewma_correlation,
    fetch_adjusted_close,
    iv_based_regime_correlation,
    rolling_correlation,
    smooth_vol_adjusted_correlation,
    volatility_regime_correlation,
)
from iv_surfaces import apply_iv_estimates, estimate_atm_ivs
from market_data import apply_market_caps, fetch_market_caps
from model import default_company_inputs, default_correlation_matrix, run_probability_engine


st.set_page_config(page_title="IV Analysis", layout="wide")
st.title("IV Analysis")
st.caption("Sensitivity view: isolate how ranking probabilities react to implied-volatility assumptions.")

IV_SHOCKS = [-0.20, -0.15, -0.10, -0.05, 0.00, 0.05, 0.10, 0.15, 0.20]
CORRELATION_METHODS = [
    "EWMA historical correlation",
    "Vol-adjusted smooth correlation",
    "Rolling historical correlation",
    "Low-vol regime correlation",
    "High-vol regime correlation",
    "IV-based hard-switch regime correlation",
    "Manual/default correlation matrix",
]


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


def prepare_inputs(company_inputs: pd.DataFrame, market_cap_source: str, iv_source: str, target_date_value: date) -> tuple[pd.DataFrame, pd.DataFrame | None, pd.DataFrame | None]:
    tickers = [ticker for ticker in company_inputs["Ticker"].astype(str).str.strip().tolist() if ticker]
    simulation_inputs = company_inputs.copy()

    market_caps = None
    if market_cap_source == "Yahoo Finance current market cap":
        market_caps = load_yahoo_market_caps(tuple(tickers))
        simulation_inputs = apply_market_caps(simulation_inputs, market_caps)

    iv_estimates = None
    if iv_source == "Yahoo option-chain near-ATM IV":
        iv_estimates = load_yahoo_atm_ivs(tuple(tickers), target_date_value.isoformat())
        simulation_inputs = apply_iv_estimates(simulation_inputs, iv_estimates)

    return simulation_inputs, market_caps, iv_estimates


def correlation_matrix_for_method(
    method: str,
    simulation_inputs: pd.DataFrame,
    price_history_period: str,
    ewma_lambda: float,
    rolling_lookback: int,
    regime_vol_window: int,
    smooth_low_quantile: float,
    smooth_high_quantile: float,
    hard_switch_threshold: float,
    min_regime_observations: int,
) -> tuple[pd.DataFrame, str]:
    tickers = simulation_inputs["Ticker"].astype(str).tolist()

    if method == "Manual/default correlation matrix":
        return default_correlation_matrix(tickers), "Manual/default correlation matrix"

    prices = load_adjusted_close(tuple(tickers), price_history_period)

    if method == "EWMA historical correlation":
        return ewma_correlation(prices, ewma_lambda), f"EWMA historical correlation, lambda={ewma_lambda}"

    if method == "Vol-adjusted smooth correlation":
        current_ivs = simulation_inputs.set_index("Ticker")["Implied volatility"].astype(float)
        corr, _ = smooth_vol_adjusted_correlation(
            prices,
            current_ivs,
            vol_window=regime_vol_window,
            low_quantile=smooth_low_quantile,
            high_quantile=smooth_high_quantile,
            min_observations=min_regime_observations,
        )
        return corr, "Vol-adjusted smooth correlation"

    if method == "Rolling historical correlation":
        return rolling_correlation(prices, rolling_lookback), f"Rolling historical correlation, {rolling_lookback} trading days"

    if method == "Low-vol regime correlation":
        corr, _ = volatility_regime_correlation(
            prices,
            vol_window=regime_vol_window,
            vol_threshold=hard_switch_threshold,
            regime="low",
            min_observations=min_regime_observations,
        )
        return corr, "Low-vol regime correlation"

    if method == "High-vol regime correlation":
        corr, _ = volatility_regime_correlation(
            prices,
            vol_window=regime_vol_window,
            vol_threshold=hard_switch_threshold,
            regime="high",
            min_observations=min_regime_observations,
        )
        return corr, "High-vol regime correlation"

    current_ivs = simulation_inputs.set_index("Ticker")["Implied volatility"].astype(float)
    corr, _, _ = iv_based_regime_correlation(
        prices,
        current_ivs,
        vol_window=regime_vol_window,
        vol_threshold=hard_switch_threshold,
        min_observations=min_regime_observations,
    )
    return corr, "IV-based hard-switch regime correlation"


def apply_global_iv_shock(inputs: pd.DataFrame, shock: float) -> pd.DataFrame:
    shocked = inputs.copy()
    shocked["Implied volatility"] = (shocked["Implied volatility"].astype(float) + shock).clip(lower=0.0001)
    return shocked


def apply_ticker_iv_shock(inputs: pd.DataFrame, tickers: list[str], shock: float) -> pd.DataFrame:
    shocked = inputs.copy()
    mask = shocked["Ticker"].astype(str).isin(tickers)
    shocked.loc[mask, "Implied volatility"] = (shocked.loc[mask, "Implied volatility"].astype(float) + shock).clip(lower=0.0001)
    return shocked


def run_shock_grid(
    base_inputs: pd.DataFrame,
    corr: pd.DataFrame,
    days_to_target: int,
    simulations: int,
    seed: int,
    mode: str,
    shocked_tickers: list[str] | None = None,
) -> pd.DataFrame:
    rows = []
    for shock in IV_SHOCKS:
        if mode == "global":
            shocked_inputs = apply_global_iv_shock(base_inputs, shock)
        else:
            shocked_inputs = apply_ticker_iv_shock(base_inputs, shocked_tickers or [], shock)

        result = run_probability_engine(shocked_inputs, corr, days_to_target=days_to_target, simulations=simulations, seed=seed)
        for _, row in result.results.iterrows():
            rows.append(
                {
                    "Shock": shock,
                    "Mode": mode,
                    "Shocked tickers": ", ".join(shocked_tickers or base_inputs["Ticker"].astype(str).tolist()),
                    "Ticker": row["Ticker"],
                    "Model probability": row["Model probability"],
                    "Average rank": row["Average rank"],
                    "Top 2": row["Probability Top 2"],
                    "Top 3": row["Probability Top 3"],
                }
            )
    return pd.DataFrame(rows)


def display_sensitivity(table: pd.DataFrame) -> pd.DataFrame:
    display = table.copy()
    display["Shock"] = display["Shock"].map(lambda value: f"{value:+.0%} vol pts")
    for column in ["Model probability", "Top 2", "Top 3"]:
        if column in display.columns:
            display[column] = display[column].map(pct)
    if "Average rank" in display.columns:
        display["Average rank"] = display["Average rank"].map(lambda value: f"{value:.2f}")
    return display


def display_market_caps(market_caps: pd.DataFrame) -> pd.DataFrame:
    display = market_caps.copy().rename(columns={"ticker": "Ticker", "yahoo_ticker": "Yahoo ticker", "market_cap": "Market cap", "source": "Source"})
    display["Market cap"] = display["Market cap"].map(dollars_trillions)
    return display[["Ticker", "Yahoo ticker", "Market cap", "Source"]]


def display_iv_estimates(iv_estimates: pd.DataFrame) -> pd.DataFrame:
    display = iv_estimates.copy().rename(
        columns={
            "ticker": "Ticker",
            "yahoo_ticker": "Yahoo ticker",
            "expiry": "Option expiry used",
            "target_date": "Target date",
            "spot": "Spot",
            "atm_strike": "ATM strike",
            "implied_volatility": "ATM IV",
            "call_iv": "Call IV",
            "put_iv": "Put IV",
        }
    )
    for column in ["ATM IV", "Call IV", "Put IV"]:
        display[column] = display[column].map(pct)
    return display


with st.sidebar:
    st.header("IV analysis controls")
    target_date = st.date_input("Target date / maturity", value=date.today() + timedelta(days=365), min_value=date.today() + timedelta(days=1))
    days_to_target = max((target_date - date.today()).days, 1)
    st.caption(f"Horizon: {days_to_target} days ({days_to_target / 365:.2f} years)")
    simulations = st.number_input("Monte Carlo simulations per shock", min_value=1_000, max_value=1_000_000, value=100_000, step=10_000)
    seed = st.number_input("Random seed", min_value=0, value=42, step=1)

    st.header("Data sources")
    market_cap_source = st.selectbox("Market cap source", ["Yahoo Finance current market cap", "Manual market cap inputs"], index=0)
    iv_source = st.selectbox("Base IV source", ["Manual IV inputs", "Yahoo option-chain near-ATM IV"], index=0)

    st.header("Correlation assumption")
    correlation_method = st.selectbox("Correlation method", CORRELATION_METHODS, index=0)
    price_history_period = st.selectbox("Yahoo Finance price history", ["2y", "5y", "10y"], index=1)
    ewma_lambda = st.selectbox("EWMA lambda", [0.94, 0.97], index=1)
    rolling_lookback = st.selectbox("Rolling lookback days", [63, 126, 252, 504, 756], index=2)
    regime_vol_window = st.selectbox("Regime realized-vol window", [20, 63], index=1)
    smooth_low_quantile = st.selectbox("Smooth low-vol bucket", [0.30, 0.40, 0.50], index=1, format_func=lambda value: f"{value:.0%}")
    smooth_high_quantile = st.selectbox("Smooth high-vol bucket", [0.50, 0.60, 0.70], index=1, format_func=lambda value: f"{value:.0%}")
    hard_switch_threshold = st.number_input("Hard-switch vol / IV threshold", min_value=0.05, max_value=2.0, value=0.50, step=0.05, format="%.2f")
    min_regime_observations = st.number_input("Min observations per pair regime", min_value=10, max_value=252, value=30, step=10)

if "iv_analysis_company_inputs" not in st.session_state:
    st.session_state.iv_analysis_company_inputs = default_company_inputs()

st.subheader("Company inputs")
st.write("Manual market cap / IV values are used only when manual sources are selected. IV shocks are additive vol-point shocks on top of the base IV inputs.")
company_inputs = st.data_editor(
    st.session_state.iv_analysis_company_inputs,
    num_rows="dynamic",
    use_container_width=True,
    hide_index=True,
    column_config={
        "Ticker": st.column_config.TextColumn(required=True),
        "Current market cap": st.column_config.NumberColumn(min_value=1.0, step=10_000_000_000.0),
        "Implied volatility": st.column_config.NumberColumn(min_value=0.0001, max_value=5.0, step=0.01),
        "Polymarket YES price": st.column_config.NumberColumn(min_value=0.0, max_value=1.0, step=0.01),
    },
)
st.session_state.iv_analysis_company_inputs = company_inputs

base_tickers = [ticker for ticker in company_inputs["Ticker"].astype(str).str.strip().tolist() if ticker]
selected_single_ticker = st.selectbox("Single-name IV shock ticker", base_tickers, index=0 if base_tickers else None)
pair_left, pair_right = st.columns(2)
with pair_left:
    pair_ticker_1 = st.selectbox("Pair shock ticker 1", base_tickers, index=0 if base_tickers else None)
with pair_right:
    pair_default_index = 1 if len(base_tickers) > 1 else 0
    pair_ticker_2 = st.selectbox("Pair shock ticker 2", base_tickers, index=pair_default_index if base_tickers else None)

run_button = st.button("Run IV analysis", type="primary")

if run_button:
    with st.spinner("Running IV sensitivity shocks..."):
        try:
            base_inputs, market_caps, iv_estimates = prepare_inputs(company_inputs, market_cap_source, iv_source, target_date)
            corr, corr_label = correlation_matrix_for_method(
                correlation_method,
                base_inputs,
                price_history_period,
                float(ewma_lambda),
                int(rolling_lookback),
                int(regime_vol_window),
                float(smooth_low_quantile),
                float(smooth_high_quantile),
                float(hard_switch_threshold),
                int(min_regime_observations),
            )
            global_table = run_shock_grid(base_inputs, corr, int(days_to_target), int(simulations), int(seed), "global")
            single_table = run_shock_grid(base_inputs, corr, int(days_to_target), int(simulations), int(seed), "single-name", [selected_single_ticker])
            pair_table = run_shock_grid(base_inputs, corr, int(days_to_target), int(simulations), int(seed), "pair", [pair_ticker_1, pair_ticker_2])

            st.session_state.iv_analysis_global_table = global_table
            st.session_state.iv_analysis_single_table = single_table
            st.session_state.iv_analysis_pair_table = pair_table
            st.session_state.iv_analysis_base_inputs = base_inputs
            st.session_state.iv_analysis_market_caps = market_caps
            st.session_state.iv_analysis_iv_estimates = iv_estimates
            st.session_state.iv_analysis_corr_label = corr_label
            st.session_state.iv_analysis_error = None
        except Exception as exc:
            st.session_state.iv_analysis_error = str(exc)
            st.session_state.iv_analysis_global_table = None
            st.session_state.iv_analysis_single_table = None
            st.session_state.iv_analysis_pair_table = None

if st.session_state.get("iv_analysis_error"):
    st.error(st.session_state.iv_analysis_error)

global_table = st.session_state.get("iv_analysis_global_table")
single_table = st.session_state.get("iv_analysis_single_table")
pair_table = st.session_state.get("iv_analysis_pair_table")

if global_table is None or single_table is None or pair_table is None:
    st.info("Run the IV analysis to see how ranking probabilities react to implied-volatility shocks.")
else:
    market_caps = st.session_state.get("iv_analysis_market_caps")
    iv_estimates = st.session_state.get("iv_analysis_iv_estimates")
    corr_label = st.session_state.get("iv_analysis_corr_label")

    if market_caps is not None:
        with st.expander("Yahoo market caps used"):
            st.dataframe(display_market_caps(market_caps), use_container_width=True, hide_index=True)
    if iv_estimates is not None:
        with st.expander("Yahoo IV estimates used"):
            st.dataframe(display_iv_estimates(iv_estimates), use_container_width=True, hide_index=True)

    st.caption(f"Correlation assumption: {corr_label}")

    st.subheader("Global IV shock")
    st.write("All companies receive the same additive IV shock. This tests whether probabilities are fragile to the overall volatility level.")
    global_probability_pivot = global_table.pivot(index="Shock", columns="Ticker", values="Model probability")
    st.dataframe(global_probability_pivot.map(pct), use_container_width=True)

    selected_global_ticker = st.selectbox("Ticker for global IV shock chart", global_probability_pivot.columns.tolist(), key="global_iv_ticker")
    global_slice = global_table[global_table["Ticker"] == selected_global_ticker].sort_values("Shock")
    left, right = st.columns(2)
    with left:
        st.plotly_chart(px.line(global_slice, x="Shock", y="Model probability", markers=True, title=f"{selected_global_ticker}: P(#1) vs global IV shock"), use_container_width=True, key="global_iv_probability_line")
    with right:
        st.plotly_chart(px.line(global_slice, x="Shock", y="Average rank", markers=True, title=f"{selected_global_ticker}: average rank vs global IV shock"), use_container_width=True, key="global_iv_rank_line")

    st.subheader("Single-name IV shock")
    shocked_name = single_table["Shocked tickers"].iloc[0]
    st.write(f"Only **{shocked_name}** receives the IV shock. This tests sensitivity to one company's relative volatility.")
    single_probability_pivot = single_table.pivot(index="Shock", columns="Ticker", values="Model probability")
    st.dataframe(single_probability_pivot.map(pct), use_container_width=True)

    selected_single_chart_ticker = st.selectbox("Ticker for single-name IV shock chart", single_probability_pivot.columns.tolist(), key="single_iv_ticker")
    single_slice = single_table[single_table["Ticker"] == selected_single_chart_ticker].sort_values("Shock")
    left, right = st.columns(2)
    with left:
        st.plotly_chart(px.line(single_slice, x="Shock", y="Model probability", markers=True, title=f"{selected_single_chart_ticker}: P(#1) under {shocked_name} IV shock"), use_container_width=True, key="single_iv_probability_line")
    with right:
        st.plotly_chart(px.line(single_slice, x="Shock", y="Average rank", markers=True, title=f"{selected_single_chart_ticker}: average rank under {shocked_name} IV shock"), use_container_width=True, key="single_iv_rank_line")

    st.subheader("Pair IV shock")
    shocked_pair = pair_table["Shocked tickers"].iloc[0]
    st.write(f"Both **{shocked_pair}** receive the same IV shock. This tests relative-vol sensitivity for a specific pair.")
    pair_probability_pivot = pair_table.pivot(index="Shock", columns="Ticker", values="Model probability")
    st.dataframe(pair_probability_pivot.map(pct), use_container_width=True)

    selected_pair_chart_ticker = st.selectbox("Ticker for pair IV shock chart", pair_probability_pivot.columns.tolist(), key="pair_iv_ticker")
    pair_slice = pair_table[pair_table["Ticker"] == selected_pair_chart_ticker].sort_values("Shock")
    left, right = st.columns(2)
    with left:
        st.plotly_chart(px.line(pair_slice, x="Shock", y="Model probability", markers=True, title=f"{selected_pair_chart_ticker}: P(#1) under pair IV shock"), use_container_width=True, key="pair_iv_probability_line")
    with right:
        st.plotly_chart(px.line(pair_slice, x="Shock", y="Average rank", markers=True, title=f"{selected_pair_chart_ticker}: average rank under pair IV shock"), use_container_width=True, key="pair_iv_rank_line")

    with st.expander("Full global IV shock table"):
        st.dataframe(display_sensitivity(global_table), use_container_width=True, hide_index=True)
    with st.expander("Full single-name IV shock table"):
        st.dataframe(display_sensitivity(single_table), use_container_width=True, hide_index=True)
    with st.expander("Full pair IV shock table"):
        st.dataframe(display_sensitivity(pair_table), use_container_width=True, hide_index=True)

    st.subheader("Interpretation")
    st.write("If probabilities move materially under small IV shocks, the result is IV-sensitive and should not be read as precise. If probabilities barely move, the ranking is mostly driven by current market-cap gaps rather than volatility assumptions.")
