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


st.set_page_config(page_title="Correlation Comparison", layout="wide")
st.title("Correlation Comparison")
st.caption("Sensitivity view: compare ranking probabilities across correlation assumptions.")

CORRELATION_METHODS = [
    "EWMA historical correlation",
    "Vol-adjusted smooth correlation",
    "Rolling historical correlation",
    "Low-vol regime correlation",
    "High-vol regime correlation",
    "IV-based hard-switch regime correlation",
    "Manual/default correlation matrix",
]

FIXED_CORRELATION_LEVELS = [level / 100 for level in range(0, 100, 5)]


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


def prepare_inputs(company_inputs: pd.DataFrame, market_cap_source: str, iv_source: str, target_date: date) -> tuple[pd.DataFrame, pd.DataFrame | None, pd.DataFrame | None]:
    tickers = [ticker for ticker in company_inputs["Ticker"].astype(str).str.strip().tolist() if ticker]
    simulation_inputs = company_inputs.copy()

    market_caps = None
    if market_cap_source == "Yahoo Finance current market cap":
        market_caps = load_yahoo_market_caps(tuple(tickers))
        simulation_inputs = apply_market_caps(simulation_inputs, market_caps)

    iv_estimates = None
    if iv_source == "Yahoo option-chain near-ATM IV":
        iv_estimates = load_yahoo_atm_ivs(tuple(tickers), target_date.isoformat())
        simulation_inputs = apply_iv_estimates(simulation_inputs, iv_estimates)

    return simulation_inputs, market_caps, iv_estimates


def constant_correlation_matrix(tickers: list[str], rho: float) -> pd.DataFrame:
    corr = pd.DataFrame(rho, index=tickers, columns=tickers, dtype=float)
    for ticker in tickers:
        corr.loc[ticker, ticker] = 1.0
    return corr


def correlation_matrix_for_method(
    method: str,
    prices: pd.DataFrame | None,
    simulation_inputs: pd.DataFrame,
    ewma_lambda: float,
    rolling_lookback: int,
    regime_vol_window: int,
    smooth_low_quantile: float,
    smooth_high_quantile: float,
    hard_switch_threshold: float,
    min_regime_observations: int,
) -> tuple[pd.DataFrame, pd.DataFrame | None]:
    tickers = simulation_inputs["Ticker"].tolist()

    if method == "Manual/default correlation matrix":
        return default_correlation_matrix(tickers), None

    if prices is None:
        raise ValueError(f"{method} requires historical prices.")

    if method == "EWMA historical correlation":
        return ewma_correlation(prices, ewma_lambda), None

    if method == "Vol-adjusted smooth correlation":
        current_ivs = simulation_inputs.set_index("Ticker")["Implied volatility"].astype(float)
        return smooth_vol_adjusted_correlation(
            prices,
            current_ivs,
            vol_window=regime_vol_window,
            low_quantile=smooth_low_quantile,
            high_quantile=smooth_high_quantile,
            min_observations=min_regime_observations,
        )

    if method == "Rolling historical correlation":
        return rolling_correlation(prices, rolling_lookback), None

    if method == "Low-vol regime correlation":
        return volatility_regime_correlation(
            prices,
            vol_window=regime_vol_window,
            vol_threshold=hard_switch_threshold,
            regime="low",
            min_observations=min_regime_observations,
        )

    if method == "High-vol regime correlation":
        return volatility_regime_correlation(
            prices,
            vol_window=regime_vol_window,
            vol_threshold=hard_switch_threshold,
            regime="high",
            min_observations=min_regime_observations,
        )

    if method == "IV-based hard-switch regime correlation":
        current_ivs = simulation_inputs.set_index("Ticker")["Implied volatility"].astype(float)
        corr, diagnostics, _ = iv_based_regime_correlation(
            prices,
            current_ivs,
            vol_window=regime_vol_window,
            vol_threshold=hard_switch_threshold,
            min_observations=min_regime_observations,
        )
        return corr, diagnostics

    raise ValueError(f"Unknown correlation method: {method}")


def run_comparison(
    company_inputs: pd.DataFrame,
    selected_methods: list[str],
    market_cap_source: str,
    iv_source: str,
    target_date: date,
    simulations: int,
    seed: int,
    price_history_period: str,
    ewma_lambda: float,
    rolling_lookback: int,
    regime_vol_window: int,
    smooth_low_quantile: float,
    smooth_high_quantile: float,
    hard_switch_threshold: float,
    min_regime_observations: int,
) -> tuple[pd.DataFrame, dict[str, pd.DataFrame], pd.DataFrame | None, pd.DataFrame | None, pd.DataFrame]:
    simulation_inputs, market_caps, iv_estimates = prepare_inputs(company_inputs, market_cap_source, iv_source, target_date)
    tickers = simulation_inputs["Ticker"].tolist()
    days_to_target = max((target_date - date.today()).days, 1)

    needs_prices = any(method != "Manual/default correlation matrix" for method in selected_methods)
    prices = load_adjusted_close(tuple(tickers), price_history_period) if needs_prices else None

    rows = []
    diagnostics_by_method: dict[str, pd.DataFrame] = {}
    for method in selected_methods:
        corr, diagnostics = correlation_matrix_for_method(
            method,
            prices,
            simulation_inputs,
            ewma_lambda,
            rolling_lookback,
            regime_vol_window,
            smooth_low_quantile,
            smooth_high_quantile,
            hard_switch_threshold,
            min_regime_observations,
        )
        result = run_probability_engine(
            simulation_inputs,
            corr,
            days_to_target=days_to_target,
            simulations=simulations,
            seed=seed,
        )
        if diagnostics is not None:
            diagnostics_by_method[method] = diagnostics
        for _, row in result.results.iterrows():
            rows.append(
                {
                    "Method": method,
                    "Ticker": row["Ticker"],
                    "Current market cap": row["Current market cap"],
                    "IV": row["Implied volatility"],
                    "Model probability": row["Model probability"],
                    "Average rank": row["Average rank"],
                    "Top 2": row["Probability Top 2"],
                    "Top 3": row["Probability Top 3"],
                    "Edge": row["Edge"],
                }
            )

    fixed_correlation_sensitivity = run_fixed_correlation_sensitivity(
        simulation_inputs,
        days_to_target=days_to_target,
        simulations=simulations,
        seed=seed,
    )

    return pd.DataFrame(rows), diagnostics_by_method, market_caps, iv_estimates, fixed_correlation_sensitivity


def run_fixed_correlation_sensitivity(
    simulation_inputs: pd.DataFrame,
    days_to_target: int,
    simulations: int,
    seed: int,
) -> pd.DataFrame:
    tickers = simulation_inputs["Ticker"].tolist()
    rows = []
    for rho in FIXED_CORRELATION_LEVELS:
        corr = constant_correlation_matrix(tickers, rho)
        result = run_probability_engine(
            simulation_inputs,
            corr,
            days_to_target=days_to_target,
            simulations=simulations,
            seed=seed,
        )
        for _, row in result.results.iterrows():
            rows.append(
                {
                    "Fixed correlation": rho,
                    "Ticker": row["Ticker"],
                    "Model probability": row["Model probability"],
                    "Average rank": row["Average rank"],
                    "Top 2": row["Probability Top 2"],
                    "Top 3": row["Probability Top 3"],
                    "Edge": row["Edge"],
                }
            )
    return pd.DataFrame(rows)


def display_comparison_table(table: pd.DataFrame) -> pd.DataFrame:
    display = table.copy()
    display["Current market cap"] = display["Current market cap"].map(dollars_trillions)
    for column in ["IV", "Model probability", "Top 2", "Top 3", "Edge"]:
        display[column] = display[column].map(pct)
    display["Average rank"] = display["Average rank"].map(lambda value: f"{value:.2f}")
    return display


def display_sensitivity_table(table: pd.DataFrame) -> pd.DataFrame:
    display = table.copy()
    display["Fixed correlation"] = display["Fixed correlation"].map(pct)
    for column in ["Model probability", "Top 2", "Top 3", "Edge"]:
        display[column] = display[column].map(pct)
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
    st.header("Comparison controls")
    target_date = st.date_input("Target date / maturity", value=date.today() + timedelta(days=365), min_value=date.today() + timedelta(days=1))
    simulations = st.number_input("Monte Carlo simulations per method", min_value=1_000, max_value=1_000_000, value=100_000, step=10_000)
    seed = st.number_input("Random seed", min_value=0, value=42, step=1)

    st.header("Data sources")
    market_cap_source = st.selectbox("Market cap source", ["Yahoo Finance current market cap", "Manual market cap inputs"], index=0)
    iv_source = st.selectbox("IV source", ["Manual IV inputs", "Yahoo option-chain near-ATM IV"], index=0)

    st.header("Correlation settings")
    selected_methods = st.multiselect(
        "Methods to compare",
        CORRELATION_METHODS,
        default=["EWMA historical correlation", "Vol-adjusted smooth correlation", "Rolling historical correlation", "Low-vol regime correlation", "High-vol regime correlation", "IV-based hard-switch regime correlation"],
    )
    price_history_period = st.selectbox("Yahoo Finance price history", ["2y", "5y", "10y"], index=1)
    ewma_lambda = st.selectbox("EWMA lambda", [0.94, 0.97], index=1)
    rolling_lookback = st.selectbox("Rolling lookback days", [63, 126, 252, 504, 756], index=2)
    regime_vol_window = st.selectbox("Regime realized-vol window", [20, 63], index=1)
    smooth_low_quantile = st.selectbox("Smooth low-vol bucket", [0.30, 0.40, 0.50], index=1, format_func=lambda value: f"{value:.0%}")
    smooth_high_quantile = st.selectbox("Smooth high-vol bucket", [0.50, 0.60, 0.70], index=1, format_func=lambda value: f"{value:.0%}")
    hard_switch_threshold = st.number_input("Hard-switch vol / IV threshold", min_value=0.05, max_value=2.0, value=0.50, step=0.05, format="%.2f")
    min_regime_observations = st.number_input("Min observations per pair regime", min_value=10, max_value=252, value=30, step=10)

if "comparison_company_inputs" not in st.session_state:
    st.session_state.comparison_company_inputs = default_company_inputs()

st.subheader("Company inputs")
st.write("These inputs are independent from the main dashboard page. Manual market cap / IV values are used only when manual sources are selected.")
company_inputs = st.data_editor(
    st.session_state.comparison_company_inputs,
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
st.session_state.comparison_company_inputs = company_inputs

run_button = st.button("Run correlation comparison", type="primary")

if run_button:
    if not selected_methods:
        st.error("Select at least one correlation method.")
    else:
        with st.spinner("Running comparison across correlation methods..."):
            try:
                table, diagnostics_by_method, market_caps, iv_estimates, fixed_sensitivity = run_comparison(
                    company_inputs,
                    selected_methods,
                    market_cap_source,
                    iv_source,
                    target_date,
                    int(simulations),
                    int(seed),
                    price_history_period,
                    float(ewma_lambda),
                    int(rolling_lookback),
                    int(regime_vol_window),
                    float(smooth_low_quantile),
                    float(smooth_high_quantile),
                    float(hard_switch_threshold),
                    int(min_regime_observations),
                )
                st.session_state.correlation_comparison_table = table
                st.session_state.correlation_comparison_diagnostics = diagnostics_by_method
                st.session_state.correlation_comparison_market_caps = market_caps
                st.session_state.correlation_comparison_iv_estimates = iv_estimates
                st.session_state.fixed_correlation_sensitivity = fixed_sensitivity
            except Exception as exc:
                st.session_state.correlation_comparison_table = None
                st.session_state.fixed_correlation_sensitivity = None
                st.error(str(exc))

comparison_table = st.session_state.get("correlation_comparison_table")
if comparison_table is None:
    st.info("Run the comparison to see method-by-method probability sensitivity.")
else:
    market_caps = st.session_state.get("correlation_comparison_market_caps")
    iv_estimates = st.session_state.get("correlation_comparison_iv_estimates")
    diagnostics_by_method = st.session_state.get("correlation_comparison_diagnostics", {})
    fixed_sensitivity = st.session_state.get("fixed_correlation_sensitivity")

    if market_caps is not None:
        with st.expander("Yahoo market caps used"):
            st.dataframe(display_market_caps(market_caps), use_container_width=True, hide_index=True)
    if iv_estimates is not None:
        with st.expander("Yahoo IV estimates used"):
            st.dataframe(display_iv_estimates(iv_estimates), use_container_width=True, hide_index=True)

    st.subheader("Comparison table")
    st.dataframe(display_comparison_table(comparison_table), use_container_width=True, hide_index=True)

    probability_pivot = comparison_table.pivot(index="Ticker", columns="Method", values="Model probability")
    rank_pivot = comparison_table.pivot(index="Ticker", columns="Method", values="Average rank")

    st.subheader("Model probability by method")
    st.dataframe(probability_pivot.map(pct), use_container_width=True)

    st.subheader("Average rank by method")
    st.dataframe(rank_pivot.map(lambda value: f"{value:.2f}"), use_container_width=True)

    selected_ticker = st.selectbox("Ticker to inspect", probability_pivot.index.tolist())
    ticker_slice = comparison_table[comparison_table["Ticker"] == selected_ticker].sort_values("Model probability", ascending=False)

    left, right = st.columns(2)
    with left:
        st.plotly_chart(
            px.bar(ticker_slice, x="Method", y="Model probability", title=f"{selected_ticker}: Model Probability by Correlation Method"),
            use_container_width=True,
            key="method_probability_bar",
        )
    with right:
        st.plotly_chart(
            px.bar(ticker_slice, x="Method", y="Average rank", title=f"{selected_ticker}: Average Rank by Correlation Method"),
            use_container_width=True,
            key="method_rank_bar",
        )

    st.subheader("Probability heatmap")
    st.plotly_chart(
        px.imshow(probability_pivot, text_auto=".1%", color_continuous_scale="Blues", title="Model Probability Sensitivity"),
        use_container_width=True,
        key="method_probability_heatmap",
    )

    if fixed_sensitivity is not None and not fixed_sensitivity.empty:
        st.subheader("Fixed correlation sensitivity")
        st.write("This isolates the impact of the absolute correlation level by forcing every pairwise correlation to the same value, from 0% to 95% in 5 percentage-point steps.")

        fixed_probability_pivot = fixed_sensitivity.pivot(index="Fixed correlation", columns="Ticker", values="Model probability")
        fixed_rank_pivot = fixed_sensitivity.pivot(index="Fixed correlation", columns="Ticker", values="Average rank")

        st.dataframe(fixed_probability_pivot.map(pct), use_container_width=True)

        sensitivity_ticker = st.selectbox("Ticker for fixed-correlation sensitivity", fixed_probability_pivot.columns.tolist(), index=0)
        sensitivity_slice = fixed_sensitivity[fixed_sensitivity["Ticker"] == sensitivity_ticker].sort_values("Fixed correlation")

        left, right = st.columns(2)
        with left:
            st.plotly_chart(
                px.line(
                    sensitivity_slice,
                    x="Fixed correlation",
                    y="Model probability",
                    markers=True,
                    title=f"{sensitivity_ticker}: P(#1) vs Fixed Correlation",
                ),
                use_container_width=True,
                key="fixed_corr_probability_line",
            )
        with right:
            st.plotly_chart(
                px.line(
                    sensitivity_slice,
                    x="Fixed correlation",
                    y="Average rank",
                    markers=True,
                    title=f"{sensitivity_ticker}: Average Rank vs Fixed Correlation",
                ),
                use_container_width=True,
                key="fixed_corr_rank_line",
            )

        st.plotly_chart(
            px.imshow(
                fixed_probability_pivot,
                text_auto=".1%",
                color_continuous_scale="Blues",
                title="P(#1) Under Constant Pairwise Correlation Stress",
            ),
            use_container_width=True,
            key="fixed_corr_probability_heatmap",
        )

        with st.expander("Full fixed-correlation sensitivity table"):
            st.dataframe(display_sensitivity_table(fixed_sensitivity), use_container_width=True, hide_index=True)

        with st.expander("Average-rank fixed-correlation sensitivity"):
            st.dataframe(fixed_rank_pivot.map(lambda value: f"{value:.2f}"), use_container_width=True)

    if diagnostics_by_method:
        st.subheader("Correlation diagnostics")
        method = st.selectbox("Diagnostics method", list(diagnostics_by_method.keys()))
        diagnostics = diagnostics_by_method[method].copy()
        for column in ["Average current IV", "Historical pair-vol percentile", "Low-vol cutoff", "High-vol cutoff", "Low-regime correlation", "High-regime correlation", "Blend weight", "Selected correlation"]:
            if column in diagnostics.columns:
                diagnostics[column] = diagnostics[column].map(pct)
        st.dataframe(diagnostics, use_container_width=True, hide_index=True)
