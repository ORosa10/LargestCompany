from __future__ import annotations

from datetime import date, datetime, timedelta
from math import erf, sqrt
import zlib

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
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


st.set_page_config(page_title="LargestCompany", layout="wide")

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


def normal_cdf(value: float) -> float:
    return 0.5 * (1.0 + erf(value / sqrt(2.0)))


def dollars_trillions(value: float) -> str:
    return f"${value / 1e12:,.2f}T"


def pct(value: float) -> str:
    return "" if pd.isna(value) else f"{value:.2%}"


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
    for column in ["Spot", "ATM strike"]:
        display[column] = display[column].map(lambda value: f"${value:,.2f}" if pd.notna(value) else "")
    for column in ["ATM IV", "Call IV", "Put IV"]:
        display[column] = display[column].map(pct)
    return display[["Ticker", "Yahoo ticker", "Target date", "Option expiry used", "Spot", "ATM strike", "ATM IV", "Call IV", "Put IV"]]


def display_results(results: pd.DataFrame) -> pd.DataFrame:
    display = results.copy().rename(
        columns={
            "Current market cap": "Mkt cap",
            "Implied volatility": "IV",
            "Polymarket YES price": "Poly price",
            "Model probability": "Model prob",
            "Average rank": "Avg rank",
            "Probability Top 2": "Top 2",
            "Probability Top 3": "Top 3",
        }
    )
    display["Mkt cap"] = display["Mkt cap"].map(dollars_trillions)
    for column in ["IV", "Poly price", "Model prob", "Edge", "Top 2", "Top 3"]:
        display[column] = display[column].map(pct)
    display["Avg rank"] = display["Avg rank"].map(lambda value: f"{value:.2f}")
    return display[["Ticker", "Mkt cap", "IV", "Poly price", "Model prob", "Edge", "Avg rank", "Top 2", "Top 3"]]


def display_regime_diagnostics(diagnostics: pd.DataFrame | None) -> pd.DataFrame | None:
    if diagnostics is None or diagnostics.empty:
        return None
    display = diagnostics.copy()
    for column in [
        "Average current IV",
        "Historical pair-vol percentile",
        "Low-vol cutoff",
        "High-vol cutoff",
        "Low-regime correlation",
        "High-regime correlation",
        "Blend weight",
        "Selected correlation",
    ]:
        if column in display.columns:
            display[column] = display[column].map(pct)
    return display


def monte_carlo_precision_table(results: pd.DataFrame, simulations: int) -> pd.DataFrame:
    rows = []
    n = max(int(simulations), 1)
    for _, row in results.iterrows():
        p = float(row["Model probability"])
        se = sqrt(max(p * (1.0 - p), 0.0) / n)
        rows.append(
            {
                "Ticker": row["Ticker"],
                "Model probability": p,
                "MC standard error": se,
                "Approx. 95% low": max(p - 1.96 * se, 0.0),
                "Approx. 95% high": min(p + 1.96 * se, 1.0),
            }
        )
    return pd.DataFrame(rows)


def display_monte_carlo_precision(table: pd.DataFrame) -> pd.DataFrame:
    display = table.copy()
    for column in ["Model probability", "MC standard error", "Approx. 95% low", "Approx. 95% high"]:
        display[column] = display[column].map(pct)
    return display


def constant_correlation_matrix(tickers: list[str], rho: float) -> pd.DataFrame:
    corr = pd.DataFrame(rho, index=tickers, columns=tickers, dtype=float)
    for ticker in tickers:
        corr.loc[ticker, ticker] = 1.0
    return corr


def fixed_correlation_sensitivity(company_inputs: pd.DataFrame, days_to_target: int, simulations: int, seed: int) -> pd.DataFrame:
    tickers = company_inputs["Ticker"].astype(str).tolist()
    rows = []
    for rho in FIXED_CORRELATION_LEVELS:
        corr = constant_correlation_matrix(tickers, rho)
        result = run_probability_engine(company_inputs, corr, days_to_target=days_to_target, simulations=simulations, seed=seed)
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


def display_fixed_sensitivity(table: pd.DataFrame) -> pd.DataFrame:
    display = table.copy()
    display["Fixed correlation"] = display["Fixed correlation"].map(pct)
    for column in ["Model probability", "Top 2", "Top 3", "Edge"]:
        display[column] = display[column].map(pct)
    display["Average rank"] = display["Average rank"].map(lambda value: f"{value:.2f}")
    return display


def correlation_summary(corr: pd.DataFrame) -> pd.DataFrame:
    values = corr.to_numpy(dtype=float, copy=True)
    mask = ~np.eye(values.shape[0], dtype=bool)
    off_diag = values[mask]
    return pd.DataFrame(
        [
            {"Metric": "Average pairwise correlation", "Value": off_diag.mean()},
            {"Metric": "Median pairwise correlation", "Value": np.median(off_diag)},
            {"Metric": "Minimum pairwise correlation", "Value": off_diag.min()},
            {"Metric": "Maximum pairwise correlation", "Value": off_diag.max()},
        ]
    )


def display_correlation_summary(summary: pd.DataFrame) -> pd.DataFrame:
    display = summary.copy()
    display["Value"] = display["Value"].map(pct)
    return display


def market_cap_percentile_table(result) -> pd.DataFrame:
    rows = []
    percentiles = {"P1": 0.01, "P5": 0.05, "P25": 0.25, "P50": 0.50, "P75": 0.75, "P95": 0.95, "P99": 0.99}
    for ticker in result.terminal_market_caps.columns:
        caps = result.terminal_market_caps[ticker]
        row = {"Ticker": ticker, "Mean": caps.mean(), "Std dev": caps.std()}
        for label, quantile in percentiles.items():
            row[label] = caps.quantile(quantile)
        rows.append(row)
    table = pd.DataFrame(rows)
    table["Average rank"] = table["Ticker"].map(result.results.set_index("Ticker")["Average rank"])
    return table.sort_values("P50", ascending=False, ignore_index=True)


def display_market_cap_percentiles(percentiles: pd.DataFrame) -> pd.DataFrame:
    display = percentiles.copy()
    for column in ["Mean", "Std dev", "P1", "P5", "P25", "P50", "P75", "P95", "P99"]:
        display[column] = display[column].map(dollars_trillions)
    display["Average rank"] = display["Average rank"].map(lambda value: f"{value:.2f}")
    return display


def rank_probability_matrix(result) -> pd.DataFrame:
    rows = []
    for ticker in result.ranks.columns:
        row = {"Ticker": ticker}
        for rank in range(1, len(result.ranks.columns) + 1):
            row[f"Rank {rank}"] = (result.ranks[ticker] == rank).mean()
        rows.append(row)
    return pd.DataFrame(rows)


def display_rank_probability_matrix(rank_probs: pd.DataFrame) -> pd.DataFrame:
    display = rank_probs.copy()
    for column in display.columns:
        if column != "Ticker":
            display[column] = display[column].map(pct)
    return display


def selected_rank_probabilities(result, ticker: str) -> pd.DataFrame:
    rank_probs = rank_probability_matrix(result).set_index("Ticker").loc[ticker]
    return pd.DataFrame({"Rank": rank_probs.index, "Probability": rank_probs.values})


def selected_ticker_summary(result, ticker: str) -> pd.DataFrame:
    caps = result.terminal_market_caps[ticker]
    ranks = result.ranks[ticker]
    rows = [
        ("Simulated market cap mean", dollars_trillions(caps.mean())),
        ("Simulated market cap median", dollars_trillions(caps.median())),
        ("1st percentile market cap", dollars_trillions(caps.quantile(0.01))),
        ("5th percentile market cap", dollars_trillions(caps.quantile(0.05))),
        ("25th percentile market cap", dollars_trillions(caps.quantile(0.25))),
        ("75th percentile market cap", dollars_trillions(caps.quantile(0.75))),
        ("95th percentile market cap", dollars_trillions(caps.quantile(0.95))),
        ("99th percentile market cap", dollars_trillions(caps.quantile(0.99))),
        ("Average simulated rank", f"{ranks.mean():.2f}"),
        ("Median simulated rank", f"{ranks.median():.0f}"),
    ]
    return pd.DataFrame([{"Metric": metric, "Value": value} for metric, value in rows])


def ticker_row(result, ticker: str) -> pd.Series:
    return result.results.set_index("Ticker").loc[ticker]


def terminal_cap_long(result, tickers: list[str]) -> pd.DataFrame:
    cap_long = result.terminal_market_caps[tickers].melt(var_name="Ticker", value_name="Simulated market cap")
    cap_long["Simulated market cap ($T)"] = cap_long["Simulated market cap"] / 1e12
    return cap_long


def pairwise_probability_audit(result, ticker: str, days_to_target: int) -> pd.DataFrame:
    rows = []
    selected = ticker_row(result, ticker)
    selected_cap = float(selected["Current market cap"])
    selected_iv = float(selected["Implied volatility"])
    years = max(days_to_target, 1) / 365.0

    for _, other_row in result.results.iterrows():
        other = other_row["Ticker"]
        if other == ticker:
            continue
        other_cap = float(other_row["Current market cap"])
        other_iv = float(other_row["Implied volatility"])
        rho = float(result.cleaned_correlation.loc[ticker, other])
        relative_var = selected_iv**2 + other_iv**2 - 2.0 * rho * selected_iv * other_iv
        relative_vol = sqrt(max(relative_var, 0.0))
        log_gap_now = np.log(selected_cap / other_cap)
        mean_log_gap = log_gap_now - 0.5 * (selected_iv**2 - other_iv**2) * years
        horizon_vol = relative_vol * sqrt(years)
        z_score = np.inf if horizon_vol <= 1e-12 and mean_log_gap > 0 else mean_log_gap / horizon_vol
        pairwise_probability = 1.0 if horizon_vol <= 1e-12 and mean_log_gap > 0 else normal_cdf(z_score)
        rows.append(
            {
                "Competitor": other,
                "Selected cap": selected_cap,
                "Competitor cap": other_cap,
                "Market-cap gap": selected_cap - other_cap,
                "Log gap now": log_gap_now,
                "Selected IV": selected_iv,
                "Competitor IV": other_iv,
                "Correlation": rho,
                "Relative annual vol": relative_vol,
                "Horizon vol": horizon_vol,
                "Z-score": z_score,
                "P(selected > competitor)": pairwise_probability,
            }
        )
    return pd.DataFrame(rows).sort_values("Market-cap gap", ascending=True, ignore_index=True)


def display_pairwise_probability_audit(pairwise: pd.DataFrame) -> pd.DataFrame:
    display = pairwise.copy()
    for column in ["Selected cap", "Competitor cap", "Market-cap gap"]:
        display[column] = display[column].map(dollars_trillions)
    for column in ["Selected IV", "Competitor IV", "Correlation", "Relative annual vol", "Horizon vol", "P(selected > competitor)"]:
        display[column] = display[column].map(pct)
    display["Log gap now"] = display["Log gap now"].map(pct)
    display["Z-score"] = display["Z-score"].map(lambda value: f"{value:.2f}")
    return display


def simulate_marginal_paths(current_cap: float, sigma: float, days: int, path_count: int, seed: int, ticker: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    steps = max(int(days), 1)
    dt = 1.0 / 365.0
    rng = np.random.default_rng(int(seed) + zlib.crc32(ticker.encode("utf-8")))
    shocks = rng.standard_normal((path_count, steps))
    log_increments = (-0.5 * sigma**2 * dt) + sigma * np.sqrt(dt) * shocks
    caps = current_cap * np.exp(np.cumsum(log_increments, axis=1))
    caps = np.column_stack([np.full(path_count, current_cap), caps])
    day_grid = np.arange(steps + 1)

    sample_size = min(path_count, 200)
    sample_paths = pd.DataFrame(caps[:sample_size].T, index=day_grid)
    sample_paths.index.name = "Day"
    sample_long = sample_paths.reset_index().melt(id_vars="Day", var_name="Path", value_name="Market cap")
    sample_long["Market cap ($T)"] = sample_long["Market cap"] / 1e12

    percentile_rows = []
    for day_idx, day in enumerate(day_grid):
        values = caps[:, day_idx]
        percentile_rows.append(
            {
                "Day": day,
                "P5": np.quantile(values, 0.05) / 1e12,
                "P25": np.quantile(values, 0.25) / 1e12,
                "P50": np.quantile(values, 0.50) / 1e12,
                "P75": np.quantile(values, 0.75) / 1e12,
                "P95": np.quantile(values, 0.95) / 1e12,
            }
        )
    return sample_long, pd.DataFrame(percentile_rows)


def path_fan_chart(sample_long: pd.DataFrame, percentile_paths: pd.DataFrame, ticker: str) -> go.Figure:
    fig = go.Figure()
    for _, path_data in sample_long.groupby("Path"):
        fig.add_trace(go.Scatter(x=path_data["Day"], y=path_data["Market cap ($T)"], mode="lines", line={"color": "rgba(80, 130, 190, 0.08)", "width": 1}, showlegend=False, hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=percentile_paths["Day"], y=percentile_paths["P95"], mode="lines", line={"width": 0}, showlegend=False, hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=percentile_paths["Day"], y=percentile_paths["P5"], mode="lines", fill="tonexty", fillcolor="rgba(30, 120, 220, 0.16)", line={"width": 0}, name="P5-P95"))
    fig.add_trace(go.Scatter(x=percentile_paths["Day"], y=percentile_paths["P75"], mode="lines", line={"width": 0}, showlegend=False, hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=percentile_paths["Day"], y=percentile_paths["P25"], mode="lines", fill="tonexty", fillcolor="rgba(30, 120, 220, 0.28)", line={"width": 0}, name="P25-P75"))
    fig.add_trace(go.Scatter(x=percentile_paths["Day"], y=percentile_paths["P50"], mode="lines", line={"color": "#111827", "width": 3}, name="Median"))
    fig.update_layout(title=f"Illustrative marginal market-cap paths: {ticker}", xaxis_title="Days from today", yaxis_title="Market cap ($T)", hovermode="x unified")
    return fig


def terminal_distribution_chart(result, ticker: str) -> go.Figure:
    caps_t = result.terminal_market_caps[ticker] / 1e12
    fig = px.histogram(x=caps_t, nbins=90, title=f"Terminal market-cap distribution: {ticker}")
    fig.update_layout(xaxis_title="Terminal market cap ($T)", yaxis_title="Simulation count")
    for q, value in caps_t.quantile([0.01, 0.05, 0.25, 0.50, 0.75, 0.95, 0.99]).items():
        fig.add_vline(x=value, line_dash="dash", annotation_text=f"P{int(q * 100)}", annotation_position="top")
    return fig


def prepare_simulation_inputs(company_inputs: pd.DataFrame, market_cap_source: str, iv_source: str, target_date_value: date) -> tuple[pd.DataFrame, pd.DataFrame | None, pd.DataFrame | None, str, str]:
    clean_tickers = [ticker for ticker in company_inputs["Ticker"].astype(str).str.strip().tolist() if ticker]
    simulation_inputs = company_inputs.copy()

    market_caps = None
    if market_cap_source == "Yahoo Finance current market cap":
        market_caps = load_yahoo_market_caps(tuple(clean_tickers))
        simulation_inputs = apply_market_caps(simulation_inputs, market_caps)
        market_cap_source_label = "Yahoo Finance current market cap"
    else:
        market_cap_source_label = "Manual market cap inputs"

    iv_estimates = None
    if iv_source == "Yahoo option-chain near-ATM IV":
        iv_estimates = load_yahoo_atm_ivs(tuple(clean_tickers), target_date_value.isoformat())
        simulation_inputs = apply_iv_estimates(simulation_inputs, iv_estimates)
        iv_source_label = "Yahoo Finance option-chain near-ATM IV; expiry closest to target date"
    else:
        iv_source_label = "Manual IV inputs"

    return simulation_inputs, market_caps, iv_estimates, market_cap_source_label, iv_source_label


def select_correlation_matrix(
    correlation_source: str,
    simulation_inputs: pd.DataFrame,
    manual_correlation_matrix: pd.DataFrame,
    price_history_period: str,
    ewma_lambda: float,
    rolling_lookback: int,
    regime_vol_window: int,
    smooth_low_quantile: float,
    smooth_high_quantile: float,
    regime_vol_threshold: float,
    min_regime_observations: int,
) -> tuple[pd.DataFrame, str, dict | None, pd.DataFrame | None]:
    clean_tickers = simulation_inputs["Ticker"].astype(str).tolist()
    price_info = None
    regime_diagnostics = None

    if correlation_source == "Manual correlation matrix":
        return manual_correlation_matrix, "Manual correlation matrix", price_info, regime_diagnostics

    prices = load_adjusted_close(tuple(clean_tickers), price_history_period)
    price_info = {"rows": len(prices), "start": prices.index.min().date().isoformat(), "end": prices.index.max().date().isoformat(), "period": price_history_period}

    if correlation_source == "EWMA historical correlation":
        return ewma_correlation(prices, float(ewma_lambda)), f"EWMA historical correlation, lambda={ewma_lambda}, Yahoo Finance {price_history_period}", price_info, regime_diagnostics

    if correlation_source == "Vol-adjusted smooth correlation":
        current_ivs = simulation_inputs.set_index("Ticker")["Implied volatility"].astype(float)
        corr, regime_diagnostics = smooth_vol_adjusted_correlation(
            prices,
            current_ivs,
            vol_window=int(regime_vol_window),
            low_quantile=float(smooth_low_quantile),
            high_quantile=float(smooth_high_quantile),
            min_observations=int(min_regime_observations),
        )
        label = f"Vol-adjusted smooth correlation, window={regime_vol_window}D, low/high buckets={smooth_low_quantile:.0%}/{smooth_high_quantile:.0%}"
        return corr, label, price_info, regime_diagnostics

    if correlation_source == "Rolling historical correlation":
        return rolling_correlation(prices, int(rolling_lookback)), f"Rolling historical correlation, {rolling_lookback} trading days, Yahoo Finance {price_history_period}", price_info, regime_diagnostics

    if correlation_source == "Low-vol regime correlation":
        corr, counts = volatility_regime_correlation(prices, vol_window=int(regime_vol_window), vol_threshold=float(regime_vol_threshold), regime="low", min_observations=int(min_regime_observations))
        label = f"Low-vol regime correlation, realized-vol threshold={regime_vol_threshold:.0%}, window={regime_vol_window}D"
        return corr, label, price_info, counts.reset_index().rename(columns={"index": "Ticker"})

    if correlation_source == "High-vol regime correlation":
        corr, counts = volatility_regime_correlation(prices, vol_window=int(regime_vol_window), vol_threshold=float(regime_vol_threshold), regime="high", min_observations=int(min_regime_observations))
        label = f"High-vol regime correlation, realized-vol threshold={regime_vol_threshold:.0%}, window={regime_vol_window}D"
        return corr, label, price_info, counts.reset_index().rename(columns={"index": "Ticker"})

    current_ivs = simulation_inputs.set_index("Ticker")["Implied volatility"].astype(float)
    corr, regime_diagnostics, _ = iv_based_regime_correlation(prices, current_ivs, vol_window=int(regime_vol_window), vol_threshold=float(regime_vol_threshold), min_observations=int(min_regime_observations))
    label = f"IV-based hard-switch regime correlation, pair avg IV threshold={regime_vol_threshold:.0%}, realized-vol window={regime_vol_window}D"
    return corr, label, price_info, regime_diagnostics


st.title("LargestCompany")
st.caption("Phase 1: statistical probability engine for largest future market capitalization.")
st.warning("Prototype status: Polymarket price inputs are still manual placeholders. Market caps, correlations, and near-ATM IV can be sourced from Yahoo Finance.")
st.info("This app does not predict stock prices. It translates current market caps, implied volatility, target-date horizon, and correlation assumptions into fair ranking probabilities.")

if "company_inputs" not in st.session_state:
    st.session_state.company_inputs = default_company_inputs()
if "correlation_matrix" not in st.session_state:
    tickers = st.session_state.company_inputs["Ticker"].tolist()
    st.session_state.correlation_matrix = default_correlation_matrix(tickers)
for key in [
    "last_result",
    "last_error",
    "last_run",
    "last_corr_source",
    "last_price_info",
    "last_iv_source",
    "last_iv_estimates",
    "last_market_cap_source",
    "last_market_caps",
    "last_regime_diagnostics",
    "last_fixed_sensitivity",
]:
    if key not in st.session_state:
        st.session_state[key] = None


today = date.today()
default_target_date = today + timedelta(days=365)

with st.sidebar:
    st.header("Simulation controls")
    target_date = st.date_input("Target date / maturity", value=default_target_date, min_value=today + timedelta(days=1))
    days_to_target = max((target_date - today).days, 1)
    horizon_years = days_to_target / 365.0
    st.caption(f"Horizon: {days_to_target} days ({horizon_years:.2f} years)")
    simulations = st.number_input("Monte Carlo simulations", min_value=1_000, max_value=2_000_000, value=100_000, step=10_000)
    seed = st.number_input("Random seed", min_value=0, value=42, step=1)

    st.header("Market cap source")
    market_cap_source = st.selectbox("Current market capitalization source", ["Yahoo Finance current market cap", "Manual market cap inputs"], index=0)

    st.header("IV source")
    iv_source = st.selectbox("Implied volatility source", ["Manual IV inputs", "Yahoo option-chain near-ATM IV"], index=0)

    st.header("Correlation source")
    correlation_source = st.selectbox(
        "Correlation method",
        [
            "EWMA historical correlation",
            "Vol-adjusted smooth correlation",
            "Rolling historical correlation",
            "Low-vol regime correlation",
            "High-vol regime correlation",
            "IV-based hard-switch regime correlation",
            "Manual correlation matrix",
        ],
        index=0,
    )
    price_history_period = st.selectbox("Yahoo Finance price history", ["2y", "5y", "10y"], index=1)
    ewma_lambda = st.selectbox("EWMA lambda", [0.94, 0.97], index=1)
    rolling_lookback = st.selectbox("Rolling lookback days", [63, 126, 252, 504, 756], index=2)
    regime_vol_window = st.selectbox("Regime realized-vol window", [20, 63], index=1)
    smooth_low_quantile = st.selectbox("Smooth low-vol bucket", [0.30, 0.40, 0.50], index=1, format_func=lambda value: f"{value:.0%}")
    smooth_high_quantile = st.selectbox("Smooth high-vol bucket", [0.50, 0.60, 0.70], index=1, format_func=lambda value: f"{value:.0%}")
    regime_vol_threshold = st.number_input("Hard-switch vol / IV threshold", min_value=0.05, max_value=2.0, value=0.50, step=0.05, format="%.2f")
    min_regime_observations = st.number_input("Min observations per pair regime", min_value=10, max_value=252, value=30, step=10)

    selected_ticker_sidebar = st.selectbox("Selected ticker for diagnostics", st.session_state.company_inputs["Ticker"].astype(str).tolist())
    run_button = st.button("Run / refresh simulation", type="primary", use_container_width=True)


results_tab, correlation_tab, inputs_tab, diagnostics_tab, methodology_tab = st.tabs(["Results", "Correlation Analysis", "Inputs & Data", "Simulation Diagnostics", "Methodology"])

with inputs_tab:
    st.subheader("Data provenance")
    st.dataframe(
        pd.DataFrame(
            [
                {"Input": "Current market capitalization", "Current source": "Yahoo Finance current market cap or manual input", "Future source": "Configurable market data provider"},
                {"Input": "Annualized implied volatility", "Current source": "Manual input or Yahoo option-chain near-ATM IV", "Future source": "Robust IV surface provider"},
                {"Input": "Polymarket YES price", "Current source": "Manual placeholder", "Future source": "Polymarket market API or manual override"},
                {"Input": "Correlation matrix", "Current source": "Yahoo historical prices: EWMA, rolling, smooth vol-adjusted, vol-regime, or manual", "Future source": "Configurable institutional data provider"},
                {"Input": "Target date / maturity", "Current source": "User-selected date", "Future source": "Parsed from prediction-market event rules"},
            ]
        ),
        use_container_width=True,
        hide_index=True,
    )

    st.subheader("Company inputs")
    st.write("Manual market cap / IV values are used only when manual sources are selected in the sidebar.")
    company_inputs = st.data_editor(
        st.session_state.company_inputs,
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
    st.session_state.company_inputs = company_inputs

    tickers = [ticker for ticker in company_inputs["Ticker"].astype(str).str.strip().tolist() if ticker]
    if tickers:
        current_corr = st.session_state.correlation_matrix.reindex(index=tickers, columns=tickers)
        current_corr = current_corr.fillna(default_correlation_matrix(tickers))
    else:
        current_corr = st.session_state.correlation_matrix

    st.subheader("Manual correlation matrix")
    st.write("Used only when **Manual correlation matrix** is selected in the sidebar.")
    st.session_state.correlation_matrix = st.data_editor(
        current_corr,
        use_container_width=True,
        column_config={ticker: st.column_config.NumberColumn(min_value=-1.0, max_value=1.0, step=0.05) for ticker in tickers},
    )


company_inputs = st.session_state.company_inputs
manual_correlation_matrix = st.session_state.correlation_matrix

if run_button or st.session_state.last_result is None:
    with st.spinner("Running Monte Carlo simulation and correlation analysis..."):
        try:
            simulation_inputs, market_caps, iv_estimates, market_cap_source_label, iv_source_label = prepare_simulation_inputs(company_inputs, market_cap_source, iv_source, target_date)
            selected_correlation_matrix, corr_source_label, price_info, regime_diagnostics = select_correlation_matrix(
                correlation_source,
                simulation_inputs,
                manual_correlation_matrix,
                price_history_period,
                float(ewma_lambda),
                int(rolling_lookback),
                int(regime_vol_window),
                float(smooth_low_quantile),
                float(smooth_high_quantile),
                float(regime_vol_threshold),
                int(min_regime_observations),
            )

            st.session_state.last_result = run_probability_engine(simulation_inputs, selected_correlation_matrix, days_to_target=int(days_to_target), simulations=int(simulations), seed=int(seed))
            st.session_state.last_fixed_sensitivity = fixed_correlation_sensitivity(simulation_inputs, int(days_to_target), int(simulations), int(seed))
            st.session_state.last_error = None
            st.session_state.last_corr_source = corr_source_label
            st.session_state.last_price_info = price_info
            st.session_state.last_iv_source = iv_source_label
            st.session_state.last_iv_estimates = iv_estimates
            st.session_state.last_market_cap_source = market_cap_source_label
            st.session_state.last_market_caps = market_caps
            st.session_state.last_regime_diagnostics = regime_diagnostics
            st.session_state.last_run = {"time": datetime.now().strftime("%H:%M:%S"), "target_date": target_date.isoformat(), "days_to_target": int(days_to_target), "horizon_years": horizon_years, "simulations": int(simulations), "seed": int(seed)}
        except Exception as exc:
            st.session_state.last_result = None
            st.session_state.last_fixed_sensitivity = None
            st.session_state.last_error = str(exc)
            st.session_state.last_run = None
            st.session_state.last_corr_source = None
            st.session_state.last_price_info = None
            st.session_state.last_iv_source = None
            st.session_state.last_iv_estimates = None
            st.session_state.last_market_cap_source = None
            st.session_state.last_market_caps = None
            st.session_state.last_regime_diagnostics = None


result = st.session_state.last_result

with results_tab:
    if st.session_state.last_error:
        st.error(st.session_state.last_error)
    elif result is None:
        st.warning("No simulation result yet. Check inputs and click Run / refresh simulation.")
    else:
        run = st.session_state.last_run or {}
        available_tickers = result.results["Ticker"].tolist()
        if selected_ticker_sidebar not in available_tickers:
            selected_ticker_sidebar = available_tickers[0]

        st.success(
            "Simulation completed"
            f" | target {run.get('target_date', target_date.isoformat())}"
            f" | {run.get('days_to_target', days_to_target)} days"
            f" | {run.get('simulations', simulations):,} paths"
            f" | seed {run.get('seed', seed)}"
            f" | last run {run.get('time', 'now')}"
        )
        st.caption(f"Market cap source: {st.session_state.last_market_cap_source}")
        st.caption(f"IV source: {st.session_state.last_iv_source}")
        st.caption(f"Correlation source: {st.session_state.last_corr_source}")
        if st.session_state.last_price_info:
            info = st.session_state.last_price_info
            st.caption(f"Yahoo adjusted close sample: {info['rows']} daily rows from {info['start']} to {info['end']} ({info['period']}).")

        if st.session_state.last_market_caps is not None:
            with st.expander("Yahoo market caps used"):
                st.dataframe(display_market_caps(st.session_state.last_market_caps), use_container_width=True, hide_index=True)
        if st.session_state.last_iv_estimates is not None:
            with st.expander("Yahoo option-chain IV estimates used"):
                st.dataframe(display_iv_estimates(st.session_state.last_iv_estimates), use_container_width=True, hide_index=True)
        if st.session_state.last_regime_diagnostics is not None:
            with st.expander("Volatility-adjusted correlation diagnostics"):
                shown = display_regime_diagnostics(st.session_state.last_regime_diagnostics)
                st.dataframe(shown if shown is not None else st.session_state.last_regime_diagnostics, use_container_width=True, hide_index=True)

        for warning in result.warnings:
            st.warning(warning)

        probability_sum = result.results["Model probability"].sum()
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Highest model probability", result.results.sort_values("Model probability", ascending=False).iloc[0]["Ticker"])
        c2.metric("Largest positive edge", result.most_undervalued["Ticker"], f'{result.most_undervalued["Edge"]:.2%}')
        c3.metric("Largest negative edge", result.most_overvalued["Ticker"], f'{result.most_overvalued["Edge"]:.2%}')
        c4.metric("Probability check", f"{probability_sum:.2%}")

        st.subheader("Statistical ranking probabilities")
        results_display = display_results(result.results)
        selected_from_table = selected_ticker_sidebar
        table_event = st.dataframe(results_display, use_container_width=True, hide_index=True, on_select="rerun", selection_mode="single-row", key="results_probability_table")
        if table_event.selection.rows:
            selected_from_table = results_display.iloc[table_event.selection.rows[0]]["Ticker"]

        with st.expander("Monte Carlo precision"):
            st.write("Sampling error from finite simulations only. This does not include model error, data-source error, IV surface error, or correlation-model uncertainty.")
            precision = monte_carlo_precision_table(result.results, int(run.get("simulations", simulations)))
            st.dataframe(display_monte_carlo_precision(precision), use_container_width=True, hide_index=True)

        st.subheader(f"Company detail: {selected_from_table}")
        selected_row = ticker_row(result, selected_from_table)
        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric("Model probability", f'{selected_row["Model probability"]:.2%}')
        m2.metric("Top 2 probability", f'{selected_row["Probability Top 2"]:.2%}')
        m3.metric("Top 3 probability", f'{selected_row["Probability Top 3"]:.2%}')
        m4.metric("Average rank", f'{selected_row["Average rank"]:.2f}')
        m5.metric("Current market cap", dollars_trillions(float(selected_row["Current market cap"])))

        detail_left, detail_right = st.columns([1, 2])
        with detail_left:
            inspect_ticker = st.selectbox("Detail ticker", available_tickers, index=available_tickers.index(selected_from_table), key="detail_ticker_select")
            st.dataframe(selected_ticker_summary(result, inspect_ticker), use_container_width=True, hide_index=True)
            st.subheader("Exact rank probabilities")
            rank_detail = selected_rank_probabilities(result, inspect_ticker)
            rank_detail["Probability"] = rank_detail["Probability"].map(pct)
            st.dataframe(rank_detail, use_container_width=True, hide_index=True)
        with detail_right:
            rank_data = result.rank_distribution[result.rank_distribution["Ticker"] == inspect_ticker]
            st.plotly_chart(px.bar(rank_data, x="Rank", y="Probability", title=f"Rank Distribution: {inspect_ticker}"), use_container_width=True, key="results_rank_distribution_chart")

        st.subheader("Pairwise probability audit")
        st.write("This table explains the #1 probability pressure one competitor at a time. It is analytical and pairwise, so it is a diagnostic, not the full joint winner probability.")
        pairwise = pairwise_probability_audit(result, inspect_ticker, int(run.get("days_to_target", days_to_target)))
        st.dataframe(display_pairwise_probability_audit(pairwise), use_container_width=True, hide_index=True)

        st.subheader("Terminal distribution and path diagnostics")
        path_count = st.slider("Illustrative path count", min_value=50, max_value=2000, value=500, step=50)
        selected_row = ticker_row(result, inspect_ticker)
        path_sample, path_percentiles = simulate_marginal_paths(float(selected_row["Current market cap"]), float(selected_row["Implied volatility"]), int(run.get("days_to_target", days_to_target)), int(path_count), int(run.get("seed", seed)), inspect_ticker)
        path_left, path_right = st.columns(2)
        with path_left:
            st.plotly_chart(path_fan_chart(path_sample, path_percentiles, inspect_ticker), use_container_width=True, key="results_path_fan_chart")
        with path_right:
            st.plotly_chart(terminal_distribution_chart(result, inspect_ticker), use_container_width=True, key="results_terminal_histogram")

        st.subheader("Compare simulated market-cap distributions")
        default_compare = available_tickers[: min(5, len(available_tickers))]
        compare_tickers = st.multiselect("Companies to compare", available_tickers, default=default_compare)
        if compare_tickers:
            st.plotly_chart(px.box(terminal_cap_long(result, compare_tickers), x="Ticker", y="Simulated market cap ($T)", points=False, title="Selected Companies: Terminal Market-Cap Distributions"), use_container_width=True, key="results_compare_box_chart")

with correlation_tab:
    if st.session_state.last_error:
        st.error(st.session_state.last_error)
    elif result is None:
        st.warning("Run the simulation first to see correlation analysis.")
    else:
        run = st.session_state.last_run or {}
        available_tickers = result.results["Ticker"].tolist()
        st.subheader("Selected correlation model")
        st.caption(f"Correlation source: {st.session_state.last_corr_source}")
        if st.session_state.last_price_info:
            info = st.session_state.last_price_info
            st.caption(f"Yahoo adjusted close sample: {info['rows']} daily rows from {info['start']} to {info['end']} ({info['period']}).")

        c1, c2 = st.columns([1, 2])
        with c1:
            st.dataframe(display_correlation_summary(correlation_summary(result.cleaned_correlation)), use_container_width=True, hide_index=True)
        with c2:
            st.plotly_chart(px.imshow(result.cleaned_correlation, zmin=-1, zmax=1, color_continuous_scale="RdBu", title="Selected Correlation Matrix", text_auto=".2f"), use_container_width=True, key="correlation_analysis_heatmap")

        if st.session_state.last_regime_diagnostics is not None:
            with st.expander("Correlation model diagnostics"):
                shown = display_regime_diagnostics(st.session_state.last_regime_diagnostics)
                st.dataframe(shown if shown is not None else st.session_state.last_regime_diagnostics, use_container_width=True, hide_index=True)

        st.subheader("Fixed correlation sensitivity")
        st.write("This reruns the same market caps, IVs, target date, seed, and simulation count with every off-diagonal pairwise correlation forced to 0%, 5%, 10%, ... 95%.")
        fixed_sensitivity = st.session_state.last_fixed_sensitivity
        if fixed_sensitivity is not None and not fixed_sensitivity.empty:
            fixed_probability_pivot = fixed_sensitivity.pivot(index="Fixed correlation", columns="Ticker", values="Model probability")
            st.dataframe(fixed_probability_pivot.map(pct), use_container_width=True)

            sensitivity_ticker = st.selectbox("Ticker for correlation sensitivity", fixed_probability_pivot.columns.tolist(), index=0, key="correlation_sensitivity_ticker")
            sensitivity_slice = fixed_sensitivity[fixed_sensitivity["Ticker"] == sensitivity_ticker].sort_values("Fixed correlation")
            left, right = st.columns(2)
            with left:
                st.plotly_chart(px.line(sensitivity_slice, x="Fixed correlation", y="Model probability", markers=True, title=f"{sensitivity_ticker}: P(#1) vs fixed correlation"), use_container_width=True, key="correlation_fixed_probability_line")
            with right:
                st.plotly_chart(px.line(sensitivity_slice, x="Fixed correlation", y="Average rank", markers=True, title=f"{sensitivity_ticker}: average rank vs fixed correlation"), use_container_width=True, key="correlation_fixed_rank_line")
            with st.expander("Full fixed-correlation sensitivity table"):
                st.dataframe(display_fixed_sensitivity(fixed_sensitivity), use_container_width=True, hide_index=True)

        st.subheader("Correlation caveat")
        st.write("Correlation is an input assumption, not an observed future fact. The fixed-correlation table is meant to show whether the model probability is robust or fragile to the absolute correlation level. It does not solve tail dependence or crisis-correlation behavior yet.")

with diagnostics_tab:
    if result is None:
        st.warning("Run the simulation first to see diagnostics.")
    else:
        available_tickers = result.results["Ticker"].tolist()
        if selected_ticker_sidebar not in available_tickers:
            selected_ticker_sidebar = available_tickers[0]

        st.subheader("Terminal market-cap distribution percentiles")
        st.dataframe(display_market_cap_percentiles(market_cap_percentile_table(result)), use_container_width=True, hide_index=True)

        st.subheader("Rank probability matrix")
        st.dataframe(display_rank_probability_matrix(rank_probability_matrix(result)), use_container_width=True, hide_index=True)

        st.subheader(f"Simulation detail: {selected_ticker_sidebar}")
        st.dataframe(selected_ticker_summary(result, selected_ticker_sidebar), use_container_width=True, hide_index=True)

        st.subheader(f"Pairwise probability audit: {selected_ticker_sidebar}")
        st.dataframe(display_pairwise_probability_audit(pairwise_probability_audit(result, selected_ticker_sidebar, int((st.session_state.last_run or {}).get("days_to_target", days_to_target)))), use_container_width=True, hide_index=True)

        st.plotly_chart(px.box(terminal_cap_long(result, available_tickers), x="Ticker", y="Simulated market cap ($T)", points=False, title="Simulated Terminal Market-Cap Distributions by Ticker"), use_container_width=True, key="diagnostics_all_box_chart")

        chart_left, chart_right = st.columns(2)
        with chart_left:
            probability_chart = px.scatter(result.results, x="Polymarket YES price", y="Model probability", text="Ticker", title="Model Probability vs Polymarket Probability")
            probability_chart.add_shape(type="line", x0=0, y0=0, x1=1, y1=1, line={"dash": "dash", "color": "gray"})
            probability_chart.update_traces(textposition="top center")
            st.plotly_chart(probability_chart, use_container_width=True, key="diagnostics_probability_scatter")
        with chart_right:
            st.plotly_chart(px.bar(result.results.sort_values("Edge"), x="Ticker", y="Edge", title="Model Probability minus Polymarket Price", color="Edge", color_continuous_scale="RdYlGn"), use_container_width=True, key="diagnostics_edge_chart")

        heatmap_left, dist_right = st.columns(2)
        with heatmap_left:
            st.plotly_chart(px.imshow(result.cleaned_correlation, zmin=-1, zmax=1, color_continuous_scale="RdBu", title="Selected Correlation Matrix", text_auto=".2f"), use_container_width=True, key="diagnostics_corr_heatmap")
        with dist_right:
            rank_data = result.rank_distribution[result.rank_distribution["Ticker"] == selected_ticker_sidebar]
            st.plotly_chart(px.bar(rank_data, x="Rank", y="Probability", title=f"Rank Distribution: {selected_ticker_sidebar}"), use_container_width=True, key="diagnostics_rank_distribution")
        st.plotly_chart(terminal_distribution_chart(result, selected_ticker_sidebar), use_container_width=True, key="diagnostics_terminal_histogram")

with methodology_tab:
    st.subheader("Phase 1 model")
    st.code("MC_T = MC_0 * exp((-0.5 * sigma^2) * T + sigma * sqrt(T) * Z)")
    st.write("The target date determines the time horizon `T = days_to_target / 365`. For each simulation path, the app simulates one future market capitalization per company and ranks companies from largest to smallest.")

    st.subheader("Monte Carlo precision")
    st.write("The Results tab reports standard error from finite simulation count. This only measures simulation noise, not model uncertainty or data-input uncertainty.")

    st.subheader("Correlation analysis")
    st.write("The Correlation Analysis tab shows the selected correlation matrix and a direct fixed-correlation stress test. This isolates whether ranking probabilities are robust to the correlation level.")

    st.subheader("Vol-adjusted smooth correlation")
    st.write("This mode maps current average pair IV into the historical distribution of pair realized volatility. The resulting percentile becomes the blend weight between low-vol and high-vol historical pair correlations. No manual switch threshold is required for the blend weight.")
    st.code("w = percentile_rank(avg_current_IV_ij, historical_pair_realized_vol_ij)\nCorr_ij = (1 - w) * Corr_low_ij + w * Corr_high_ij")

    st.subheader("Pairwise probability audit")
    st.write("The pairwise audit computes P(selected company market cap > competitor market cap) analytically under the same lognormal inputs. It is diagnostic and does not replace the full joint Monte Carlo ranking probability.")

    st.subheader("Market cap source")
    st.write("Yahoo market cap mode uses yfinance fast_info.market_cap with a fallback to info['marketCap']. Manual market caps remain available for overrides.")

    st.subheader("IV source")
    st.write("Manual IV remains available. Yahoo option-chain mode selects the expiry closest to the target date, finds the strike nearest spot, and averages call/put implied volatility at that strike. This is a near-ATM IV estimate, not a full smile/surface calibration.")

    st.subheader("Correlation estimation")
    st.code("r_t = log(P_t / P_{t-1})\nCov_t = lambda * Cov_{t-1} + (1 - lambda) * r_t r_t'\nCorr_ij = Cov_ij / sqrt(Cov_ii * Cov_jj)")

    st.subheader("What is not modeled yet")
    st.write("Volatility skew/smile is not fully modeled yet. Future versions should ingest full option chains, clean bid/ask quotes, and calibrate a full IV surface or terminal risk-neutral distribution.")
