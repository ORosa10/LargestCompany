from __future__ import annotations

from datetime import date, datetime, timedelta
from math import erf, sqrt

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from correlations import (
    ewma_correlation,
    fetch_adjusted_close,
    rolling_correlation,
    smooth_vol_adjusted_correlation,
)
from iv_surfaces import apply_iv_estimates, estimate_atm_ivs
from market_data import apply_market_caps, fetch_market_caps
from model import (
    SHOCK_MODELS,
    SimulationResult,
    build_rank_distribution,
    cholesky_with_jitter,
    clean_correlation_matrix,
    default_company_inputs,
    default_correlation_matrix,
    forward_log_carry,
    rank_descending,
    standard_shocks,
    validate_company_inputs,
)


st.set_page_config(page_title="LargestCompany", layout="wide")

CORRELATION_METHODS = [
    "EWMA historical correlation",
    "Vol-adjusted smooth correlation",
    "Rolling historical correlation",
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


def normal_cdf(value: float) -> float:
    return 0.5 * (1.0 + erf(value / sqrt(2.0)))


def display_results(results: pd.DataFrame) -> pd.DataFrame:
    display = results.copy().rename(
        columns={
            "Current market cap": "Market cap",
            "Implied volatility": "IV",
            "Polymarket YES price": "Polymarket price",
            "Model probability": "Fair probability",
            "Average rank": "Avg rank",
            "Probability Top 2": "Top 2",
            "Probability Top 3": "Top 3",
        }
    )
    display["Market cap"] = display["Market cap"].map(dollars_trillions)
    for column in ["IV", "Polymarket price", "Fair probability", "Edge", "Top 2", "Top 3"]:
        display[column] = display[column].map(pct)
    display["Avg rank"] = display["Avg rank"].map(lambda value: f"{value:.2f}")
    return display[["Ticker", "Market cap", "IV", "Polymarket price", "Fair probability", "Edge", "Avg rank", "Top 2", "Top 3"]]


def display_base_inputs(inputs: pd.DataFrame) -> pd.DataFrame:
    display = inputs.copy()
    display["Current market cap"] = display["Current market cap"].map(dollars_trillions)
    display["Implied volatility"] = display["Implied volatility"].map(pct)
    display["Polymarket YES price"] = display["Polymarket YES price"].map(pct)
    return display


def display_market_caps(market_caps: pd.DataFrame | None) -> pd.DataFrame | None:
    if market_caps is None or market_caps.empty:
        return None
    display = market_caps.copy().rename(columns={"ticker": "Ticker", "yahoo_ticker": "Yahoo ticker", "market_cap": "Market cap", "source": "Source"})
    display["Market cap"] = display["Market cap"].map(dollars_trillions)
    return display[["Ticker", "Yahoo ticker", "Market cap", "Source"]]


def display_iv_estimates(iv_estimates: pd.DataFrame | None) -> pd.DataFrame | None:
    if iv_estimates is None or iv_estimates.empty:
        return None
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


def prepare_simulation_inputs(company_inputs: pd.DataFrame, market_cap_source: str, iv_source: str, target_date_value: date) -> tuple[pd.DataFrame, pd.DataFrame | None, pd.DataFrame | None, str, str]:
    tickers = [ticker for ticker in company_inputs["Ticker"].astype(str).str.strip().tolist() if ticker]
    simulation_inputs = company_inputs.copy()

    market_caps = None
    if market_cap_source == "Yahoo Finance current market cap":
        market_caps = load_yahoo_market_caps(tuple(tickers))
        simulation_inputs = apply_market_caps(simulation_inputs, market_caps)
        market_cap_label = "Yahoo Finance current market cap"
    else:
        market_cap_label = "Manual market cap inputs"

    iv_estimates = None
    if iv_source == "Yahoo option-chain near-ATM IV":
        iv_estimates = load_yahoo_atm_ivs(tuple(tickers), target_date_value.isoformat())
        simulation_inputs = apply_iv_estimates(simulation_inputs, iv_estimates)
        iv_label = "Yahoo option-chain near-ATM IV"
    else:
        iv_label = "Manual IV inputs"

    return simulation_inputs, market_caps, iv_estimates, market_cap_label, iv_label


def select_correlation_matrix(
    method: str,
    simulation_inputs: pd.DataFrame,
    manual_correlation_matrix: pd.DataFrame,
    price_history_period: str,
    ewma_lambda: float,
    rolling_lookback: int,
    smooth_low_quantile: float,
    smooth_high_quantile: float,
) -> tuple[pd.DataFrame, str, dict | None]:
    tickers = simulation_inputs["Ticker"].astype(str).tolist()
    if method == "Manual/default correlation matrix":
        return manual_correlation_matrix.reindex(index=tickers, columns=tickers).fillna(default_correlation_matrix(tickers)), "Manual/default correlation matrix", None

    prices = load_adjusted_close(tuple(tickers), price_history_period)
    price_info = {"rows": len(prices), "start": prices.index.min().date().isoformat(), "end": prices.index.max().date().isoformat(), "period": price_history_period}

    if method == "EWMA historical correlation":
        return ewma_correlation(prices, float(ewma_lambda)), f"EWMA historical correlation, lambda={ewma_lambda}, Yahoo Finance {price_history_period}", price_info
    if method == "Vol-adjusted smooth correlation":
        current_ivs = simulation_inputs.set_index("Ticker")["Implied volatility"].astype(float)
        corr, _ = smooth_vol_adjusted_correlation(prices, current_ivs, vol_window=63, low_quantile=float(smooth_low_quantile), high_quantile=float(smooth_high_quantile), min_observations=30)
        return corr, f"Vol-adjusted smooth correlation, low/high buckets={smooth_low_quantile:.0%}/{smooth_high_quantile:.0%}", price_info
    return rolling_correlation(prices, int(rolling_lookback)), f"Rolling historical correlation, {rolling_lookback} trading days, Yahoo Finance {price_history_period}", price_info


def run_engine(company_inputs: pd.DataFrame, correlation_matrix: pd.DataFrame, *, days_to_target: int, simulations: int, seed: int, shock_model: str) -> SimulationResult:
    clean_inputs = validate_company_inputs(company_inputs)
    tickers = clean_inputs["Ticker"].tolist()
    cleaned_corr, corr_warnings = clean_correlation_matrix(correlation_matrix, tickers)
    cholesky, chol_warnings = cholesky_with_jitter(cleaned_corr.to_numpy(dtype=float))

    rng = np.random.default_rng(seed)
    horizon_years = days_to_target / 365.0
    independent_shocks = standard_shocks(rng, simulations, len(tickers), shock_model)
    correlated_shocks = independent_shocks @ cholesky.T

    market_caps_0 = clean_inputs["Current market cap"].to_numpy(dtype=float)
    volatilities = clean_inputs["Implied volatility"].to_numpy(dtype=float)
    log_carry = forward_log_carry(clean_inputs, tickers)
    drift = log_carry - 0.5 * np.square(volatilities) * horizon_years
    diffusion = volatilities * np.sqrt(horizon_years) * correlated_shocks
    terminal_caps = market_caps_0 * np.exp(drift + diffusion)

    ranks_array = rank_descending(terminal_caps)
    yes_price = clean_inputs["Polymarket YES price"].to_numpy(dtype=float)
    model_probability = (ranks_array == 1).mean(axis=0)
    results = pd.DataFrame(
        {
            "Ticker": tickers,
            "Current market cap": market_caps_0,
            "Implied volatility": volatilities,
            "Polymarket YES price": yes_price,
            "Model probability": model_probability,
            "Edge": model_probability - yes_price,
            "Expected value": model_probability - yes_price,
            "ROI": np.nan,
            "Average rank": ranks_array.mean(axis=0),
            "Probability Top 2": (ranks_array <= 2).mean(axis=0),
            "Probability Top 3": (ranks_array <= 3).mean(axis=0),
        }
    ).sort_values("Edge", ascending=False, ignore_index=True)

    return SimulationResult(
        results=results,
        terminal_market_caps=pd.DataFrame(terminal_caps, columns=tickers),
        ranks=pd.DataFrame(ranks_array, columns=tickers),
        rank_distribution=build_rank_distribution(ranks_array, tickers),
        cleaned_correlation=cleaned_corr,
        warnings=tuple(corr_warnings + chol_warnings),
    )


def market_cap_percentile_table(result: SimulationResult) -> pd.DataFrame:
    rows = []
    for ticker in result.terminal_market_caps.columns:
        caps = result.terminal_market_caps[ticker]
        rows.append(
            {
                "Ticker": ticker,
                "Mean": caps.mean(),
                "P5": caps.quantile(0.05),
                "P25": caps.quantile(0.25),
                "P50": caps.quantile(0.50),
                "P75": caps.quantile(0.75),
                "P95": caps.quantile(0.95),
                "Average rank": result.results.set_index("Ticker").loc[ticker, "Average rank"],
            }
        )
    return pd.DataFrame(rows).sort_values("P50", ascending=False, ignore_index=True)


def display_percentiles(table: pd.DataFrame) -> pd.DataFrame:
    display = table.copy()
    for column in ["Mean", "P5", "P25", "P50", "P75", "P95"]:
        display[column] = display[column].map(dollars_trillions)
    display["Average rank"] = display["Average rank"].map(lambda value: f"{value:.2f}")
    return display


def rank_probability_matrix(result: SimulationResult) -> pd.DataFrame:
    rows = []
    for ticker in result.ranks.columns:
        row = {"Ticker": ticker}
        for rank in range(1, len(result.ranks.columns) + 1):
            row[f"Rank {rank}"] = (result.ranks[ticker] == rank).mean()
        rows.append(row)
    return pd.DataFrame(rows)


def display_rank_matrix(table: pd.DataFrame) -> pd.DataFrame:
    display = table.copy()
    for column in display.columns:
        if column != "Ticker":
            display[column] = display[column].map(pct)
    return display


def terminal_cap_long(result: SimulationResult, tickers: list[str]) -> pd.DataFrame:
    cap_long = result.terminal_market_caps[tickers].melt(var_name="Ticker", value_name="Market cap")
    cap_long["Market cap ($T)"] = cap_long["Market cap"] / 1e12
    return cap_long


def pairwise_probability_audit(result: SimulationResult, selected: str, competitor: str, days_to_target: int) -> pd.DataFrame:
    rows = []
    result_by_ticker = result.results.set_index("Ticker")
    selected_row = result_by_ticker.loc[selected]
    competitor_row = result_by_ticker.loc[competitor]
    years = max(days_to_target, 1) / 365.0
    selected_cap = float(selected_row["Current market cap"])
    competitor_cap = float(competitor_row["Current market cap"])
    selected_iv = float(selected_row["Implied volatility"])
    competitor_iv = float(competitor_row["Implied volatility"])
    rho = float(result.cleaned_correlation.loc[selected, competitor])
    relative_var = selected_iv**2 + competitor_iv**2 - 2.0 * rho * selected_iv * competitor_iv
    relative_vol = sqrt(max(relative_var, 0.0))
    log_gap_now = np.log(selected_cap / competitor_cap)
    mean_log_gap = log_gap_now - 0.5 * (selected_iv**2 - competitor_iv**2) * years
    horizon_vol = relative_vol * sqrt(years)
    z_score = np.inf if horizon_vol <= 1e-12 and mean_log_gap > 0 else mean_log_gap / horizon_vol
    rows.append(
        {
            "Selected": selected,
            "Competitor": competitor,
            "Selected cap": selected_cap,
            "Competitor cap": competitor_cap,
            "Market-cap gap": selected_cap - competitor_cap,
            "Correlation": rho,
            "Selected IV": selected_iv,
            "Competitor IV": competitor_iv,
            "Relative annual vol": relative_vol,
            "Horizon vol": horizon_vol,
            "Z-score": z_score,
            "P(selected > competitor)": normal_cdf(z_score),
        }
    )
    return pd.DataFrame(rows)


def display_pairwise(table: pd.DataFrame) -> pd.DataFrame:
    display = table.copy()
    for column in ["Selected cap", "Competitor cap", "Market-cap gap"]:
        display[column] = display[column].map(dollars_trillions)
    for column in ["Correlation", "Selected IV", "Competitor IV", "Relative annual vol", "Horizon vol", "P(selected > competitor)"]:
        display[column] = display[column].map(pct)
    display["Z-score"] = display["Z-score"].map(lambda value: f"{value:.2f}")
    return display


def model_comparison(simulation_inputs: pd.DataFrame, manual_corr: pd.DataFrame, days_to_target: int, simulations: int, seed: int, corr_methods: list[str], shock_models: list[str], price_period: str, ewma_lambda: float, rolling_lookback: int, smooth_low: float, smooth_high: float) -> pd.DataFrame:
    rows = []
    for corr_method in corr_methods:
        corr, corr_label, _ = select_correlation_matrix(corr_method, simulation_inputs, manual_corr, price_period, ewma_lambda, rolling_lookback, smooth_low, smooth_high)
        for shock_model in shock_models:
            result = run_engine(simulation_inputs, corr, days_to_target=days_to_target, simulations=simulations, seed=seed, shock_model=shock_model)
            for _, row in result.results.iterrows():
                rows.append(
                    {
                        "Correlation method": corr_method,
                        "Shock model": shock_model,
                        "Ticker": row["Ticker"],
                        "Model probability": row["Model probability"],
                        "Average rank": row["Average rank"],
                        "Edge": row["Edge"],
                    }
                )
    return pd.DataFrame(rows)


st.title("LargestCompany")
st.caption("Phase 1 probability cockpit for largest future market capitalization.")
st.info("Main view: baseline fair probabilities and model-comparison controls. Detailed correlation, IV, and return-shape diagnostics live in the separate pages on the left.")

if "company_inputs" not in st.session_state:
    st.session_state.company_inputs = default_company_inputs()
if "correlation_matrix" not in st.session_state:
    st.session_state.correlation_matrix = default_correlation_matrix(st.session_state.company_inputs["Ticker"].astype(str).tolist())
for key in ["last_result", "last_error", "last_run", "last_simulation_inputs", "last_market_caps", "last_iv_estimates", "last_sources", "last_corr_label", "last_price_info", "last_comparison"]:
    if key not in st.session_state:
        st.session_state[key] = None


today = date.today()
with st.sidebar:
    st.header("Baseline model")
    target_date = st.date_input("Target date / maturity", value=today + timedelta(days=365), min_value=today + timedelta(days=1))
    days_to_target = max((target_date - today).days, 1)
    st.caption(f"Horizon: {days_to_target} days ({days_to_target / 365:.2f} years)")
    simulations = st.number_input("Monte Carlo simulations", min_value=1_000, max_value=2_000_000, value=100_000, step=10_000)
    seed = st.number_input("Random seed", min_value=0, value=42, step=1)

    market_cap_source = st.selectbox("Market cap source", ["Yahoo Finance current market cap", "Manual market cap inputs"], index=0)
    iv_source = st.selectbox("IV source", ["Manual IV inputs", "Yahoo option-chain near-ATM IV"], index=0)
    st.caption("Polymarket YES prices are manual inputs for now.")

    correlation_method = st.selectbox("Correlation method", CORRELATION_METHODS, index=0)
    shock_model = st.selectbox("Shock distribution", SHOCK_MODELS, index=0)
    price_history_period = st.selectbox("Yahoo price history", ["1y", "3y", "5y", "10y"], index=2)
    ewma_lambda = st.selectbox("EWMA lambda", [0.94, 0.97], index=1)
    rolling_lookback = st.selectbox("Rolling lookback days", [63, 126, 252, 504, 756], index=2)
    smooth_low_quantile = st.selectbox("Smooth low-vol bucket", [0.30, 0.40, 0.50], index=1, format_func=lambda value: f"{value:.0%}")
    smooth_high_quantile = st.selectbox("Smooth high-vol bucket", [0.50, 0.60, 0.70], index=1, format_func=lambda value: f"{value:.0%}")

    st.header("Comparison")
    comparison_sims = st.number_input("Comparison simulations", min_value=1_000, max_value=300_000, value=30_000, step=5_000)
    comparison_corr_methods = st.multiselect("Compare correlations", CORRELATION_METHODS, default=["EWMA historical correlation", "Vol-adjusted smooth correlation"])
    comparison_shock_models = st.multiselect("Compare shock models", SHOCK_MODELS, default=["Normal shocks", "Student-t df=10"])

    run_button = st.button("Run / refresh", type="primary", use_container_width=True)

inputs_tab, overview_tab, ticker_tab, pair_tab, comparison_tab, data_tab, methodology_tab = st.tabs(["Inputs", "Overview", "Ticker Detail", "Pair Detail", "Model Comparison", "Data Used", "Methodology"])

with inputs_tab:
    st.subheader("Editable company inputs")
    st.write("Market caps can be overridden by Yahoo. IV and Polymarket prices remain manual unless Yahoo near-ATM IV is explicitly selected.")
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
    current_corr = st.session_state.correlation_matrix.reindex(index=tickers, columns=tickers).fillna(default_correlation_matrix(tickers))
    st.subheader("Manual/default correlation matrix")
    st.write("Used only when Manual/default correlation matrix is selected.")
    st.session_state.correlation_matrix = st.data_editor(current_corr, use_container_width=True)

company_inputs = st.session_state.company_inputs
manual_correlation_matrix = st.session_state.correlation_matrix

if run_button or st.session_state.last_result is None:
    with st.spinner("Running baseline and comparison simulations..."):
        try:
            simulation_inputs, market_caps, iv_estimates, market_cap_label, iv_label = prepare_simulation_inputs(company_inputs, market_cap_source, iv_source, target_date)
            corr, corr_label, price_info = select_correlation_matrix(correlation_method, simulation_inputs, manual_correlation_matrix, price_history_period, float(ewma_lambda), int(rolling_lookback), float(smooth_low_quantile), float(smooth_high_quantile))
            result = run_engine(simulation_inputs, corr, days_to_target=int(days_to_target), simulations=int(simulations), seed=int(seed), shock_model=shock_model)
            comparison = model_comparison(simulation_inputs, manual_correlation_matrix, int(days_to_target), int(comparison_sims), int(seed), comparison_corr_methods, comparison_shock_models, price_history_period, float(ewma_lambda), int(rolling_lookback), float(smooth_low_quantile), float(smooth_high_quantile)) if comparison_corr_methods and comparison_shock_models else pd.DataFrame()

            st.session_state.last_result = result
            st.session_state.last_error = None
            st.session_state.last_simulation_inputs = simulation_inputs
            st.session_state.last_market_caps = market_caps
            st.session_state.last_iv_estimates = iv_estimates
            st.session_state.last_sources = {"Market cap": market_cap_label, "IV": iv_label, "Polymarket": "Manual inputs", "Shock distribution": shock_model}
            st.session_state.last_corr_label = corr_label
            st.session_state.last_price_info = price_info
            st.session_state.last_comparison = comparison
            st.session_state.last_run = {"time": datetime.now().strftime("%H:%M:%S"), "target_date": target_date.isoformat(), "days_to_target": int(days_to_target), "simulations": int(simulations), "comparison_sims": int(comparison_sims), "seed": int(seed)}
        except Exception as exc:
            st.session_state.last_result = None
            st.session_state.last_error = str(exc)

result = st.session_state.last_result

for tab in [overview_tab, ticker_tab, pair_tab, comparison_tab, data_tab]:
    with tab:
        if st.session_state.last_error:
            st.error(st.session_state.last_error)

if result is not None:
    run = st.session_state.last_run or {}
    sources = st.session_state.last_sources or {}
    tickers = result.results["Ticker"].astype(str).tolist()

    with overview_tab:
        st.success(f"Baseline run complete | target {run.get('target_date')} | {run.get('days_to_target')} days | {run.get('simulations'):,} paths | seed {run.get('seed')} | last run {run.get('time')}")
        a1, a2, a3, a4 = st.columns(4)
        a1.metric("Market caps", sources.get("Market cap"))
        a2.metric("IV", sources.get("IV"))
        a3.metric("Correlation", correlation_method)
        a4.metric("Shock model", sources.get("Shock distribution"))
        st.caption(f"Correlation detail: {st.session_state.last_corr_label}")

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Highest fair probability", result.results.sort_values("Model probability", ascending=False).iloc[0]["Ticker"])
        c2.metric("Largest positive gap", result.most_undervalued["Ticker"], f'{result.most_undervalued["Edge"]:.2%}')
        c3.metric("Largest negative gap", result.most_overvalued["Ticker"], f'{result.most_overvalued["Edge"]:.2%}')
        c4.metric("Probability check", f'{result.results["Model probability"].sum():.2%}')

        st.subheader("Fair ranking probabilities")
        st.dataframe(display_results(result.results), use_container_width=True, hide_index=True)

        left, right = st.columns(2)
        with left:
            chart = px.scatter(result.results, x="Polymarket YES price", y="Model probability", text="Ticker", title="Fair probability vs Polymarket price")
            chart.add_shape(type="line", x0=0, y0=0, x1=1, y1=1, line={"dash": "dash", "color": "gray"})
            chart.update_traces(textposition="top center")
            st.plotly_chart(chart, use_container_width=True, key="overview_prob_scatter")
        with right:
            st.plotly_chart(px.bar(result.results.sort_values("Edge"), x="Ticker", y="Edge", color="Edge", color_continuous_scale="RdYlGn", title="Fair probability minus Polymarket price"), use_container_width=True, key="overview_edge_bar")

    with ticker_tab:
        selected_ticker = st.selectbox("Ticker", tickers, index=tickers.index(result.results.iloc[0]["Ticker"]), key="main_ticker_detail")
        row = result.results.set_index("Ticker").loc[selected_ticker]
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Fair probability", f'{row["Model probability"]:.2%}')
        c2.metric("Polymarket price", f'{row["Polymarket YES price"]:.2%}')
        c3.metric("Gap", f'{row["Edge"]:.2%}')
        c4.metric("Top 2", f'{row["Probability Top 2"]:.2%}')
        c5.metric("Average rank", f'{row["Average rank"]:.2f}')

        left, right = st.columns(2)
        with left:
            caps_t = result.terminal_market_caps[selected_ticker] / 1e12
            hist = px.histogram(caps_t, nbins=80, title=f"Terminal market-cap distribution: {selected_ticker}")
            hist.update_layout(xaxis_title="Market cap ($T)", yaxis_title="Simulation count")
            st.plotly_chart(hist, use_container_width=True, key="ticker_hist")
        with right:
            rank_data = result.rank_distribution[result.rank_distribution["Ticker"] == selected_ticker]
            st.plotly_chart(px.bar(rank_data, x="Rank", y="Probability", title=f"Rank distribution: {selected_ticker}"), use_container_width=True, key="ticker_rank")

        st.subheader("Market-cap distribution percentiles")
        percentiles = market_cap_percentile_table(result)
        st.dataframe(display_percentiles(percentiles[percentiles["Ticker"] == selected_ticker]), use_container_width=True, hide_index=True)

    with pair_tab:
        default_selected = "NVDA" if "NVDA" in tickers else tickers[0]
        default_competitor = "GOOGL" if "GOOGL" in tickers else tickers[min(1, len(tickers) - 1)]
        p1, p2 = st.columns(2)
        with p1:
            selected = st.selectbox("Selected ticker", tickers, index=tickers.index(default_selected), key="pair_selected")
        with p2:
            competitor = st.selectbox("Competitor", tickers, index=tickers.index(default_competitor), key="pair_competitor")
        if selected == competitor:
            st.warning("Choose two different tickers.")
        else:
            audit = pairwise_probability_audit(result, selected, competitor, int(run.get("days_to_target", days_to_target)))
            st.dataframe(display_pairwise(audit), use_container_width=True, hide_index=True)
            compare_caps = terminal_cap_long(result, [selected, competitor])
            st.plotly_chart(px.box(compare_caps, x="Ticker", y="Market cap ($T)", points=False, title="Pair terminal market-cap distributions"), use_container_width=True, key="pair_box")

    with comparison_tab:
        comparison = st.session_state.last_comparison
        if comparison is None or comparison.empty:
            st.info("Select at least one correlation method and one shock model in the sidebar, then run refresh.")
        else:
            st.subheader("Model probability by approach")
            selected_compare_ticker = st.selectbox("Ticker for approach comparison", tickers, index=tickers.index("NVDA") if "NVDA" in tickers else 0, key="compare_ticker")
            ticker_comparison = comparison[comparison["Ticker"] == selected_compare_ticker].copy()
            pivot = ticker_comparison.pivot(index="Correlation method", columns="Shock model", values="Model probability")
            st.dataframe(pivot.map(pct), use_container_width=True)
            st.plotly_chart(px.bar(ticker_comparison, x="Correlation method", y="Model probability", color="Shock model", barmode="group", title=f"{selected_compare_ticker}: probability by correlation and shock model"), use_container_width=True, key="compare_bar")

            st.subheader("All tickers comparison")
            display = comparison.copy()
            display["Model probability"] = display["Model probability"].map(pct)
            display["Edge"] = display["Edge"].map(pct)
            display["Average rank"] = display["Average rank"].map(lambda value: f"{value:.2f}")
            st.dataframe(display, use_container_width=True, hide_index=True)

    with data_tab:
        st.subheader("Base inputs actually used")
        st.dataframe(display_base_inputs(st.session_state.last_simulation_inputs), use_container_width=True, hide_index=True)
        market_caps_display = display_market_caps(st.session_state.last_market_caps)
        if market_caps_display is not None:
            st.subheader("Yahoo market caps used")
            st.dataframe(market_caps_display, use_container_width=True, hide_index=True)
        iv_display = display_iv_estimates(st.session_state.last_iv_estimates)
        if iv_display is not None:
            st.subheader("Yahoo IV estimates used")
            st.dataframe(iv_display, use_container_width=True, hide_index=True)
        if st.session_state.last_price_info:
            info = st.session_state.last_price_info
            st.caption(f"Yahoo adjusted close sample for correlation: {info['rows']} rows from {info['start']} to {info['end']} ({info['period']}).")
        st.subheader("Selected correlation matrix")
        st.plotly_chart(px.imshow(result.cleaned_correlation, zmin=-1, zmax=1, color_continuous_scale="RdBu", text_auto=".2f"), use_container_width=True, key="data_corr_heatmap")
        st.subheader("Rank probability matrix")
        st.dataframe(display_rank_matrix(rank_probability_matrix(result)), use_container_width=True, hide_index=True)
else:
    with overview_tab:
        st.warning("No result yet. Check inputs and click Run / refresh.")

with methodology_tab:
    st.subheader("Baseline model")
    st.code("MC_T = MC_0 * exp((-0.5 * sigma^2) * T + sigma * sqrt(T) * Z)")
    st.write("Market caps and correlations are linked to stock market data when Yahoo sources are selected. IV and Polymarket prices remain manual unless Yahoo near-ATM IV is explicitly selected.")
    st.subheader("Shock distribution")
    st.write("Normal shocks are the baseline. Independent Student-t shocks (df=10, df=6) are a fat-tail marginal sensitivity. The Student-t copula (df=5) shares one mixing variable across tickers, adding joint tail dependence so names crash together; IV still sets the volatility scale.")
    st.subheader("No strategy layer yet")
    st.write("This phase reports statistical fair probabilities and model-vs-market gaps only. Hedging, ROI strategy, and payoff construction are intentionally left for later phases.")
