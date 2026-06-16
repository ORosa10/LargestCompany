from __future__ import annotations

from datetime import date, datetime, timedelta

import pandas as pd
import plotly.express as px
import streamlit as st

from correlations import ewma_correlation, fetch_adjusted_close, rolling_correlation
from iv_surfaces import apply_iv_estimates, estimate_atm_ivs
from model import default_company_inputs, default_correlation_matrix, run_probability_engine


st.set_page_config(page_title="LargestCompany", layout="wide")

st.title("LargestCompany")
st.caption("Phase 1: statistical probability engine for largest future market capitalization.")

st.warning(
    "Prototype status: market cap and Polymarket price inputs are still manual placeholders. "
    "Correlations can be estimated from Yahoo historical prices. IV can be manual or estimated "
    "from Yahoo option-chain near-ATM implied vols."
)

st.info(
    "This app does not predict stock prices. It translates current market caps, implied "
    "volatility, target-date horizon, and correlation assumptions into fair ranking probabilities, "
    "then compares them with Polymarket YES prices."
)


@st.cache_data(show_spinner=False, ttl=60 * 60)
def load_adjusted_close(tickers: tuple[str, ...], period: str) -> pd.DataFrame:
    return fetch_adjusted_close(list(tickers), period=period)


@st.cache_data(show_spinner=False, ttl=30 * 60)
def load_yahoo_atm_ivs(tickers: tuple[str, ...], target_date_iso: str) -> pd.DataFrame:
    return estimate_atm_ivs(list(tickers), date.fromisoformat(target_date_iso))


def dollars_trillions(value: float) -> str:
    return f"${value / 1e12:,.2f}T"


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
        display[column] = display[column].map(lambda value: "" if value != value else f"{value:.2%}")
    display["Avg rank"] = display["Avg rank"].map(lambda value: f"{value:.2f}")
    return display[["Ticker", "Mkt cap", "IV", "Poly price", "Model prob", "Edge", "Avg rank", "Top 2", "Top 3"]]


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
        display[column] = display[column].map(lambda value: f"${value:,.2f}")
    for column in ["ATM IV", "Call IV", "Put IV"]:
        display[column] = display[column].map(lambda value: "" if pd.isna(value) else f"{value:.2%}")
    return display[["Ticker", "Yahoo ticker", "Target date", "Option expiry used", "Spot", "ATM strike", "ATM IV", "Call IV", "Put IV"]]


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
    rank_lookup = result.results.set_index("Ticker")["Average rank"]
    table["Average rank"] = table["Ticker"].map(rank_lookup)
    return table.sort_values("P50", ascending=False, ignore_index=True)


def display_market_cap_percentiles(percentiles: pd.DataFrame) -> pd.DataFrame:
    display = percentiles.copy()
    for column in ["Mean", "Std dev", "P1", "P5", "P25", "P50", "P75", "P95", "P99"]:
        display[column] = display[column].map(dollars_trillions)
    display["Average rank"] = display["Average rank"].map(lambda value: f"{value:.2f}")
    return display


def selected_ticker_summary(result, ticker: str) -> pd.DataFrame:
    caps = result.terminal_market_caps[ticker]
    ranks = result.ranks[ticker]
    return pd.DataFrame(
        [
            {"Metric": "Simulated market cap mean", "Value": dollars_trillions(caps.mean())},
            {"Metric": "Simulated market cap median", "Value": dollars_trillions(caps.median())},
            {"Metric": "1st percentile market cap", "Value": dollars_trillions(caps.quantile(0.01))},
            {"Metric": "5th percentile market cap", "Value": dollars_trillions(caps.quantile(0.05))},
            {"Metric": "95th percentile market cap", "Value": dollars_trillions(caps.quantile(0.95))},
            {"Metric": "99th percentile market cap", "Value": dollars_trillions(caps.quantile(0.99))},
            {"Metric": "Average simulated rank", "Value": f"{ranks.mean():.2f}"},
            {"Metric": "Median simulated rank", "Value": f"{ranks.median():.0f}"},
        ]
    )


if "company_inputs" not in st.session_state:
    st.session_state.company_inputs = default_company_inputs()

if "correlation_matrix" not in st.session_state:
    tickers = st.session_state.company_inputs["Ticker"].tolist()
    st.session_state.correlation_matrix = default_correlation_matrix(tickers)

if "last_result" not in st.session_state:
    st.session_state.last_result = None
    st.session_state.last_error = None
    st.session_state.last_run = None
    st.session_state.last_corr_source = None
    st.session_state.last_price_info = None
    st.session_state.last_iv_source = None
    st.session_state.last_iv_estimates = None


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

    st.header("IV source")
    iv_source = st.selectbox(
        "Implied volatility source",
        ["Manual IV inputs", "Yahoo option-chain near-ATM IV"],
        index=0,
    )

    st.header("Correlation source")
    correlation_source = st.selectbox(
        "Correlation method",
        ["EWMA historical correlation", "Rolling historical correlation", "Manual correlation matrix"],
        index=0,
    )
    price_history_period = st.selectbox("Yahoo Finance price history", ["2y", "5y", "10y"], index=1)
    ewma_lambda = st.selectbox("EWMA lambda", [0.94, 0.97], index=1)
    rolling_lookback = st.selectbox("Rolling lookback days", [63, 126, 252, 504, 756], index=2)

    selected_ticker = st.selectbox("Selected ticker for diagnostics", st.session_state.company_inputs["Ticker"].astype(str).tolist())
    run_button = st.button("Run / refresh simulation", type="primary", use_container_width=True)


results_tab, inputs_tab, diagnostics_tab, methodology_tab = st.tabs(["Results", "Inputs & Data", "Simulation Diagnostics", "Methodology"])

with inputs_tab:
    st.subheader("Data provenance")
    st.dataframe(
        pd.DataFrame(
            [
                {"Input": "Current market capitalization", "Current source": "Manual placeholder", "Future source": "Market data API or uploaded snapshot"},
                {"Input": "Annualized implied volatility", "Current source": "Manual input or Yahoo option-chain near-ATM IV", "Future source": "Robust IV surface provider"},
                {"Input": "Polymarket YES price", "Current source": "Manual placeholder", "Future source": "Polymarket market API or manual override"},
                {"Input": "Correlation matrix", "Current source": "Yahoo Finance historical adjusted close prices or manual input", "Future source": "Configurable institutional data provider"},
                {"Input": "Target date / maturity", "Current source": "User-selected date", "Future source": "Parsed from prediction-market event rules"},
            ]
        ),
        use_container_width=True,
        hide_index=True,
    )

    st.subheader("Company inputs")
    st.write("Manual IV values are used only when **Manual IV inputs** is selected in the sidebar.")
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

    tickers = company_inputs["Ticker"].astype(str).str.strip().tolist()
    tickers = [ticker for ticker in tickers if ticker]
    if tickers:
        current_corr = st.session_state.correlation_matrix.reindex(index=tickers, columns=tickers)
        fallback_corr = default_correlation_matrix(tickers)
        current_corr = current_corr.fillna(fallback_corr)
    else:
        current_corr = st.session_state.correlation_matrix

    st.subheader("Manual correlation matrix")
    st.write("Used only when **Manual correlation matrix** is selected in the sidebar.")
    correlation_matrix = st.data_editor(
        current_corr,
        use_container_width=True,
        column_config={ticker: st.column_config.NumberColumn(min_value=-1.0, max_value=1.0, step=0.05) for ticker in tickers},
    )
    st.session_state.correlation_matrix = correlation_matrix


company_inputs = st.session_state.company_inputs
manual_correlation_matrix = st.session_state.correlation_matrix

if run_button or st.session_state.last_result is None:
    with st.spinner("Running Monte Carlo simulation..."):
        try:
            clean_tickers = company_inputs["Ticker"].astype(str).str.strip().tolist()
            clean_tickers = [ticker for ticker in clean_tickers if ticker]

            iv_estimates = None
            simulation_inputs = company_inputs.copy()
            if iv_source == "Yahoo option-chain near-ATM IV":
                iv_estimates = load_yahoo_atm_ivs(tuple(clean_tickers), target_date.isoformat())
                simulation_inputs = apply_iv_estimates(simulation_inputs, iv_estimates)
                iv_source_label = "Yahoo Finance option-chain near-ATM IV; expiry closest to target date"
            else:
                iv_source_label = "Manual IV inputs"

            price_info = None
            if correlation_source == "Manual correlation matrix":
                selected_correlation_matrix = manual_correlation_matrix
                corr_source_label = "Manual correlation matrix"
            else:
                prices = load_adjusted_close(tuple(clean_tickers), price_history_period)
                price_info = {
                    "rows": len(prices),
                    "start": prices.index.min().date().isoformat(),
                    "end": prices.index.max().date().isoformat(),
                    "period": price_history_period,
                }
                if correlation_source == "EWMA historical correlation":
                    selected_correlation_matrix = ewma_correlation(prices, float(ewma_lambda))
                    corr_source_label = f"EWMA historical correlation, lambda={ewma_lambda}, Yahoo Finance {price_history_period}"
                else:
                    selected_correlation_matrix = rolling_correlation(prices, int(rolling_lookback))
                    corr_source_label = f"Rolling historical correlation, {rolling_lookback} trading days, Yahoo Finance {price_history_period}"

            st.session_state.last_result = run_probability_engine(
                simulation_inputs,
                selected_correlation_matrix,
                days_to_target=int(days_to_target),
                simulations=int(simulations),
                seed=int(seed),
            )
            st.session_state.last_error = None
            st.session_state.last_corr_source = corr_source_label
            st.session_state.last_price_info = price_info
            st.session_state.last_iv_source = iv_source_label
            st.session_state.last_iv_estimates = iv_estimates
            st.session_state.last_run = {
                "time": datetime.now().strftime("%H:%M:%S"),
                "target_date": target_date.isoformat(),
                "days_to_target": int(days_to_target),
                "horizon_years": horizon_years,
                "simulations": int(simulations),
                "seed": int(seed),
            }
        except Exception as exc:
            st.session_state.last_result = None
            st.session_state.last_error = str(exc)
            st.session_state.last_run = None
            st.session_state.last_corr_source = None
            st.session_state.last_price_info = None
            st.session_state.last_iv_source = None
            st.session_state.last_iv_estimates = None


result = st.session_state.last_result

with results_tab:
    if st.session_state.last_error:
        st.error(st.session_state.last_error)
    elif result is None:
        st.warning("No simulation result yet. Check inputs and click Run / refresh simulation.")
    else:
        run = st.session_state.last_run or {}
        st.success(
            "Simulation completed"
            f" | target {run.get('target_date', target_date.isoformat())}"
            f" | {run.get('days_to_target', days_to_target)} days"
            f" | {run.get('horizon_years', horizon_years):.2f} years"
            f" | {run.get('simulations', simulations):,} paths"
            f" | seed {run.get('seed', seed)}"
            f" | last run {run.get('time', 'now')}"
        )
        st.caption(f"IV source: {st.session_state.last_iv_source}")
        st.caption(f"Correlation source: {st.session_state.last_corr_source}")
        if st.session_state.last_price_info:
            info = st.session_state.last_price_info
            st.caption(f"Yahoo adjusted close sample: {info['rows']} daily rows from {info['start']} to {info['end']} ({info['period']}).")

        if st.session_state.last_iv_estimates is not None:
            with st.expander("Yahoo option-chain IV estimates used"):
                st.dataframe(display_iv_estimates(st.session_state.last_iv_estimates), use_container_width=True, hide_index=True)

        for warning in result.warnings:
            st.warning(warning)

        probability_sum = result.results["Model probability"].sum()
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Highest model probability", result.results.sort_values("Model probability", ascending=False).iloc[0]["Ticker"])
        c2.metric("Largest positive edge", result.most_undervalued["Ticker"], f'{result.most_undervalued["Edge"]:.2%}')
        c3.metric("Largest negative edge", result.most_overvalued["Ticker"], f'{result.most_overvalued["Edge"]:.2%}')
        c4.metric("Probability check", f"{probability_sum:.2%}")

        st.subheader("Statistical ranking probabilities")
        st.dataframe(display_results(result.results), use_container_width=True, hide_index=True)

        best = result.most_undervalued
        worst = result.most_overvalued
        st.subheader("Interpretation")
        st.write(
            f"Under the current assumptions and target date **{run.get('target_date', target_date.isoformat())}**, "
            f"**{best['Ticker']}** has the largest positive model-vs-Polymarket gap. "
            f"**{worst['Ticker']}** has the largest negative gap. This is a statistical relative-value comparison under the supplied inputs."
        )

with diagnostics_tab:
    if result is None:
        st.warning("Run the simulation first to see diagnostics.")
    else:
        available_tickers = result.results["Ticker"].tolist()
        if selected_ticker not in available_tickers:
            selected_ticker = available_tickers[0]

        percentiles = market_cap_percentile_table(result)
        st.subheader("Terminal market-cap distribution percentiles")
        st.dataframe(display_market_cap_percentiles(percentiles), use_container_width=True, hide_index=True)

        st.subheader(f"Simulation detail: {selected_ticker}")
        st.dataframe(selected_ticker_summary(result, selected_ticker), use_container_width=True, hide_index=True)

        cap_long = result.terminal_market_caps.melt(var_name="Ticker", value_name="Simulated market cap")
        cap_long["Simulated market cap ($T)"] = cap_long["Simulated market cap"] / 1e12
        box_chart = px.box(cap_long, x="Ticker", y="Simulated market cap ($T)", points=False, title="Simulated Terminal Market-Cap Distributions by Ticker")
        st.plotly_chart(box_chart, use_container_width=True)

        chart_left, chart_right = st.columns(2)
        with chart_left:
            probability_chart = px.scatter(
                result.results,
                x="Polymarket YES price",
                y="Model probability",
                text="Ticker",
                title="Model Probability vs Polymarket Probability",
                range_x=[0, max(0.01, result.results["Polymarket YES price"].max() * 1.15)],
                range_y=[0, max(0.01, result.results["Model probability"].max() * 1.15)],
            )
            probability_chart.add_shape(type="line", x0=0, y0=0, x1=1, y1=1, line={"dash": "dash", "color": "gray"})
            probability_chart.update_traces(textposition="top center")
            st.plotly_chart(probability_chart, use_container_width=True)

        with chart_right:
            edge_chart = px.bar(result.results.sort_values("Edge"), x="Ticker", y="Edge", title="Model Probability minus Polymarket Price", color="Edge", color_continuous_scale="RdYlGn")
            st.plotly_chart(edge_chart, use_container_width=True)

        heatmap_left, dist_right = st.columns(2)
        with heatmap_left:
            corr_chart = px.imshow(result.cleaned_correlation, zmin=-1, zmax=1, color_continuous_scale="RdBu", title="Selected Correlation Matrix", text_auto=".2f")
            st.plotly_chart(corr_chart, use_container_width=True)

        with dist_right:
            rank_data = result.rank_distribution[result.rank_distribution["Ticker"] == selected_ticker]
            rank_chart = px.bar(rank_data, x="Rank", y="Probability", title=f"Rank Distribution: {selected_ticker}")
            st.plotly_chart(rank_chart, use_container_width=True)

        cap_chart = px.histogram(result.terminal_market_caps, x=selected_ticker, nbins=80, title=f"Simulated Market Capitalization Distribution: {selected_ticker}")
        cap_chart.update_layout(xaxis_title="Simulated market capitalization")
        st.plotly_chart(cap_chart, use_container_width=True)

with methodology_tab:
    st.subheader("Phase 1 model")
    st.code("MC_T = MC_0 * exp((-0.5 * sigma^2) * T + sigma * sqrt(T) * Z)")
    st.write(
        "The target date determines the time horizon `T = days_to_target / 365`. For each simulation path, the app simulates one future market capitalization per company, ranks companies from largest to smallest, and stores the winner and full rank vector."
    )

    st.subheader("IV source")
    st.write(
        "Manual IV remains available. The Yahoo option-chain mode fetches option chains with yfinance, selects the expiry closest to the target date, finds the strike nearest spot, and averages call/put implied volatility at that strike. This is a near-ATM IV estimate, not a full smile/surface calibration."
    )

    st.subheader("Correlation estimation")
    st.write(
        "EWMA is the default method. The app downloads adjusted close prices from Yahoo Finance via yfinance, calculates daily log returns, demeans returns, estimates EWMA covariance, and converts covariance to correlation. Rolling correlation uses Pearson correlation of log returns over the selected trailing lookback window."
    )
    st.code("r_t = log(P_t / P_{t-1})\nCov_t = lambda * Cov_{t-1} + (1 - lambda) * r_t r_t'\nCorr_ij = Cov_ij / sqrt(Cov_ii * Cov_jj)")

    st.subheader("Distribution statistics")
    st.write("The diagnostics tab reports terminal market-cap percentiles for every company: P1, P5, P25, P50, P75, P95, and P99.")

    st.subheader("What is not modeled yet")
    st.write(
        "Volatility skew/smile is not fully modeled yet. Yahoo option-chain mode gives a first live near-ATM IV estimate, but future versions should ingest full option chains, clean bid/ask quotes, and calibrate a full IV surface or terminal risk-neutral distribution."
    )
