from __future__ import annotations

from datetime import datetime

import pandas as pd
import plotly.express as px
import streamlit as st

from model import default_company_inputs, default_correlation_matrix, run_probability_engine


st.set_page_config(page_title="LargestCompany", layout="wide")

st.title("LargestCompany")
st.caption("Phase 1: statistical probability engine for largest future market capitalization.")

st.warning(
    "Prototype status: all market cap, implied volatility, Polymarket price, and correlation "
    "inputs are manual placeholders until data connectors are added. Treat current values as "
    "editable assumptions, not live market data."
)

st.info(
    "This app does not predict stock prices. It translates current market caps, implied "
    "volatility, and correlation assumptions into fair ranking probabilities, then compares "
    "them with Polymarket YES prices."
)


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
    display["Mkt cap"] = display["Mkt cap"].map(lambda value: f"${value / 1e12:,.2f}T")
    for column in ["IV", "Poly price", "Model prob", "Edge", "Top 2", "Top 3"]:
        display[column] = display[column].map(lambda value: "" if value != value else f"{value:.2%}")
    display["Avg rank"] = display["Avg rank"].map(lambda value: f"{value:.2f}")
    return display[["Ticker", "Mkt cap", "IV", "Poly price", "Model prob", "Edge", "Avg rank", "Top 2", "Top 3"]]


def selected_ticker_summary(result, ticker: str) -> pd.DataFrame:
    caps = result.terminal_market_caps[ticker]
    ranks = result.ranks[ticker]
    return pd.DataFrame(
        [
            {"Metric": "Simulated market cap mean", "Value": f"${caps.mean() / 1e12:,.2f}T"},
            {"Metric": "Simulated market cap median", "Value": f"${caps.median() / 1e12:,.2f}T"},
            {"Metric": "5th percentile market cap", "Value": f"${caps.quantile(0.05) / 1e12:,.2f}T"},
            {"Metric": "95th percentile market cap", "Value": f"${caps.quantile(0.95) / 1e12:,.2f}T"},
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


with st.sidebar:
    st.header("Simulation controls")
    days_to_target = st.number_input("Days to target date", min_value=1, value=365, step=1)
    simulations = st.number_input(
        "Monte Carlo simulations",
        min_value=1_000,
        max_value=2_000_000,
        value=100_000,
        step=10_000,
    )
    seed = st.number_input("Random seed", min_value=0, value=42, step=1)
    selected_ticker = st.selectbox(
        "Selected ticker for diagnostics",
        st.session_state.company_inputs["Ticker"].astype(str).tolist(),
    )
    run_button = st.button("Run / refresh simulation", type="primary", use_container_width=True)


results_tab, inputs_tab, diagnostics_tab, methodology_tab = st.tabs(
    ["Results", "Inputs & Data", "Simulation Diagnostics", "Methodology"]
)


with inputs_tab:
    st.subheader("Data provenance")
    st.write(
        "Current version has no live data pipeline. The default rows are seed assumptions "
        "inside `model.py`, and every value should be edited or replaced by the user."
    )
    st.dataframe(
        pd.DataFrame(
            [
                {"Input": "Current market capitalization", "Current source": "Manual placeholder", "Future source": "Market data API or uploaded snapshot"},
                {"Input": "Annualized implied volatility", "Current source": "Manual placeholder", "Future source": "Option chain / IV surface feed"},
                {"Input": "Polymarket YES price", "Current source": "Manual placeholder", "Future source": "Polymarket market API"},
                {"Input": "Correlation matrix", "Current source": "Manual assumption", "Future source": "Historical return correlation estimator"},
            ]
        ),
        use_container_width=True,
        hide_index=True,
    )

    st.subheader("Company inputs")
    st.write("Edit the assumptions below, then click **Run / refresh simulation** in the sidebar.")
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

    st.subheader("Correlation matrix")
    st.write(
        "This matrix is currently an assumption, not a historical estimate. It should be symmetric, "
        "have 1.0 on the diagonal, and values between -1 and 1."
    )
    correlation_matrix = st.data_editor(
        current_corr,
        use_container_width=True,
        column_config={
            ticker: st.column_config.NumberColumn(min_value=-1.0, max_value=1.0, step=0.05)
            for ticker in tickers
        },
    )
    st.session_state.correlation_matrix = correlation_matrix


company_inputs = st.session_state.company_inputs
correlation_matrix = st.session_state.correlation_matrix

if run_button or st.session_state.last_result is None:
    with st.spinner("Running Monte Carlo simulation..."):
        try:
            st.session_state.last_result = run_probability_engine(
                company_inputs,
                correlation_matrix,
                days_to_target=int(days_to_target),
                simulations=int(simulations),
                seed=int(seed),
            )
            st.session_state.last_error = None
            st.session_state.last_run = {
                "time": datetime.now().strftime("%H:%M:%S"),
                "days_to_target": int(days_to_target),
                "simulations": int(simulations),
                "seed": int(seed),
            }
        except Exception as exc:
            st.session_state.last_result = None
            st.session_state.last_error = str(exc)
            st.session_state.last_run = None


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
            f" | {run.get('simulations', simulations):,} paths"
            f" | {run.get('days_to_target', days_to_target)} days"
            f" | seed {run.get('seed', seed)}"
            f" | last run {run.get('time', 'now')}"
        )

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

        st.subheader("Interpretation")
        best = result.most_undervalued
        worst = result.most_overvalued
        st.write(
            f"Under the current assumptions, **{best['Ticker']}** has the largest positive "
            f"model-vs-Polymarket gap. **{worst['Ticker']}** has the largest negative gap. "
            "This is a statistical relative-value comparison under the supplied inputs."
        )

with diagnostics_tab:
    if result is None:
        st.warning("Run the simulation first to see diagnostics.")
    else:
        available_tickers = result.results["Ticker"].tolist()
        if selected_ticker not in available_tickers:
            selected_ticker = available_tickers[0]

        st.subheader(f"Simulation detail: {selected_ticker}")
        st.dataframe(selected_ticker_summary(result, selected_ticker), use_container_width=True, hide_index=True)

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
            edge_chart = px.bar(
                result.results.sort_values("Edge"),
                x="Ticker",
                y="Edge",
                title="Model Probability minus Polymarket Price",
                color="Edge",
                color_continuous_scale="RdYlGn",
            )
            st.plotly_chart(edge_chart, use_container_width=True)

        heatmap_left, dist_right = st.columns(2)
        with heatmap_left:
            corr_chart = px.imshow(
                result.cleaned_correlation,
                zmin=-1,
                zmax=1,
                color_continuous_scale="RdBu",
                title="Correlation Assumption Matrix",
                text_auto=".2f",
            )
            st.plotly_chart(corr_chart, use_container_width=True)

        with dist_right:
            rank_data = result.rank_distribution[result.rank_distribution["Ticker"] == selected_ticker]
            rank_chart = px.bar(rank_data, x="Rank", y="Probability", title=f"Rank Distribution: {selected_ticker}")
            st.plotly_chart(rank_chart, use_container_width=True)

        cap_chart = px.histogram(
            result.terminal_market_caps,
            x=selected_ticker,
            nbins=80,
            title=f"Simulated Market Capitalization Distribution: {selected_ticker}",
        )
        cap_chart.update_layout(xaxis_title="Simulated market capitalization")
        st.plotly_chart(cap_chart, use_container_width=True)

with methodology_tab:
    st.subheader("Phase 1 model")
    st.code("MC_T = MC_0 * exp((-0.5 * sigma^2) * T + sigma * sqrt(T) * Z)")
    st.write(
        "For each simulation path, the app simulates one future market capitalization per company, "
        "ranks companies from largest to smallest, and stores the winner and full rank vector. "
        "The reported probabilities are Monte Carlo frequencies."
    )

    st.subheader("What is not modeled yet")
    st.write(
        "Volatility skew/smile is not used in this Phase 1 prototype. Each company currently has "
        "one annualized implied volatility input. A later version should replace that flat IV with "
        "an option-surface calibration module."
    )
    st.write(
        "Historical correlations are not computed yet. The current correlation matrix is a user-supplied "
        "model assumption. A later version should estimate correlations from historical log returns over "
        "an explicit lookback window and allow stress scenarios."
    )
