from __future__ import annotations

import plotly.express as px
import streamlit as st

from model import default_company_inputs, default_correlation_matrix, run_probability_engine


st.set_page_config(page_title="LargestCompany", layout="wide")

st.title("LargestCompany")
st.caption("Phase 1: probability engine for largest future market capitalization.")


def format_results_for_display(results):
    display = results.copy()
    display["Current market cap"] = display["Current market cap"].map(
        lambda value: f"${value / 1e12:,.2f}T"
    )

    percent_columns = [
        "Implied volatility",
        "Polymarket YES price",
        "Model probability",
        "Edge",
        "Expected value",
        "ROI",
        "Probability Top 2",
        "Probability Top 3",
    ]
    for column in percent_columns:
        display[column] = display[column].map(
            lambda value: "" if value != value else f"{value:.2%}"
        )

    display["Average rank"] = display["Average rank"].map(lambda value: f"{value:.2f}")
    return display


if "company_inputs" not in st.session_state:
    st.session_state.company_inputs = default_company_inputs()

if "correlation_matrix" not in st.session_state:
    tickers = st.session_state.company_inputs["Ticker"].tolist()
    st.session_state.correlation_matrix = default_correlation_matrix(tickers)


with st.sidebar:
    st.header("Simulation")
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
        "Selected ticker",
        st.session_state.company_inputs["Ticker"].astype(str).tolist(),
    )
    run_button = st.button("Run simulation", type="primary", use_container_width=True)


st.subheader("Company Inputs")
company_inputs = st.data_editor(
    st.session_state.company_inputs,
    num_rows="dynamic",
    use_container_width=True,
    column_config={
        "Ticker": st.column_config.TextColumn(required=True),
        "Current market cap": st.column_config.NumberColumn(
            min_value=1.0,
            step=10_000_000_000.0,
        ),
        "Implied volatility": st.column_config.NumberColumn(
            min_value=0.0001,
            max_value=5.0,
            step=0.01,
        ),
        "Polymarket YES price": st.column_config.NumberColumn(
            min_value=0.0,
            max_value=1.0,
            step=0.01,
        ),
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

st.subheader("Correlation Matrix")
correlation_matrix = st.data_editor(
    current_corr,
    use_container_width=True,
    column_config={
        ticker: st.column_config.NumberColumn(min_value=-1.0, max_value=1.0, step=0.05)
        for ticker in tickers
    },
)
st.session_state.correlation_matrix = correlation_matrix


if run_button or "last_result" not in st.session_state:
    try:
        st.session_state.last_result = run_probability_engine(
            company_inputs,
            correlation_matrix,
            days_to_target=int(days_to_target),
            simulations=int(simulations),
            seed=int(seed),
        )
        st.session_state.last_error = None
    except Exception as exc:
        st.session_state.last_result = None
        st.session_state.last_error = str(exc)


if st.session_state.get("last_error"):
    st.error(st.session_state.last_error)

result = st.session_state.get("last_result")
if result is not None:
    available_tickers = result.results["Ticker"].tolist()
    if selected_ticker not in available_tickers:
        selected_ticker = available_tickers[0]

    for warning in result.warnings:
        st.warning(warning)

    left, right = st.columns(2)
    with left:
        st.metric(
            "Most undervalued",
            result.most_undervalued["Ticker"],
            f'{result.most_undervalued["Edge"]:.2%} edge',
        )
    with right:
        st.metric(
            "Most overvalued",
            result.most_overvalued["Ticker"],
            f'{result.most_overvalued["Edge"]:.2%} edge',
        )

    st.subheader("Results")
    st.dataframe(format_results_for_display(result.results), use_container_width=True)

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
        probability_chart.add_shape(
            type="line",
            x0=0,
            y0=0,
            x1=1,
            y1=1,
            line={"dash": "dash", "color": "gray"},
        )
        probability_chart.update_traces(textposition="top center")
        st.plotly_chart(probability_chart, use_container_width=True)

    with chart_right:
        edge_chart = px.bar(
            result.results.sort_values("Edge"),
            x="Ticker",
            y="Edge",
            title="Edge by Ticker",
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
            title="Correlation Matrix",
            text_auto=".2f",
        )
        st.plotly_chart(corr_chart, use_container_width=True)

    with dist_right:
        rank_data = result.rank_distribution[result.rank_distribution["Ticker"] == selected_ticker]
        rank_chart = px.bar(
            rank_data,
            x="Rank",
            y="Probability",
            title=f"Rank Distribution: {selected_ticker}",
        )
        st.plotly_chart(rank_chart, use_container_width=True)

    cap_chart = px.histogram(
        result.terminal_market_caps,
        x=selected_ticker,
        nbins=80,
        title=f"Simulated Market Capitalization Distribution: {selected_ticker}",
    )
    cap_chart.update_layout(xaxis_title="Simulated market capitalization")
    st.plotly_chart(cap_chart, use_container_width=True)
