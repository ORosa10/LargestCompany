from __future__ import annotations

from statistics import NormalDist

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from correlations import fetch_adjusted_close
from model import default_company_inputs


st.set_page_config(page_title="Return Diagnostics", layout="wide")
st.title("Return Diagnostics")
st.caption("Historical return diagnostics for normality, skewness, and fat tails. Diagnostic only; the main probability engine still defaults to option-IV lognormal shocks.")

NORMAL = NormalDist()
NORMAL_TAIL_2SIGMA = 0.0455
NORMAL_TAIL_3SIGMA = 0.0027


@st.cache_data(show_spinner=False, ttl=60 * 60)
def load_prices(tickers: tuple[str, ...], period: str) -> pd.DataFrame:
    return fetch_adjusted_close(list(tickers), period=period)


def pct(value: float) -> str:
    return "" if pd.isna(value) else f"{value:.2%}"


def calculate_log_returns(prices: pd.DataFrame) -> pd.DataFrame:
    clean_prices = prices.sort_index().dropna(how="all")
    return np.log(clean_prices / clean_prices.shift(1)).dropna(how="all")


def student_t_df_from_kurtosis(excess_kurtosis: float) -> float | None:
    if excess_kurtosis <= 0:
        return None
    return max(4.5, min(60.0, 4.0 + 6.0 / excess_kurtosis))


def distribution_recommendation(skew: float, excess_kurtosis: float, jb_pvalue: float, tail_3sigma: float) -> tuple[str, str, str]:
    df = student_t_df_from_kurtosis(excess_kurtosis)
    df_text = "" if df is None else f"df approx {df:.1f}"

    if abs(skew) >= 0.75 and (excess_kurtosis >= 1.0 or tail_3sigma > 2.0 * NORMAL_TAIL_3SIGMA):
        return "Historical residual bootstrap", "Captures both skew and fat tails from empirical standardized residuals.", df_text
    if excess_kurtosis >= 2.0 or tail_3sigma > 2.0 * NORMAL_TAIL_3SIGMA:
        return "Student-t fat-tail shocks", "Main issue is excess kurtosis / tail frequency; keep IV as scale and replace normal Z with standardized Student-t.", df_text
    if jb_pvalue < 0.05 or excess_kurtosis >= 0.75:
        return "Mild Student-t or empirical residual check", "Normality is questionable, but tail evidence is moderate. Use as sensitivity, not default.", df_text
    return "Normal shocks acceptable", "Historical sample does not show strong evidence against normal shocks for this diagnostic window.", df_text


def return_diagnostics(returns: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for ticker in returns.columns:
        series = returns[ticker].dropna()
        n = len(series)
        if n < 20:
            continue
        mean_daily = float(series.mean())
        std_daily = float(series.std(ddof=1))
        centered = series - mean_daily
        skew = float(series.skew())
        excess_kurtosis = float(series.kurtosis())
        jarque_bera = n / 6.0 * (skew**2 + 0.25 * excess_kurtosis**2)
        jb_pvalue = float(np.exp(-0.5 * jarque_bera))
        tail_2sigma = float((centered.abs() > 2 * std_daily).mean())
        tail_3sigma = float((centered.abs() > 3 * std_daily).mean())
        recommendation, rationale, df_text = distribution_recommendation(skew, excess_kurtosis, jb_pvalue, tail_3sigma)
        rows.append(
            {
                "Ticker": ticker,
                "Observations": n,
                "Mean daily return": mean_daily,
                "Annualized realized vol": std_daily * np.sqrt(252),
                "Skewness": skew,
                "Excess kurtosis": excess_kurtosis,
                "Jarque-Bera stat": jarque_bera,
                "JB p-value": jb_pvalue,
                "Normality flag": "Reject normality" if jb_pvalue < 0.05 else "Not rejected",
                "Empirical >2 sigma": tail_2sigma,
                "Empirical >3 sigma": tail_3sigma,
                "Suggested shock model": recommendation,
                "Suggested t df": df_text,
                "Reason": rationale,
                "P1 daily return": float(series.quantile(0.01)),
                "P5 daily return": float(series.quantile(0.05)),
                "P50 daily return": float(series.quantile(0.50)),
                "P95 daily return": float(series.quantile(0.95)),
                "P99 daily return": float(series.quantile(0.99)),
            }
        )
    return pd.DataFrame(rows)


def display_diagnostics(table: pd.DataFrame) -> pd.DataFrame:
    display = table.copy()
    percent_columns = [
        "Mean daily return",
        "Annualized realized vol",
        "JB p-value",
        "Empirical >2 sigma",
        "Empirical >3 sigma",
        "P1 daily return",
        "P5 daily return",
        "P50 daily return",
        "P95 daily return",
        "P99 daily return",
    ]
    for column in percent_columns:
        display[column] = display[column].map(pct)
    for column in ["Skewness", "Excess kurtosis", "Jarque-Bera stat"]:
        display[column] = display[column].map(lambda value: f"{value:.2f}")
    return display


def qq_data(series: pd.Series) -> pd.DataFrame:
    clean = series.dropna().sort_values().to_numpy(dtype=float)
    n = len(clean)
    if n == 0:
        return pd.DataFrame(columns=["Normal theoretical quantile", "Historical return quantile"])
    probabilities = (np.arange(1, n + 1) - 0.5) / n
    theoretical = np.array([NORMAL.inv_cdf(float(p)) for p in probabilities])
    standardized = (clean - clean.mean()) / clean.std(ddof=1)
    return pd.DataFrame({"Normal theoretical quantile": theoretical, "Historical return quantile": standardized})


def empirical_tail_table(summary: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, row in summary.iterrows():
        rows.append(
            {
                "Ticker": row["Ticker"],
                "P(|return| > 2 sigma) empirical": row["Empirical >2 sigma"],
                "P(|return| > 3 sigma) empirical": row["Empirical >3 sigma"],
                "Normal benchmark >2 sigma": NORMAL_TAIL_2SIGMA,
                "Normal benchmark >3 sigma": NORMAL_TAIL_3SIGMA,
            }
        )
    return pd.DataFrame(rows)


def display_tail_table(table: pd.DataFrame) -> pd.DataFrame:
    display = table.copy()
    for column in display.columns:
        if column != "Ticker":
            display[column] = display[column].map(pct)
    return display


def recommendation_table(summary: pd.DataFrame) -> pd.DataFrame:
    columns = ["Ticker", "Suggested shock model", "Suggested t df", "Reason", "Skewness", "Excess kurtosis", "Empirical >3 sigma"]
    table = summary[columns].copy()
    table["Skewness"] = table["Skewness"].map(lambda value: f"{value:.2f}")
    table["Excess kurtosis"] = table["Excess kurtosis"].map(lambda value: f"{value:.2f}")
    table["Empirical >3 sigma"] = table["Empirical >3 sigma"].map(pct)
    return table


company_inputs = default_company_inputs()
default_tickers = company_inputs["Ticker"].astype(str).tolist()

with st.sidebar:
    st.header("Return sample")
    selected_tickers = st.multiselect("Tickers", default_tickers, default=default_tickers)
    period = st.selectbox("Yahoo Finance history", ["1y", "3y", "5y", "10y"], index=2)
    selected_ticker = st.selectbox("Ticker detail", selected_tickers, index=0 if selected_tickers else None)

if not selected_tickers:
    st.warning("Select at least one ticker.")
    st.stop()

with st.spinner("Loading Yahoo adjusted close prices and calculating log returns..."):
    prices = load_prices(tuple(selected_tickers), period)
    returns = calculate_log_returns(prices)

st.caption(f"Sample: {len(returns)} daily log-return observations from {returns.index.min().date()} to {returns.index.max().date()} ({period}).")

summary = return_diagnostics(returns)
tails = empirical_tail_table(summary)

st.subheader("Suggested shock model by ticker")
st.write("This is a modelling suggestion, not a replacement for the option-IV baseline. IV would remain the volatility scale; only the standardized shock shape would change.")
st.dataframe(recommendation_table(summary), use_container_width=True, hide_index=True)

st.subheader("Normality and fat-tail diagnostics")
st.write("Excess kurtosis above 0 means fatter tails than a normal distribution. Jarque-Bera is a quick normality diagnostic based on skewness and excess kurtosis.")
st.dataframe(display_diagnostics(summary), use_container_width=True, hide_index=True)

st.subheader("Empirical tail frequency")
st.write("This compares realized historical tail events with a normal benchmark. If empirical >3 sigma events are materially above 0.27%, the series has fatter tails than normal in this sample.")
st.dataframe(display_tail_table(tails), use_container_width=True, hide_index=True)

if selected_ticker in returns.columns:
    series = returns[selected_ticker].dropna()
    selected_summary = summary.set_index("Ticker").loc[selected_ticker]
    c1, c2, c3 = st.columns(3)
    c1.metric("Suggested shock model", selected_summary["Suggested shock model"])
    c2.metric("Skewness", f"{selected_summary['Skewness']:.2f}")
    c3.metric("Excess kurtosis", f"{selected_summary['Excess kurtosis']:.2f}")

    left, right = st.columns(2)
    with left:
        hist = px.histogram(series, nbins=80, title=f"Daily log-return distribution: {selected_ticker}")
        hist.update_layout(xaxis_title="Daily log return", yaxis_title="Observation count")
        st.plotly_chart(hist, use_container_width=True, key="return_histogram")
    with right:
        qq = qq_data(series)
        fig = px.scatter(qq, x="Normal theoretical quantile", y="Historical return quantile", title=f"QQ plot vs normal: {selected_ticker}")
        lo = min(qq["Normal theoretical quantile"].min(), qq["Historical return quantile"].min())
        hi = max(qq["Normal theoretical quantile"].max(), qq["Historical return quantile"].max())
        fig.add_trace(go.Scatter(x=[lo, hi], y=[lo, hi], mode="lines", name="Normal line", line={"dash": "dash", "color": "gray"}))
        st.plotly_chart(fig, use_container_width=True, key="qq_plot")

    st.subheader(f"Return percentiles: {selected_ticker}")
    percentile_table = pd.DataFrame(
        [
            {"Percentile": "P1", "Daily return": series.quantile(0.01)},
            {"Percentile": "P5", "Daily return": series.quantile(0.05)},
            {"Percentile": "P25", "Daily return": series.quantile(0.25)},
            {"Percentile": "P50", "Daily return": series.quantile(0.50)},
            {"Percentile": "P75", "Daily return": series.quantile(0.75)},
            {"Percentile": "P95", "Daily return": series.quantile(0.95)},
            {"Percentile": "P99", "Daily return": series.quantile(0.99)},
        ]
    )
    percentile_table["Daily return"] = percentile_table["Daily return"].map(pct)
    st.dataframe(percentile_table, use_container_width=True, hide_index=True)

st.subheader("How this should feed the model later")
st.write("For now this page is diagnostic only. A later optional engine mode can keep IV as the volatility scale but replace normal shocks with standardized Student-t shocks or historical standardized residual bootstrap shocks. That preserves the option-implied volatility input while allowing fatter tails and skew where history strongly suggests it.")
