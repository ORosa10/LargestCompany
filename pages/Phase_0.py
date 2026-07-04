from __future__ import annotations

from statistics import NormalDist

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from correlations import fetch_adjusted_close
from iv_surface_model import SURFACE_AS_OF, SURFACE_EXPIRY, default_surface_nodes
from model import default_company_inputs


st.set_page_config(page_title="Phase 0 - Input Diagnostics", layout="wide")
st.title("Phase 0: Input Diagnostics")
st.caption("Prepare and audit the historical return sample and the option-implied volatility surfaces before running Phase 1.")

NORMAL = NormalDist()


@st.cache_data(show_spinner=False, ttl=60 * 60)
def load_prices(tickers: tuple[str, ...], period: str) -> pd.DataFrame:
    return fetch_adjusted_close(list(tickers), period=period)


def pct(value: float) -> str:
    return "" if pd.isna(value) else f"{value:.2%}"


def calculate_log_returns(prices: pd.DataFrame) -> pd.DataFrame:
    return np.log(prices.sort_index() / prices.sort_index().shift(1)).dropna(how="all")


def return_summary(returns: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for ticker in returns.columns:
        series = returns[ticker].dropna()
        if len(series) < 20:
            continue
        centered = series - series.mean()
        daily_std = float(series.std(ddof=1))
        skew = float(series.skew())
        kurtosis = float(series.kurtosis())
        jb = len(series) / 6.0 * (skew**2 + 0.25 * kurtosis**2)
        rows.append(
            {
                "Ticker": ticker,
                "Observations": len(series),
                "Annualized realized vol": daily_std * np.sqrt(252),
                "Skewness": skew,
                "Excess kurtosis": kurtosis,
                "JB p-value": float(np.exp(-0.5 * jb)),
                "Empirical >3 sigma": float((centered.abs() > 3 * daily_std).mean()),
                "P1": float(series.quantile(0.01)),
                "P5": float(series.quantile(0.05)),
                "P50": float(series.quantile(0.50)),
                "P95": float(series.quantile(0.95)),
                "P99": float(series.quantile(0.99)),
            }
        )
    return pd.DataFrame(rows)


def qq_data(series: pd.Series) -> pd.DataFrame:
    clean = series.dropna().sort_values().to_numpy(dtype=float)
    probabilities = (np.arange(1, len(clean) + 1) - 0.5) / len(clean)
    theoretical = np.array([NORMAL.inv_cdf(float(value)) for value in probabilities])
    standardized = (clean - clean.mean()) / clean.std(ddof=1)
    return pd.DataFrame({"Normal quantile": theoretical, "Historical standardized return": standardized})


returns_tab, surface_tab = st.tabs(["Return Diagnostics", "IV Surface Calibration"])

with returns_tab:
    defaults = default_company_inputs()["Ticker"].astype(str).tolist()
    c1, c2 = st.columns([3, 1])
    with c1:
        selected_tickers = st.multiselect("Tickers", defaults, default=defaults, key="phase0_return_tickers")
    with c2:
        period = st.selectbox("Yahoo history", ["1y", "3y", "5y", "10y"], index=2, key="phase0_return_period")

    if not selected_tickers:
        st.info("Select at least one ticker.")
    else:
        with st.spinner("Loading Yahoo adjusted-close history..."):
            prices = load_prices(tuple(selected_tickers), period)
            returns = calculate_log_returns(prices)
        summary = return_summary(returns)
        st.caption(f"{len(returns):,} daily log-return observations, {returns.index.min().date()} to {returns.index.max().date()}.")

        display = summary.copy()
        for column in ["Annualized realized vol", "JB p-value", "Empirical >3 sigma", "P1", "P5", "P50", "P95", "P99"]:
            display[column] = display[column].map(pct)
        for column in ["Skewness", "Excess kurtosis"]:
            display[column] = display[column].map(lambda value: f"{value:.2f}")
        st.dataframe(display, use_container_width=True, hide_index=True)

        detail_ticker = st.selectbox("Ticker detail", summary["Ticker"].tolist(), key="phase0_return_detail")
        series = returns[detail_ticker].dropna()
        left, right = st.columns(2)
        with left:
            st.plotly_chart(
                px.histogram(series, nbins=80, title=f"Daily log-return distribution: {detail_ticker}"),
                use_container_width=True,
                key="phase0_return_hist",
            )
        with right:
            qq = qq_data(series)
            chart = px.scatter(qq, x="Normal quantile", y="Historical standardized return", title=f"QQ plot vs normal: {detail_ticker}")
            low = min(qq.min())
            high = max(qq.max())
            chart.add_trace(go.Scatter(x=[low, high], y=[low, high], mode="lines", name="Normal line", line={"dash": "dash", "color": "gray"}))
            st.plotly_chart(chart, use_container_width=True, key="phase0_return_qq")

with surface_tab:
    st.caption(f"Manual calibration snapshot: {SURFACE_AS_OF}; option expiry: {SURFACE_EXPIRY}.")
    nodes = default_surface_nodes()
    selected_surface_tickers = st.multiselect(
        "Surface tickers",
        nodes["Ticker"].drop_duplicates().tolist(),
        default=nodes["Ticker"].drop_duplicates().tolist(),
        key="phase0_surface_tickers",
    )
    filtered = nodes[nodes["Ticker"].isin(selected_surface_tickers)].copy()
    if filtered.empty:
        st.info("Select at least one surface ticker.")
    else:
        chart = px.line(
            filtered,
            x="Strike",
            y="IV",
            color="Ticker",
            symbol="Wing",
            markers=True,
            facet_col="Ticker",
            facet_col_wrap=1,
            title="Calibrated implied-volatility smiles",
        )
        chart.update_yaxes(tickformat=".0%")
        chart.update_layout(height=300 * len(selected_surface_tickers), showlegend=True)
        st.plotly_chart(chart, use_container_width=True, key="phase0_surface_chart")

        display_nodes = filtered.copy()
        display_nodes["Observed spot"] = display_nodes["Observed spot"].map(lambda value: f"${value:,.2f}")
        display_nodes["Moneyness"] = display_nodes["Moneyness"].map(pct)
        display_nodes["IV"] = display_nodes["IV"].map(pct)
        st.dataframe(display_nodes, use_container_width=True, hide_index=True)

        st.info("Phase 0 calibrates and audits inputs. Phase 1 consumes these surfaces and runs the joint ranking simulation once.")
