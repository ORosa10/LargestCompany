"""Live payoff-by-terminal-price diagnostics for the Phase 5 manual portfolio."""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st


def fixed_price_bin_profile(
    terminal_prices: np.ndarray,
    payoffs: np.ndarray,
    *,
    bin_width: float = 5.0,
) -> pd.DataFrame:
    prices = np.asarray(terminal_prices, dtype=float)
    values = np.asarray(payoffs, dtype=float)
    core_low = np.floor(np.quantile(prices, 0.01) / bin_width) * bin_width
    core_high = np.ceil(np.quantile(prices, 0.99) / bin_width) * bin_width
    finite_edges = np.arange(core_low, core_high + bin_width * 0.5, bin_width)
    edges = np.concatenate(([-np.inf], finite_edges, [np.inf]))
    frame = pd.DataFrame({"Terminal price": prices, "Payoff": values})
    frame["Price bin"] = pd.cut(frame["Terminal price"], edges, include_lowest=True)
    rows = []
    for interval, group in frame.groupby("Price bin", observed=True):
        bin_payoffs = group["Payoff"].to_numpy(dtype=float)
        if not np.isfinite(interval.left):
            label = f"<{interval.right:.0f}%"
        elif not np.isfinite(interval.right):
            label = f">={interval.left:.0f}%"
        else:
            label = f"{interval.left:.0f}-{interval.right:.0f}%"
        mean = float(bin_payoffs.mean())
        sd = float(bin_payoffs.std(ddof=0))
        rows.append(
            {
                "Price bin": label,
                "Scenario probability": len(group) / len(frame),
                "Expected payoff": mean,
                "Payoff SD": sd,
                "Mean minus 1 SD": mean - sd,
                "Payoff P5": float(np.quantile(bin_payoffs, 0.05)),
                "Payoff P1": float(np.quantile(bin_payoffs, 0.01)),
            }
        )
    return pd.DataFrame(rows)


def live_manual_profile_figure(
    baseline_profile: pd.DataFrame,
    manual_profile: pd.DataFrame,
) -> go.Figure:
    labels = manual_profile["Price bin"]
    colors = np.where(manual_profile["Expected payoff"] >= 0, "#16a34a", "#dc2626")
    probability_text = manual_profile["Scenario probability"].map(lambda value: f"{value:.1%}")
    figure = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.07,
        row_heights=[0.74, 0.26],
    )
    figure.add_trace(
        go.Scatter(
            x=labels,
            y=baseline_profile["Expected payoff"],
            name="Polymarket-only mean",
            mode="lines",
            line=dict(color="#94a3b8", dash="dash", width=2),
        ),
        row=1,
        col=1,
    )
    figure.add_trace(
        go.Bar(
            x=labels,
            y=manual_profile["Expected payoff"],
            name="Manual portfolio mean",
            marker_color=colors,
        ),
        row=1,
        col=1,
    )
    figure.add_trace(
        go.Scatter(
            x=labels,
            y=manual_profile["Mean minus 1 SD"],
            name="Mean - 1 SD",
            mode="lines+markers",
            line=dict(color="#7c3aed", dash="dash"),
        ),
        row=1,
        col=1,
    )
    figure.add_trace(
        go.Scatter(
            x=labels,
            y=manual_profile["Payoff P5"],
            name="P5 payoff",
            mode="lines+markers",
            line=dict(color="#f59e0b", dash="dash"),
        ),
        row=1,
        col=1,
    )
    figure.add_trace(
        go.Scatter(
            x=labels,
            y=manual_profile["Payoff P1"],
            name="P1 payoff",
            mode="lines+markers",
            line=dict(color="#dc2626", dash="dot"),
        ),
        row=1,
        col=1,
    )
    figure.add_trace(
        go.Bar(
            x=labels,
            y=manual_profile["Scenario probability"],
            text=probability_text,
            textposition="outside",
            name="Scenario probability",
            marker_color="#60a5fa",
        ),
        row=2,
        col=1,
    )
    figure.add_hline(y=0, line_dash="dash", line_color="black", row=1, col=1)
    figure.update_yaxes(title_text="Payoff", row=1, col=1)
    figure.update_yaxes(title_text="Probability", tickformat=".1%", row=2, col=1)
    figure.update_xaxes(
        title_text="Terminal stock price / current price",
        tickangle=-45,
        row=2,
        col=1,
    )
    figure.update_layout(
        title="Live manual payoff profile and scenario probability",
        height=780,
        barmode="overlay",
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        margin=dict(l=50, r=30, t=100, b=100),
    )
    return figure


def render_live_manual_profile(
    terminal_prices: np.ndarray,
    baseline_payoff: np.ndarray,
    manual_payoff: np.ndarray,
    *,
    bin_width: float = 5.0,
) -> None:
    st.subheader("Live payoff profile by terminal price")
    st.caption(
        "Bars show conditional mean payoff. Lines show downside dispersion inside each price bin. "
        "Probability bars below use the exact same bins."
    )
    baseline_profile = fixed_price_bin_profile(
        terminal_prices, baseline_payoff, bin_width=bin_width
    )
    manual_profile = fixed_price_bin_profile(
        terminal_prices, manual_payoff, bin_width=bin_width
    )
    st.plotly_chart(
        live_manual_profile_figure(baseline_profile, manual_profile),
        use_container_width=True,
        key="phase5_live_manual_profile",
    )
    with st.expander("Show live price-bin values"):
        display = manual_profile.copy()
        display["Scenario probability"] = display["Scenario probability"].map(
            lambda value: f"{value:.2%}"
        )
        st.dataframe(display, use_container_width=True, hide_index=True)
