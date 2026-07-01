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
    selected_wins: np.ndarray | None = None,
    bin_width: float = 5.0,
) -> pd.DataFrame:
    prices = np.asarray(terminal_prices, dtype=float)
    values = np.asarray(payoffs, dtype=float)
    wins = np.zeros(len(prices), dtype=bool) if selected_wins is None else np.asarray(selected_wins, dtype=bool)
    core_low = np.floor(np.quantile(prices, 0.01) / bin_width) * bin_width
    core_high = np.ceil(np.quantile(prices, 0.99) / bin_width) * bin_width
    finite_edges = np.arange(core_low, core_high + bin_width * 0.5, bin_width)
    edges = np.concatenate(([-np.inf], finite_edges, [np.inf]))
    frame = pd.DataFrame({"Terminal price": prices, "Payoff": values, "Selected wins": wins})
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
        rows.append({
            "Price bin": label,
            "Scenario probability": len(group) / len(frame),
            "Conditional P(#1)": float(group["Selected wins"].mean()),
            "Expected payoff": mean,
            "Payoff SD": sd,
            "Mean minus 1 SD": mean - sd,
            "Payoff P5": float(np.quantile(bin_payoffs, 0.05)),
            "Payoff P1": float(np.quantile(bin_payoffs, 0.01)),
        })
    return pd.DataFrame(rows)


def live_manual_profile_figure(baseline_profile: pd.DataFrame, manual_profile: pd.DataFrame) -> go.Figure:
    labels = manual_profile["Price bin"]
    colors = np.where(manual_profile["Expected payoff"] >= 0, "#16a34a", "#dc2626")
    probability_text = manual_profile["Scenario probability"].map(lambda value: f"{value:.1%}")
    figure = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.07,
        row_heights=[0.72, 0.28],
        specs=[[{}], [{"secondary_y": True}]],
    )
    figure.add_trace(go.Scatter(x=labels, y=baseline_profile["Expected payoff"], name="Polymarket-only mean", mode="lines", line=dict(color="#94a3b8", dash="dash", width=2)), row=1, col=1)
    figure.add_trace(go.Bar(x=labels, y=manual_profile["Expected payoff"], name="Manual portfolio mean", marker_color=colors), row=1, col=1)
    figure.add_trace(go.Scatter(x=labels, y=manual_profile["Mean minus 1 SD"], name="Mean - 1 SD", mode="lines+markers", line=dict(color="#7c3aed", dash="dash")), row=1, col=1)
    figure.add_trace(go.Scatter(x=labels, y=manual_profile["Payoff P5"], name="P5 payoff", mode="lines+markers", line=dict(color="#f59e0b", dash="dash")), row=1, col=1)
    figure.add_trace(go.Scatter(x=labels, y=manual_profile["Payoff P1"], name="P1 payoff", mode="lines+markers", line=dict(color="#dc2626", dash="dot")), row=1, col=1)
    figure.add_trace(go.Bar(x=labels, y=manual_profile["Scenario probability"], text=probability_text, textposition="outside", name="Scenario probability", marker_color="#60a5fa"), row=2, col=1, secondary_y=False)
    figure.add_trace(go.Scatter(x=labels, y=manual_profile["Conditional P(#1)"], name="Conditional P(#1)", mode="lines+markers", line=dict(color="#dc2626", width=3)), row=2, col=1, secondary_y=True)
    figure.add_trace(go.Scatter(x=labels, y=np.full(len(labels), 0.01), name="1% threshold", mode="lines", line=dict(color="#64748b", dash="dot")), row=2, col=1, secondary_y=True)
    figure.add_trace(go.Scatter(x=labels, y=np.full(len(labels), 0.05), name="5% threshold", mode="lines", line=dict(color="#64748b", dash="dash")), row=2, col=1, secondary_y=True)
    figure.add_hline(y=0, line_dash="dash", line_color="black", row=1, col=1)
    figure.update_yaxes(title_text="Payoff", row=1, col=1)
    figure.update_yaxes(title_text="Bin probability", tickformat=".1%", row=2, col=1, secondary_y=False)
    figure.update_yaxes(title_text="Conditional P(#1)", tickformat=".0%", range=[0, 1], row=2, col=1, secondary_y=True)
    figure.update_xaxes(title_text="Terminal stock price / current price", tickangle=-45, row=2, col=1)
    figure.update_layout(title="Live manual payoff profile, bin probability, and conditional win probability", height=820, barmode="overlay", legend=dict(orientation="h", yanchor="bottom", y=1.02), margin=dict(l=50, r=50, t=110, b=100))
    return figure


def render_live_manual_profile(
    terminal_prices: np.ndarray,
    baseline_payoff: np.ndarray,
    manual_payoff: np.ndarray,
    *,
    polymarket_side: str,
    bin_width: float = 5.0,
) -> None:
    st.subheader("Live payoff profile by terminal price")
    st.caption("Bars show conditional mean payoff. P1/P5 show stress payoff inside each price bin. The lower panel shows both the probability of reaching the bin and P(selected ticker finishes #1 | price bin).")
    base = np.asarray(baseline_payoff, dtype=float)
    selected_wins = base > 0 if str(polymarket_side).upper() == "YES" else base < 0
    baseline_profile = fixed_price_bin_profile(terminal_prices, base, selected_wins=selected_wins, bin_width=bin_width)
    manual_profile = fixed_price_bin_profile(terminal_prices, manual_payoff, selected_wins=selected_wins, bin_width=bin_width)
    st.plotly_chart(live_manual_profile_figure(baseline_profile, manual_profile), use_container_width=True, key="phase5_live_manual_profile")
    with st.expander("Show live price-bin values"):
        display = manual_profile.copy()
        display["Scenario probability"] = display["Scenario probability"].map(lambda value: f"{value:.2%}")
        display["Conditional P(#1)"] = display["Conditional P(#1)"].map(lambda value: f"{value:.2%}")
        st.dataframe(display, use_container_width=True, hide_index=True)
