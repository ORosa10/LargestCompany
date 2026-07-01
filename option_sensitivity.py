"""Boundary-confidence and option-quantity sensitivity for Phase 5."""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from interactive_portfolio import strike_at_confidence
from manual_profile_chart import render_live_manual_profile
from option_construction import black_scholes_price, option_payoff


DEFAULT_CONFIDENCES = list(range(40, 100, 5)) + [99]
DEFAULT_QUANTITIES = [round(value, 2) for value in np.arange(0.0, 2.01, 0.10)]


def _unit_leg_payoff(
    terminal_prices: np.ndarray,
    *,
    option_type: str,
    position: str,
    strike: float,
    volatility: float,
    time_to_expiry: float,
    risk_free_rate: float,
    include_premiums: bool,
    contract_multiplier: float,
    normalized_spot: float,
) -> np.ndarray:
    premium = black_scholes_price(
        spot=normalized_spot,
        strike=strike,
        time_to_expiry=time_to_expiry,
        volatility=volatility,
        risk_free_rate=risk_free_rate,
        option_type=option_type,
    )
    return option_payoff(
        option_type,
        position,
        strike,
        terminal_prices,
        premium=premium if include_premiums else 0.0,
    ) * float(contract_multiplier)


def calculate_boundary_quantity_sensitivity(
    base_payoff: np.ndarray,
    terminal_prices: np.ndarray,
    curve: pd.DataFrame,
    *,
    polymarket_side: str,
    volatility: float,
    time_to_expiry: float,
    risk_free_rate: float,
    include_premiums: bool,
    contract_multiplier: float,
    normalized_spot: float = 100.0,
    confidences: list[float] | None = None,
    quantities: list[float] | None = None,
) -> pd.DataFrame:
    """Evaluate total portfolio EV and SD for boundary/quantity combinations."""
    confidences = confidences or DEFAULT_CONFIDENCES
    quantities = quantities or DEFAULT_QUANTITIES
    side = str(polymarket_side).upper()
    if side not in {"YES", "NO"}:
        raise ValueError("Polymarket side must be YES or NO.")

    call_position = "Short" if side == "YES" else "Long"
    put_position = "Long" if side == "YES" else "Short"
    rows = []
    for confidence_pct in confidences:
        confidence = float(confidence_pct) / 100.0
        call_strike = strike_at_confidence(curve, confidence, boundary_type="Win boundary", normalized_spot=normalized_spot)
        put_strike = strike_at_confidence(curve, confidence, boundary_type="Loss boundary", normalized_spot=normalized_spot)
        call_payoff = _unit_leg_payoff(terminal_prices, option_type="Call", position=call_position, strike=call_strike, volatility=volatility, time_to_expiry=time_to_expiry, risk_free_rate=risk_free_rate, include_premiums=include_premiums, contract_multiplier=contract_multiplier, normalized_spot=normalized_spot)
        put_payoff = _unit_leg_payoff(terminal_prices, option_type="Put", position=put_position, strike=put_strike, volatility=volatility, time_to_expiry=time_to_expiry, risk_free_rate=risk_free_rate, include_premiums=include_premiums, contract_multiplier=contract_multiplier, normalized_spot=normalized_spot)
        for quantity in quantities:
            for structure, option_component in (("Call + Put", call_payoff + put_payoff), ("Call only", call_payoff), ("Put only", put_payoff)):
                total = np.asarray(base_payoff, dtype=float) + float(quantity) * option_component
                expected_payoff = float(total.mean())
                payoff_sd = float(total.std(ddof=0))
                rows.append({
                    "Structure": structure,
                    "Boundary confidence (%)": float(confidence_pct),
                    "Contracts per leg": float(quantity),
                    "Call strike": float(call_strike),
                    "Put strike": float(put_strike),
                    "Expected payoff": expected_payoff,
                    "Payoff SD": payoff_sd,
                    "EV / SD": expected_payoff / payoff_sd if payoff_sd > 0 else np.nan,
                    "Probability of loss": float((total < 0).mean()),
                })
    sensitivity = pd.DataFrame(rows)
    sensitivity.attrs["terminal_prices"] = np.asarray(terminal_prices, dtype=float)
    sensitivity.attrs["base_payoff"] = np.asarray(base_payoff, dtype=float)
    return sensitivity


def sensitivity_heatmap(sensitivity: pd.DataFrame, *, structure: str, metric: str) -> go.Figure:
    selected = sensitivity[sensitivity["Structure"] == structure]
    pivot = selected.pivot(index="Contracts per leg", columns="Boundary confidence (%)", values=metric).sort_index(ascending=False)
    values = pivot.to_numpy(dtype=float)
    if metric == "EV / SD":
        text = np.vectorize(lambda value: f"{value:.2f}")(values)
        color_scale = "RdYlGn"
    else:
        text = np.vectorize(lambda value: f"${value:,.1f}")(values)
        color_scale = "RdYlGn" if metric == "Expected payoff" else "RdYlGn_r"
    fig = go.Figure(go.Heatmap(
        z=values,
        x=[f"{value:.0f}%" for value in pivot.columns],
        y=[f"{value:.2f}" for value in pivot.index],
        text=text,
        texttemplate="%{text}",
        colorscale=color_scale,
        colorbar=dict(title=metric, thickness=10),
        hovertemplate="Boundary confidence: %{x}<br>Contracts per leg: %{y}<br>" + metric + ": %{text}<extra></extra>",
    ))
    fig.update_layout(title=metric, xaxis_title="Boundary confidence", yaxis_title="Contracts per leg", height=690, margin=dict(l=20, r=20, t=60, b=40), font=dict(size=11))
    return fig


def render_boundary_quantity_sensitivity(sensitivity: pd.DataFrame, *, polymarket_side: str) -> None:
    """Render the live manual profile followed by structure sensitivity tabs."""
    terminal_prices = sensitivity.attrs.get("terminal_prices")
    base_payoff = sensitivity.attrs.get("base_payoff")
    manual_payoff = st.session_state.get("phase5_manual_total_payoff")
    if terminal_prices is not None and base_payoff is not None and manual_payoff is not None and len(manual_payoff) == len(terminal_prices):
        render_live_manual_profile(terminal_prices, base_payoff, manual_payoff, bin_width=5.0)

    st.subheader("Boundary confidence x option quantity sensitivity")
    st.caption("Each cell reuses the stored Monte Carlo scenarios. Quantity applies to each active leg. EV / SD is a simple payoff-efficiency ratio, not a Sharpe ratio. For YES the hedge is short call / long put; for NO it is long call / short put.")
    structures = ["Call + Put", "Call only", "Put only"]
    structure_tabs = st.tabs(structures)
    for structure, tab in zip(structures, structure_tabs):
        with tab:
            columns = st.columns(3)
            for column, metric, key_suffix in zip(columns, ["Expected payoff", "Payoff SD", "EV / SD"], ["ev", "sd", "efficiency"]):
                with column:
                    st.plotly_chart(sensitivity_heatmap(sensitivity, structure=structure, metric=metric), use_container_width=True, key=f"phase5_sensitivity_{key_suffix}_{polymarket_side}_{structure}")
            detail = sensitivity[sensitivity["Structure"] == structure].copy()
            detail["Expected payoff"] = detail["Expected payoff"].map(lambda x: f"${x:,.2f}")
            detail["Payoff SD"] = detail["Payoff SD"].map(lambda x: f"${x:,.2f}")
            detail["EV / SD"] = detail["EV / SD"].map(lambda x: f"{x:.3f}")
            detail["Probability of loss"] = detail["Probability of loss"].map(lambda x: f"{x:.2%}")
            detail["Call strike"] = detail["Call strike"].map(lambda x: f"${x:,.2f}")
            detail["Put strike"] = detail["Put strike"].map(lambda x: f"${x:,.2f}")
            with st.expander("Show sensitivity values and strikes"):
                st.dataframe(detail, use_container_width=True, hide_index=True)
