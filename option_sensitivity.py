"""Boundary-confidence and option-quantity sensitivity for Phase 5."""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from interactive_portfolio import strike_at_confidence
from option_construction import black_scholes_price, option_payoff


DEFAULT_CONFIDENCES = [50, 60, 70, 80, 90, 95, 99]
DEFAULT_QUANTITIES = [0.0, 0.25, 0.50, 0.75, 1.0, 1.50, 2.0]


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
        call_strike = strike_at_confidence(
            curve,
            confidence,
            boundary_type="Win boundary",
            normalized_spot=normalized_spot,
        )
        put_strike = strike_at_confidence(
            curve,
            confidence,
            boundary_type="Loss boundary",
            normalized_spot=normalized_spot,
        )
        call_payoff = _unit_leg_payoff(
            terminal_prices,
            option_type="Call",
            position=call_position,
            strike=call_strike,
            volatility=volatility,
            time_to_expiry=time_to_expiry,
            risk_free_rate=risk_free_rate,
            include_premiums=include_premiums,
            contract_multiplier=contract_multiplier,
            normalized_spot=normalized_spot,
        )
        put_payoff = _unit_leg_payoff(
            terminal_prices,
            option_type="Put",
            position=put_position,
            strike=put_strike,
            volatility=volatility,
            time_to_expiry=time_to_expiry,
            risk_free_rate=risk_free_rate,
            include_premiums=include_premiums,
            contract_multiplier=contract_multiplier,
            normalized_spot=normalized_spot,
        )

        for quantity in quantities:
            for structure, option_component in (
                ("Call + Put", call_payoff + put_payoff),
                ("Call only", call_payoff),
                ("Put only", put_payoff),
            ):
                total = np.asarray(base_payoff, dtype=float) + float(quantity) * option_component
                rows.append(
                    {
                        "Structure": structure,
                        "Boundary confidence (%)": float(confidence_pct),
                        "Contracts per leg": float(quantity),
                        "Call strike": float(call_strike),
                        "Put strike": float(put_strike),
                        "Expected payoff": float(total.mean()),
                        "Payoff SD": float(total.std(ddof=0)),
                        "Probability of loss": float((total < 0).mean()),
                    }
                )
    return pd.DataFrame(rows)


def sensitivity_heatmap(
    sensitivity: pd.DataFrame,
    *,
    structure: str,
    metric: str,
) -> go.Figure:
    selected = sensitivity[sensitivity["Structure"] == structure]
    pivot = selected.pivot(
        index="Contracts per leg",
        columns="Boundary confidence (%)",
        values=metric,
    ).sort_index(ascending=False)
    values = pivot.to_numpy(dtype=float)
    text = np.vectorize(lambda value: f"${value:,.2f}")(values)
    fig = go.Figure(
        go.Heatmap(
            z=values,
            x=[f"{value:.0f}%" for value in pivot.columns],
            y=[f"{value:.2f}" for value in pivot.index],
            text=text,
            texttemplate="%{text}",
            colorscale="RdYlGn" if metric == "Expected payoff" else "RdYlGn_r",
            colorbar=dict(title=metric),
            hovertemplate=(
                "Boundary confidence: %{x}<br>Contracts per leg: %{y}<br>"
                + metric
                + ": %{text}<extra></extra>"
            ),
        )
    )
    fig.update_layout(
        title=f"{structure}: {metric}",
        xaxis_title="Boundary confidence",
        yaxis_title="Contracts per leg",
        height=470,
        margin=dict(l=30, r=30, t=70, b=40),
    )
    return fig


def render_boundary_quantity_sensitivity(
    sensitivity: pd.DataFrame,
    *,
    polymarket_side: str,
) -> None:
    """Render three structure tabs with EV and SD heatmaps."""
    st.subheader("Boundary confidence x option quantity sensitivity")
    st.caption(
        "Each cell reuses the stored Monte Carlo scenarios. Quantity applies to each active leg. "
        "For YES the hedge is short call / long put; for NO it is long call / short put."
    )
    structure_tabs = st.tabs(["Call + Put", "Call only", "Put only"])
    for structure, tab in zip(["Call + Put", "Call only", "Put only"], structure_tabs):
        with tab:
            left, right = st.columns(2)
            with left:
                st.plotly_chart(
                    sensitivity_heatmap(
                        sensitivity,
                        structure=structure,
                        metric="Expected payoff",
                    ),
                    use_container_width=True,
                    key=f"phase5_sensitivity_ev_{polymarket_side}_{structure}",
                )
            with right:
                st.plotly_chart(
                    sensitivity_heatmap(
                        sensitivity,
                        structure=structure,
                        metric="Payoff SD",
                    ),
                    use_container_width=True,
                    key=f"phase5_sensitivity_sd_{polymarket_side}_{structure}",
                )

            detail = sensitivity[sensitivity["Structure"] == structure].copy()
            detail["Expected payoff"] = detail["Expected payoff"].map(lambda x: f"${x:,.2f}")
            detail["Payoff SD"] = detail["Payoff SD"].map(lambda x: f"${x:,.2f}")
            detail["Probability of loss"] = detail["Probability of loss"].map(lambda x: f"{x:.2%}")
            detail["Call strike"] = detail["Call strike"].map(lambda x: f"${x:,.2f}")
            detail["Put strike"] = detail["Put strike"].map(lambda x: f"${x:,.2f}")
            with st.expander("Show sensitivity values"):
                st.dataframe(detail, use_container_width=True, hide_index=True)
