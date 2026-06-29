from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from payoff_surface import payoff_summary


HEDGE_TEMPLATES = [
    "All constructed legs",
    "Polymarket only",
    "Protective put only",
    "Short call only",
    "Collar",
]


def pct(value: float) -> str:
    return "" if pd.isna(value) else f"{value:.2%}"


def dollars(value: float) -> str:
    return "" if pd.isna(value) else f"${value:,.2f}"


def dollars_trillions(value: float) -> str:
    return "" if pd.isna(value) else f"${value / 1e12:,.2f}T"


def apply_hedge_template(option_legs: pd.DataFrame, template: str) -> pd.DataFrame:
    """Return all constructed legs with inactive template legs sized to zero."""
    active = option_legs.copy()
    if "Quantity" not in active.columns:
        active["Quantity"] = 0.0

    if template == "All constructed legs":
        return active
    if template == "Polymarket only":
        active["Quantity"] = 0.0
        return active

    option_type = active["Option type"].astype(str)
    position = active["Position"].astype(str)
    if template == "Protective put only":
        mask = option_type.eq("Put") & position.eq("Long")
    elif template == "Short call only":
        mask = option_type.eq("Call") & position.eq("Short")
    elif template == "Collar":
        mask = (option_type.eq("Put") & position.eq("Long")) | (option_type.eq("Call") & position.eq("Short"))
    else:
        raise ValueError(f"Unknown hedge template: {template}")

    active.loc[~mask, "Quantity"] = 0.0
    return active


def editable_option_legs_view(table: pd.DataFrame) -> pd.DataFrame:
    display = table.copy()
    if "Boundary market cap" in display.columns:
        display["Boundary market cap"] = display["Boundary market cap"].map(dollars_trillions)
    if "Boundary / current cap" in display.columns:
        display["Boundary / current cap"] = display["Boundary / current cap"].map(pct)
        display = display.rename(columns={"Boundary / current cap": "Boundary / current cap (%)"})
    return display


def merge_edited_quantities(original: pd.DataFrame, edited_view: pd.DataFrame) -> pd.DataFrame:
    updated = original.copy()
    if "Quantity" in edited_view.columns:
        updated["Quantity"] = pd.to_numeric(edited_view["Quantity"], errors="coerce").fillna(0.0)
    return updated


def display_option_legs(table: pd.DataFrame) -> pd.DataFrame:
    display = table.copy()
    for column in ["Strike", "Spot", "Theoretical premium"]:
        if column in display.columns:
            display[column] = display[column].map(dollars)
    if "Boundary market cap" in display.columns:
        display["Boundary market cap"] = display["Boundary market cap"].map(dollars_trillions)
    for column in ["Boundary / current cap", "Model IV", "Risk-free rate"]:
        if column in display.columns:
            display[column] = display[column].map(pct)
    if "Time to expiry" in display.columns:
        display["Time to expiry"] = display["Time to expiry"].map(lambda value: f"{value:.2f}y")
    return display


def display_scenarios(table: pd.DataFrame) -> pd.DataFrame:
    display = table.copy()
    if "Selected terminal market cap" in display.columns:
        display["Selected terminal market cap"] = display["Selected terminal market cap"].map(dollars_trillions)
    for column in ["Selected terminal stock price", "Polymarket payoff", "Option payoff", "Total payoff"]:
        if column in display.columns:
            display[column] = display[column].map(dollars)
    return display


def display_profile(table: pd.DataFrame) -> pd.DataFrame:
    display = table.copy()
    for column in ["selected_ratio_low", "selected_ratio_high", "selected_ratio", "win_probability", "scenario_probability"]:
        display[column] = display[column].map(pct)
    display["selected_market_cap"] = display["selected_market_cap"].map(dollars_trillions)
    for column in ["selected_stock_price", "expected_polymarket_payoff", "expected_option_payoff", "expected_payoff", "payoff_standard_deviation", "weighted_payoff_contribution"]:
        display[column] = display[column].map(dollars)
    renamed = display.rename(columns={
        "bin_label": "Selected terminal cap bin",
        "selected_ratio_low": "Bin low / current",
        "selected_ratio_high": "Bin high / current",
        "selected_ratio": "Avg cap / current",
        "selected_market_cap": "Avg terminal market cap",
        "selected_stock_price": "Avg terminal stock price",
        "win_probability": "Conditional P(#1)",
        "expected_polymarket_payoff": "Avg Polymarket payoff",
        "expected_option_payoff": "Avg option payoff",
        "expected_payoff": "Avg total payoff",
        "payoff_standard_deviation": "Payoff SD inside bin",
        "scenario_probability": "Scenario probability",
        "weighted_payoff_contribution": "Contribution to expected payoff",
        "scenario_count": "Scenario count",
    })
    order = [
        "Selected terminal cap bin", "Scenario probability", "Conditional P(#1)", "Avg total payoff",
        "Contribution to expected payoff", "Avg option payoff", "Avg Polymarket payoff", "Payoff SD inside bin",
        "Scenario count", "Avg cap / current", "Avg terminal market cap", "Avg terminal stock price",
        "Bin low / current", "Bin high / current",
    ]
    return renamed[[column for column in order if column in renamed.columns]]


def display_risk_summary(summary: pd.Series) -> pd.DataFrame:
    return pd.DataFrame([
        {"Metric": "Expected payoff", "Value": dollars(float(summary["Expected payoff"])), "How to read": "Probability-weighted average payoff across all scenarios."},
        {"Metric": "Payoff SD", "Value": dollars(float(summary["Payoff standard deviation"])), "How to read": "Dispersion of total payoff across scenarios."},
        {"Metric": "Median payoff", "Value": dollars(float(summary["Median payoff"])), "How to read": "Middle scenario payoff."},
        {"Metric": "P(loss)", "Value": pct(float(summary["Probability of loss"])), "How to read": "Share of scenarios with negative payoff."},
        {"Metric": "Expected shortfall 5%", "Value": dollars(float(summary["Expected shortfall 5%"])), "How to read": "Average payoff in the worst 5% of scenarios."},
        {"Metric": "Worst payoff", "Value": dollars(float(summary["Worst payoff"])), "How to read": "Worst simulated payoff."},
    ])


def comparison_table(baseline_scenarios: pd.DataFrame, hedge_scenarios: pd.DataFrame, hedge_label: str) -> pd.DataFrame:
    rows = []
    for label, scenarios in [("Polymarket only", baseline_scenarios), (hedge_label, hedge_scenarios)]:
        summary = payoff_summary(scenarios)
        rows.append({
            "Payoff profile": label,
            "Expected payoff": dollars(float(summary["Expected payoff"])),
            "Payoff SD": dollars(float(summary["Payoff standard deviation"])),
            "Median payoff": dollars(float(summary["Median payoff"])),
            "P(loss)": pct(float(summary["Probability of loss"])),
            "Expected shortfall 5%": dollars(float(summary["Expected shortfall 5%"])),
            "Worst payoff": dollars(float(summary["Worst payoff"])),
        })
    return pd.DataFrame(rows)


def payoff_profile_figure(profile: pd.DataFrame, selected_ticker: str) -> go.Figure:
    fig = go.Figure()
    fig.add_bar(x=profile["bin_label"], y=profile["scenario_probability"], name="Scenario probability", marker_color="#7aa6ff", yaxis="y")
    fig.add_bar(x=profile["bin_label"], y=profile["weighted_payoff_contribution"], name="Contribution to expected payoff", marker_color="#22c55e", opacity=0.6, yaxis="y2")
    fig.add_scatter(x=profile["bin_label"], y=profile["expected_payoff"], name="Avg payoff in bin", mode="lines+markers", line=dict(color="#1f3a8a", width=3), yaxis="y2")
    fig.update_layout(
        title=f"{selected_ticker}: payoff profile by terminal market-cap bin",
        xaxis_title=f"{selected_ticker} terminal market cap / current market cap",
        yaxis=dict(title="Scenario probability", tickformat=".0%"),
        yaxis2=dict(title="Payoff / expected payoff contribution", overlaying="y", side="right", tickprefix="$"),
        barmode="group",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
    )
    return fig


def payoff_by_bin_figure(profile: pd.DataFrame, selected_ticker: str) -> go.Figure:
    colors = ["#16a34a" if value >= 0 else "#dc2626" for value in profile["expected_payoff"]]
    fig = go.Figure(go.Bar(
        x=profile["bin_label"],
        y=profile["expected_payoff"],
        marker_color=colors,
        hovertemplate="Terminal cap bin=%{x}<br>Avg payoff=$%{y:,.2f}<extra></extra>",
    ))
    fig.add_hline(y=0, line_dash="dash", line_color="#6b7280")
    fig.update_layout(
        title=f"{selected_ticker}: average payoff by terminal market-cap bin",
        xaxis_title=f"{selected_ticker} terminal market cap / current market cap",
        yaxis_title="Average payoff in bin",
        yaxis=dict(tickprefix="$"),
        height=420,
        showlegend=False,
    )
    return fig


def single_option_payoff(option_type: str, position: str, terminal_price: float, strike: float, premium: float) -> float:
    intrinsic = max(terminal_price - strike, 0.0) if option_type == "Call" else max(strike - terminal_price, 0.0)
    return intrinsic - premium if position == "Long" else premium - intrinsic


def polymarket_conditional_ev_and_sd(side: str, win_probability: float, entry_price: float, quantity: float) -> tuple[float, float]:
    p = min(max(float(win_probability), 0.0), 1.0)
    if side == "YES":
        win_payoff, lose_payoff = 1.0 - entry_price, -entry_price
    else:
        win_payoff, lose_payoff = -entry_price, 1.0 - entry_price
    ev = p * win_payoff + (1.0 - p) * lose_payoff
    variance = p * (win_payoff - ev) ** 2 + (1.0 - p) * (lose_payoff - ev) ** 2
    return ev * quantity, variance ** 0.5 * quantity


def manual_option_calculator(option_legs: pd.DataFrame | None, profile: pd.DataFrame | None, side: str, entry_price: float, polymarket_quantity: float) -> None:
    st.subheader("Manual option payoff intuition calculator")
    st.write("Option payoff uses manual inputs. Scenario probability and P(#1) use the closest Phase 4 terminal-price bin.")

    spot_default = 200.0
    call_strike_default, call_premium_default, call_quantity_default = 260.0, 2.0, 0.01
    put_strike_default, put_premium_default, put_quantity_default = 140.0, 3.0, 0.01
    if option_legs is not None and not option_legs.empty:
        legs = option_legs.copy()
        if "Spot" in legs and legs["Spot"].notna().any():
            spot_default = float(legs["Spot"].dropna().iloc[0])
        calls = legs[(legs["Option type"] == "Call") & legs["Strike"].notna()]
        puts = legs[(legs["Option type"] == "Put") & legs["Strike"].notna()]
        if not calls.empty:
            leg = calls.iloc[0]
            call_strike_default = float(leg["Strike"])
            call_premium_default = float(leg.get("Theoretical premium", 0.0))
            call_quantity_default = float(leg.get("Quantity", 0.01))
        if not puts.empty:
            leg = puts.iloc[0]
            put_strike_default = float(leg["Strike"])
            put_premium_default = float(leg.get("Theoretical premium", 0.0))
            put_quantity_default = float(leg.get("Quantity", 0.01))

    row = st.columns(4)
    spot = row[0].number_input("Current stock price", min_value=0.01, value=spot_default, step=1.0)
    multiplier = row[1].number_input("Shares per contract", min_value=1.0, value=100.0, step=1.0)
    low_pct = row[2].number_input("Grid low (% of spot)", min_value=1.0, value=50.0, step=5.0)
    high_pct = row[3].number_input("Grid high (% of spot)", min_value=1.0, value=180.0, step=5.0)

    row = st.columns(4)
    call_strike = row[0].number_input("Short call strike", min_value=0.01, value=call_strike_default, step=1.0)
    call_premium = row[1].number_input("Call premium received", min_value=0.0, value=call_premium_default, step=0.1)
    call_quantity = row[2].number_input("Short call quantity", min_value=0.0, value=call_quantity_default, step=0.01)
    include_call = row[3].checkbox("Include short call", value=True)

    row = st.columns(4)
    put_strike = row[0].number_input("Long put strike", min_value=0.01, value=put_strike_default, step=1.0)
    put_premium = row[1].number_input("Put premium paid", min_value=0.0, value=put_premium_default, step=0.1)
    put_quantity = row[2].number_input("Long put quantity", min_value=0.0, value=put_quantity_default, step=0.01)
    include_put = row[3].checkbox("Include long put", value=True)

    low, high = sorted([low_pct / 100.0 * spot, high_pct / 100.0 * spot])
    rows = []
    for terminal_price in [low + (high - low) * i / 24 for i in range(25)]:
        call_payoff = single_option_payoff("Call", "Short", terminal_price, call_strike, call_premium) * call_quantity * multiplier if include_call else 0.0
        put_payoff = single_option_payoff("Put", "Long", terminal_price, put_strike, put_premium) * put_quantity * multiplier if include_put else 0.0
        matched = None
        if profile is not None and not profile.empty:
            matched = profile.loc[(profile["selected_stock_price"].astype(float) - terminal_price).abs().idxmin()]
        probability = float(matched["scenario_probability"]) if matched is not None else 0.0
        win_probability = float(matched["win_probability"]) if matched is not None else 0.0
        pm_ev, pm_sd = polymarket_conditional_ev_and_sd(side, win_probability, entry_price, polymarket_quantity)
        option_payoff = call_payoff + put_payoff
        total_ev = pm_ev + option_payoff
        rows.append({
            "Terminal stock price": terminal_price,
            "Terminal / spot": terminal_price / spot,
            "Nearest MC bin probability": probability,
            "P(Polymarket wins | bin)": win_probability,
            "Polymarket EV | bin": pm_ev,
            "Polymarket SD | bin": pm_sd,
            "Short call payoff": call_payoff,
            "Long put payoff": put_payoff,
            "Total option payoff": option_payoff,
            "Total conditional EV | bin": total_ev,
            "Contribution to global EV": total_ev * probability,
        })
    table = pd.DataFrame(rows)

    fig = go.Figure()
    fig.add_scatter(x=table["Terminal stock price"], y=table["Total option payoff"], mode="lines", name="Option payoff")
    fig.add_scatter(x=table["Terminal stock price"], y=table["Total conditional EV | bin"], mode="lines", name="Total conditional EV", line=dict(width=4))
    fig.add_bar(x=table["Terminal stock price"], y=table["Nearest MC bin probability"], name="Scenario probability", opacity=0.35, yaxis="y2")
    fig.add_hline(y=0, line_dash="dash", line_color="#6b7280")
    fig.update_layout(
        title="Manual payoff with nearest Monte Carlo probability lookup",
        xaxis_title="Terminal stock price",
        yaxis=dict(title="Payoff / EV", tickprefix="$"),
        yaxis2=dict(title="Scenario probability", overlaying="y", side="right", tickformat=".0%"),
        height=500,
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
    )
    st.plotly_chart(fig, use_container_width=True)

    display = table.copy()
    for column in ["Terminal / spot", "Nearest MC bin probability", "P(Polymarket wins | bin)"]:
        display[column] = display[column].map(pct)
    for column in ["Terminal stock price", "Polymarket EV | bin", "Polymarket SD | bin", "Short call payoff", "Long put payoff", "Total option payoff", "Total conditional EV | bin", "Contribution to global EV"]:
        display[column] = display[column].map(dollars)
    st.dataframe(display, use_container_width=True, hide_index=True)
