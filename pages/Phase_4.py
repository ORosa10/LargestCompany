from __future__ import annotations

from datetime import date, timedelta

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from boundaries import calculate_boundaries_for_all_tickers
from correlations import ewma_correlation, fetch_adjusted_close, rolling_correlation, smooth_vol_adjusted_correlation
from market_data import apply_market_caps, fetch_market_caps, fetch_spot_prices
from model import default_company_inputs, run_probability_engine
from option_construction import attach_theoretical_premiums, construct_candidate_option_structure
from payoff_surface import calculate_scenario_payoffs, payoff_summary, selected_payoff_profile_bins


st.set_page_config(page_title="Phase 4", layout="wide")
st.title("Phase 4")
st.caption("Payoff Profile Engine. This phase combines Polymarket payoff and candidate option legs across Monte Carlo scenarios. It does not optimize hedge ratios.")

CORRELATION_METHODS = ["EWMA historical correlation", "Vol-adjusted smooth correlation", "Rolling historical correlation"]
CONFIDENCE_LEVELS = [0.80, 0.90, 0.95, 0.99]
CONSTRUCTION_MODE_LABELS = {
    "Selected-only hedge": "selected_only",
    "Selected + single competitor diagnostic": "single_competitor",
    "Selected + full universe competitors": "full_universe",
}


@st.cache_data(show_spinner=False, ttl=15 * 60)
def load_yahoo_market_caps(tickers: tuple[str, ...]) -> pd.DataFrame:
    return fetch_market_caps(list(tickers))


@st.cache_data(show_spinner=False, ttl=15 * 60)
def load_spot_prices(tickers: tuple[str, ...]) -> pd.DataFrame:
    return fetch_spot_prices(list(tickers))


@st.cache_data(show_spinner=False, ttl=60 * 60)
def load_adjusted_close(tickers: tuple[str, ...], period: str) -> pd.DataFrame:
    return fetch_adjusted_close(list(tickers), period=period)


def pct(value: float) -> str:
    return "" if pd.isna(value) else f"{value:.2%}"


def dollars(value: float) -> str:
    return "" if pd.isna(value) else f"${value:,.2f}"


def dollars_trillions(value: float) -> str:
    return "" if pd.isna(value) else f"${value / 1e12:,.2f}T"


def years(days: int) -> float:
    return max(int(days), 1) / 365.0


def build_correlation_matrix(method: str, prices: pd.DataFrame, simulation_inputs: pd.DataFrame, ewma_lambda: float, rolling_lookback: int, smooth_low_quantile: float, smooth_high_quantile: float) -> pd.DataFrame:
    if method == "EWMA historical correlation":
        return ewma_correlation(prices, ewma_lambda)
    if method == "Rolling historical correlation":
        return rolling_correlation(prices, rolling_lookback)
    if method == "Vol-adjusted smooth correlation":
        current_ivs = simulation_inputs.set_index("Ticker")["Implied volatility"].astype(float)
        corr, _ = smooth_vol_adjusted_correlation(
            prices,
            current_ivs,
            vol_window=63,
            low_quantile=smooth_low_quantile,
            high_quantile=smooth_high_quantile,
            min_observations=30,
        )
        return corr
    raise ValueError(f"Unknown correlation method: {method}")


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


def display_scenarios(table: pd.DataFrame) -> pd.DataFrame:
    display = table.copy()
    for column in ["Selected terminal market cap"]:
        display[column] = display[column].map(dollars_trillions)
    for column in ["Selected terminal stock price", "Polymarket payoff", "Option payoff", "Total payoff"]:
        display[column] = display[column].map(dollars)
    return display


def display_profile(table: pd.DataFrame) -> pd.DataFrame:
    display = table.copy()
    for column in ["selected_ratio_low", "selected_ratio_high", "selected_ratio", "win_probability", "scenario_probability"]:
        display[column] = display[column].map(pct)
    for column in ["selected_market_cap"]:
        display[column] = display[column].map(dollars_trillions)
    for column in ["selected_stock_price", "expected_polymarket_payoff", "expected_option_payoff", "expected_payoff", "payoff_standard_deviation", "weighted_payoff_contribution"]:
        display[column] = display[column].map(dollars)
    renamed = display.rename(
        columns={
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
        }
    )
    preferred_order = [
        "Selected terminal cap bin", "Scenario probability", "Conditional P(#1)", "Avg total payoff", "Contribution to expected payoff",
        "Avg option payoff", "Avg Polymarket payoff", "Payoff SD inside bin", "Scenario count", "Avg cap / current",
        "Avg terminal market cap", "Avg terminal stock price", "Bin low / current", "Bin high / current",
    ]
    return renamed[[column for column in preferred_order if column in renamed.columns]]


def display_risk_summary(summary: pd.Series) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"Metric": "Expected payoff", "Value": dollars(float(summary["Expected payoff"])), "How to read": "Probability-weighted average payoff across all simulated scenarios."},
            {"Metric": "Payoff SD", "Value": dollars(float(summary["Payoff standard deviation"])), "How to read": "Dispersion of total payoff across scenarios. Higher means a more volatile payoff profile."},
            {"Metric": "Median payoff", "Value": dollars(float(summary["Median payoff"])), "How to read": "Middle scenario payoff."},
            {"Metric": "P(loss)", "Value": pct(float(summary["Probability of loss"])), "How to read": "Share of scenarios with negative total payoff."},
            {"Metric": "Expected shortfall 5%", "Value": dollars(float(summary["Expected shortfall 5%"])), "How to read": "Average payoff inside the worst 5% of simulated scenarios."},
            {"Metric": "Worst payoff", "Value": dollars(float(summary["Worst payoff"])), "How to read": "Worst single simulated payoff."},
        ]
    )


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
    fig = go.Figure()
    fig.add_bar(x=profile["bin_label"], y=profile["expected_payoff"], marker_color=colors, hovertemplate="Terminal cap bin=%{x}<br>Avg payoff=$%{y:,.2f}<extra></extra>", name="Avg payoff")
    fig.add_hline(y=0, line_dash="dash", line_color="#6b7280")
    fig.update_layout(title=f"{selected_ticker}: average payoff by terminal market-cap bin", xaxis_title=f"{selected_ticker} terminal market cap / current market cap", yaxis_title="Average payoff in bin", yaxis=dict(tickprefix="$"), height=420, showlegend=False)
    return fig


def single_option_payoff(option_type: str, position: str, terminal_price: float, strike: float, premium: float) -> float:
    intrinsic = max(float(terminal_price) - float(strike), 0.0) if option_type == "Call" else max(float(strike) - float(terminal_price), 0.0)
    return intrinsic - float(premium) if position == "Long" else float(premium) - intrinsic


def manual_calculator_defaults(option_legs: pd.DataFrame | None) -> dict[str, float]:
    defaults = {"spot": 200.0, "call_strike": 260.0, "call_premium": 2.0, "call_quantity": 0.01, "put_strike": 140.0, "put_premium": 3.0, "put_quantity": 0.01, "multiplier": 100.0}
    if option_legs is None or option_legs.empty:
        return defaults
    legs = option_legs.copy()
    if "Spot" in legs.columns and legs["Spot"].notna().any():
        defaults["spot"] = float(legs["Spot"].dropna().iloc[0])
    call = legs[(legs["Option type"] == "Call") & (legs["Strike"].notna())]
    put = legs[(legs["Option type"] == "Put") & (legs["Strike"].notna())]
    if not call.empty:
        first_call = call.iloc[0]
        defaults["call_strike"] = float(first_call["Strike"])
        defaults["call_premium"] = float(first_call.get("Theoretical premium", 0.0))
        defaults["call_quantity"] = float(first_call.get("Quantity", defaults["call_quantity"]))
    if not put.empty:
        first_put = put.iloc[0]
        defaults["put_strike"] = float(first_put["Strike"])
        defaults["put_premium"] = float(first_put.get("Theoretical premium", 0.0))
        defaults["put_quantity"] = float(first_put.get("Quantity", defaults["put_quantity"]))
    return defaults


def polymarket_conditional_ev_and_sd(side: str, win_probability: float, entry_price: float, quantity: float) -> tuple[float, float]:
    p_win = min(max(float(win_probability), 0.0), 1.0)
    if side == "YES":
        win_payoff = 1.0 - float(entry_price)
        lose_payoff = -float(entry_price)
        ev_per_share = p_win * win_payoff + (1.0 - p_win) * lose_payoff
        variance_per_share = p_win * (win_payoff - ev_per_share) ** 2 + (1.0 - p_win) * (lose_payoff - ev_per_share) ** 2
    else:
        win_payoff = -float(entry_price)
        lose_payoff = 1.0 - float(entry_price)
        ev_per_share = p_win * win_payoff + (1.0 - p_win) * lose_payoff
        variance_per_share = p_win * (win_payoff - ev_per_share) ** 2 + (1.0 - p_win) * (lose_payoff - ev_per_share) ** 2
    return ev_per_share * float(quantity), variance_per_share ** 0.5 * float(quantity)


def nearest_profile_row(profile: pd.DataFrame | None, terminal_price: float) -> pd.Series | None:
    if profile is None or profile.empty or "selected_stock_price" not in profile.columns:
        return None
    distances = (profile["selected_stock_price"].astype(float) - float(terminal_price)).abs()
    return profile.loc[distances.idxmin()]


def manual_option_calculator(option_legs: pd.DataFrame | None, profile: pd.DataFrame | None, polymarket_side: str, polymarket_entry_price: float, polymarket_quantity: float) -> None:
    st.subheader("Manual option payoff intuition calculator")
    st.write("This is a price-grid calculator plus a nearest-bin Monte Carlo lookup. The option payoff is calculated from your manual inputs; scenario probability and P(#1) come from the closest Phase 4 simulated terminal-price bin.")
    defaults = manual_calculator_defaults(option_legs)

    input_cols = st.columns(4)
    spot = input_cols[0].number_input("Current stock price", min_value=0.01, value=defaults["spot"], step=1.0, format="%.2f")
    multiplier = input_cols[1].number_input("Shares per contract", min_value=1.0, value=defaults["multiplier"], step=1.0)
    price_min_pct = input_cols[2].number_input("Terminal grid low (% of spot)", min_value=1.0, value=50.0, step=5.0)
    price_max_pct = input_cols[3].number_input("Terminal grid high (% of spot)", min_value=1.0, value=180.0, step=5.0)

    call_cols = st.columns(4)
    call_strike = call_cols[0].number_input("Short call strike", min_value=0.01, value=defaults["call_strike"], step=1.0, format="%.2f")
    call_premium = call_cols[1].number_input("Call premium received", min_value=0.0, value=defaults["call_premium"], step=0.1, format="%.2f")
    call_quantity = call_cols[2].number_input("Short call quantity", min_value=0.0, value=defaults["call_quantity"], step=0.01, format="%.2f")
    include_call = call_cols[3].checkbox("Include short call", value=True)

    put_cols = st.columns(4)
    put_strike = put_cols[0].number_input("Long put strike", min_value=0.01, value=defaults["put_strike"], step=1.0, format="%.2f")
    put_premium = put_cols[1].number_input("Put premium paid", min_value=0.0, value=defaults["put_premium"], step=0.1, format="%.2f")
    put_quantity = put_cols[2].number_input("Long put quantity", min_value=0.0, value=defaults["put_quantity"], step=0.01, format="%.2f")
    include_put = put_cols[3].checkbox("Include long put", value=True)

    low = min(price_min_pct, price_max_pct) / 100.0 * spot
    high = max(price_min_pct, price_max_pct) / 100.0 * spot
    terminal_prices = [low + (high - low) * i / 24 for i in range(25)]
    rows = []
    for terminal_price in terminal_prices:
        call_payoff = single_option_payoff("Call", "Short", terminal_price, call_strike, call_premium) * call_quantity * multiplier if include_call else 0.0
        put_payoff = single_option_payoff("Put", "Long", terminal_price, put_strike, put_premium) * put_quantity * multiplier if include_put else 0.0
        option_payoff = call_payoff + put_payoff
        matched = nearest_profile_row(profile, terminal_price)
        scenario_probability = float(matched["scenario_probability"]) if matched is not None else 0.0
        win_probability = float(matched["win_probability"]) if matched is not None else 0.0
        matched_sd = float(matched.get("payoff_standard_deviation", 0.0)) if matched is not None else 0.0
        pm_ev, pm_sd = polymarket_conditional_ev_and_sd(polymarket_side, win_probability, polymarket_entry_price, polymarket_quantity)
        conditional_ev = pm_ev + option_payoff
        rows.append(
            {
                "Terminal stock price": terminal_price,
                "Terminal / spot": terminal_price / spot,
                "Nearest MC bin probability": scenario_probability,
                "P(Polymarket wins | bin)": win_probability,
                "Polymarket EV | bin": pm_ev,
                "Polymarket SD | bin": pm_sd,
                "Short call payoff": call_payoff,
                "Long put payoff": put_payoff,
                "Total option payoff": option_payoff,
                "Total conditional EV | bin": conditional_ev,
                "Contribution to global EV": conditional_ev * scenario_probability,
                "Matched MC bin payoff SD": matched_sd,
            }
        )
    table = pd.DataFrame(rows)

    fig = go.Figure()
    fig.add_scatter(x=table["Terminal stock price"], y=table["Total option payoff"], mode="lines", name="Manual option payoff", line=dict(width=3))
    fig.add_scatter(x=table["Terminal stock price"], y=table["Total conditional EV | bin"], mode="lines", name="Total EV using nearest MC bin", line=dict(width=4))
    fig.add_bar(x=table["Terminal stock price"], y=table["Nearest MC bin probability"], name="Nearest-bin scenario probability", opacity=0.35, yaxis="y2")
    fig.add_hline(y=0, line_dash="dash", line_color="#6b7280")
    fig.add_vline(x=spot, line_dash="dot", line_color="#6b7280", annotation_text="spot")
    fig.add_vline(x=call_strike, line_dash="dash", line_color="#dc2626", annotation_text="call strike")
    fig.add_vline(x=put_strike, line_dash="dash", line_color="#2563eb", annotation_text="put strike")
    fig.update_layout(
        title="Manual payoff with nearest Monte Carlo probability lookup",
        xaxis_title="Terminal stock price",
        yaxis=dict(title="Payoff / EV", tickprefix="$"),
        yaxis2=dict(title="Scenario probability", overlaying="y", side="right", tickformat=".0%"),
        height=500,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
    )
    st.plotly_chart(fig, use_container_width=True)

    display = table.copy()
    for column in ["Terminal / spot", "Nearest MC bin probability", "P(Polymarket wins | bin)"]:
        display[column] = display[column].map(pct)
    for column in ["Terminal stock price", "Polymarket EV | bin", "Polymarket SD | bin", "Short call payoff", "Long put payoff", "Total option payoff", "Total conditional EV | bin", "Contribution to global EV", "Matched MC bin payoff SD"]:
        display[column] = display[column].map(dollars)
    st.dataframe(display, use_container_width=True, hide_index=True)

    st.markdown(
        """
How to read this calculator:

- `Nearest MC bin probability` is the approximate probability that the selected ticker finishes near that terminal price zone.
- `P(Polymarket wins | bin)` is the conditional win probability from the nearest simulated bin.
- `Polymarket EV | bin` is the expected Polymarket payoff conditional on that bin.
- `Total option payoff` is deterministic for the manual terminal price you entered.
- `Total conditional EV | bin` combines the conditional Polymarket EV and manual option payoff.
- `Contribution to global EV` multiplies that conditional EV by the bin probability.
- `Matched MC bin payoff SD` is the realized payoff dispersion inside the nearest simulated bin from the full Phase 4 run.
        """
    )


with st.sidebar:
    st.header("Phase 4 controls")
    target_date = st.date_input("Target date / maturity", value=date.today() + timedelta(days=365), min_value=date.today() + timedelta(days=1))
    days_to_target = max((target_date - date.today()).days, 1)
    st.caption(f"Horizon: {days_to_target} days ({days_to_target / 365:.2f} years)")

    simulations = st.number_input("Monte Carlo simulations", min_value=2_000, max_value=1_000_000, value=100_000, step=10_000)
    seed = st.number_input("Random seed", min_value=0, value=42, step=1)
    boundary_bins = st.slider("Phase 2 market-cap quantile bins", min_value=10, max_value=100, value=30, step=5)
    profile_bins = st.slider("Payoff profile bins", min_value=5, max_value=40, value=20, step=1)
    confidence_level = st.selectbox("Boundary confidence level", CONFIDENCE_LEVELS, index=3, format_func=lambda value: f"{value:.0%}")

    st.header("Polymarket position")
    polymarket_side = st.selectbox("Side", ["YES", "NO"], index=0)
    polymarket_quantity = st.number_input("Polymarket shares", min_value=0.0, value=100.0, step=10.0)

    st.header("Option construction")
    construction_mode_label = st.selectbox("Option construction mode", list(CONSTRUCTION_MODE_LABELS), index=0)
    construction_mode = CONSTRUCTION_MODE_LABELS[construction_mode_label]
    include_competitor_short_puts = st.checkbox("Include competitor short puts", value=construction_mode == "single_competitor")
    risk_free_rate = st.number_input("Risk-free rate", min_value=0.0, max_value=0.20, value=0.04, step=0.005, format="%.3f")
    contract_multiplier = st.number_input("Shares per option contract (multiplier)", min_value=1.0, value=100.0, step=1.0, help="Usually 100 for listed US equity options. This is not the number of contracts; use Quantity in the option-leg table for that.")
    default_option_quantity = st.number_input("Default contracts per valid option leg", min_value=0.0, value=0.01, step=0.01, format="%.2f", help="Analytical preview size. One listed option contract can dwarf a small Polymarket position, so the default is fractional for research scaling.")
    include_option_premiums = st.checkbox("Include theoretical option premiums", value=True)
    st.caption("Quantity is the number of contracts. Multiplier is shares per contract, usually 100. One full listed option contract can be much larger than 100 Polymarket shares.")

    st.header("Correlation")
    correlation_method = st.selectbox("Correlation method", CORRELATION_METHODS, index=0)
    price_history_period = st.selectbox("Yahoo price history", ["1y", "3y", "5y", "10y"], index=2)
    ewma_lambda = st.selectbox("EWMA lambda", [0.94, 0.97], index=1)
    rolling_lookback = st.selectbox("Rolling lookback days", [63, 126, 252, 504, 756], index=2)
    smooth_low_quantile = st.selectbox("Smooth low-vol bucket", [0.30, 0.40, 0.50], index=1, format_func=lambda value: f"{value:.0%}")
    smooth_high_quantile = st.selectbox("Smooth high-vol bucket", [0.50, 0.60, 0.70], index=1, format_func=lambda value: f"{value:.0%}")

    run_button = st.button("Build payoff profile", type="primary", use_container_width=True)

summary_tab, profile_tab, calculator_tab, scenarios_tab, methodology_tab = st.tabs(["Payoff Summary", "Payoff Profile", "Manual Calculator", "Scenario Table", "Methodology"])

with summary_tab:
    st.subheader("Inputs")
    if "phase4_company_inputs" not in st.session_state:
        st.session_state.phase4_company_inputs = default_company_inputs()

    stored_inputs = st.session_state.phase4_company_inputs.copy()
    manual_inputs = stored_inputs[["Ticker", "Implied volatility", "Polymarket YES price"]]
    edited_manual_inputs = st.data_editor(
        manual_inputs,
        num_rows="dynamic",
        use_container_width=True,
        hide_index=True,
        column_config={
            "Ticker": st.column_config.TextColumn(required=True),
            "Implied volatility": st.column_config.NumberColumn("Manual IV", min_value=0.0001, max_value=5.0, step=0.01),
            "Polymarket YES price": st.column_config.NumberColumn("Manual Polymarket YES", min_value=0.0, max_value=1.0, step=0.01),
        },
    )
    fallback_caps = stored_inputs.set_index("Ticker")["Current market cap"].to_dict()
    company_inputs = edited_manual_inputs.copy()
    company_inputs["Current market cap"] = company_inputs["Ticker"].map(fallback_caps).fillna(1_000_000_000_000.0)
    company_inputs = company_inputs[["Ticker", "Current market cap", "Implied volatility", "Polymarket YES price"]]
    st.session_state.phase4_company_inputs = company_inputs

    tickers = [ticker for ticker in company_inputs["Ticker"].astype(str).str.strip().tolist() if ticker]
    selected_ticker = st.selectbox("Selected Polymarket ticker", tickers, index=0)
    competitor_options = [ticker for ticker in tickers if ticker != selected_ticker]
    competitor_ticker = None
    if construction_mode == "single_competitor":
        competitor_ticker = st.selectbox("Single competitor for diagnostic option legs", competitor_options, index=0)

    default_entry = float(company_inputs.loc[company_inputs["Ticker"] == selected_ticker, "Polymarket YES price"].iloc[0])
    polymarket_entry_price = st.number_input("Polymarket entry price", min_value=0.0, max_value=1.0, value=default_entry, step=0.01, format="%.2f")

    st.caption("Boundary confidence affects expected payoff through the option strikes and premiums. If every option quantity is zero, expected payoff is just the Polymarket payoff and boundary confidence has no effect.")

    if run_button:
        with st.spinner("Running scenarios, constructing option candidates, and calculating payoff profile..."):
            try:
                market_caps = load_yahoo_market_caps(tuple(tickers))
                spots = load_spot_prices(tuple(tickers))
                simulation_inputs = apply_market_caps(company_inputs, market_caps)
                prices = load_adjusted_close(tuple(tickers), price_history_period)
                corr = build_correlation_matrix(correlation_method, prices, simulation_inputs, float(ewma_lambda), int(rolling_lookback), float(smooth_low_quantile), float(smooth_high_quantile))
                result = run_probability_engine(simulation_inputs, corr, days_to_target=int(days_to_target), simulations=int(simulations), seed=int(seed))
                current_caps = simulation_inputs.set_index("Ticker")["Current market cap"]
                boundaries = calculate_boundaries_for_all_tickers(result.terminal_market_caps, current_caps, [float(confidence_level)], ranks=result.ranks, n_bins=int(boundary_bins))
                spot_series = spots.set_index("ticker")["spot_price"]
                structure = construct_candidate_option_structure(boundaries, result.results, current_caps, spot_series, selected_ticker=selected_ticker, competitor_ticker=competitor_ticker, confidence_level=float(confidence_level), construction_mode=construction_mode, include_competitor_short_puts=bool(include_competitor_short_puts))
                valued_structure = attach_theoretical_premiums(structure, simulation_inputs.set_index("Ticker")["Implied volatility"], time_to_expiry=years(int(days_to_target)), risk_free_rate=float(risk_free_rate))
                valued_structure["Quantity"] = 0.0
                valid_strikes = valued_structure["Strike"].notna()
                valued_structure.loc[valid_strikes, "Quantity"] = float(default_option_quantity)
                st.session_state.phase4_inputs_used = simulation_inputs
                st.session_state.phase4_result = result
                st.session_state.phase4_spots = spots
                st.session_state.phase4_boundaries = boundaries
                st.session_state.phase4_option_legs = valued_structure
                st.session_state.phase4_selected_ticker = selected_ticker
                st.session_state.phase4_error = None
            except Exception as exc:
                st.session_state.phase4_error = str(exc)

    if st.session_state.get("phase4_error"):
        st.error(st.session_state.phase4_error)

    option_legs = st.session_state.get("phase4_option_legs")
    result = st.session_state.get("phase4_result")
    simulation_inputs = st.session_state.get("phase4_inputs_used")
    spots = st.session_state.get("phase4_spots")

    if option_legs is None or result is None or simulation_inputs is None or spots is None:
        st.info("Build the payoff profile to generate scenario-level payoff outputs.")
    else:
        st.subheader("Candidate option legs and quantities")
        st.caption("Quantity is editable and means number of contracts. These are construction-preview inputs, not optimized hedge ratios.")
        editable_legs = option_legs.copy()
        edited_view = st.data_editor(
            editable_option_legs_view(editable_legs),
            use_container_width=True,
            hide_index=True,
            column_config={"Quantity": st.column_config.NumberColumn("Quantity", step=0.01, format="%.2f"), "Strike": st.column_config.NumberColumn("Strike", format="$%.2f"), "Spot": st.column_config.NumberColumn("Spot", format="$%.2f"), "Theoretical premium": st.column_config.NumberColumn("Theoretical premium", format="$%.2f")},
            disabled=[column for column in editable_option_legs_view(editable_legs).columns if column != "Quantity"],
        )
        edited_legs = merge_edited_quantities(editable_legs, edited_view)
        st.session_state.phase4_option_legs = edited_legs

        selected = st.session_state.phase4_selected_ticker
        current_caps = simulation_inputs.set_index("Ticker")["Current market cap"]
        spot_series = spots.set_index("ticker")["spot_price"]
        try:
            scenario = calculate_scenario_payoffs(result.terminal_market_caps, result.ranks, current_caps, spot_series, edited_legs, selected_ticker=selected, polymarket_side=polymarket_side, polymarket_entry_price=float(polymarket_entry_price), polymarket_quantity=float(polymarket_quantity), contract_multiplier=float(contract_multiplier), include_option_premiums=bool(include_option_premiums))
            profile = selected_payoff_profile_bins(scenario, result.terminal_market_caps, current_caps, selected_ticker=selected, bins=int(profile_bins))
            st.session_state.phase4_scenario_payoffs = scenario
            st.session_state.phase4_profile = profile

            summary = payoff_summary(scenario)
            cols = st.columns(6)
            cols[0].metric("Expected payoff", dollars(float(summary["Expected payoff"])))
            cols[1].metric("Payoff SD", dollars(float(summary["Payoff standard deviation"])))
            cols[2].metric("Median payoff", dollars(float(summary["Median payoff"])))
            cols[3].metric("P(loss)", pct(float(summary["Probability of loss"])))
            cols[4].metric("Expected shortfall 5%", dollars(float(summary["Expected shortfall 5%"])))
            cols[5].metric("Worst payoff", dollars(float(summary["Worst payoff"])))

            st.subheader("Risk metrics")
            st.dataframe(display_risk_summary(summary), use_container_width=True, hide_index=True)

            st.subheader("Payoff components")
            component_summary = scenario[["Polymarket payoff", "Option payoff", "Total payoff"]].mean().to_frame("Expected payoff").reset_index().rename(columns={"index": "Component"})
            component_summary["Expected payoff"] = component_summary["Expected payoff"].map(dollars)
            st.dataframe(component_summary, use_container_width=True, hide_index=True)
        except Exception as exc:
            st.error(str(exc))

        with st.expander("Option legs used"):
            st.dataframe(display_option_legs(edited_legs), use_container_width=True, hide_index=True)

with profile_tab:
    profile = st.session_state.get("phase4_profile")
    if profile is None:
        st.info("Build the payoff profile first.")
    else:
        selected = st.session_state.phase4_selected_ticker
        st.plotly_chart(payoff_profile_figure(profile, selected), use_container_width=True)
        st.plotly_chart(payoff_by_bin_figure(profile, selected), use_container_width=True)
        st.subheader("Probability-weighted payoff bins")
        st.write("Each row is a selected-ticker terminal price/cap zone. Scenario probability times average payoff gives that bin's contribution to total expected payoff. The contributions sum to the global expected payoff.")
        st.dataframe(display_profile(profile), use_container_width=True, hide_index=True)

with calculator_tab:
    manual_option_calculator(st.session_state.get("phase4_option_legs"), st.session_state.get("phase4_profile"), polymarket_side, float(polymarket_entry_price), float(polymarket_quantity))

with scenarios_tab:
    scenario = st.session_state.get("phase4_scenario_payoffs")
    if scenario is None:
        st.info("Build the payoff profile first.")
    else:
        st.subheader("Scenario-level payoff sample")
        st.dataframe(display_scenarios(scenario.head(500)), use_container_width=True, hide_index=True)
        st.caption("Showing first 500 simulated scenarios only.")

with methodology_tab:
    st.subheader("Methodology")
    st.markdown(
        """
Phase 4 evaluates payoff, but does not optimize anything.

Workflow:

- Run the same Monte Carlo scenario engine as Phase 1.
- Recalculate Phase 2 boundaries for the selected confidence level.
- Construct Phase 3 candidate option legs.
- Assign a default manual quantity to valid option legs so the payoff preview is active.
- Let the user edit option quantities manually.
- Calculate Polymarket payoff in each scenario.
- Convert terminal market caps into terminal stock prices and calculate option payoff in each scenario.
- Add Polymarket payoff and option payoff into total scenario payoff.
- Bin scenarios by the selected ticker's terminal market-cap ratio.
- Calculate scenario probability, conditional win probability, average payoff, payoff dispersion, and weighted contribution in each bin.

The Manual Calculator tab uses manual option-payoff inputs and then looks up the nearest Monte Carlo bin from the Phase 4 payoff profile. It is an intuition tool, not a separate pricing engine.

Expected payoff bridge:

```text
Global expected payoff = sum(bin scenario probability * average payoff in bin)
```

Payoff standard deviation:

```text
Payoff SD = sqrt(sum(probability_scenario * (payoff_scenario - expected payoff)^2))
```

Quantity versus multiplier:

- Quantity is the number of option contracts for a leg.
- The option contract multiplier is shares per contract, usually 100 for listed US equity options.
- One full listed option contract can be much larger than a small Polymarket position, so fractional quantities are allowed here as analytical preview sizing.

Premium versus payoff dispersion:

- Option premium shifts expected payoff because it is paid or received up front.
- Premium alone does not eliminate scenario dispersion.
- Deep out-of-the-money options can still create large tail payoffs when a rare simulated scenario crosses the strike.
- Short options can collect premium in most scenarios but create large losses in tail scenarios.
        """
    )
