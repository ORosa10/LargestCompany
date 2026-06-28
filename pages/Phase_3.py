from __future__ import annotations

from datetime import date, timedelta

import pandas as pd
import plotly.express as px
import streamlit as st

from boundaries import calculate_boundaries_for_all_tickers
from correlations import ewma_correlation, fetch_adjusted_close, rolling_correlation, smooth_vol_adjusted_correlation
from market_data import apply_market_caps, fetch_market_caps, fetch_spot_prices
from model import default_company_inputs, run_probability_engine
from option_construction import attach_theoretical_premiums, construct_candidate_option_structure, payoff_grid_for_leg


st.set_page_config(page_title="Phase 3", layout="wide")
st.title("Phase 3")
st.caption("Option Construction Engine. This phase constructs candidate vanilla option building blocks from Phase 2 probability boundaries. It does not optimize hedge ratios.")

CORRELATION_METHODS = [
    "EWMA historical correlation",
    "Vol-adjusted smooth correlation",
    "Rolling historical correlation",
]
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


def theoretical_cashflow(structure: pd.DataFrame) -> float:
    signs = structure["Premium direction"].map({"Credit": 1.0, "Debit": -1.0}).fillna(0.0)
    return float((structure["Theoretical premium"].astype(float) * signs).sum())


def build_correlation_matrix(
    method: str,
    prices: pd.DataFrame,
    simulation_inputs: pd.DataFrame,
    ewma_lambda: float,
    rolling_lookback: int,
    smooth_low_quantile: float,
    smooth_high_quantile: float,
) -> pd.DataFrame:
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


def display_structure(table: pd.DataFrame) -> pd.DataFrame:
    display = table.copy()
    display["Strike"] = display["Strike"].map(dollars)
    display["Boundary market cap"] = display["Boundary market cap"].map(dollars_trillions)
    display["Boundary / current cap"] = display["Boundary / current cap"].map(pct)
    display["Spot"] = display["Spot"].map(dollars)
    if "Model IV" in display.columns:
        display["Model IV"] = display["Model IV"].map(pct)
    if "Risk-free rate" in display.columns:
        display["Risk-free rate"] = display["Risk-free rate"].map(pct)
    if "Time to expiry" in display.columns:
        display["Time to expiry"] = display["Time to expiry"].map(lambda value: "" if pd.isna(value) else f"{value:.2f}y")
    if "Theoretical premium" in display.columns:
        display["Theoretical premium"] = display["Theoretical premium"].map(dollars)
    return display


def display_market_caps(market_caps: pd.DataFrame) -> pd.DataFrame:
    display = market_caps.copy().rename(columns={"ticker": "Ticker", "yahoo_ticker": "Yahoo ticker", "market_cap": "Market cap", "source": "Source"})
    display["Market cap"] = display["Market cap"].map(dollars_trillions)
    return display[["Ticker", "Yahoo ticker", "Market cap", "Source"]]


def display_spots(spots: pd.DataFrame) -> pd.DataFrame:
    display = spots.copy().rename(columns={"ticker": "Ticker", "yahoo_ticker": "Yahoo ticker", "spot_price": "Spot", "source": "Source"})
    display["Spot"] = display["Spot"].map(dollars)
    return display[["Ticker", "Yahoo ticker", "Spot", "Source"]]


with st.sidebar:
    st.header("Phase 3 controls")
    target_date = st.date_input("Target date / maturity", value=date.today() + timedelta(days=365), min_value=date.today() + timedelta(days=1))
    days_to_target = max((target_date - date.today()).days, 1)
    st.caption(f"Horizon: {days_to_target} days ({days_to_target / 365:.2f} years)")

    simulations = st.number_input("Monte Carlo simulations", min_value=2_000, max_value=1_000_000, value=100_000, step=10_000)
    seed = st.number_input("Random seed", min_value=0, value=42, step=1)
    n_bins = st.slider("Phase 2 market-cap quantile bins", min_value=10, max_value=100, value=30, step=5)
    confidence_level = st.selectbox("Boundary confidence level", CONFIDENCE_LEVELS, index=3, format_func=lambda value: f"{value:.0%}")

    st.header("Construction")
    construction_mode_label = st.selectbox("Option construction mode", list(CONSTRUCTION_MODE_LABELS), index=0)
    construction_mode = CONSTRUCTION_MODE_LABELS[construction_mode_label]
    include_competitor_short_puts = st.checkbox(
        "Include competitor short puts",
        value=construction_mode == "single_competitor",
        help="Competitor short puts are optional income legs, not pure protection. They can be useful diagnostics but may add unwanted exposure.",
    )

    st.header("Pricing")
    risk_free_rate = st.number_input(
        "Risk-free rate for theoretical premiums",
        min_value=0.0,
        max_value=0.20,
        value=0.04,
        step=0.005,
        format="%.3f",
    )

    st.header("Correlation")
    correlation_method = st.selectbox("Correlation method", CORRELATION_METHODS, index=0)
    price_history_period = st.selectbox("Yahoo price history", ["1y", "3y", "5y", "10y"], index=2)
    ewma_lambda = st.selectbox("EWMA lambda", [0.94, 0.97], index=1)
    rolling_lookback = st.selectbox("Rolling lookback days", [63, 126, 252, 504, 756], index=2)
    smooth_low_quantile = st.selectbox("Smooth low-vol bucket", [0.30, 0.40, 0.50], index=1, format_func=lambda value: f"{value:.0%}")
    smooth_high_quantile = st.selectbox("Smooth high-vol bucket", [0.50, 0.60, 0.70], index=1, format_func=lambda value: f"{value:.0%}")

    run_button = st.button("Construct option building blocks", type="primary", use_container_width=True)

construction_tab, payoff_tab, methodology_tab = st.tabs(["Boundary Strikes", "Standalone Payoffs", "Methodology"])

with construction_tab:
    st.subheader("Manual scenario inputs")
    st.write("Market caps and spot prices are refreshed from Yahoo Finance. IV and Polymarket prices remain manual assumptions. Phase 3 uses Phase 2 boundaries as strike anchors.")

    if "phase3_company_inputs" not in st.session_state:
        st.session_state.phase3_company_inputs = default_company_inputs()

    stored_inputs = st.session_state.phase3_company_inputs.copy()
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
    st.session_state.phase3_company_inputs = company_inputs

    if run_button:
        with st.spinner("Running Phase 2 boundaries and constructing option building blocks..."):
            try:
                tickers = [ticker for ticker in company_inputs["Ticker"].astype(str).str.strip().tolist() if ticker]
                market_caps = load_yahoo_market_caps(tuple(tickers))
                spots = load_spot_prices(tuple(tickers))
                simulation_inputs = apply_market_caps(company_inputs, market_caps)
                prices = load_adjusted_close(tuple(tickers), price_history_period)
                corr = build_correlation_matrix(
                    correlation_method,
                    prices,
                    simulation_inputs,
                    float(ewma_lambda),
                    int(rolling_lookback),
                    float(smooth_low_quantile),
                    float(smooth_high_quantile),
                )
                result = run_probability_engine(
                    simulation_inputs,
                    corr,
                    days_to_target=int(days_to_target),
                    simulations=int(simulations),
                    seed=int(seed),
                )
                current_caps = simulation_inputs.set_index("Ticker")["Current market cap"]
                all_boundaries = calculate_boundaries_for_all_tickers(
                    result.terminal_market_caps,
                    current_caps,
                    [float(confidence_level)],
                    ranks=result.ranks,
                    n_bins=int(n_bins),
                )
                st.session_state.phase3_inputs_used = simulation_inputs
                st.session_state.phase3_market_caps = market_caps
                st.session_state.phase3_spots = spots
                st.session_state.phase3_result = result
                st.session_state.phase3_boundaries = all_boundaries
                st.session_state.phase3_error = None
            except Exception as exc:
                st.session_state.phase3_error = str(exc)

    if st.session_state.get("phase3_error"):
        st.error(st.session_state.phase3_error)

    simulation_inputs = st.session_state.get("phase3_inputs_used")
    result = st.session_state.get("phase3_result")
    boundaries = st.session_state.get("phase3_boundaries")
    spots = st.session_state.get("phase3_spots")
    market_caps = st.session_state.get("phase3_market_caps")

    if simulation_inputs is None or result is None or boundaries is None or spots is None:
        st.info("Run Phase 3 construction to generate option building blocks.")
    else:
        tickers = simulation_inputs["Ticker"].tolist()
        selected_ticker = st.selectbox("Selected Polymarket ticker", tickers, index=0)
        competitor_options = [ticker for ticker in tickers if ticker != selected_ticker]
        competitor_ticker = None
        if construction_mode == "single_competitor":
            auto_competitor = result.results[result.results["Ticker"] != selected_ticker].sort_values("Model probability", ascending=False).iloc[0]["Ticker"]
            competitor_index = competitor_options.index(auto_competitor) if auto_competitor in competitor_options else 0
            competitor_ticker = st.selectbox("Competitor ticker", competitor_options, index=competitor_index)
        elif construction_mode == "full_universe":
            st.caption("Full universe mode adds competitor protection legs for every non-selected ticker, sorted by model win probability.")
        else:
            st.caption("Selected-only mode hedges only the underlying ticker behind the YES bet. It does not assume which competitor wins if the selected ticker loses.")

        current_caps = simulation_inputs.set_index("Ticker")["Current market cap"]
        spot_series = spots.set_index("ticker")["spot_price"]
        structure = construct_candidate_option_structure(
            boundaries,
            result.results,
            current_caps,
            spot_series,
            selected_ticker=selected_ticker,
            competitor_ticker=competitor_ticker,
            confidence_level=float(confidence_level),
            construction_mode=construction_mode,
            include_competitor_short_puts=bool(include_competitor_short_puts),
        )
        iv_series = simulation_inputs.set_index("Ticker")["Implied volatility"]
        valued_structure = attach_theoretical_premiums(
            structure,
            iv_series,
            time_to_expiry=years(int(days_to_target)),
            risk_free_rate=float(risk_free_rate),
        )
        st.session_state.phase3_structure = valued_structure

        st.subheader("Construction summary")
        debit_count = int((valued_structure["Premium direction"] == "Debit").sum())
        credit_count = int((valued_structure["Premium direction"] == "Credit").sum())
        net_cashflow = theoretical_cashflow(valued_structure)
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Selected ticker", selected_ticker)
        col2.metric("Mode", construction_mode_label)
        col3.metric("Option legs", f"{len(valued_structure)}")
        col4.metric("Net theoretical premium", dollars(net_cashflow), help="Positive means net credit. Negative means net debit. This ignores contract multipliers and hedge ratios.")
        st.caption(f"Premium mix: {credit_count} credit leg(s), {debit_count} debit leg(s). Values are per share/option model unit before contract multipliers and before any quantity optimization.")

        st.subheader("Suggested option structure")
        st.dataframe(display_structure(valued_structure), use_container_width=True, hide_index=True)

        st.subheader("Interpretation")
        if construction_mode == "selected_only":
            st.write(
                "Selected-only mode is the cleanest hedge construction for a YES bet on one ticker finishing #1. It proposes option legs only on the selected stock, without pretending that one specific competitor is the only risk."
            )
        elif construction_mode == "single_competitor":
            st.write(
                "Single competitor mode is a diagnostic pair view. It helps inspect one threat, but it is not a complete hedge for a winner-takes-all universe market."
            )
        else:
            st.write(
                "Full universe mode lists candidate competitor protection legs for every rival. This is more complete, but it can become expensive and will need Phase 5 optimization before being treated as a portfolio."
            )
        st.caption(
            "Theoretical premiums use simplified Black-Scholes with one fixed IV per ticker. They are not live option quotes, and this is still not an optimized hedge package."
        )

        with st.expander("Phase 2 boundaries used"):
            boundary_display = boundaries.copy()
            for column in ["Lower loss boundary", "Upper win boundary"]:
                boundary_display[column] = boundary_display[column].map(dollars_trillions)
            for column in ["Confidence level", "Lower loss boundary / current", "Upper win boundary / current"]:
                boundary_display[column] = boundary_display[column].map(pct)
            st.dataframe(boundary_display, use_container_width=True, hide_index=True)

        with st.expander("Yahoo spot prices used"):
            st.dataframe(display_spots(spots), use_container_width=True, hide_index=True)

        with st.expander("Yahoo market caps used"):
            st.dataframe(display_market_caps(market_caps), use_container_width=True, hide_index=True)

with payoff_tab:
    st.subheader("Standalone option payoff functions")
    structure = st.session_state.get("phase3_structure")
    if structure is None or structure.empty:
        st.info("Construct option building blocks first.")
    else:
        include_premium = st.toggle("Include theoretical premium", value=True)
        selected_instrument = st.selectbox("Instrument", structure["Instrument"].tolist())
        leg = structure[structure["Instrument"] == selected_instrument].iloc[0]
        payoff = payoff_grid_for_leg(leg, premium=None if include_premium else 0.0)
        fig = px.line(
            payoff,
            x="Terminal price",
            y="Payoff",
            title=f"Standalone payoff: {selected_instrument}",
            labels={"Terminal price": "Underlying terminal price", "Payoff": "Option payoff before strategy combination"},
        )
        fig.add_hline(y=0.0, line_dash="dot")
        fig.add_vline(x=float(leg["Strike"]), line_dash="dash", annotation_text="strike")
        fig.add_vline(x=float(leg["Spot"]), line_dash="dot", annotation_text="spot")
        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(payoff, use_container_width=True, hide_index=True)

with methodology_tab:
    st.subheader("Methodology")
    st.write("Phase 3 translates Phase 2 market-cap boundaries into option strikes. It does not choose quantities, combine legs, optimize hedge ratios, or calculate a full portfolio payoff surface.")
    st.markdown(
        """
Construction modes:

- Selected-only hedge: Short Call and Long Put on the ticker behind the YES bet. This is the clean default because it does not assume which competitor wins if the selected ticker loses.
- Selected + single competitor diagnostic: adds candidate Long Call and optional Short Put legs for one chosen rival. This is useful for pairwise threat analysis, not a full universe hedge.
- Selected + full universe competitors: adds competitor protection legs for every non-selected ticker. This is more complete but likely too broad until Phase 5 optimization chooses quantities and filters.

Core selected-ticker rules:

- Selected ticker upper win boundary -> Short Call
- Selected ticker lower loss boundary -> Long Put

Competitor extension rules:

- Competitor upper win boundary -> Long Call
- Competitor lower loss boundary -> optional Short Put

Market-cap boundaries are converted to stock-price strikes with:

```text
strike = spot price * boundary market cap / current market cap
```

Theoretical premiums are diagnostic only. They use simplified Black-Scholes assumptions:

- current spot price from Yahoo Finance
- strike from the probability boundary conversion above
- target-date maturity from the sidebar
- one manual ticker-level IV from the input table
- one simplified risk-free rate from the sidebar
- no dividends and no volatility smile/surface yet

The output is a set of option building blocks. Phase 4 will combine these with Polymarket payoff surfaces. Phase 5 will optimize quantities and strike offsets.
        """
    )
