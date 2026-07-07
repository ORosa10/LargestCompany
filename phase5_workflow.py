"""Phase 5 workflow locked to saved Phase 1-4 artifacts."""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from interactive_portfolio import (
    confidence_at_strike,
    default_interactive_rows,
    optimized_legs_to_interactive_rows,
    render_interactive_leg_editor,
)
from iv_surface_model import default_surface_nodes
from manual_portfolio import manual_option_payoffs_and_analytics, resolve_manual_option_legs
from optimization import build_candidate_option_universe, long_option_payoff_matrix, payoff_metrics
from option_valuation import reprice_option_legs
from payoff_surface import polymarket_payoff, selected_payoff_profile_bins, terminal_stock_prices, winner_from_ranks
from phase4_ui import display_profile, dollars, pct
from robust_optimizer import aligned_profile_figure, price_bin_profile, render_robust_optimizer
from simulation_store import load_phase_artifact, load_simulation_snapshot

NORMALIZED_SPOT = 100.0
OPTION_QUANTITY_MULTIPLIER = 1.0
DEFAULT_MANUAL_QUANTITY = 1.0


def matching_metadata(left: dict, right: dict) -> bool:
    return all(left.get(key) == right.get(key) for key in ["target_date", "days_to_target", "simulations", "seed"])


def metrics_comparison(baseline: pd.Series, portfolio: pd.Series, name: str) -> pd.DataFrame:
    rows = []
    for label, metrics in [("Polymarket only", baseline), (name, portfolio)]:
        ev = float(metrics["Expected payoff"])
        sd = float(metrics["Payoff standard deviation"])
        rows.append({
            "Portfolio": label,
            "Expected payoff": dollars(ev),
            "Payoff SD": dollars(sd),
            "EV / SD": f"{ev / sd:.3f}" if sd > 0 else "n/a",
            "Median payoff": dollars(float(metrics["Median payoff"])),
            "P(loss)": pct(float(metrics["Probability of loss"])),
            "Expected shortfall 5%": dollars(float(metrics["Expected shortfall 5%"])),
            "Worst payoff": dollars(float(metrics["Worst payoff"])),
        })
    return pd.DataFrame(rows)


def display_legs(legs: pd.DataFrame) -> pd.DataFrame:
    display = legs.copy()
    for column in ["Strike", "Spot", "Theoretical premium", "Execution cost estimate"]:
        if column in display.columns:
            display[column] = display[column].map(dollars)
    for column in ["Strike / spot", "Model IV", "Implied dividend yield"]:
        if column in display.columns:
            display[column] = display[column].map(pct)
    order = [
        "Instrument", "Ticker", "Option type", "Position", "Quantity", "Strike",
        "Strike / spot", "Strike source", "Boundary used", "Model IV", "IV source",
        "Theoretical premium", "Execution cost estimate",
    ]
    return display[[column for column in order if column in display.columns]]


def distribution_figure(baseline, portfolio, name) -> go.Figure:
    fig = go.Figure()
    fig.add_histogram(x=baseline, name="Polymarket only", opacity=0.55, nbinsx=80, histnorm="probability")
    fig.add_histogram(x=portfolio, name=name, opacity=0.55, nbinsx=80, histnorm="probability")
    fig.update_layout(
        title="Payoff distribution comparison", xaxis_title="Terminal payoff",
        yaxis_title="Scenario probability", barmode="overlay", yaxis_tickformat=".1%",
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
    )
    return fig


def manual_diagnostic_figure(base, total, terminal_prices) -> go.Figure:
    base_profile = price_bin_profile(terminal_prices, base, bin_width=5.0)
    portfolio_profile = price_bin_profile(terminal_prices, total, bin_width=5.0)
    fig = aligned_profile_figure(base_profile, portfolio_profile)
    fig.add_trace(
        go.Scatter(
            x=portfolio_profile["Price bin"],
            y=portfolio_profile["Expected payoff"] - portfolio_profile["Payoff SD"],
            name="Manual mean minus SD",
            mode="lines+markers",
            line=dict(color="#7c3aed", dash="dash"),
        ),
        row=1,
        col=1,
    )
    fig.update_layout(title="Manual portfolio payoff, stress lines, and scenario probability")
    return fig


def clear_leg_state() -> None:
    for key in list(st.session_state):
        if str(key).startswith("leg_"):
            del st.session_state[key]


def default_manual_rows(ticker: str, pricing_iv: float) -> list[dict]:
    rows = default_interactive_rows(ticker, pricing_iv)
    for row in rows:
        row["quantity"] = DEFAULT_MANUAL_QUANTITY
    return rows


def phase4_rows(phase4: dict | None, fallback_iv: float) -> list[dict] | None:
    if not phase4 or phase4.get("active_option_legs") is None:
        return None
    legs = phase4["active_option_legs"].copy()
    if legs.empty:
        return None
    legs["Strike"] = NORMALIZED_SPOT * legs["Strike"].astype(float) / legs["Spot"].astype(float)
    legs["Spot"] = NORMALIZED_SPOT
    return optimized_legs_to_interactive_rows(legs, fallback_iv)


def make_profile(result, current_caps, normalized_prices, winners, selected_ticker, total, option, base):
    scenario = pd.DataFrame({
        "Winner": winners,
        "Selected terminal market cap": result.terminal_market_caps[selected_ticker],
        "Selected terminal stock price": normalized_prices[selected_ticker],
        "Polymarket payoff": base,
        "Option payoff": option,
        "Total payoff": total,
    })
    return selected_payoff_profile_bins(
        scenario, result.terminal_market_caps, current_caps,
        selected_ticker=selected_ticker, bins=20,
    )


def render() -> None:
    st.set_page_config(page_title="Phase 5", layout="wide")
    st.title("Phase 5: Portfolio Design")
    st.caption("Manual construction and robust optimization using the locked outputs of Phases 1-4. Phase 5 never refreshes Yahoo data or reruns Monte Carlo.")

    snapshot = load_simulation_snapshot()
    phase2 = load_phase_artifact("phase2")
    phase3 = load_phase_artifact("phase3")
    phase4 = load_phase_artifact("phase4")
    if snapshot is None:
        st.error("No saved Phase 1 snapshot. Run Phase 1 first.")
        st.stop()
    run = snapshot.get("run_metadata") or {}
    for name, artifact in [("Phase 2", phase2), ("Phase 3", phase3)]:
        if artifact is None or not matching_metadata(run, artifact.get("run_metadata") or {}):
            st.error(f"{name} is missing or belongs to another Phase 1 run. Open the phases in order before Phase 5.")
            st.stop()

    result = snapshot["result"]
    inputs = snapshot["simulation_inputs"].copy()
    curves = phase2.get("curves") or {}
    tickers = result.terminal_market_caps.columns.astype(str).tolist()
    relevant = inputs.loc[inputs["Polymarket YES price"].astype(float) > 0, "Ticker"].astype(str).tolist()
    relevant = [ticker for ticker in relevant if ticker in curves]
    if not relevant:
        st.error("No relevant Phase 1 Polymarket outcomes have saved Phase 2 curves.")
        st.stop()

    current_caps = inputs.set_index("Ticker")["Current market cap"].astype(float)
    input_by_ticker = inputs.set_index("Ticker")
    normalized_spots = pd.Series(NORMALIZED_SPOT, index=tickers, dtype=float)
    normalized_prices = terminal_stock_prices(result.terminal_market_caps, current_caps, normalized_spots)
    winners = winner_from_ranks(result.ranks)
    probabilities = winners.value_counts(normalize=True).reindex(tickers, fill_value=0.0).astype(float)
    days = int(run.get("days_to_target", 1))
    time_to_expiry = max(days, 1) / 365.0
    risk_free_rate = float(phase3.get("risk_free_rate", 0.04))
    use_surface = bool(phase3.get("use_surface_pricing", False))
    fallback_ivs = input_by_ticker["Implied volatility"].astype(float)
    forward_ratios = input_by_ticker["Forward / spot"].astype(float) if "Forward / spot" in input_by_ticker.columns else None
    surface_tickers = set(default_surface_nodes()["Ticker"].astype(str)) if use_surface else set()

    default_selected = str((phase4 or {}).get("selected_ticker", phase3.get("selected_ticker", relevant[0])))
    if default_selected not in relevant:
        default_selected = relevant[0]

    with st.sidebar:
        st.header("Locked upstream position")
        selected = st.selectbox("Polymarket ticker", relevant, index=relevant.index(default_selected))
        side = st.radio("Side", ["YES", "NO"], horizontal=True)
        yes_price = float(input_by_ticker.loc[selected, "Polymarket YES price"])
        default_entry = yes_price if side == "YES" else 1.0 - yes_price
        entry = st.number_input(f"{selected} {side} entry price", 0.0, 1.0, default_entry, 0.001, format="%.3f")
        shares = st.number_input("Polymarket shares", 0.0, value=100.0, step=10.0)
        st.header("Hedge universe")
        threshold = st.number_input("Minimum Phase 1 P(#1)", 0.0, 1.0, 0.10, 0.01, format="%.2f")
        st.header("Candidate chain")
        grid_points = st.number_input("Strike grid points", 7, 61, 25, 2, help="Number of candidate strikes generated between the lower and upper terminal-price quantiles for every eligible ticker.")
        lower_q = st.number_input("Lower terminal-price quantile", 0.001, 0.25, 0.01, 0.005, format="%.3f", help="Lowest simulated terminal-price quantile used to start the candidate strike grid. 0.010 means the 1st percentile.")
        upper_q = st.number_input("Upper terminal-price quantile", 0.75, 0.999, 0.99, 0.005, format="%.3f", help="Highest simulated terminal-price quantile used to end the candidate strike grid. 0.990 means the 99th percentile.")
        quantity_step = st.number_input("Quantity step", 0.1, value=1.0, step=1.0, format="%.3f", help="Optimizer search increment in option-share equivalents. 1 = one share-equivalent; 100 = one standard listed option contract.")
        max_quantity = st.number_input("Maximum absolute quantity per leg", 0.0, value=100.0, step=10.0, format="%.3f", help="Maximum long or short option-share equivalents per leg. 100 equals one standard listed option contract.")
        max_total = st.number_input("Maximum total absolute quantity", 0.0, value=200.0, step=10.0)
        max_legs = st.number_input("Maximum active legs", 1, 5, 4, 1)
        optimization_paths = st.number_input("Stored paths used in search", min_value=min(500, len(result.terminal_market_caps)), max_value=len(result.terminal_market_caps), value=min(20_000, len(result.terminal_market_caps)), step=min(500, len(result.terminal_market_caps)))

    st.success(f"Frozen close snapshot | target {run.get('target_date', 'n/a')} | {len(result.terminal_market_caps):,} paths | Phase 2 curves loaded | pricing: {'strike-specific surface' if use_surface else 'Phase 1 ATM fallback'}")
    st.caption("Prices are normalized to 100. Option quantities are option-share equivalents: 1 = one share-equivalent; 100 = one standard listed option contract. No Yahoo request occurs on this page.")

    eligible = [ticker for ticker in tickers if float(probabilities.loc[ticker]) >= float(threshold)]
    if selected not in eligible:
        eligible.append(selected)
    eligible = sorted(set(eligible), key=lambda ticker: float(probabilities.loc[ticker]), reverse=True)
    eligible = [ticker for ticker in eligible if ticker in curves]

    base = polymarket_payoff(winners, selected_ticker=selected, side=side, entry_price=float(entry), quantity=float(shares)).to_numpy(float)
    baseline_metrics = payoff_metrics(base)

    candidate_tables, matrices = [], []
    for ticker in eligible:
        terminal = normalized_prices[ticker].to_numpy(float)
        lo, hi = np.quantile(terminal, [float(lower_q), float(upper_q)])
        multipliers = np.unique(np.append(np.linspace(lo / 100.0, hi / 100.0, int(grid_points)), 1.0))
        chain = build_candidate_option_universe(ticker=ticker, spot=100.0, volatility=float(fallback_ivs.loc[ticker]), time_to_expiry=time_to_expiry, risk_free_rate=risk_free_rate, strike_multipliers=multipliers, include_calls=True, include_puts=True)
        chain = reprice_option_legs(chain, fallback_ivs, forward_ratios=forward_ratios, time_to_expiry=time_to_expiry, risk_free_rate=risk_free_rate, use_surface=use_surface)
        labels = []
        for _, leg in chain.iterrows():
            boundary_type = "Win boundary" if leg["Option type"] == "Call" else "Loss boundary"
            confidence = confidence_at_strike(curves[ticker], float(leg["Strike"]), boundary_type=boundary_type, normalized_spot=100.0)
            labels.append(f"{confidence:.1%} {boundary_type.lower()}")
        chain["Boundary used"] = labels
        candidate_tables.append(chain)
        matrices.append(long_option_payoff_matrix(normalized_prices[ticker], chain, contract_multiplier=OPTION_QUANTITY_MULTIPLIER, include_premiums=True))
    candidates = pd.concat(candidate_tables, ignore_index=True)
    option_matrix = np.concatenate(matrices, axis=1)
    optimizer_candidates = candidates.copy()
    optimizer_candidates["Theoretical premium"] = optimizer_candidates["Theoretical premium"].astype(float) * OPTION_QUANTITY_MULTIPLIER
    optimizer_candidates["Premium unit"] = "per share-equivalent"

    manual_tab, optimizer_tab, chain_tab, methodology_tab = st.tabs(["Manual Portfolio", "Optimizer 2", "Option Chain", "Methodology"])

    with manual_tab:
        default_iv = float(fallback_ivs.loc[selected])
        if "phase5_interactive_rows" not in st.session_state:
            st.session_state.phase5_interactive_rows = default_manual_rows(selected, default_iv)
        a, b, _ = st.columns([1, 1, 3])
        if a.button("Reset portfolio"):
            st.session_state.phase5_interactive_rows = default_manual_rows(selected, default_iv)
            clear_leg_state()
            st.rerun()
        upstream_rows = phase4_rows(phase4, default_iv)
        if b.button("Load Phase 4 portfolio", disabled=upstream_rows is None):
            st.session_state.phase5_interactive_rows = upstream_rows
            clear_leg_state()
            st.rerun()
        st.info("Option quantity sizing: 1 = one share-equivalent exposure; 100 = one standard listed option contract.")
        manual_inputs = render_interactive_leg_editor(tickers=eligible, curves=curves, default_ticker=selected, default_iv=default_iv, iv_by_ticker=fallback_ivs, normalized_spot=100.0, auto_surface_tickers=surface_tickers)
        try:
            legs = resolve_manual_option_legs(manual_inputs, pd.DataFrame(), time_to_expiry=time_to_expiry, risk_free_rate=risk_free_rate, normalized_spot=100.0)
            metadata = manual_inputs[manual_inputs["Active"]].reset_index(drop=True)
            if not legs.empty:
                legs["Strike source"] = metadata["Definition mode"].to_numpy()
                legs["Boundary used"] = metadata.apply(lambda row: f"{row['Implied confidence (%)']:.1f}% {row['Boundary type']}" if row["Definition mode"] == "Strike" else f"{row['Boundary confidence (%)']:.1f}% {row['Boundary type']}", axis=1).to_numpy()
                legs = reprice_option_legs(legs, fallback_ivs, forward_ratios=forward_ratios, time_to_expiry=time_to_expiry, risk_free_rate=risk_free_rate, use_surface=use_surface)
            option_payoff, analytics = manual_option_payoffs_and_analytics(legs, normalized_prices, contract_multiplier=OPTION_QUANTITY_MULTIPLIER, include_premiums=True)
            total = base + option_payoff
            manual_metrics = payoff_metrics(total)
            profile = make_profile(result, current_caps, normalized_prices, winners, selected, total, option_payoff, base)
            st.dataframe(metrics_comparison(baseline_metrics, manual_metrics, "Manual portfolio"), width="stretch", hide_index=True)
            st.subheader("Live payoff profile")
            st.caption("Mean, mean minus SD, P5, P1, and scenario probability update with every manual leg.")
            st.plotly_chart(manual_diagnostic_figure(base, total, normalized_prices[selected].to_numpy(float)), width="stretch", key="phase5_manual_diagnostic_v2")
            with st.expander("Payoff distribution and detailed bin table"):
                st.plotly_chart(distribution_figure(base, total, "Manual portfolio"), width="stretch", key="phase5_manual_distribution")
                st.dataframe(display_profile(profile), width="stretch", hide_index=True)
            st.subheader("Resolved portfolio")
            st.dataframe(display_legs(legs), width="stretch", hide_index=True)
            st.subheader("Standalone leg analytics")
            st.dataframe(analytics, width="stretch", hide_index=True)
        except Exception as exc:
            st.error(str(exc))

    with optimizer_tab:
        render_robust_optimizer(base_payoff=base, option_payoff_matrix=option_matrix, candidates=optimizer_candidates, terminal_prices=normalized_prices[selected].to_numpy(float), quantity_min=-float(max_quantity), quantity_max=float(max_quantity), quantity_step=float(quantity_step), max_legs=int(max_legs), max_total_quantity=float(max_total), default_minimum_ev=float(baseline_metrics["Expected payoff"]), optimization_scenarios=int(optimization_paths), seed=int(run.get("seed", 42)))

    with chain_tab:
        st.subheader("Locked normalized candidate chain")
        filters = st.multiselect("Tickers", eligible, default=eligible)
        st.dataframe(display_legs(candidates[candidates["Ticker"].isin(filters)]), width="stretch", hide_index=True)

    with methodology_tab:
        st.markdown("""
Phase 5 is downstream-only:

- Phase 1 supplies the frozen joint scenarios, ranks, market caps, forward carry, and fallback distribution IV.
- Phase 2 supplies the saved conditional boundary curves; Phase 5 does not estimate them again.
- Phase 3 supplies the locked risk-free rate and decides whether the calibrated surface matches the target expiry.
- Phase 4 can seed the manual portfolio, but Phase 5 may change quantities and strikes.
- Candidate premiums use strike-specific surface IV when available. Distribution IV and pricing IV remain separate.
- Manual Portfolio and Optimizer 2 use option-share-equivalent quantities. Quantity 1 is one share-equivalent; quantity 100 is one standard listed option contract.
- Optimizer 2 deducts execution cost, enforces EV and ES5 floors, and caps active option legs at five.
        """)
