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
from option_construction import option_payoff
from option_valuation import reprice_option_legs
from payoff_surface import polymarket_payoff, selected_payoff_profile_bins, terminal_stock_prices, winner_from_ranks
from phase4_ui import display_profile, dollars, pct
from robust_optimizer import aligned_profile_figure, price_bin_profile, render_profile_trace_controls, render_robust_optimizer
from simulation_store import load_phase_artifact, load_simulation_snapshot

NORMALIZED_SPOT = 100.0
OPTION_QUANTITY_MULTIPLIER = 1.0
DEFAULT_MANUAL_QUANTITY = 1.0
PORTFOLIO_SLOTS = ["A", "B", "C", "D", "E"]


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


def slot_metrics_row(slot: str, rows: list[dict], metrics: pd.Series) -> dict:
    ev = float(metrics["Expected payoff"])
    sd = float(metrics["Payoff standard deviation"])
    active_legs = sum(1 for row in rows if bool(row.get("active", False)))
    return {
        "Slot": slot,
        "Active legs": active_legs,
        "Expected payoff": dollars(ev),
        "Payoff SD": dollars(sd),
        "EV / SD": f"{ev / sd:.3f}" if sd > 0 else "n/a",
        "Median payoff": dollars(float(metrics["Median payoff"])),
        "P(loss)": pct(float(metrics["Probability of loss"])),
        "Expected shortfall 5%": dollars(float(metrics["Expected shortfall 5%"])),
        "Worst payoff": dollars(float(metrics["Worst payoff"])),
    }


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


def display_contribution_table(table: pd.DataFrame) -> pd.DataFrame:
    display = table.copy()
    money_columns = [
        column for column in display.columns
        if column not in {"Price bin", "Scenario probability"}
    ]
    for column in money_columns:
        display[column] = display[column].map(dollars)
    display["Scenario probability"] = display["Scenario probability"].map(pct)
    return display


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


def contribution_stacked_figure(table: pd.DataFrame, axis_ticker: str) -> go.Figure:
    fig = go.Figure()
    if table.empty:
        return fig
    x = table["Price bin"]
    leg_columns = [
        column for column in table.columns
        if column not in {"Price bin", "Scenario probability", "Total option payoff"}
    ]
    for column in leg_columns:
        fig.add_bar(
            x=x,
            y=table[column],
            name=column,
            hovertemplate="%{x}<br>%{fullData.name}<br>%{y:$,.2f}<extra></extra>",
        )
    fig.add_trace(
        go.Scatter(
            x=x,
            y=table["Total option payoff"],
            name="Total option payoff",
            mode="lines+markers",
            line=dict(color="#111827", width=2),
            hovertemplate="%{x}<br>Total option payoff<br>%{y:$,.2f}<extra></extra>",
        )
    )
    fig.add_hline(y=0, line_dash="dash", line_color="#111827")
    fig.update_layout(
        title=f"Leg contribution by {axis_ticker} bin",
        barmode="relative",
        height=520,
        yaxis_title="Average option payoff contribution",
        xaxis_title=f"{axis_ticker} terminal stock price / current price",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        margin=dict(t=110, r=40, b=90, l=80),
    )
    fig.update_xaxes(tickangle=-45)
    return fig


def manual_diagnostic_figure(base, total, terminal_prices, name: str = "Manual portfolio", trace_visibility: dict | None = None, axis_ticker: str | None = None) -> go.Figure:
    base_profile = price_bin_profile(terminal_prices, base, bin_width=5.0)
    portfolio_profile = price_bin_profile(terminal_prices, total, bin_width=5.0)
    fig = aligned_profile_figure(base_profile, portfolio_profile, trace_visibility)

    rename_map = {
        "Optimizer 2 mean": f"{name} mean",
        "Optimizer 2 P5": f"{name} P5",
        "Optimizer 2 P1": f"{name} P1",
    }
    for trace in fig.data:
        if trace.name in rename_map:
            trace.name = rename_map[trace.name]

    show_mean_sd = trace_visibility is None or bool(trace_visibility.get("Portfolio mean - SD", False))
    if show_mean_sd:
        fig.add_trace(
            go.Scatter(
                x=portfolio_profile["Price bin"],
                y=portfolio_profile["Expected payoff"] - portfolio_profile["Payoff SD"],
                name=f"{name} mean - SD",
                mode="lines+markers",
                line=dict(color="#7c3aed", dash="dash"),
            ),
            row=1,
            col=1,
        )
    axis_label = axis_ticker or "selected ticker"
    fig.update_xaxes(title_text=f"{axis_label} terminal stock price / current price", row=2, col=1)
    fig.update_layout(
        title=dict(text=f"{name} payoff by {axis_label} terminal-price bin", y=0.98),
        height=820,
        legend=dict(orientation="h", yanchor="bottom", y=1.12, xanchor="left", x=0),
        margin=dict(t=150, r=40, b=90, l=80),
    )
    return fig


def clear_leg_state(prefix: str | None = None) -> None:
    for key in list(st.session_state):
        text_key = str(key)
        if text_key.startswith("leg_") or (prefix and text_key.startswith(f"{prefix}_leg_")):
            del st.session_state[key]


def copy_rows(rows: list[dict]) -> list[dict]:
    return [dict(row) for row in rows]


def default_manual_rows(ticker: str, pricing_iv: float) -> list[dict]:
    rows = default_interactive_rows(ticker, pricing_iv)
    for row in rows:
        row["quantity"] = DEFAULT_MANUAL_QUANTITY
    return rows


def rows_to_manual_inputs(rows: list[dict]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "Active": row["active"],
                "Ticker": row["ticker"],
                "Option type": row["option_type"],
                "Position": row["position"],
                "Quantity": row["quantity"],
                "Strike source": "Manual strike",
                "Boundary confidence (%)": row["confidence_pct"],
                "Manual strike": row["strike"],
                "Pricing IV": row["pricing_iv"],
                "Definition mode": row["define_by"],
                "Boundary type": row["boundary_type"],
                "Implied confidence (%)": row["confidence_pct"],
            }
            for row in rows
        ]
    )


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


def leg_contribution_by_axis(legs: pd.DataFrame, normalized_prices: pd.DataFrame, axis_prices: pd.Series, *, bin_width: float = 5.0) -> pd.DataFrame:
    if legs.empty:
        return pd.DataFrame()
    prices = axis_prices.to_numpy(float)
    low = np.floor(np.quantile(prices, 0.01) / bin_width) * bin_width
    high = np.ceil(np.quantile(prices, 0.99) / bin_width) * bin_width
    edges = np.concatenate(([-np.inf], np.arange(low, high + 0.5 * bin_width, bin_width), [np.inf]))
    frame = pd.DataFrame({"Axis price": prices})
    frame["Price bin"] = pd.cut(frame["Axis price"], edges, include_lowest=True)

    for _, leg in legs.iterrows():
        ticker = str(leg["Ticker"])
        premium = float(leg["Theoretical premium"])
        payoff = option_payoff(
            str(leg["Option type"]),
            str(leg["Position"]),
            float(leg["Strike"]),
            normalized_prices[ticker].to_numpy(float),
            premium=premium,
        ) * float(leg["Quantity"]) * OPTION_QUANTITY_MULTIPLIER
        frame[str(leg["Instrument"])] = payoff

    rows = []
    leg_columns = [column for column in frame.columns if column not in {"Axis price", "Price bin"}]
    for interval, group in frame.groupby("Price bin", observed=True):
        label = f"<{interval.right:.0f}%" if not np.isfinite(interval.left) else (f">={interval.left:.0f}%" if not np.isfinite(interval.right) else f"{interval.left:.0f}-{interval.right:.0f}%")
        row = {"Price bin": label, "Scenario probability": len(group) / len(frame)}
        row["Total option payoff"] = float(group[leg_columns].sum(axis=1).mean())
        for column in leg_columns:
            row[column] = float(group[column].mean())
        rows.append(row)
    return pd.DataFrame(rows)


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
    if "GOOG" not in tickers and "GOOGL" in tickers:
        st.caption("Google appears as GOOGL in this model, not GOOG.")

    eligible = [ticker for ticker in tickers if float(probabilities.loc[ticker]) >= float(threshold)]
    for ticker in relevant:
        if ticker not in eligible:
            eligible.append(ticker)
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

    def evaluate_manual_rows(rows: list[dict]) -> tuple[pd.DataFrame, np.ndarray, pd.Series, pd.DataFrame]:
        manual_inputs = rows_to_manual_inputs(rows)
        legs = resolve_manual_option_legs(manual_inputs, pd.DataFrame(), time_to_expiry=time_to_expiry, risk_free_rate=risk_free_rate, normalized_spot=100.0)
        metadata = manual_inputs[manual_inputs["Active"]].reset_index(drop=True)
        if not legs.empty:
            legs["Strike source"] = metadata["Definition mode"].to_numpy()
            legs["Boundary used"] = metadata.apply(lambda row: f"{row['Implied confidence (%)']:.1f}% {row['Boundary type']}" if row["Definition mode"] == "Strike" else f"{row['Boundary confidence (%)']:.1f}% {row['Boundary type']}", axis=1).to_numpy()
            legs = reprice_option_legs(legs, fallback_ivs, forward_ratios=forward_ratios, time_to_expiry=time_to_expiry, risk_free_rate=risk_free_rate, use_surface=use_surface)
        option_payoff, analytics = manual_option_payoffs_and_analytics(legs, normalized_prices, contract_multiplier=OPTION_QUANTITY_MULTIPLIER, include_premiums=True)
        total = base + option_payoff
        return legs, option_payoff, payoff_metrics(total), analytics

    def render_manual_portfolio_workspace(name: str, state_key: str, chart_key_prefix: str, use_sliders: bool = False) -> None:
        default_iv = float(fallback_ivs.loc[selected])
        slots_key = f"{state_key}_saved_slots"
        active_slot_key = f"{state_key}_active_slot"
        loaded_slot_key = f"{state_key}_loaded_slot"
        if state_key not in st.session_state:
            st.session_state[state_key] = default_manual_rows(selected, default_iv)
        if slots_key not in st.session_state:
            st.session_state[slots_key] = {}
        if active_slot_key not in st.session_state:
            st.session_state[active_slot_key] = PORTFOLIO_SLOTS[0]
        saved_slots = st.session_state[slots_key]

        reset_col, phase4_col, slot_col, save_col = st.columns([1, 1, 1, 1])
        if reset_col.button("Reset portfolio", key=f"{chart_key_prefix}_reset"):
            st.session_state[state_key] = default_manual_rows(selected, default_iv)
            clear_leg_state(state_key)
            st.rerun()
        upstream_rows = phase4_rows(phase4, default_iv)
        if phase4_col.button("Load Phase 4 portfolio", disabled=upstream_rows is None, key=f"{chart_key_prefix}_load_phase4"):
            st.session_state[state_key] = upstream_rows
            clear_leg_state(state_key)
            st.rerun()

        slot_labels = [f"{slot} saved" if slot in saved_slots else f"{slot} empty" for slot in PORTFOLIO_SLOTS]
        selected_slot_label = slot_col.selectbox(
            "Layout slot",
            slot_labels,
            index=PORTFOLIO_SLOTS.index(st.session_state[active_slot_key]),
            key=f"{chart_key_prefix}_slot_select",
        )
        selected_slot = selected_slot_label.split()[0]
        st.session_state[active_slot_key] = selected_slot
        if st.session_state.get(loaded_slot_key) != selected_slot:
            st.session_state[loaded_slot_key] = selected_slot
            if selected_slot in saved_slots:
                st.session_state[state_key] = copy_rows(saved_slots[selected_slot])
                clear_leg_state(state_key)
                st.rerun()

        st.info("Option quantity sizing: 1 = one share-equivalent exposure; 100 = one standard listed option contract.")
        manual_inputs = render_interactive_leg_editor(
            tickers=eligible,
            curves=curves,
            default_ticker=selected,
            default_iv=default_iv,
            iv_by_ticker=fallback_ivs,
            normalized_spot=100.0,
            state_key=state_key,
            key_prefix=state_key,
            add_button_label="Add another option leg",
            auto_surface_tickers=surface_tickers,
            use_sliders=use_sliders,
        )
        if save_col.button(f"Save to {selected_slot}", key=f"{chart_key_prefix}_save_slot"):
            st.session_state[slots_key][selected_slot] = copy_rows(st.session_state[state_key])
            st.session_state[loaded_slot_key] = selected_slot
            st.rerun()

        try:
            legs, option_payoff, manual_metrics, analytics = evaluate_manual_rows(st.session_state[state_key])
            total = base + option_payoff
            profile = make_profile(result, current_caps, normalized_prices, winners, selected, total, option_payoff, base)
            st.dataframe(metrics_comparison(baseline_metrics, manual_metrics, name), width="stretch", hide_index=True)

            if saved_slots:
                comparison_rows = []
                for slot in PORTFOLIO_SLOTS:
                    if slot not in saved_slots:
                        continue
                    try:
                        _, _, slot_metrics, _ = evaluate_manual_rows(saved_slots[slot])
                        comparison_rows.append(slot_metrics_row(slot, saved_slots[slot], slot_metrics))
                    except Exception as slot_exc:
                        comparison_rows.append({"Slot": slot, "Active legs": "error", "Expected payoff": str(slot_exc)})
                st.subheader("Saved layout comparison")
                st.dataframe(pd.DataFrame(comparison_rows), width="stretch", hide_index=True)

            st.subheader("Live payoff profile")
            st.caption("Mean, mean minus SD, P5, P1, and scenario probability update with every manual leg.")
            control_cols = st.columns([1, 2])
            axis_options = [ticker for ticker in tickers if ticker in normalized_prices.columns]
            default_axis_index = axis_options.index(selected) if selected in axis_options else 0
            axis_ticker = control_cols[0].selectbox("Profile axis ticker", axis_options, index=default_axis_index, key=f"{chart_key_prefix}_axis_ticker")
            with control_cols[1]:
                trace_visibility = render_profile_trace_controls(chart_key_prefix, include_mean_sd=True)
            axis_prices = normalized_prices[axis_ticker]
            st.plotly_chart(manual_diagnostic_figure(base, total, axis_prices.to_numpy(float), name, trace_visibility, axis_ticker), width="stretch", key=f"{chart_key_prefix}_diagnostic")
            contribution = leg_contribution_by_axis(legs, normalized_prices, axis_prices)
            if not contribution.empty:
                with st.expander(f"Leg contribution by {axis_ticker} bin", expanded=True):
                    st.plotly_chart(contribution_stacked_figure(contribution, axis_ticker), width="stretch", key=f"{chart_key_prefix}_contribution_stack")
                    st.dataframe(display_contribution_table(contribution), width="stretch", hide_index=True)
            with st.expander("Payoff distribution and detailed bin table"):
                st.plotly_chart(distribution_figure(base, total, name), width="stretch", key=f"{chart_key_prefix}_distribution")
                st.dataframe(display_profile(profile), width="stretch", hide_index=True)
            st.subheader("Resolved portfolio")
            st.dataframe(display_legs(legs), width="stretch", hide_index=True)
            st.subheader("Standalone leg analytics")
            st.dataframe(analytics, width="stretch", hide_index=True)
        except Exception as exc:
            st.error(str(exc))

    manual_tab, manual2_tab, optimizer_tab, chain_tab, methodology_tab = st.tabs(["Manual Portfolio", "Manual Portfolio 2", "Optimizer 2", "Option Chain", "Methodology"])

    with manual_tab:
        render_manual_portfolio_workspace("Manual portfolio", "phase5_interactive_rows", "phase5_manual")

    with manual2_tab:
        render_manual_portfolio_workspace("Manual portfolio 2", "phase5_interactive_rows_2", "phase5_manual2", use_sliders=True)

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
- Phase 4 can seed either manual portfolio, but Phase 5 may change quantities and strikes.
- Candidate premiums use strike-specific surface IV when available. Distribution IV and pricing IV remain separate.
- Manual Portfolio, Manual Portfolio 2, and Optimizer 2 use option-share-equivalent quantities. Quantity 1 is one share-equivalent; quantity 100 is one standard listed option contract.
- Manual Portfolio and Manual Portfolio 2 can save A-E in-session layouts for fast payoff comparison across the same locked market scenarios.
- Optimizer 2 deducts execution cost, enforces EV and ES5 floors, and caps active option legs at five.
        """)
