from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from execution_mapping import (
    choose_expiration,
    default_strike_step,
    fetch_option_chain_quotes_for_expiries,
    fetch_option_expirations,
    infer_listed_strike_step,
    map_normalized_legs,
)
from manual_portfolio import manual_option_payoffs_and_analytics
from market_data import fetch_spot_prices
from optimization import payoff_metrics
from payoff_surface import polymarket_payoff, terminal_stock_prices, winner_from_ranks
from phase5_workflow import distribution_figure
from robust_optimizer import render_profile_trace_controls
from simulation_store import load_phase_artifact, load_simulation_snapshot, save_phase_artifact

CONTRACT_MULTIPLIER = 100.0
PHASE5_AUTOSAVE_ARTIFACT = "phase5_manual_autosave"
PHASE6_CANDIDATE_ARTIFACT = "phase6_execution_candidate"

st.set_page_config(page_title="Phase 6", layout="wide")
st.title("Phase 6: execution mapping")
st.caption("Map the selected Phase 5 research portfolio into real spots, real listed strikes, editable real execution premiums, and execution-sized payoffs.")


def available_snapshot() -> dict | None:
    if st.session_state.get("last_result") is not None and st.session_state.get("last_simulation_inputs") is not None:
        return {
            "result": st.session_state.last_result,
            "simulation_inputs": st.session_state.last_simulation_inputs,
            "run_metadata": st.session_state.get("last_run") or {},
        }
    return load_simulation_snapshot()


def matching_metadata(left: dict, right: dict) -> bool:
    return all(left.get(key) == right.get(key) for key in ["target_date", "days_to_target", "simulations", "seed"])


def load_phase5_autosave(run_metadata: dict) -> dict:
    payload = load_phase_artifact(PHASE5_AUTOSAVE_ARTIFACT)
    if not isinstance(payload, dict):
        return {}
    if not matching_metadata(run_metadata, payload.get("run_metadata") or {}):
        return {}
    return payload


def target_from_metadata(metadata: dict) -> date:
    raw = metadata.get("target_date")
    if raw:
        try:
            return pd.Timestamp(raw).date()
        except Exception:
            pass
    return date.today() + timedelta(days=int(metadata.get("days_to_target", 365)))


def metric_table(base, mapped) -> pd.DataFrame:
    rows = []
    for name, values in [("Polymarket only", base), ("Phase 6 real-market portfolio", mapped)]:
        metrics = payoff_metrics(values)
        ev = float(metrics["Expected payoff"])
        sd = float(metrics["Payoff standard deviation"])
        rows.append({
            "Portfolio": name,
            "Expected payoff": f"${ev:,.2f}",
            "Payoff SD": f"${sd:,.2f}",
            "EV / SD": f"{ev / sd:.3f}" if sd else "n/a",
            "P(loss)": f"{metrics['Probability of loss']:.2%}",
            "Expected shortfall 5%": f"${metrics['Expected shortfall 5%']:,.2f}",
            "Worst payoff": f"${metrics['Worst payoff']:,.2f}",
        })
    return pd.DataFrame(rows)


def real_execution_legs(mapping: pd.DataFrame, original_legs: pd.DataFrame, *, time_to_expiry: float, risk_free_rate: float) -> pd.DataFrame:
    rows = []
    original = original_legs.reset_index(drop=True)
    active_mapping = mapping[mapping["Use"]].reset_index(drop=True)
    for _, mapped in active_mapping.iterrows():
        source = original.iloc[int(mapped["Leg"]) - 1]
        spot = float(mapped["Current spot"])
        strike = float(mapped["Executable strike"])
        premium = float(mapped["Execution premium"])
        rows.append({
            "Instrument": f"{mapped['Position']} {mapped['Ticker']} {mapped['Option type']} {strike:.2f}",
            "Ticker": str(mapped["Ticker"]),
            "Option type": str(mapped["Option type"]),
            "Position": str(mapped["Position"]),
            "Quantity": float(mapped["Quantity"]),
            "Strike": strike,
            "Strike / spot": strike / spot,
            "Strike source": "Phase 6 real listed strike",
            "Boundary used": source.get("Boundary used", "Mapped from Phase 5"),
            "Spot": spot,
            "Model IV": float(mapped.get("Model IV", np.nan)),
            "Risk-free rate": risk_free_rate,
            "Time to expiry": time_to_expiry,
            "Theoretical premium": premium,
            "Execution premium": premium,
            "Contract symbol": mapped.get("Contract symbol", ""),
        })
    return pd.DataFrame(rows)


def real_price_bin_profile(terminal_prices, payoffs, *, bin_width=5.0) -> pd.DataFrame:
    prices, values = np.asarray(terminal_prices, float), np.asarray(payoffs, float)
    low = np.floor(np.quantile(prices, 0.01) / bin_width) * bin_width
    high = np.ceil(np.quantile(prices, 0.99) / bin_width) * bin_width
    edges = np.concatenate(([-np.inf], np.arange(low, high + 0.5 * bin_width, bin_width), [np.inf]))
    frame = pd.DataFrame({"Terminal price": prices, "Payoff": values})
    frame["Price bin"] = pd.cut(frame["Terminal price"], edges, include_lowest=True)
    rows = []
    for interval, group in frame.groupby("Price bin", observed=True):
        x = group["Payoff"].to_numpy(float)
        if not np.isfinite(interval.left):
            label = f"<${interval.right:,.0f}"
        elif not np.isfinite(interval.right):
            label = f">=${interval.left:,.0f}"
        else:
            label = f"${interval.left:,.0f}-${interval.right:,.0f}"
        rows.append({
            "Price bin": label,
            "Price midpoint": group["Terminal price"].mean(),
            "Scenario probability": len(group) / len(frame),
            "Expected payoff": float(x.mean()),
            "Payoff SD": float(x.std(ddof=0)),
            "Payoff P1": float(np.quantile(x, 0.01)),
            "Payoff P5": float(np.quantile(x, 0.05)),
        })
    return pd.DataFrame(rows)


def strike_marker_rows(profile: pd.DataFrame, legs: pd.DataFrame, axis_ticker: str) -> pd.DataFrame:
    if legs.empty:
        return pd.DataFrame(columns=["Price bin", "Strikes"])
    axis_legs = legs[legs["Ticker"].astype(str) == str(axis_ticker)]
    if axis_legs.empty:
        return pd.DataFrame(columns=["Price bin", "Strikes"])
    rows = []
    for _, profile_row in profile.iterrows():
        label = str(profile_row["Price bin"])
        midpoint = float(profile_row["Price midpoint"])
        strikes = []
        for _, leg in axis_legs.iterrows():
            strike = float(leg["Strike"])
            if abs(strike - midpoint) <= 3.0:
                strikes.append(f"{strike:.2f}")
        if strikes:
            rows.append({"Price bin": label, "Strikes": ", ".join(dict.fromkeys(strikes))})
    return pd.DataFrame(rows)


def phase6_profile_figure(base, total, terminal_prices, name: str, trace_visibility: dict, axis_ticker: str, legs: pd.DataFrame) -> go.Figure:
    base_profile = real_price_bin_profile(terminal_prices, base, bin_width=5.0)
    portfolio_profile = real_price_bin_profile(terminal_prices, total, bin_width=5.0)
    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.12,
        row_heights=[0.72, 0.28],
        specs=[[{"secondary_y": False}], [{"secondary_y": False}]],
    )
    x = portfolio_profile["Price bin"]
    if trace_visibility.get("Polymarket-only mean", True):
        fig.add_trace(go.Scatter(x=base_profile["Price bin"], y=base_profile["Expected payoff"], name="Polymarket-only mean", mode="lines", line=dict(color="#94a3b8", dash="dash")), row=1, col=1)
    if trace_visibility.get("Portfolio mean", True):
        fig.add_trace(go.Bar(x=x, y=portfolio_profile["Expected payoff"], name=f"{name} mean", marker_color="#16a34a"), row=1, col=1)
    if trace_visibility.get("Portfolio P5", True):
        fig.add_trace(go.Scatter(x=x, y=portfolio_profile["Payoff P5"], name=f"{name} P5", mode="lines+markers", line=dict(color="#f59e0b", dash="dash")), row=1, col=1)
    if trace_visibility.get("Portfolio P1", False):
        fig.add_trace(go.Scatter(x=x, y=portfolio_profile["Payoff P1"], name=f"{name} P1", mode="lines+markers", line=dict(color="#ef4444", dash="dot")), row=1, col=1)
    if trace_visibility.get("Portfolio mean - SD", True):
        fig.add_trace(go.Scatter(x=x, y=portfolio_profile["Expected payoff"] - portfolio_profile["Payoff SD"], name=f"{name} mean - SD", mode="lines+markers", line=dict(color="#7c3aed", dash="dash")), row=1, col=1)
    markers = strike_marker_rows(portfolio_profile, legs, axis_ticker)
    if not markers.empty:
        fig.add_trace(go.Scatter(x=markers["Price bin"], y=np.zeros(len(markers)), name=f"{axis_ticker} listed strikes", mode="markers+text", text=markers["Strikes"], textposition="top center", marker=dict(symbol="diamond", size=8, color="#111827"), hovertemplate="%{x}<br>Strikes: %{text}<extra></extra>"), row=1, col=1)
    if trace_visibility.get("Scenario probability", True):
        fig.add_trace(go.Bar(x=x, y=portfolio_profile["Scenario probability"], name="Scenario probability", marker_color="#60a5fa"), row=2, col=1)
    fig.add_hline(y=0, line_dash="dash", line_color="#111827", row=1, col=1)
    fig.update_yaxes(title_text="Payoff", row=1, col=1)
    fig.update_yaxes(title_text="Probability", tickformat=".1%", row=2, col=1)
    fig.update_xaxes(title_text=f"{axis_ticker} terminal stock price", row=2, col=1, tickangle=-45)
    fig.update_layout(title=f"{name} payoff by {axis_ticker} real terminal-price bin", height=850, legend=dict(orientation="h", yanchor="bottom", y=1.08, xanchor="left", x=0), margin=dict(t=130, r=40, b=110, l=80))
    return fig


def order_of_magnitude_polymarket_shares(mapping: pd.DataFrame) -> float:
    active = mapping[mapping["Use"]] if "Use" in mapping.columns else mapping
    quantities = pd.to_numeric(active.get("Quantity", pd.Series(dtype=float)), errors="coerce").abs().replace([np.inf, -np.inf], np.nan).dropna()
    if quantities.empty:
        return 100.0
    return float(max(100.0, np.ceil(float(quantities.max())) * CONTRACT_MULTIPLIER))


snapshot = available_snapshot()
if snapshot is None:
    st.error("No stored Monte Carlo snapshot is available. Run Phase 1 first.")
    st.stop()

portfolio_sources = {}
manual_legs = st.session_state.get("phase5_manual_legs")
manual_total = st.session_state.get("phase5_manual_total_payoff")
if manual_legs is not None and not manual_legs.empty and manual_total is not None:
    portfolio_sources["Phase 5 manual portfolio"] = (manual_legs.copy(), np.asarray(manual_total, dtype=float))
manual2_legs = st.session_state.get("phase5_manual2_legs")
manual2_total = st.session_state.get("phase5_manual2_total_payoff")
if manual2_legs is not None and not manual2_legs.empty and manual2_total is not None:
    portfolio_sources["Phase 5 manual portfolio 2"] = (manual2_legs.copy(), np.asarray(manual2_total, dtype=float))
robust_result = st.session_state.get("phase5_robust_optimization")
if robust_result is not None and not robust_result.selected_legs.empty:
    portfolio_sources["Phase 5 Optimizer 2"] = (robust_result.selected_legs.copy(), np.asarray(robust_result.payoffs, dtype=float))

if not portfolio_sources:
    st.error("Build a Manual Portfolio or run Optimizer 2 in Phase 5, then return here in the same Streamlit session.")
    st.stop()

source = st.radio("Phase 5 portfolio source", list(portfolio_sources), horizontal=True)
original_legs, original_total_payoff = portfolio_sources[source]
tickers = sorted(original_legs["Ticker"].astype(str).unique())
metadata = snapshot.get("run_metadata") or {}
target_date = target_from_metadata(metadata)
days_to_target = max((target_date - date.today()).days, 1)
time_to_target = days_to_target / 365.0
risk_free_rate = float(original_legs.get("Risk-free rate", pd.Series([0.04])).iloc[0])
result = snapshot["result"]
inputs = snapshot["simulation_inputs"].copy()
current_caps = inputs.set_index("Ticker")["Current market cap"].astype(float)
phase5_sidebar = (load_phase5_autosave(metadata).get("sidebar") or {})

st.success(f"Loaded {source}: {len(original_legs)} option legs across {len(tickers)} ticker(s). Event target: {target_date}.")
st.info("Phase 6 sizing: option quantity is listed contracts; option premium is still per share; option payoff is contracts x 100 shares x per-share payoff.")

fetch_left, fetch_right, _ = st.columns([1, 1, 3])
if fetch_left.button("Fetch current spots", type="primary"):
    try:
        st.session_state.phase6_spots = fetch_spot_prices(tickers)
        st.session_state.phase6_spot_error = None
    except Exception as exc:
        st.session_state.phase6_spot_error = str(exc)
if fetch_right.button("Fetch listed expirations"):
    try:
        st.session_state.phase6_expirations = fetch_option_expirations(tickers)
        st.session_state.phase6_expiry_error = None
    except Exception as exc:
        st.session_state.phase6_expiry_error = str(exc)
if st.session_state.get("phase6_spot_error"):
    st.error(st.session_state.phase6_spot_error)
if st.session_state.get("phase6_expiry_error"):
    st.error(st.session_state.phase6_expiry_error)

spot_data = st.session_state.get("phase6_spots")
if spot_data is None or not set(tickers).issubset(set(spot_data["ticker"].astype(str))):
    spot_table = pd.DataFrame({"Ticker": tickers, "Current spot": [100.0] * len(tickers), "Strike step": [1.0] * len(tickers), "Source": ["Placeholder - fetch or edit"] * len(tickers)})
else:
    lookup = spot_data.set_index("ticker")
    spot_table = pd.DataFrame({
        "Ticker": tickers,
        "Current spot": [float(lookup.loc[ticker, "spot_price"]) for ticker in tickers],
        "Strike step": [default_strike_step(float(lookup.loc[ticker, "spot_price"])) for ticker in tickers],
        "Source": [str(lookup.loc[ticker, "source"]) for ticker in tickers],
    })

st.subheader("Current underlyings and provisional strike grids")
edited_spots = st.data_editor(
    spot_table,
    use_container_width=True,
    hide_index=True,
    key=f"phase6_spots_{source}",
    column_config={
        "Ticker": st.column_config.TextColumn(disabled=True),
        "Current spot": st.column_config.NumberColumn(min_value=0.01, step=0.01, format="$%.2f"),
        "Strike step": st.column_config.NumberColumn(min_value=0.01, step=0.5, format="%.2f"),
        "Source": st.column_config.TextColumn(disabled=True),
    },
)
spot_lookup = edited_spots.set_index("Ticker")["Current spot"].astype(float)
step_lookup = edited_spots.set_index("Ticker")["Strike step"].astype(float)

st.subheader("Expiration alignment")
policy = st.selectbox(
    "Expiration policy",
    ["Last Friday before month end", "First expiry on/after target", "Nearest listed expiry", "Last expiry on/before target"],
    help="Default Phase 6 execution policy: use the listed expiry closest to the last Friday before the event month-end, falling back to the nearest listed expiry before that date if needed.",
)
expirations = st.session_state.get("phase6_expirations") or {}
expiry_rows = []
for ticker in tickers:
    listed = expirations.get(ticker, [])
    selected_expiry = choose_expiration(listed, target_date, policy)
    gap = (selected_expiry - target_date).days if selected_expiry else np.nan
    expiry_rows.append({
        "Ticker": ticker,
        "Event target": target_date,
        "Selected option expiry": selected_expiry,
        "Gap days": gap,
        "Listed expirations found": len(listed),
        "Treatment": "Month-end Friday execution mapping" if policy == "Last Friday before month end" else ("Event-date mark-to-market" if pd.notna(gap) and gap > 0 else ("Close package at option expiry" if pd.notna(gap) and gap < 0 else "Intrinsic at aligned expiry")),
    })
expiry_table = pd.DataFrame(expiry_rows)
st.dataframe(expiry_table, use_container_width=True, hide_index=True)
if expiry_table["Gap days"].isna().any():
    st.warning("Some expiration calendars are not loaded. Fetch listed expirations before relying on the mapped chains.")
elif policy == "Last Friday before month end":
    st.info("Phase 6 is using the last listed Friday before the target month-end as the execution expiry. Gap days only shows distance versus the event target date.")
elif (expiry_table["Gap days"] > 0).any():
    st.info("Selected expiry is after the event target, so residual option value remains at the event date.")
if (expiry_table["Gap days"].abs() > 7).any():
    st.warning("At least one option expiry differs from the event by more than seven days. Phase 7 should handle the residual event/expiry timing risk explicitly.")

selected_expiry_by_ticker = {
    str(row["Ticker"]): row["Selected option expiry"]
    for _, row in expiry_table.dropna(subset=["Selected option expiry"]).iterrows()
}
chain_left, chain_right, _ = st.columns([1, 1, 3])
if chain_left.button("Fetch selected option chains", disabled=not bool(selected_expiry_by_ticker)):
    try:
        st.session_state.phase6_option_chains = fetch_option_chain_quotes_for_expiries(selected_expiry_by_ticker)
        st.session_state.phase6_chain_error = None
    except Exception as exc:
        st.session_state.phase6_chain_error = str(exc)
if chain_right.button("Clear option chains"):
    st.session_state.phase6_option_chains = {}
if st.session_state.get("phase6_chain_error"):
    st.error(st.session_state.phase6_chain_error)

option_chains = st.session_state.get("phase6_option_chains") or {}
if option_chains:
    spacing_rows = []
    for ticker, chain in option_chains.items():
        spot = float(spot_lookup.loc[ticker]) if ticker in spot_lookup.index else np.nan
        spacing_rows.append({
            "Ticker": ticker,
            "Expiry": str(chain["Expiry"].iloc[0]) if "Expiry" in chain.columns and not chain.empty else "",
            "Listed strikes": int(chain["strike"].nunique()) if "strike" in chain.columns else 0,
            "Inferred local strike step": infer_listed_strike_step(chain["strike"], spot),
            "Min strike": float(chain["strike"].min()),
            "Max strike": float(chain["strike"].max()),
        })
    st.subheader("Listed chain reality check")
    st.dataframe(pd.DataFrame(spacing_rows), use_container_width=True, hide_index=True)

mapping = map_normalized_legs(original_legs, spot_lookup, step_lookup, option_chains=option_chains)
pricing_basis = st.radio(
    "Default execution premium basis",
    ["Bid/ask mid", "Conservative executable"],
    horizontal=True,
    disabled=not bool(option_chains),
    help="Conservative executable uses ask for long legs and bid for short legs. You can still override each premium manually below.",
)
if "Execution premium" not in mapping.columns:
    mapping["Execution premium"] = np.nan
if option_chains:
    if pricing_basis == "Conservative executable" and "Conservative premium normalized" in mapping.columns:
        mapping["Reference premium"] = mapping["Conservative premium normalized"] * mapping["Current spot"] / 100.0
    elif "Mid" in mapping.columns:
        mapping["Reference premium"] = mapping["Mid"]
    else:
        mapping["Reference premium"] = np.nan
else:
    original_premiums = pd.to_numeric(original_legs.reset_index(drop=True).get("Theoretical premium"), errors="coerce")
    mapping["Reference premium"] = original_premiums.to_numpy(float) * mapping["Current spot"] / 100.0
mapping["Execution premium"] = mapping["Reference premium"]

st.subheader("Real-strike portfolio mapping")
st.caption("Execution premium is the real per-share option quote. Edit it to the expected fill. Quantity is listed contracts; Phase 6 multiplies option payoff by 100 shares per contract.")
edited_mapping = st.data_editor(
    mapping,
    use_container_width=True,
    hide_index=True,
    key=f"phase6_mapping_{source}",
    column_config={
        "Leg": st.column_config.NumberColumn(disabled=True),
        "Use": st.column_config.CheckboxColumn(),
        "Ticker": st.column_config.TextColumn(disabled=True),
        "Option type": st.column_config.TextColumn(disabled=True),
        "Position": st.column_config.TextColumn(disabled=True),
        "Quantity": st.column_config.NumberColumn(disabled=True, format="%.3f", help="Listed option contracts. 1 = one contract = 100 shares."),
        "Phase 5 normalized strike": st.column_config.NumberColumn(disabled=True, format="%.2f"),
        "Current spot": st.column_config.NumberColumn(disabled=True, format="$%.2f"),
        "Raw real strike": st.column_config.NumberColumn(disabled=True, format="$%.2f"),
        "Strike step": st.column_config.NumberColumn(disabled=True, format="%.2f"),
        "Executable strike": st.column_config.NumberColumn(min_value=0.01, step=0.5, format="$%.2f"),
        "Listed strike": st.column_config.NumberColumn(disabled=True, format="$%.2f"),
        "Mapped normalized strike": None,
        "Strike mapping error": st.column_config.NumberColumn(disabled=True, format="%+.2f"),
        "Bid": st.column_config.NumberColumn(disabled=True, format="$%.2f"),
        "Ask": st.column_config.NumberColumn(disabled=True, format="$%.2f"),
        "Mid": st.column_config.NumberColumn(disabled=True, format="$%.2f"),
        "Bid/ask spread": st.column_config.NumberColumn(disabled=True, format="$%.2f"),
        "Reference premium": st.column_config.NumberColumn(disabled=True, format="$%.2f"),
        "Execution premium": st.column_config.NumberColumn(min_value=0.0, step=0.01, format="$%.2f", help="Per-share option premium, as quoted in the listed option chain."),
        "Market premium normalized": None,
        "Conservative premium normalized": None,
        "Market IV": st.column_config.NumberColumn(disabled=True, format="%.2f"),
        "Model IV": st.column_config.NumberColumn(min_value=0.0001, max_value=5.0, step=0.01, format="%.2f"),
        "Volume": st.column_config.NumberColumn(disabled=True, format="%.0f"),
        "Open interest": st.column_config.NumberColumn(disabled=True, format="%.0f"),
        "Contract symbol": st.column_config.TextColumn(disabled=True),
    },
)

st.subheader("Phase 6 Polymarket sizing")
input_by_ticker = inputs.set_index("Ticker")
rank_tickers = result.ranks.columns.astype(str).tolist()
pm_ticker_options = [ticker for ticker in rank_tickers if ticker in input_by_ticker.index]
default_pm_ticker = str(phase5_sidebar.get("selected_ticker") or (pm_ticker_options[0] if pm_ticker_options else ""))
if default_pm_ticker not in pm_ticker_options and pm_ticker_options:
    default_pm_ticker = pm_ticker_options[0]
default_side = str(phase5_sidebar.get("side") or "NO")
if default_side not in ["YES", "NO"]:
    default_side = "NO"
suggested_pm_shares = order_of_magnitude_polymarket_shares(edited_mapping)
size_cols = st.columns([1, 1, 1, 2])
pm_ticker = size_cols[0].selectbox("Polymarket ticker", pm_ticker_options, index=pm_ticker_options.index(default_pm_ticker) if default_pm_ticker in pm_ticker_options else 0, key=f"phase6_pm_ticker_{source}")
pm_side = size_cols[1].radio("Side", ["YES", "NO"], index=["YES", "NO"].index(default_side), horizontal=True, key=f"phase6_pm_side_{source}")
yes_price = float(input_by_ticker.loc[pm_ticker, "Polymarket YES price"]) if pm_ticker in input_by_ticker.index and "Polymarket YES price" in input_by_ticker.columns else 0.5
default_entry = yes_price if pm_side == "YES" else 1.0 - yes_price
saved_entry_matches = phase5_sidebar.get("selected_ticker") == pm_ticker and phase5_sidebar.get("side") == pm_side
saved_entry = float(phase5_sidebar.get("entry", default_entry)) if saved_entry_matches else default_entry
saved_entry = min(max(saved_entry, 0.0), 1.0)
pm_entry = size_cols[2].number_input("Entry price", 0.0, 1.0, saved_entry, 0.001, format="%.3f", key=f"phase6_pm_entry_{source}")
pm_shares = size_cols[3].number_input("Polymarket shares", min_value=0.0, value=suggested_pm_shares, step=100.0, format="%.0f", key=f"phase6_pm_shares_{source}")
st.caption(f"Auto default is order-of-magnitude only: max absolute option contracts x 100 = {suggested_pm_shares:,.0f} Polymarket shares. Override it if the actual ticket size differs.")

try:
    mapped_legs = real_execution_legs(edited_mapping, original_legs, time_to_expiry=time_to_target, risk_free_rate=risk_free_rate)
    real_terminal = terminal_stock_prices(result.terminal_market_caps, current_caps, spot_lookup)
    mapped_option_payoff, _ = manual_option_payoffs_and_analytics(mapped_legs, real_terminal, contract_multiplier=CONTRACT_MULTIPLIER, include_premiums=True)
    winners = winner_from_ranks(result.ranks)
    base_payoff = polymarket_payoff(winners, selected_ticker=pm_ticker, side=pm_side, entry_price=float(pm_entry), quantity=float(pm_shares)).to_numpy(float)
    mapped_total_payoff = base_payoff + mapped_option_payoff
    metrics_display = metric_table(base_payoff, mapped_total_payoff)
    st.subheader("Phase 6 real-market payoff metrics")
    st.dataframe(metrics_display, use_container_width=True, hide_index=True)
    st.caption("Phase 6 uses real-dollar option terms. It is no longer normalized to spot=100; this is the execution view.")

    save_payload = {
        "saved_at": pd.Timestamp.utcnow().isoformat(),
        "run_metadata": metadata,
        "source": source,
        "target_date": str(target_date),
        "pricing_basis": pricing_basis,
        "contract_multiplier": CONTRACT_MULTIPLIER,
        "unit_notes": "Quantity = listed contracts; execution premium = per-share option quote; option payoff = contracts x 100 x per-share payoff.",
        "polymarket": {
            "selected_ticker": pm_ticker,
            "side": pm_side,
            "entry": float(pm_entry),
            "shares": float(pm_shares),
            "sizing_default": suggested_pm_shares,
        },
        "spots": edited_spots.copy(),
        "expiry_table": expiry_table.copy(),
        "mapping_table": edited_mapping.copy(),
        "mapped_legs": mapped_legs.copy(),
        "metrics": metrics_display.copy(),
        "payoffs": {
            "polymarket": base_payoff,
            "options": mapped_option_payoff,
            "total": mapped_total_payoff,
        },
    }
    if st.button("Save Phase 6 execution candidate", type="primary"):
        path = save_phase_artifact(PHASE6_CANDIDATE_ARTIFACT, save_payload)
        st.success(f"Saved Phase 6 execution candidate to {path}.")

    st.subheader("Phase 6 real-market payoff profile")
    control_cols = st.columns([1, 2])
    axis_options = [ticker for ticker in real_terminal.columns if ticker in tickers]
    if not axis_options:
        axis_options = real_terminal.columns.astype(str).tolist()
    axis_ticker = control_cols[0].selectbox("Profile axis ticker", axis_options, index=0, key=f"phase6_axis_{source}")
    with control_cols[1]:
        trace_visibility = render_profile_trace_controls("phase6_mapped", include_mean_sd=True)
    st.plotly_chart(
        phase6_profile_figure(
            base_payoff,
            mapped_total_payoff,
            real_terminal[axis_ticker].to_numpy(float),
            "Phase 6 real-market portfolio",
            trace_visibility,
            axis_ticker,
            mapped_legs,
        ),
        use_container_width=True,
        key=f"phase6_profile_{source}",
    )

    st.subheader("Phase 6 payoff probability distribution")
    distribution_traces = st.multiselect(
        "Payoff distribution traces",
        ["Phase 6 real-market portfolio", "Polymarket only"],
        default=["Phase 6 real-market portfolio"],
        key=f"phase6_distribution_traces_{source}",
    )
    st.plotly_chart(
        distribution_figure(
            base_payoff,
            mapped_total_payoff,
            "Phase 6 real-market portfolio",
            show_baseline="Polymarket only" in distribution_traces,
            show_portfolio="Phase 6 real-market portfolio" in distribution_traces,
        ),
        use_container_width=True,
        key=f"phase6_distribution_{source}",
    )

    st.session_state.phase6_mapped_legs = mapped_legs
    st.session_state.phase6_mapping_table = edited_mapping
    st.session_state.phase6_expiry_table = expiry_table
    st.session_state.phase6_mapped_total_payoff = mapped_total_payoff
    st.session_state.phase6_polymarket_payoff = base_payoff
    st.session_state.phase6_option_payoff = mapped_option_payoff
except Exception as exc:
    st.error(f"Could not evaluate the mapped portfolio: {exc}")

with st.expander("Phase 6 expiry convention", expanded=True):
    st.markdown("""
Phase 6 now maps execution to the listed expiry selected by **Last Friday before month end**. This keeps the execution screen tied to the practical month-end option package.

Phase 7 will handle the separate question of event-date versus option-expiry timing risk. For Phase 6, the table intentionally shows the gap in days but does not try to model the residual timing mismatch.
    """)
