from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import pandas as pd
import streamlit as st

from execution_mapping import choose_expiration, default_strike_step, fetch_option_expirations, map_normalized_legs, rebuild_normalized_legs
from manual_portfolio import manual_option_payoffs_and_analytics
from market_data import fetch_spot_prices
from optimization import payoff_metrics
from payoff_surface import terminal_stock_prices
from simulation_store import load_simulation_snapshot

st.set_page_config(page_title="Phase 6", layout="wide")
st.title("Phase 6: execution mapping")
st.caption("Translate a selected Phase 5 research portfolio from normalized spot=100 into current spots, executable strike grids, and listed expiration dates.")


def available_snapshot() -> dict | None:
    if st.session_state.get("last_result") is not None and st.session_state.get("last_simulation_inputs") is not None:
        return {
            "result": st.session_state.last_result,
            "simulation_inputs": st.session_state.last_simulation_inputs,
            "run_metadata": st.session_state.get("last_run") or {},
        }
    return load_simulation_snapshot()


def target_from_metadata(metadata: dict) -> date:
    raw = metadata.get("target_date")
    if raw:
        try:
            return pd.Timestamp(raw).date()
        except Exception:
            pass
    return date.today() + timedelta(days=int(metadata.get("days_to_target", 365)))


def metric_table(base, original, mapped) -> pd.DataFrame:
    rows = []
    for name, values in [("Polymarket only", base), ("Phase 5 portfolio", original), ("Phase 6 mapped portfolio", mapped)]:
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


snapshot = available_snapshot()
if snapshot is None:
    st.error("No stored Monte Carlo snapshot is available. Run Phase 1 first.")
    st.stop()

portfolio_sources = {}
manual_legs = st.session_state.get("phase5_manual_legs")
manual_total = st.session_state.get("phase5_manual_total_payoff")
if manual_legs is not None and not manual_legs.empty and manual_total is not None:
    portfolio_sources["Phase 5 manual portfolio"] = (manual_legs.copy(), np.asarray(manual_total, dtype=float))
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

st.success(f"Loaded {source}: {len(original_legs)} option legs across {len(tickers)} ticker(s). Event target: {target_date}.")

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

mapping = map_normalized_legs(original_legs, spot_lookup, step_lookup)
st.subheader("Real-strike portfolio mapping")
st.caption("Raw real strike preserves Phase 5 moneyness exactly. Executable strike is rounded to the provisional grid and remains editable. Phase 7 will replace this grid with the actual listed chain.")
edited_mapping = st.data_editor(
    mapping,
    use_container_width=True,
    hide_index=True,
    key=f"phase6_mapping_{source}",
    column_config={
        "Leg": st.column_config.NumberColumn(disabled=True),
        "Ticker": st.column_config.TextColumn(disabled=True),
        "Option type": st.column_config.TextColumn(disabled=True),
        "Position": st.column_config.TextColumn(disabled=True),
        "Phase 5 normalized strike": st.column_config.NumberColumn(disabled=True, format="%.2f"),
        "Current spot": st.column_config.NumberColumn(disabled=True, format="$%.2f"),
        "Raw real strike": st.column_config.NumberColumn(disabled=True, format="$%.2f"),
        "Mapped normalized strike": st.column_config.NumberColumn(disabled=True, format="%.2f"),
        "Strike mapping error": st.column_config.NumberColumn(disabled=True, format="%+.2f"),
        "Executable strike": st.column_config.NumberColumn(min_value=0.01, step=0.5, format="$%.2f"),
        "Model IV": st.column_config.NumberColumn(min_value=0.0001, max_value=5.0, step=0.01, format="%.2f"),
    },
)

st.subheader("Expiration alignment")
policy = st.selectbox("Expiration policy", ["First expiry on/after target", "Nearest listed expiry", "Last expiry on/before target"])
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
        "Treatment": "Event-date mark-to-market" if pd.notna(gap) and gap > 0 else ("Close package at option expiry" if pd.notna(gap) and gap < 0 else "Intrinsic at aligned expiry"),
    })
expiry_table = pd.DataFrame(expiry_rows)
st.dataframe(expiry_table, use_container_width=True, hide_index=True)
if expiry_table["Gap days"].isna().any():
    st.warning("Some expiration calendars are not loaded. Fetch listed expirations or enter the dates later in Phase 7.")
elif (expiry_table["Gap days"] > 0).any():
    st.info("Recommended treatment: use the first expiry on/after the event and value the option mark-to-market on the event date with residual time value. Intrinsic payoff is only exact when expiry equals the event date.")
if (expiry_table["Gap days"].abs() > 7).any():
    st.warning("At least one option expiry differs from the event by more than seven days. Treat this as material model risk and do not silently use intrinsic payoff at the event date.")

result = snapshot["result"]
inputs = snapshot["simulation_inputs"].copy()
current_caps = inputs.set_index("Ticker")["Current market cap"].astype(float)
normalized_spots = pd.Series(100.0, index=result.terminal_market_caps.columns)
normalized_terminal = terminal_stock_prices(result.terminal_market_caps, current_caps, normalized_spots)

try:
    original_option_payoff, _ = manual_option_payoffs_and_analytics(original_legs, normalized_terminal, contract_multiplier=1.0, include_premiums=True)
    base_payoff = original_total_payoff - original_option_payoff
    mapped_legs = rebuild_normalized_legs(original_legs, edited_mapping, time_to_expiry=time_to_target, risk_free_rate=risk_free_rate)
    mapped_option_payoff, _ = manual_option_payoffs_and_analytics(mapped_legs, normalized_terminal, contract_multiplier=1.0, include_premiums=True)
    mapped_total_payoff = base_payoff + mapped_option_payoff
    st.subheader("Strike-rounding impact under stored Phase 5 scenarios")
    st.dataframe(metric_table(base_payoff, original_total_payoff, mapped_total_payoff), use_container_width=True, hide_index=True)
    st.caption("This comparison isolates current-spot mapping and strike rounding. It still uses theoretical premiums and the Phase 5 event-date payoff convention; Phase 7 will introduce listed contracts, live quotes, spreads, and correct event-date mark-to-market for expiry gaps.")
    st.session_state.phase6_mapped_legs = mapped_legs
    st.session_state.phase6_mapping_table = edited_mapping
    st.session_state.phase6_expiry_table = expiry_table
except Exception as exc:
    st.error(f"Could not evaluate the mapped portfolio: {exc}")

with st.expander("How to handle the option/event date mismatch", expanded=True):
    st.markdown("""
1. **Preferred:** choose the first listed expiry on or after the Polymarket event. At the event date, value each still-live option at market or with an option model using its remaining time to expiry.
2. **Before-target expiry:** close both the option package and the Polymarket position at the option expiry. This changes the effective research horizon and introduces Polymarket exit-price risk.
3. **Rolling:** use an earlier liquid expiry and roll into the next contract. This introduces roll cost, changing IV, and execution risk.

For a gap of a few days, the first approach is usually the cleanest. For longer gaps, Phase 7 must explicitly model residual option value or rolling; intrinsic payoff at the event date would be wrong.
    """)
