from __future__ import annotations

import pandas as pd
import streamlit as st

from payoff_surface import calculate_scenario_payoffs, payoff_summary, selected_payoff_profile_bins
from phase4_ui import (
    HEDGE_TEMPLATES,
    apply_hedge_template,
    comparison_table,
    display_option_legs,
    display_profile,
    display_risk_summary,
    display_scenarios,
    dollars,
    editable_option_legs_view,
    manual_option_calculator,
    merge_edited_quantities,
    payoff_by_bin_figure,
    payoff_profile_figure,
    pct,
)
from simulation_store import load_phase_artifact, load_simulation_snapshot, save_phase_artifact


st.set_page_config(page_title="Phase 4", layout="wide")
st.title("Phase 4: Payoff Profile")
st.caption("Phase 4 evaluates the saved Phase 3 option structures on the exact saved Phase 1 Monte Carlo scenarios. It does not rerun the probability model or reconstruct boundaries.")


def calculate_payoffs(
    result,
    current_caps: pd.Series,
    spot_series: pd.Series,
    option_legs: pd.DataFrame,
    selected_ticker: str,
    polymarket_side: str,
    polymarket_entry_price: float,
    polymarket_quantity: float,
    contract_multiplier: float,
    include_option_premiums: bool,
) -> pd.DataFrame:
    active = option_legs.copy()
    if "Quantity" in active.columns:
        active = active[active["Quantity"].astype(float) != 0.0]
    required_tickers = [selected_ticker]
    if not active.empty and "Ticker" in active.columns:
        required_tickers.extend(active["Ticker"].astype(str).tolist())
    required_tickers = list(dict.fromkeys(required_tickers))

    missing_caps = [ticker for ticker in required_tickers if ticker not in result.terminal_market_caps.columns]
    missing_spots = [ticker for ticker in required_tickers if ticker not in spot_series.index]
    if missing_caps:
        raise ValueError("Missing terminal market-cap scenarios for " + ", ".join(missing_caps) + ".")
    if missing_spots:
        raise ValueError("Missing current spot price for " + ", ".join(missing_spots) + ". Open Phase 3 once to refresh market spots.")

    return calculate_scenario_payoffs(
        result.terminal_market_caps[required_tickers],
        result.ranks,
        current_caps,
        spot_series,
        option_legs,
        selected_ticker=selected_ticker,
        polymarket_side=polymarket_side,
        polymarket_entry_price=polymarket_entry_price,
        polymarket_quantity=polymarket_quantity,
        contract_multiplier=contract_multiplier,
        include_option_premiums=include_option_premiums,
    )


def matching_run_metadata(left: dict, right: dict) -> bool:
    keys = ["target_date", "days_to_target", "simulations", "seed"]
    return all(left.get(key) == right.get(key) for key in keys)


snapshot = load_simulation_snapshot()
phase3_artifact = load_phase_artifact("phase3")
if snapshot is None:
    st.error("No saved Phase 1 simulation was found. Run Phase 1 first.")
    st.stop()
if phase3_artifact is None or phase3_artifact.get("structure") is None:
    st.error("No saved Phase 3 option structure was found. Open Phase 3 after the latest Phase 2 run.")
    st.stop()

result = snapshot["result"]
simulation_inputs = snapshot["simulation_inputs"].copy()
phase1_run = snapshot.get("run_metadata") or {}
phase3_run = phase3_artifact.get("run_metadata") or {}
if not matching_run_metadata(phase1_run, phase3_run):
    st.error("The saved Phase 3 structures belong to a different Phase 1 run. Open Phase 2 and Phase 3 once, then return here.")
    st.stop()

legacy_ticker = str(phase3_artifact["selected_ticker"])
structures_by_ticker = phase3_artifact.get("structures_by_ticker")
if not isinstance(structures_by_ticker, dict) or not structures_by_ticker:
    structures_by_ticker = {legacy_ticker: phase3_artifact["structure"].copy()}

relevant_from_phase1 = simulation_inputs.loc[
    simulation_inputs["Polymarket YES price"].astype(float) > 0,
    "Ticker",
].astype(str).tolist()
available_tickers = [ticker for ticker in relevant_from_phase1 if ticker in structures_by_ticker]
if not available_tickers:
    available_tickers = list(structures_by_ticker)

with st.sidebar:
    st.header("Locked event")
    default_ticker = legacy_ticker if legacy_ticker in available_tickers else available_tickers[0]
    selected_ticker = st.selectbox(
        "Selected Polymarket ticker",
        available_tickers,
        index=available_tickers.index(default_ticker),
        help="Relevant outcomes come from the saved Phase 1 Polymarket list. Phase 4 switches between the corresponding Phase 3 structures.",
    )

base_option_legs = structures_by_ticker[selected_ticker].copy()
current_caps = simulation_inputs.set_index("Ticker")["Current market cap"].astype(float)
selected_rows = simulation_inputs.loc[simulation_inputs["Ticker"].astype(str) == selected_ticker]
if selected_rows.empty:
    st.error(f"{selected_ticker} is missing from the saved Phase 1 inputs.")
    st.stop()
selected_yes_price = float(selected_rows["Polymarket YES price"].iloc[0])
if base_option_legs.empty:
    st.error(f"The saved Phase 3 boundary query did not produce valid option legs for {selected_ticker}. Choose a reachable boundary in Phase 3 first.")
    st.stop()
spot_series = base_option_legs.drop_duplicates("Ticker").set_index("Ticker")["Spot"].astype(float)

with st.sidebar:
    st.caption(
        f"Phase 1 run: {phase1_run.get('target_date', 'n/a')} | "
        f"{phase1_run.get('simulations', len(result.terminal_market_caps)):,} paths"
    )
    profile_bins = st.slider("Payoff profile bins", min_value=5, max_value=50, value=20, step=1)

    st.header(f"{selected_ticker} Polymarket position")
    polymarket_side = st.selectbox(f"{selected_ticker} side", ["YES", "NO"], index=0)
    polymarket_quantity = st.number_input(f"{selected_ticker} shares", min_value=0.0, value=100.0, step=10.0)
    default_entry = selected_yes_price if polymarket_side == "YES" else 1.0 - selected_yes_price
    polymarket_entry_price = st.number_input(
        f"{selected_ticker} {polymarket_side} entry price",
        min_value=0.0,
        max_value=1.0,
        value=float(default_entry),
        step=0.001,
        format="%.3f",
        key=f"phase4_entry_{selected_ticker}_{polymarket_side}",
    )

    st.header("Option payoff preview")
    hedge_template = st.selectbox("Payoff hedge template", HEDGE_TEMPLATES, index=0)
    contract_multiplier = st.number_input(
        "Shares per option contract",
        min_value=1.0,
        value=100.0,
        step=1.0,
        help="Usually 100 for listed US equity options.",
    )
    default_option_quantity = st.number_input(
        "Default contracts per valid leg",
        min_value=0.0,
        value=0.01,
        step=0.01,
        format="%.2f",
    )
    include_option_premiums = st.checkbox("Include theoretical option premiums", value=True)

st.success(
    f"Locked position: {polymarket_side} {selected_ticker} at {polymarket_entry_price:.3f} | "
    f"Phase 3 boundary query {phase3_artifact.get('confidence_level', float('nan')):.0%} | "
    f"pricing mode {'surface smile' if phase3_artifact.get('use_surface_pricing') else 'ATM fallback'}"
)
st.caption("Date, market caps, marginal model, IV surface, correlations, random seed, and ranks are locked upstream. Change them in Phase 1 and then refresh Phases 2 and 3.")

summary_tab, profile_tab, calculator_tab, scenarios_tab, methodology_tab = st.tabs(
    ["Payoff Summary", "Payoff Profile", "Manual Calculator", "Scenario Table", "Methodology"]
)

with summary_tab:
    option_legs = base_option_legs.copy()
    if "Quantity" not in option_legs.columns:
        option_legs["Quantity"] = float(default_option_quantity)
    state_key = "phase4_option_legs"
    state_signature = (
        phase1_run.get("target_date"),
        phase1_run.get("seed"),
        selected_ticker,
        phase3_artifact.get("confidence_level"),
        phase3_artifact.get("construction_mode"),
    )
    if st.session_state.get("phase4_structure_signature") != state_signature:
        st.session_state[state_key] = option_legs
        st.session_state.phase4_structure_signature = state_signature
    option_legs = st.session_state[state_key].copy()

    st.subheader(f"{selected_ticker} candidate option legs and quantities")
    st.caption("Strikes, IVs, carry, and premiums come from the matching Phase 3 structure. Only quantities are editable here.")
    editable_view = editable_option_legs_view(option_legs)
    edited_view = st.data_editor(
        editable_view,
        width="stretch",
        hide_index=True,
        column_config={
            "Quantity": st.column_config.NumberColumn("Quantity", step=0.01, format="%.2f"),
            "Strike": st.column_config.NumberColumn("Strike", format="$%.2f"),
            "Spot": st.column_config.NumberColumn("Spot", format="$%.2f"),
            "Theoretical premium": st.column_config.NumberColumn("Theoretical premium", format="$%.2f"),
        },
        disabled=[column for column in editable_view.columns if column != "Quantity"],
    )
    edited_legs = merge_edited_quantities(option_legs, edited_view)
    active_legs = apply_hedge_template(edited_legs, hedge_template)
    st.session_state[state_key] = edited_legs
    st.session_state.phase4_active_option_legs = active_legs

    try:
        scenario = calculate_payoffs(
            result,
            current_caps,
            spot_series,
            active_legs,
            selected_ticker,
            polymarket_side,
            float(polymarket_entry_price),
            float(polymarket_quantity),
            float(contract_multiplier),
            bool(include_option_premiums),
        )
        zero_legs = edited_legs.copy()
        zero_legs["Quantity"] = 0.0
        baseline_scenario = calculate_payoffs(
            result,
            current_caps,
            spot_series,
            zero_legs,
            selected_ticker,
            polymarket_side,
            float(polymarket_entry_price),
            float(polymarket_quantity),
            float(contract_multiplier),
            bool(include_option_premiums),
        )
        profile = selected_payoff_profile_bins(
            scenario,
            result.terminal_market_caps,
            current_caps,
            selected_ticker=selected_ticker,
            bins=int(profile_bins),
        )
        st.session_state.phase4_scenario_payoffs = scenario
        st.session_state.phase4_baseline_scenario = baseline_scenario
        st.session_state.phase4_profile = profile
        st.session_state.phase4_selected_ticker = selected_ticker

        summary = payoff_summary(scenario)
        cols = st.columns(6)
        cols[0].metric("Expected payoff", dollars(float(summary["Expected payoff"])))
        cols[1].metric("Payoff SD", dollars(float(summary["Payoff standard deviation"])))
        cols[2].metric("Median payoff", dollars(float(summary["Median payoff"])))
        cols[3].metric("P(loss)", pct(float(summary["Probability of loss"])))
        cols[4].metric("Expected shortfall 5%", dollars(float(summary["Expected shortfall 5%"])))
        cols[5].metric("Worst payoff", dollars(float(summary["Worst payoff"])))

        st.subheader("Baseline comparison")
        st.caption(f"Same Phase 1 scenarios and {selected_ticker} Polymarket position. Only the saved Phase 3 option legs differ.")
        st.dataframe(comparison_table(baseline_scenario, scenario, hedge_template), width="stretch", hide_index=True)

        st.subheader("Risk metrics")
        st.dataframe(display_risk_summary(summary), width="stretch", hide_index=True)

        st.subheader("Payoff components")
        components = scenario[["Polymarket payoff", "Option payoff", "Total payoff"]].mean().to_frame("Expected payoff").reset_index().rename(columns={"index": "Component"})
        components["Expected payoff"] = components["Expected payoff"].map(dollars)
        st.dataframe(components, width="stretch", hide_index=True)

        save_phase_artifact(
            "phase4",
            {
                "active_option_legs": active_legs,
                "scenario_payoffs": scenario,
                "baseline_scenario": baseline_scenario,
                "profile": profile,
                "selected_ticker": selected_ticker,
                "polymarket_side": polymarket_side,
                "polymarket_entry_price": float(polymarket_entry_price),
                "polymarket_quantity": float(polymarket_quantity),
                "contract_multiplier": float(contract_multiplier),
                "include_option_premiums": bool(include_option_premiums),
                "run_metadata": phase1_run,
            },
        )
    except Exception as exc:
        st.error(str(exc))

    with st.expander(f"Active {selected_ticker} option legs: {hedge_template}"):
        st.dataframe(display_option_legs(active_legs), width="stretch", hide_index=True)

with profile_tab:
    profile = st.session_state.get("phase4_profile")
    if profile is None or st.session_state.get("phase4_selected_ticker") != selected_ticker:
        st.info("Open Payoff Summary first.")
    else:
        st.plotly_chart(payoff_profile_figure(profile, selected_ticker), width="stretch")
        st.plotly_chart(payoff_by_bin_figure(profile, selected_ticker), width="stretch")
        st.subheader("Probability-weighted payoff bins")
        st.write("Scenario probability times average payoff gives each bin's contribution to global expected payoff.")
        st.dataframe(display_profile(profile), width="stretch", hide_index=True)

with calculator_tab:
    manual_option_calculator(
        st.session_state.get("phase4_active_option_legs"),
        st.session_state.get("phase4_profile"),
        polymarket_side,
        float(polymarket_entry_price),
        float(polymarket_quantity),
    )

with scenarios_tab:
    scenario = st.session_state.get("phase4_scenario_payoffs")
    if scenario is None or st.session_state.get("phase4_selected_ticker") != selected_ticker:
        st.info("Open Payoff Summary first.")
    else:
        st.subheader("Scenario-level payoff sample")
        st.dataframe(display_scenarios(scenario.head(500)), width="stretch", hide_index=True)
        st.caption("Showing the first 500 saved Phase 1 scenarios only.")

with methodology_tab:
    st.subheader("Methodology")
    st.markdown(
        """
Phase 4 is a payoff evaluation engine. It does not alter the probability model and it does not decide which hedge is optimal.

Locked workflow:

- Phase 1 supplies terminal market-cap scenarios, ranks, and the relevant Polymarket outcomes.
- Phase 2 supplies complete conditional probability curves.
- Phase 3 builds and saves a matching option structure for every relevant outcome.
- Phase 4 selects one saved outcome and changes only its Polymarket position, option quantities, and payoff display.
- Full Phase 1 ranks are retained when deciding the winner. Only tickers with active option legs require stock spot prices for payoff conversion.

Expected payoff bridge:

```text
Global expected payoff = sum(bin scenario probability * average payoff in bin)
```

Payoff standard deviation:

```text
Payoff SD = sqrt(mean((scenario payoff - expected payoff)^2))
```

A deterministic premium shifts expected payoff. Scenario-dependent option intrinsic value determines whether dispersion and tail loss rise or falls.
        """
    )
