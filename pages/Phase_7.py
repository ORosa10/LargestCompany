"""Phase 7: risk assessment and sensitivity workspace.

Phase 7 reruns the saved Phase 1 universe and the saved Phase 4 portfolio under
stressed assumptions and reports how fragile the numbers are. It never changes a
model; it quantifies the uncertainty around the models Phases 1-6 already
produced. See ``phase7.py`` for the engine and the README for the logic.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from model import SHOCK_MODELS
from phase7 import (
    PortfolioSpec,
    assessment,
    constant_correlation,
    copula_tail_stress,
    dispersion_summary,
    gap_scaling_scan,
    gap_vs_randomness,
    iv_scaling_scan,
    model_robustness,
    multi_seed_dispersion,
    robustness_summary,
)
from simulation_store import load_phase_artifact, load_simulation_snapshot

st.set_page_config(page_title="Phase 7", layout="wide")
st.title("Phase 7: Risk assessment & sensitivity")
st.caption(
    "Phase 7 stresses the inputs of the existing engine to separate a real edge "
    "from an artifact of assumptions. It reuses the Phase 1 probability engine "
    "and the Phase 4 payoff surface, so it measures exactly the numbers the rest "
    "of the app produces."
)


def dollars(value: float) -> str:
    return "n/a" if pd.isna(value) else f"${value:,.0f}"


def pct(value: float) -> str:
    return "n/a" if pd.isna(value) else f"{value:.2%}"


def signed_pct(value: float) -> str:
    return "n/a" if pd.isna(value) else f"{value:+.1%}"


# ---------------------------------------------------------------------------
# Load saved Phase 1 simulation and (optionally) the saved Phase 4 portfolio
# ---------------------------------------------------------------------------
snapshot = load_simulation_snapshot()
if snapshot is None:
    st.error("No saved Phase 1 simulation was found. Run Phase 1 first, then return here.")
    st.stop()

result = snapshot["result"]
simulation_inputs = snapshot["simulation_inputs"].copy()
run_metadata = snapshot.get("run_metadata") or {}
base_seed = int(run_metadata.get("seed", 0))
days_to_target = int(run_metadata.get("days_to_target", 90))
base_simulations = int(run_metadata.get("simulations", len(result.terminal_market_caps)))
tickers = simulation_inputs["Ticker"].astype(str).tolist()
base_correlation = result.cleaned_correlation.copy()

phase4 = load_phase_artifact("phase4")
portfolio = None
portfolio_note = ""
if phase4 is not None and isinstance(phase4.get("active_option_legs"), pd.DataFrame):
    legs = phase4["active_option_legs"].copy()
    selected_ticker = str(phase4.get("selected_ticker", tickers[0]))
    current_caps = simulation_inputs.set_index("Ticker")["Current market cap"].astype(float)
    if "Spot" in legs.columns and not legs.empty:
        spot_series = legs.drop_duplicates("Ticker").set_index("Ticker")["Spot"].astype(float)
    else:
        spot_series = pd.Series(dtype=float)
    portfolio = PortfolioSpec(
        option_legs=legs,
        current_market_caps=current_caps,
        spot_prices=spot_series,
        selected_ticker=selected_ticker,
        polymarket_side=str(phase4.get("polymarket_side", "NO")),
        polymarket_entry_price=float(phase4.get("polymarket_entry_price", 0.0)),
        polymarket_quantity=float(phase4.get("polymarket_quantity", 0.0)),
        contract_multiplier=float(phase4.get("contract_multiplier", 100.0)),
        include_option_premiums=bool(phase4.get("include_option_premiums", True)),
    )
    portfolio_note = f"Loaded saved Phase 4 portfolio for {selected_ticker} ({portfolio.polymarket_side} + option legs)."
else:
    relevant = simulation_inputs.loc[simulation_inputs["Polymarket YES price"].astype(float) > 0]
    selected_ticker = str(relevant.sort_values("Polymarket YES price", ascending=False)["Ticker"].iloc[0]) if not relevant.empty else tickers[0]
    portfolio_note = (
        "No saved Phase 4 portfolio found. Running probability-only stresses "
        f"(tail-metric tests need a saved Phase 4 payoff). Selected ticker: {selected_ticker}."
    )

st.info(portfolio_note)

# ---------------------------------------------------------------------------
# Controls
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("Phase 7 controls")
    st.caption(
        f"Phase 1 run: target {run_metadata.get('target_date', 'n/a')} | "
        f"{days_to_target} days | base seed {base_seed} | {base_simulations:,} paths."
    )
    selected_ticker = st.selectbox(
        "Selected ticker (edge / P(#1))",
        tickers,
        index=tickers.index(selected_ticker) if selected_ticker in tickers else 0,
    )
    n_seeds = st.slider("Seeds (Monte Carlo reruns)", min_value=2, max_value=20, value=6, step=1)
    simulations = st.select_slider(
        "Simulations per run",
        options=[10_000, 20_000, 30_000, 50_000, 100_000, 200_000],
        value=min(max(base_simulations, 30_000), 100_000),
        help="Tail metrics converge slowly. Raise this if the worst-case / shortfall dispersion is large.",
    )
    shortfall_probability = st.slider("Expected-shortfall tail", min_value=0.01, max_value=0.10, value=0.05, step=0.01)
    st.divider()
    st.caption("Stress families used in the robustness grid (Test 5).")
    robustness_shocks = st.multiselect("Shock models", SHOCK_MODELS, default=["Normal shocks", "Student-t copula df=5"])
    use_constant_variants = st.checkbox("Add constant-correlation variants (0.0 / 0.2 / 0.8)", value=True)

seeds = list(range(base_seed, base_seed + int(n_seeds)))

summary_tab, tab1, tab2, tab3, tab5, methodology = st.tabs(
    [
        "0. Summary",
        "1. Monte Carlo error",
        "2. Tail-dependence",
        "3. Gap vs randomness",
        "5. Model robustness",
        "Methodology",
    ]
)

with summary_tab:
    st.subheader("Overall assessment")
    st.caption(
        "Runs all four tests on the current portfolio and sidebar settings, then "
        "gives a plain-language verdict on whether the edge is a real signal and a "
        "list of things to watch before trading."
    )
    if st.button("Run full assessment", key="run_summary"):
        with st.spinner("Running all four tests..."):
            summary_factors = [0.5, 0.75, 1.0, 1.25, 1.5, 2.0]
            disp = dispersion_summary(
                multi_seed_dispersion(
                    simulation_inputs, base_correlation, portfolio,
                    days_to_target=days_to_target, simulations=int(simulations), seeds=seeds,
                    shortfall_probability=float(shortfall_probability),
                )
            )
            cop = copula_tail_stress(
                simulation_inputs, base_correlation, portfolio,
                days_to_target=days_to_target, simulations=int(simulations), seeds=seeds,
                shortfall_probability=float(shortfall_probability),
            )
            iv_scan = iv_scaling_scan(
                simulation_inputs, base_correlation, selected_ticker=selected_ticker,
                days_to_target=days_to_target, simulations=int(simulations), seed=base_seed, factors=summary_factors,
            )
            gap_scan = gap_scaling_scan(
                simulation_inputs, base_correlation, selected_ticker=selected_ticker,
                days_to_target=days_to_target, simulations=int(simulations), seed=base_seed, factors=summary_factors,
            )
            gap_verdict = gap_vs_randomness(iv_scan, gap_scan)
            variants = {"Saved correlation": base_correlation}
            if use_constant_variants:
                variants["Independent (0.0)"] = constant_correlation(tickers, 0.0)
                variants["Low constant (0.2)"] = constant_correlation(tickers, 0.2)
                variants["High constant (0.8)"] = constant_correlation(tickers, 0.8)
            shocks = robustness_shocks or ["Normal shocks"]
            grid = model_robustness(
                simulation_inputs, variants, selected_ticker=selected_ticker,
                days_to_target=days_to_target, simulations=int(simulations), seed=base_seed, shock_models=shocks,
            )
            report = assessment(disp, cop, gap_verdict, robustness_summary(grid), selected_ticker=selected_ticker)
        st.session_state.phase7_summary = report

    if "phase7_summary" in st.session_state:
        report = st.session_state.phase7_summary
        headline = report["headline"]
        if "not fully robust" in headline.lower():
            st.warning(headline)
        else:
            st.success(headline)
        st.subheader("Findings by test")
        st.dataframe(report["findings"], width="stretch", hide_index=True)
        st.subheader("What to watch")
        for item in report["watch_outs"]:
            st.warning(item)
    else:
        st.info("Set seeds, simulations, and stress families in the sidebar, then click Run full assessment.")

# ---------------------------------------------------------------------------
# Test 1 - Monte Carlo error / multi-seed reruns
# ---------------------------------------------------------------------------
with tab1:
    st.subheader("Monte Carlo error on tail metrics")
    st.caption(
        "Means and probabilities converge fast; tail metrics (expected shortfall, "
        "worst case) rest on few scenarios and converge slowly. We rerun across "
        "seeds and report the cross-seed dispersion as the +/- error. Large "
        "relative dispersion on a tail metric => raise the simulation count."
    )
    if st.button("Run multi-seed reruns", key="run_test1"):
        with st.spinner(f"Running {len(seeds)} seeds x {simulations:,} paths..."):
            per_seed = multi_seed_dispersion(
                simulation_inputs, base_correlation, portfolio,
                days_to_target=days_to_target, simulations=int(simulations), seeds=seeds,
                shortfall_probability=float(shortfall_probability),
            )
            summary = dispersion_summary(per_seed)
        st.session_state.phase7_test1 = (per_seed, summary)

    if "phase7_test1" in st.session_state:
        per_seed, summary = st.session_state.phase7_test1
        display = summary.copy()
        display["Reported as"] = display.apply(lambda r: f"{r['Mean']:.4g} +/- {r['MC error (std)']:.2g}", axis=1)
        st.dataframe(display, width="stretch", hide_index=True)
        with st.expander("Per-seed detail"):
            st.dataframe(per_seed, width="stretch", hide_index=True)
        worst_row = summary.loc[summary["Metric"] == "Worst payoff"]
        if not worst_row.empty:
            rel = float(worst_row["Relative dispersion"].iloc[0])
            if np.isnan(rel) or rel < 0.02:
                st.success("Worst-case payoff is stable across seeds (bounded by the spread construction).")
            else:
                st.warning("Worst-case payoff wanders across seeds. Raise the simulation count before trusting the tail.")

# ---------------------------------------------------------------------------
# Test 2 - Tail-dependence stress
# ---------------------------------------------------------------------------
with tab2:
    st.subheader("Tail-dependence stress: Gaussian vs Student-t copula (df=5)")
    st.caption(
        "Swap only the dependence family, keep the marginals fixed. The Gaussian "
        "copula has zero tail dependence (names never crash together in extremes); "
        "the Student-t copula shares one chi-square shock so extremes arrive "
        "jointly. Bounded-loss spreads should keep the worst case fixed; watch how "
        "much edge and expected payoff move."
    )
    if st.button("Run copula stress", key="run_test2"):
        with st.spinner("Comparing Gaussian vs Student-t copula..."):
            stress = copula_tail_stress(
                simulation_inputs, base_correlation, portfolio,
                days_to_target=days_to_target, simulations=int(simulations), seeds=seeds,
                shortfall_probability=float(shortfall_probability),
            )
        st.session_state.phase7_test2 = stress

    if "phase7_test2" in st.session_state:
        stress = st.session_state.phase7_test2
        comp = stress.comparison.copy()
        st.dataframe(comp, width="stretch", hide_index=True)
        edge_row = comp.loc[comp["Metric"] == "Edge selected"]
        if not edge_row.empty:
            change = float(edge_row["Change %"].iloc[0])
            st.metric("Edge change under tail dependence", signed_pct(change))

# ---------------------------------------------------------------------------
# Test 3 - Gap vs randomness
# ---------------------------------------------------------------------------
with tab3:
    st.subheader("Gap vs randomness decomposition")
    st.caption(
        "Scale every IV by k, and separately widen/compress the cap gaps at fixed "
        "vol. If P(#1) swings more under IV scaling, the outcome is "
        "randomness-dominated (IV is the critical lever); if it swings more under "
        "gap scaling, it is gap-dominated (structural)."
    )
    iv_factors = [0.5, 0.75, 1.0, 1.25, 1.5, 2.0]
    gap_factors = [0.5, 0.75, 1.0, 1.25, 1.5, 2.0]
    if st.button("Run sensitivity scans", key="run_test3"):
        with st.spinner("Scanning IV and gap scales..."):
            iv_scan = iv_scaling_scan(
                simulation_inputs, base_correlation, selected_ticker=selected_ticker,
                days_to_target=days_to_target, simulations=int(simulations), seed=base_seed, factors=iv_factors,
            )
            gap_scan = gap_scaling_scan(
                simulation_inputs, base_correlation, selected_ticker=selected_ticker,
                days_to_target=days_to_target, simulations=int(simulations), seed=base_seed, factors=gap_factors,
            )
        st.session_state.phase7_test3 = (iv_scan, gap_scan)

    if "phase7_test3" in st.session_state:
        iv_scan, gap_scan = st.session_state.phase7_test3
        verdict = gap_vs_randomness(iv_scan, gap_scan)
        st.dataframe(verdict, width="stretch", hide_index=True)
        st.plotly_chart(
            px.line(iv_scan, x="IV scale", y="P(#1) selected", markers=True, title=f"{selected_ticker}: P(#1) vs IV scale"),
            width="stretch",
        )
        st.plotly_chart(
            px.line(gap_scan, x="Gap scale", y="P(#1) selected", markers=True, title=f"{selected_ticker}: P(#1) vs cap-gap scale"),
            width="stretch",
        )
        with st.expander("Full per-ticker scans"):
            st.write("IV scaling")
            st.dataframe(iv_scan, width="stretch", hide_index=True)
            st.write("Gap scaling")
            st.dataframe(gap_scan, width="stretch", hide_index=True)

# ---------------------------------------------------------------------------
# Test 5 - Model robustness
# ---------------------------------------------------------------------------
with tab5:
    st.subheader("Model-uncertainty / robustness grid")
    st.caption(
        "Report the range of P(#1) and edge across a grid of plausible models "
        "(correlation variants x shock models). An edge that survives the grid is "
        "tradeable; one that flips sign or collapses is model-dependent."
    )
    if st.button("Run robustness grid", key="run_test5"):
        variants = {"Saved correlation": base_correlation}
        if use_constant_variants:
            variants["Independent (0.0)"] = constant_correlation(tickers, 0.0)
            variants["Low constant (0.2)"] = constant_correlation(tickers, 0.2)
            variants["High constant (0.8)"] = constant_correlation(tickers, 0.8)
        shocks = robustness_shocks or ["Normal shocks"]
        with st.spinner("Running model grid..."):
            grid = model_robustness(
                simulation_inputs, variants, selected_ticker=selected_ticker,
                days_to_target=days_to_target, simulations=int(simulations), seed=base_seed, shock_models=shocks,
            )
            summary = robustness_summary(grid)
        st.session_state.phase7_test5 = (grid, summary)

    if "phase7_test5" in st.session_state:
        grid, summary = st.session_state.phase7_test5
        cols = st.columns(4)
        cols[0].metric("P(#1) range", f"{pct(summary['P(#1) min'])} - {pct(summary['P(#1) max'])}")
        cols[1].metric("P(#1) spread", pct(summary["P(#1) spread"]))
        cols[2].metric("Edge range", f"{signed_pct(summary['Edge min'])} - {signed_pct(summary['Edge max'])}")
        cols[3].metric("Edge sign consistent", "Yes" if bool(summary["Edge sign consistent"]) else "No")
        if bool(summary["Edge sign consistent"]):
            st.success(f"Edge on {selected_ticker} keeps the same sign across every model in the grid.")
        else:
            st.warning(f"Edge on {selected_ticker} flips sign somewhere in the grid: model-dependent, treat with caution.")
        st.plotly_chart(
            px.bar(grid, x="Correlation", y="Edge selected", color="Shock model", barmode="group",
                   title=f"{selected_ticker}: edge across correlation variants and shock models"),
            width="stretch",
        )
        st.dataframe(grid, width="stretch", hide_index=True)

# ---------------------------------------------------------------------------
# Methodology
# ---------------------------------------------------------------------------
with methodology:
    st.markdown(
        """
Phase 7 is a risk-assessment layer. It changes no model; it stresses the inputs
of the existing engine and reports how fragile the outputs are.

**1. Monte Carlo error.** Means and ranking probabilities converge quickly, but
tail metrics (expected shortfall, worst case) rest on few scenarios and converge
slowly. Phase 7 reruns across seeds and reports the cross-seed standard
deviation as the +/- error. If a tail metric's relative dispersion is large,
raise the simulation count.

**2. Tail-dependence stress.** Marginals (single-name distributions) and
dependence (how names move together) are separated. The Gaussian copula has zero
tail dependence; the Student-t copula (df=5) shares one chi-square shock so
extremes arrive jointly. Only the dependence family changes, so any difference
is pure tail dependence. Bounded-loss spreads keep the worst case fixed by
construction; the interesting effect is on expected payoff and edge.

**3. Gap vs randomness.** Scaling every IV by k isolates randomness; widening or
compressing the cap gaps at fixed vol isolates structure
(`cap_i -> ref * (cap_i / ref)**g`). Whichever scan moves P(#1) more is the
dominant lever.

**5. Model robustness.** P(#1) and edge are recomputed across a grid of
correlation variants and shock models. An edge that keeps its sign and magnitude
across the grid is tradeable; one that depends on the model choice is not.

Test 4 (out-of-sample optimizer validation) is intentionally omitted while the
workflow uses a manual portfolio rather than an automatic optimizer.
        """
    )
