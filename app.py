from __future__ import annotations

from datetime import date
from pathlib import Path

import pandas as pd
import streamlit as st

from implied_forwards import apply_implied_forwards, estimate_implied_forwards
from iv_surface_model import (
    SURFACE_EXPIRY,
    apply_surface_atm_ivs,
    default_surface_nodes,
    run_surface_probability_engine,
)
from simulation_store import save_simulation_snapshot


@st.cache_data(show_spinner=False, ttl=15 * 60)
def load_implied_forwards(tickers: tuple[str, ...], target_date_iso: str, risk_free_rate: float) -> pd.DataFrame:
    return estimate_implied_forwards(list(tickers), date.fromisoformat(target_date_iso), risk_free_rate=float(risk_free_rate))


APP_CORE_PATH = Path(__file__).with_name("app_core.py")
source = APP_CORE_PATH.read_text(encoding="utf-8")

source = source.replace(
    'CORRELATION_METHODS = [\n    "EWMA historical correlation",\n    "Vol-adjusted smooth correlation",\n    "Rolling historical correlation",\n    "Manual/default correlation matrix",\n]',
    'CORRELATION_METHODS = [\n    "EWMA historical correlation",\n    "Vol-adjusted smooth correlation",\n    "Rolling historical correlation",\n]',
)
source = source.replace(
    '    target_date = st.date_input("Target date / maturity", value=today + timedelta(days=365), min_value=today + timedelta(days=1))',
    '    default_target = date(2026, 7, 31) if today < date(2026, 7, 31) else today + timedelta(days=365)\n    target_date = st.date_input("Target date / maturity", value=default_target, min_value=today + timedelta(days=1))',
)
source = source.replace(
    '    market_cap_source = st.selectbox("Market cap source", ["Yahoo Finance current market cap", "Manual market cap inputs"], index=0)\n',
    '    market_cap_source = "Yahoo Finance current market cap"\n    st.caption("Market caps: Yahoo Finance current market cap")\n',
)
source = source.replace(
    '    iv_source = st.selectbox("IV source", ["Manual IV inputs", "Yahoo option-chain near-ATM IV"], index=0)\n',
    '    probability_model = st.selectbox("Probability model", ["IV surface marginals", "ATM lognormal"], index=0)\n    iv_source = st.selectbox("Fallback IV source", ["Manual IV inputs", "Yahoo option-chain near-ATM IV"], index=0)\n    st.caption("Surface model: manually calibrated 2026-07-31 smiles for NVDA, AAPL, and GOOGL; ATM lognormal fallback for other tickers.")\n    forward_source = st.selectbox("Forward / dividend carry", ["Put-call parity implied forward", "Flat forward (legacy)"], index=0)\n    forward_risk_free_rate = st.number_input("Forward extraction risk-free rate", min_value=0.0, max_value=0.20, value=0.04, step=0.005, format="%.3f")\n',
)
source = source.replace(
    'inputs_tab, overview_tab, ticker_tab, pair_tab, comparison_tab, data_tab, methodology_tab = st.tabs(["Inputs", "Overview", "Ticker Detail", "Pair Detail", "Model Comparison", "Data Used", "Methodology"])',
    'inputs_tab, overview_tab, ticker_tab, pair_tab, comparison_tab, data_tab, methodology_tab = st.tabs(["Manual IV & Polymarket", "Overview", "Ticker Detail", "Pair Detail", "Model Comparison", "Data Used", "Methodology"])',
)
old_inputs_block = '''with inputs_tab:
    st.subheader("Editable company inputs")
    st.write("Market caps can be overridden by Yahoo. IV and Polymarket prices remain manual unless Yahoo near-ATM IV is explicitly selected.")
    company_inputs = st.data_editor(
        st.session_state.company_inputs,
        num_rows="dynamic",
        use_container_width=True,
        hide_index=True,
        column_config={
            "Ticker": st.column_config.TextColumn(required=True),
            "Current market cap": st.column_config.NumberColumn(min_value=1.0, step=10_000_000_000.0),
            "Implied volatility": st.column_config.NumberColumn(min_value=0.0001, max_value=5.0, step=0.01),
            "Polymarket YES price": st.column_config.NumberColumn(min_value=0.0, max_value=1.0, step=0.01),
        },
    )
    st.session_state.company_inputs = company_inputs

    tickers = [ticker for ticker in company_inputs["Ticker"].astype(str).str.strip().tolist() if ticker]
    current_corr = st.session_state.correlation_matrix.reindex(index=tickers, columns=tickers).fillna(default_correlation_matrix(tickers))
    st.subheader("Manual/default correlation matrix")
    st.write("Used only when Manual/default correlation matrix is selected.")
    st.session_state.correlation_matrix = st.data_editor(current_corr, use_container_width=True)
'''
new_inputs_block = '''with inputs_tab:
    st.subheader("Manual event inputs")
    st.write("Edit only tickers, fallback annualized IV, and Polymarket YES prices. Market caps come from Yahoo; forward carry comes from option put-call parity when selected.")
    previous_inputs = st.session_state.company_inputs.copy()
    editable_inputs = previous_inputs[["Ticker", "Implied volatility", "Polymarket YES price"]].copy()
    edited_inputs = st.data_editor(
        editable_inputs,
        num_rows="dynamic",
        use_container_width=True,
        hide_index=True,
        column_config={
            "Ticker": st.column_config.TextColumn(required=True),
            "Implied volatility": st.column_config.NumberColumn("Fallback ATM IV", min_value=0.0001, max_value=5.0, step=0.01),
            "Polymarket YES price": st.column_config.NumberColumn("Manual Polymarket YES price", min_value=0.0, max_value=1.0, step=0.01),
        },
    )
    cap_fallback = pd.concat([previous_inputs[["Ticker", "Current market cap"]], default_company_inputs()[["Ticker", "Current market cap"]]]).drop_duplicates("Ticker", keep="first")
    company_inputs = edited_inputs.merge(cap_fallback, on="Ticker", how="left")
    company_inputs["Current market cap"] = company_inputs["Current market cap"].fillna(default_company_inputs()["Current market cap"].median())
    company_inputs = company_inputs[["Ticker", "Current market cap", "Implied volatility", "Polymarket YES price"]]
    st.session_state.company_inputs = company_inputs

    tickers = [ticker for ticker in company_inputs["Ticker"].astype(str).str.strip().tolist() if ticker]
    st.session_state.correlation_matrix = default_correlation_matrix(tickers)
    if probability_model == "IV surface marginals":
        st.subheader("Calibrated IV surface nodes")
        surface_display = default_surface_nodes().copy()
        surface_display["IV"] = surface_display["IV"].map(pct)
        surface_display["Moneyness"] = surface_display["Moneyness"].map(pct)
        st.dataframe(surface_display, use_container_width=True, hide_index=True)
    st.caption("Use Data Used to inspect Yahoo market caps, implied forwards, marginal models, and the selected correlation matrix after running.")
'''
source = source.replace(old_inputs_block, new_inputs_block)
source = source.replace(
    '    drift = -0.5 * np.square(volatilities) * horizon_years\n    diffusion = volatilities * np.sqrt(horizon_years) * correlated_shocks\n    terminal_caps = market_caps_0 * np.exp(drift + diffusion)',
    '    forward_ratios = clean_inputs["Forward / spot"].to_numpy(dtype=float) if "Forward / spot" in clean_inputs.columns else np.ones(len(tickers))\n    drift = -0.5 * np.square(volatilities) * horizon_years\n    diffusion = volatilities * np.sqrt(horizon_years) * correlated_shocks\n    terminal_caps = market_caps_0 * forward_ratios * np.exp(drift + diffusion)',
)
source = source.replace(
    'for key in ["last_result", "last_error", "last_run", "last_simulation_inputs", "last_market_caps", "last_iv_estimates", "last_sources", "last_corr_label", "last_price_info", "last_comparison"]:',
    'for key in ["last_result", "last_error", "last_run", "last_simulation_inputs", "last_market_caps", "last_iv_estimates", "last_forward_estimates", "last_surface_diagnostics", "last_sources", "last_corr_label", "last_price_info", "last_comparison"]:',
)
source = source.replace(
    '            simulation_inputs, market_caps, iv_estimates, market_cap_label, iv_label = prepare_simulation_inputs(company_inputs, market_cap_source, iv_source, target_date)\n            corr, corr_label, price_info = select_correlation_matrix',
    '            simulation_inputs, market_caps, iv_estimates, market_cap_label, iv_label = prepare_simulation_inputs(company_inputs, market_cap_source, iv_source, target_date)\n            run_tickers = tuple(simulation_inputs["Ticker"].astype(str).tolist())\n            if forward_source == "Put-call parity implied forward":\n                forward_estimates = load_implied_forwards(run_tickers, target_date.isoformat(), float(forward_risk_free_rate))\n                simulation_inputs = apply_implied_forwards(simulation_inputs, forward_estimates)\n                forward_label = "Option-implied forward from put-call parity"\n            else:\n                forward_estimates = None\n                simulation_inputs["Forward / spot"] = 1.0\n                forward_label = "Flat forward (legacy)"\n            corr, corr_label, price_info = select_correlation_matrix',
)
source = source.replace(
    '            result = run_engine(simulation_inputs, corr, days_to_target=int(days_to_target), simulations=int(simulations), seed=int(seed), shock_model=shock_model)',
    '            if probability_model == "IV surface marginals":\n                if target_date.isoformat() != SURFACE_EXPIRY:\n                    raise ValueError(f"Calibrated surfaces are for {SURFACE_EXPIRY}. Choose that target date or use ATM lognormal.")\n                simulation_inputs = apply_surface_atm_ivs(simulation_inputs)\n                result, surface_diagnostics = run_surface_probability_engine(\n                    simulation_inputs, corr, days_to_target=int(days_to_target), simulations=int(simulations), seed=int(seed),\n                    surface_nodes=default_surface_nodes(), risk_free_rate=float(forward_risk_free_rate),\n                )\n                probability_model_label = "IV surface risk-neutral marginals + Gaussian copula"\n            else:\n                result = run_engine(simulation_inputs, corr, days_to_target=int(days_to_target), simulations=int(simulations), seed=int(seed), shock_model=shock_model)\n                surface_diagnostics = None\n                probability_model_label = f"ATM lognormal + {shock_model}"',
)
source = source.replace(
    '            st.session_state.last_iv_estimates = iv_estimates\n            st.session_state.last_sources = {"Market cap": market_cap_label, "IV": iv_label, "Polymarket": "Manual inputs", "Shock distribution": shock_model}',
    '            st.session_state.last_iv_estimates = iv_estimates\n            st.session_state.last_forward_estimates = forward_estimates\n            st.session_state.last_surface_diagnostics = surface_diagnostics\n            st.session_state.last_sources = {"Market cap": market_cap_label, "IV": iv_label, "Forward": forward_label, "Probability model": probability_model_label, "Polymarket": "Manual inputs", "Shock distribution": shock_model}',
)
source = source.replace(
    '        a1, a2, a3, a4 = st.columns(4)\n        a1.metric("Market caps", sources.get("Market cap"))\n        a2.metric("IV", sources.get("IV"))\n        a3.metric("Correlation", correlation_method)\n        a4.metric("Shock model", sources.get("Shock distribution"))',
    '        a1, a2, a3, a4 = st.columns(4)\n        a1.metric("Market caps", sources.get("Market cap"))\n        a2.metric("Forward carry", sources.get("Forward"))\n        a3.metric("Correlation", correlation_method)\n        a4.metric("Probability model", sources.get("Probability model"))',
)
source = source.replace(
    '            st.subheader("Model probability by approach")',
    '            st.subheader("ATM-model sensitivity by approach")\n            st.caption("This comparison tab remains an ATM correlation/shock sensitivity. The main Overview result uses the probability model selected in the sidebar.")',
)
source = source.replace(
    '        if iv_display is not None:\n            st.subheader("Yahoo IV estimates used")\n            st.dataframe(iv_display, use_container_width=True, hide_index=True)\n        if st.session_state.last_price_info:',
    '        if iv_display is not None:\n            st.subheader("Yahoo IV estimates used")\n            st.dataframe(iv_display, use_container_width=True, hide_index=True)\n        forward_display = st.session_state.last_forward_estimates\n        if forward_display is not None and not forward_display.empty:\n            st.subheader("Put-call parity implied forwards used")\n            display_forward = forward_display.copy()\n            for column in ["spot", "implied_forward"]:\n                display_forward[column] = display_forward[column].map(lambda value: f"${value:,.2f}")\n            for column in ["forward_to_spot", "annualized_implied_carry", "forward_dispersion"]:\n                display_forward[column] = display_forward[column].map(pct)\n            st.dataframe(display_forward, use_container_width=True, hide_index=True)\n        surface_diagnostics = st.session_state.last_surface_diagnostics\n        if surface_diagnostics is not None and not surface_diagnostics.empty:\n            st.subheader("Marginal distribution diagnostics")\n            surface_display = surface_diagnostics.copy()\n            for column in ["ATM IV", "Forward / spot", "Projection adjustment"]:\n                surface_display[column] = surface_display[column].map(pct)\n            st.dataframe(surface_display, use_container_width=True, hide_index=True)\n        if st.session_state.last_price_info:',
)
source = source.replace(
    '                "P5": caps.quantile(0.05),\n                "P25": caps.quantile(0.25),',
    '                "P1": caps.quantile(0.01),\n                "P5": caps.quantile(0.05),\n                "P25": caps.quantile(0.25),',
)
source = source.replace(
    '                "P75": caps.quantile(0.75),\n                "P95": caps.quantile(0.95),',
    '                "P75": caps.quantile(0.75),\n                "P95": caps.quantile(0.95),\n                "P99": caps.quantile(0.99),',
)
source = source.replace(
    '    for column in ["Mean", "P5", "P25", "P50", "P75", "P95"]:',
    '    for column in ["Mean", "P1", "P5", "P25", "P50", "P75", "P95", "P99"]:',
)
source = source.replace(
    '    st.code("MC_T = MC_0 * exp((-0.5 * sigma^2) * T + sigma * sqrt(T) * Z)")',
    '    st.code("MC_T = MC_0 * (F_0T / S_0) * X_T")\n    st.write("For NVDA, AAPL, and GOOGL, X_T is sampled from the risk-neutral marginal CDF implied by the calibrated 2026-07-31 IV smile. Other tickers use the ATM lognormal fallback. A Gaussian copula applies the selected correlation matrix across marginals.")\n    st.write("F_0T / S_0 is estimated from liquid call-put pairs around ATM and introduces option-implied dividend and financing carry without forecasting stock returns.")',
)

exec(compile(source, str(APP_CORE_PATH), "exec"), globals())

if st.session_state.get("last_result") is not None and st.session_state.get("last_simulation_inputs") is not None:
    save_simulation_snapshot(
        result=st.session_state.last_result,
        simulation_inputs=st.session_state.last_simulation_inputs,
        run_metadata=st.session_state.get("last_run"),
        source=(st.session_state.get("last_sources") or {}).get("Probability model", "Phase 1 baseline"),
    )
