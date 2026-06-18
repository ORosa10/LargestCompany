from __future__ import annotations

from pathlib import Path


APP_CORE_PATH = Path(__file__).with_name("app_core.py")

IV_HELPERS = r'''
IV_SHOCK_LEVELS = [level / 100 for level in range(-20, 25, 5)]


def iv_shock_label(value: float) -> str:
    return f"{value * 100:+.0f} vol pts"


def base_inputs_from_result(result) -> pd.DataFrame:
    return result.results[["Ticker", "Current market cap", "Implied volatility", "Polymarket YES price"]].copy()


def shock_iv_inputs(base_inputs: pd.DataFrame, shock: float, tickers: list[str] | None) -> pd.DataFrame:
    shocked = base_inputs.copy()
    if tickers is None:
        mask = pd.Series(True, index=shocked.index)
    else:
        mask = shocked["Ticker"].astype(str).isin(tickers)
    shocked.loc[mask, "Implied volatility"] = (shocked.loc[mask, "Implied volatility"].astype(float) + shock).clip(lower=0.0001)
    return shocked


def iv_sensitivity_grid(base_inputs: pd.DataFrame, corr: pd.DataFrame, days_to_target: int, simulations: int, seed: int, shocked_tickers: list[str] | None, mode: str) -> pd.DataFrame:
    rows = []
    for shock in IV_SHOCK_LEVELS:
        shocked_inputs = shock_iv_inputs(base_inputs, shock, shocked_tickers)
        shocked_result = run_probability_engine(shocked_inputs, corr, days_to_target=days_to_target, simulations=simulations, seed=seed)
        for _, row in shocked_result.results.iterrows():
            rows.append(
                {
                    "Mode": mode,
                    "Shock": shock,
                    "Shocked tickers": "All" if shocked_tickers is None else ", ".join(shocked_tickers),
                    "Ticker": row["Ticker"],
                    "Model probability": row["Model probability"],
                    "Average rank": row["Average rank"],
                    "Top 2": row["Probability Top 2"],
                    "Top 3": row["Probability Top 3"],
                }
            )
    return pd.DataFrame(rows)


def display_iv_sensitivity(table: pd.DataFrame) -> pd.DataFrame:
    display = table.copy()
    display["Shock"] = display["Shock"].map(iv_shock_label)
    for column in ["Model probability", "Top 2", "Top 3"]:
        display[column] = display[column].map(pct)
    display["Average rank"] = display["Average rank"].map(lambda value: f"{value:.2f}")
    return display
'''

IV_TAB_BLOCK = r'''
with iv_tab:
    if st.session_state.last_error:
        st.error(st.session_state.last_error)
    elif result is None:
        st.warning("Run the main simulation first to see IV analysis.")
    else:
        run = st.session_state.last_run or {}
        available_tickers = result.results["Ticker"].astype(str).tolist()
        base_inputs = base_inputs_from_result(result)

        st.subheader("IV sensitivity")
        st.write("This reruns the same market caps, selected correlation matrix, target date, and seed while shocking implied volatility. Shocks are additive annualized volatility points, e.g. +5 vol pts turns 30% IV into 35% IV.")
        st.caption(f"Base IV source: {st.session_state.last_iv_source}")
        st.caption(f"Correlation source: {st.session_state.last_corr_source}")

        controls_left, controls_mid, controls_right = st.columns(3)
        with controls_left:
            iv_sensitivity_paths = st.number_input("IV sensitivity simulations per shock", min_value=1_000, max_value=500_000, value=min(int(run.get("simulations", simulations)), 100_000), step=10_000)
        with controls_mid:
            single_iv_ticker = st.selectbox("Single-name shock ticker", available_tickers, index=available_tickers.index(selected_ticker_sidebar) if selected_ticker_sidebar in available_tickers else 0, key="iv_single_ticker")
        with controls_right:
            pair_defaults = available_tickers[:2] if len(available_tickers) >= 2 else available_tickers
            pair_iv_tickers = st.multiselect("Pair shock tickers", available_tickers, default=pair_defaults, max_selections=2, key="iv_pair_tickers")

        if st.button("Run IV sensitivity", type="primary", key="run_iv_sensitivity"):
            with st.spinner("Running IV sensitivity shocks..."):
                try:
                    st.session_state.last_global_iv_sensitivity = iv_sensitivity_grid(base_inputs, result.cleaned_correlation, int(run.get("days_to_target", days_to_target)), int(iv_sensitivity_paths), int(run.get("seed", seed)), None, "Global")
                    st.session_state.last_single_iv_sensitivity = iv_sensitivity_grid(base_inputs, result.cleaned_correlation, int(run.get("days_to_target", days_to_target)), int(iv_sensitivity_paths), int(run.get("seed", seed)), [single_iv_ticker], "Single-name")
                    st.session_state.last_pair_iv_sensitivity = iv_sensitivity_grid(base_inputs, result.cleaned_correlation, int(run.get("days_to_target", days_to_target)), int(iv_sensitivity_paths), int(run.get("seed", seed)), pair_iv_tickers, "Pair") if pair_iv_tickers else None
                    st.session_state.last_iv_sensitivity_error = None
                except Exception as exc:
                    st.session_state.last_iv_sensitivity_error = str(exc)

        if st.session_state.get("last_iv_sensitivity_error"):
            st.error(st.session_state.last_iv_sensitivity_error)

        global_iv = st.session_state.get("last_global_iv_sensitivity")
        single_iv = st.session_state.get("last_single_iv_sensitivity")
        pair_iv = st.session_state.get("last_pair_iv_sensitivity")

        if global_iv is None and single_iv is None and pair_iv is None:
            st.info("Click Run IV sensitivity to calculate the global, single-name, and pair IV shock tables.")
        else:
            if global_iv is not None and not global_iv.empty:
                st.subheader("Global IV shock")
                global_pivot = global_iv.pivot(index="Shock", columns="Ticker", values="Model probability")
                global_pivot.index = [iv_shock_label(value) for value in global_pivot.index]
                st.dataframe(global_pivot.map(pct), use_container_width=True)
                selected_global = st.selectbox("Ticker for global IV chart", global_iv["Ticker"].unique().tolist(), key="iv_global_chart_ticker")
                global_slice = global_iv[global_iv["Ticker"] == selected_global].sort_values("Shock")
                left, right = st.columns(2)
                with left:
                    st.plotly_chart(px.line(global_slice, x="Shock", y="Model probability", markers=True, title=f"{selected_global}: P(#1) vs global IV shock"), use_container_width=True, key="iv_global_probability_chart")
                with right:
                    st.plotly_chart(px.line(global_slice, x="Shock", y="Average rank", markers=True, title=f"{selected_global}: average rank vs global IV shock"), use_container_width=True, key="iv_global_rank_chart")

            if single_iv is not None and not single_iv.empty:
                shocked_name = single_iv["Shocked tickers"].iloc[0]
                st.subheader("Single-name IV shock")
                st.write(f"Only **{shocked_name}** receives the IV shock.")
                single_pivot = single_iv.pivot(index="Shock", columns="Ticker", values="Model probability")
                single_pivot.index = [iv_shock_label(value) for value in single_pivot.index]
                st.dataframe(single_pivot.map(pct), use_container_width=True)
                selected_single = st.selectbox("Ticker for single-name IV chart", single_iv["Ticker"].unique().tolist(), key="iv_single_chart_ticker")
                single_slice = single_iv[single_iv["Ticker"] == selected_single].sort_values("Shock")
                left, right = st.columns(2)
                with left:
                    st.plotly_chart(px.line(single_slice, x="Shock", y="Model probability", markers=True, title=f"{selected_single}: P(#1) under {shocked_name} IV shock"), use_container_width=True, key="iv_single_probability_chart")
                with right:
                    st.plotly_chart(px.line(single_slice, x="Shock", y="Average rank", markers=True, title=f"{selected_single}: average rank under {shocked_name} IV shock"), use_container_width=True, key="iv_single_rank_chart")

            if pair_iv is not None and not pair_iv.empty:
                shocked_pair = pair_iv["Shocked tickers"].iloc[0]
                st.subheader("Pair IV shock")
                st.write(f"Both **{shocked_pair}** receive the same IV shock.")
                pair_pivot = pair_iv.pivot(index="Shock", columns="Ticker", values="Model probability")
                pair_pivot.index = [iv_shock_label(value) for value in pair_pivot.index]
                st.dataframe(pair_pivot.map(pct), use_container_width=True)
                selected_pair = st.selectbox("Ticker for pair IV chart", pair_iv["Ticker"].unique().tolist(), key="iv_pair_chart_ticker")
                pair_slice = pair_iv[pair_iv["Ticker"] == selected_pair].sort_values("Shock")
                left, right = st.columns(2)
                with left:
                    st.plotly_chart(px.line(pair_slice, x="Shock", y="Model probability", markers=True, title=f"{selected_pair}: P(#1) under pair IV shock"), use_container_width=True, key="iv_pair_probability_chart")
                with right:
                    st.plotly_chart(px.line(pair_slice, x="Shock", y="Average rank", markers=True, title=f"{selected_pair}: average rank under pair IV shock"), use_container_width=True, key="iv_pair_rank_chart")

            with st.expander("Full IV sensitivity tables"):
                if global_iv is not None:
                    st.write("Global IV shock")
                    st.dataframe(display_iv_sensitivity(global_iv), use_container_width=True, hide_index=True)
                if single_iv is not None:
                    st.write("Single-name IV shock")
                    st.dataframe(display_iv_sensitivity(single_iv), use_container_width=True, hide_index=True)
                if pair_iv is not None:
                    st.write("Pair IV shock")
                    st.dataframe(display_iv_sensitivity(pair_iv), use_container_width=True, hide_index=True)

        st.subheader("Interpretation")
        st.write("If the probability barely moves across IV shocks, the current market-cap gap dominates. If it moves a lot, the conclusion is sensitive to volatility assumptions and should be treated as less stable.")
'''

source = APP_CORE_PATH.read_text(encoding="utf-8")
source = source.replace('\nst.title("LargestCompany")', '\n' + IV_HELPERS + '\nst.title("LargestCompany")')
source = source.replace(
    'results_tab, correlation_tab, inputs_tab, diagnostics_tab, methodology_tab = st.tabs(["Results", "Correlation Analysis", "Inputs & Data", "Simulation Diagnostics", "Methodology"])',
    'results_tab, correlation_tab, iv_tab, inputs_tab, diagnostics_tab, methodology_tab = st.tabs(["Results", "Correlation Analysis", "IV Analysis", "Inputs & Data", "Simulation Diagnostics", "Methodology"])',
)
source = source.replace('\nwith diagnostics_tab:', '\n' + IV_TAB_BLOCK + '\nwith diagnostics_tab:')
exec(compile(source, str(APP_CORE_PATH), "exec"), globals())
