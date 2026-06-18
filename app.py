from __future__ import annotations

from pathlib import Path


APP_CORE_PATH = Path(__file__).with_name("app_core.py")
source = APP_CORE_PATH.read_text(encoding="utf-8")

source = source.replace(
    'CORRELATION_METHODS = [\n    "EWMA historical correlation",\n    "Vol-adjusted smooth correlation",\n    "Rolling historical correlation",\n    "Manual/default correlation matrix",\n]',
    'CORRELATION_METHODS = [\n    "EWMA historical correlation",\n    "Vol-adjusted smooth correlation",\n    "Rolling historical correlation",\n]',
)
source = source.replace(
    '    market_cap_source = st.selectbox("Market cap source", ["Yahoo Finance current market cap", "Manual market cap inputs"], index=0)\n',
    '    market_cap_source = "Yahoo Finance current market cap"\n    st.caption("Market caps: Yahoo Finance current market cap")\n',
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
    st.write("Edit only the event-specific assumptions here: tickers, annualized IV, and Polymarket YES prices. Market caps come from Yahoo Finance and correlations come from historical stock prices.")
    previous_inputs = st.session_state.company_inputs.copy()
    editable_inputs = previous_inputs[["Ticker", "Implied volatility", "Polymarket YES price"]].copy()
    edited_inputs = st.data_editor(
        editable_inputs,
        num_rows="dynamic",
        use_container_width=True,
        hide_index=True,
        column_config={
            "Ticker": st.column_config.TextColumn(required=True),
            "Implied volatility": st.column_config.NumberColumn("Manual IV", min_value=0.0001, max_value=5.0, step=0.01),
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
    st.caption("Manual market caps and manual correlation matrices are hidden from the main flow. Use Data Used to inspect the Yahoo market caps and selected correlation matrix after running.")
'''
source = source.replace(old_inputs_block, new_inputs_block)
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

exec(compile(source, str(APP_CORE_PATH), "exec"), globals())
