from __future__ import annotations


def _replace_once(source: str, old: str, new: str, label: str) -> str:
    if old not in source:
        raise RuntimeError(f"Phase 1 UI patch target not found: {label}")
    return source.replace(old, new, 1)


def apply_phase1_ui_patches(source: str) -> str:
    source = _replace_once(
        source,
        '''    probability_model = st.selectbox("Probability model", ["IV surface marginals", "ATM lognormal"], index=0)
    iv_source = st.selectbox("Fallback IV source", ["Manual IV inputs", "Yahoo option-chain near-ATM IV"], index=0)
    st.caption("Surface model: manually calibrated 2026-07-31 smiles for NVDA, AAPL, and GOOGL; ATM lognormal fallback for other tickers.")
    forward_source = st.selectbox("Forward / dividend carry", ["Put-call parity implied forward", "Flat forward (legacy)"], index=0)''',
        '''    probability_model = st.selectbox("Probability model", ["IV surface marginals", "ATM lognormal"], index=0)
    if probability_model == "IV surface marginals":
        iv_source = "Manual IV inputs"
        st.caption("Surface model: calibrated 2026-07-31 smiles for NVDA, AAPL, and GOOGL; internal ATM fallback for the residual universe.")
    else:
        iv_source = st.selectbox("IV source", ["Manual IV inputs", "Yahoo option-chain near-ATM IV"], index=0)
    forward_source = st.selectbox("Forward / dividend carry", ["Put-call parity implied forward", "Flat forward (legacy)"], index=0)''',
        "conditional IV source",
    )

    source = _replace_once(
        source,
        '''with inputs_tab:
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
''',
        '''with inputs_tab:
    previous_inputs = st.session_state.company_inputs.copy()
    if probability_model == "IV surface marginals":
        st.subheader("Polymarket outcomes")
        st.write("Only this compact list is compared with Polymarket. Deleting a row sets its market price to zero, but the company remains in the Monte Carlo universe and correlation matrix.")
        if "surface_event_prices" not in st.session_state:
            st.session_state.surface_event_prices = JULY_2026_EVENT_PRICES.copy()
        edited_event_prices = st.data_editor(
            st.session_state.surface_event_prices,
            num_rows="dynamic",
            use_container_width=True,
            hide_index=True,
            column_config={
                "Ticker": st.column_config.SelectboxColumn("Ticker", options=previous_inputs["Ticker"].astype(str).tolist(), required=True),
                "Polymarket YES price": st.column_config.NumberColumn(
                    "Polymarket YES price", min_value=0.0, max_value=1.0, step=0.001, format="%.4f", required=True
                ),
            },
        )
        company_inputs, visible_event_prices, unknown_tickers = apply_event_prices(previous_inputs, edited_event_prices)
        st.session_state.surface_event_prices = visible_event_prices
        st.session_state.company_inputs = company_inputs
        if unknown_tickers:
            st.warning("Ignored tickers outside the simulation universe: " + ", ".join(unknown_tickers))

        st.subheader("Calibrated IV surface nodes")
        surface_display = default_surface_nodes().copy()
        surface_display["IV"] = surface_display["IV"].map(pct)
        surface_display["Moneyness"] = surface_display["Moneyness"].map(pct)
        st.dataframe(surface_display, use_container_width=True, hide_index=True)
    else:
        st.subheader("ATM lognormal inputs")
        st.write("Edit the ATM volatility and Polymarket price used by the fallback model. Market caps still come from Yahoo.")
        editable_inputs = previous_inputs[["Ticker", "Implied volatility", "Polymarket YES price"]].copy()
        edited_inputs = st.data_editor(
            editable_inputs,
            num_rows="dynamic",
            use_container_width=True,
            hide_index=True,
            column_config={
                "Ticker": st.column_config.TextColumn(required=True),
                "Implied volatility": st.column_config.NumberColumn("ATM IV", min_value=0.0001, max_value=5.0, step=0.001, format="%.4f"),
                "Polymarket YES price": st.column_config.NumberColumn("Polymarket YES price", min_value=0.0, max_value=1.0, step=0.001, format="%.4f"),
            },
        )
        cap_fallback = pd.concat([previous_inputs[["Ticker", "Current market cap"]], default_company_inputs()[["Ticker", "Current market cap"]]]).drop_duplicates("Ticker", keep="first")
        company_inputs = edited_inputs.merge(cap_fallback, on="Ticker", how="left")
        company_inputs["Current market cap"] = company_inputs["Current market cap"].fillna(default_company_inputs()["Current market cap"].median())
        company_inputs = company_inputs[["Ticker", "Current market cap", "Implied volatility", "Polymarket YES price"]]
        st.session_state.company_inputs = company_inputs

    tickers = [ticker for ticker in st.session_state.company_inputs["Ticker"].astype(str).str.strip().tolist() if ticker]
    st.session_state.correlation_matrix = default_correlation_matrix(tickers)
    st.caption("Use Data Used to inspect the full internal universe, Yahoo market caps, implied forwards, marginal models, and the selected correlation matrix after running.")
''',
        "surface event editor",
    )

    source = _replace_once(
        source,
        '''            corr, corr_label, price_info = select_correlation_matrix(correlation_method, simulation_inputs, manual_correlation_matrix, price_history_period, float(ewma_lambda), int(rolling_lookback), float(smooth_low_quantile), float(smooth_high_quantile))''',
        '''            if probability_model == "IV surface marginals":
                simulation_inputs = apply_surface_atm_ivs(simulation_inputs)
            corr, corr_label, price_info = select_correlation_matrix(correlation_method, simulation_inputs, manual_correlation_matrix, price_history_period, float(ewma_lambda), int(rolling_lookback), float(smooth_low_quantile), float(smooth_high_quantile))''',
        "surface ATM anchors before correlation",
    )

    source = _replace_once(
        source,
        '''        st.subheader("Fair ranking probabilities")
        st.dataframe(display_results(result.results), use_container_width=True, hide_index=True)

        left, right = st.columns(2)
        with left:
            chart = px.scatter(result.results, x="Polymarket YES price", y="Model probability", text="Ticker", title="Fair probability vs Polymarket price")
            chart.add_shape(type="line", x0=0, y0=0, x1=1, y1=1, line={"dash": "dash", "color": "gray"})
            chart.update_traces(textposition="top center")
            st.plotly_chart(chart, use_container_width=True, key="overview_prob_scatter")
        with right:
            st.plotly_chart(px.bar(result.results.sort_values("Edge"), x="Ticker", y="Edge", color="Edge", color_continuous_scale="RdYlGn", title="Fair probability minus Polymarket price"), use_container_width=True, key="overview_edge_bar")''',
        '''        event_results = result.results[result.results["Polymarket YES price"] > 0].copy()
        residual_results = result.results[result.results["Polymarket YES price"] <= 0].copy()
        st.subheader("Polymarket outcomes")
        st.dataframe(display_results(event_results), use_container_width=True, hide_index=True)
        if not residual_results.empty:
            with st.expander("Other companies retained in the simulation universe"):
                st.caption("These firms affect ranks and dependence, but their Polymarket comparison price is zero.")
                st.dataframe(display_results(residual_results), use_container_width=True, hide_index=True)

        left, right = st.columns(2)
        with left:
            chart = px.scatter(event_results, x="Polymarket YES price", y="Model probability", text="Ticker", title="Fair probability vs Polymarket price")
            chart.add_shape(type="line", x0=0, y0=0, x1=1, y1=1, line={"dash": "dash", "color": "gray"})
            chart.update_traces(textposition="top center")
            st.plotly_chart(chart, use_container_width=True, key="overview_prob_scatter")
        with right:
            st.plotly_chart(px.bar(event_results.sort_values("Edge"), x="Ticker", y="Edge", color="Edge", color_continuous_scale="RdYlGn", title="Fair probability minus Polymarket price"), use_container_width=True, key="overview_edge_bar")''',
        "event-only overview",
    )

    source = _replace_once(
        source,
        '''        left, right = st.columns(2)
        with left:
            caps_t = result.terminal_market_caps[selected_ticker] / 1e12
            hist = px.histogram(caps_t, nbins=80, title=f"Terminal market-cap distribution: {selected_ticker}")
            hist.update_layout(xaxis_title="Market cap ($T)", yaxis_title="Simulation count")
            st.plotly_chart(hist, use_container_width=True, key="ticker_hist")
        with right:
            rank_data = result.rank_distribution[result.rank_distribution["Ticker"] == selected_ticker]
            st.plotly_chart(px.bar(rank_data, x="Rank", y="Probability", title=f"Rank distribution: {selected_ticker}"), use_container_width=True, key="ticker_rank")

        st.subheader("Market-cap distribution percentiles")
        percentiles = market_cap_percentile_table(result)
        st.dataframe(display_percentiles(percentiles[percentiles["Ticker"] == selected_ticker]), use_container_width=True, hide_index=True)''',
        '''        current_market_cap = float(row["Current market cap"])
        current_market_cap_t = current_market_cap / 1e12
        left, right = st.columns(2)
        with left:
            caps_t = result.terminal_market_caps[selected_ticker] / 1e12
            hist = px.histogram(caps_t, nbins=80, title=f"Terminal market-cap distribution: {selected_ticker}")
            hist.add_vline(
                x=current_market_cap_t,
                line_dash="dash",
                line_color="#ff4b4b",
                annotation_text=f"Current ${current_market_cap_t:,.2f}T",
                annotation_position="top",
            )
            hist.update_layout(xaxis_title="Market cap ($T)", yaxis_title="Simulation count")
            st.plotly_chart(hist, use_container_width=True, key="ticker_hist")
        with right:
            rank_data = result.rank_distribution[result.rank_distribution["Ticker"] == selected_ticker]
            st.plotly_chart(px.bar(rank_data, x="Rank", y="Probability", title=f"Rank distribution: {selected_ticker}"), use_container_width=True, key="ticker_rank")

        st.subheader("Market-cap distribution percentiles")
        percentiles = market_cap_percentile_table(result)
        selected_percentiles = display_percentiles(percentiles[percentiles["Ticker"] == selected_ticker])
        selected_percentiles.insert(1, "Current market cap", dollars_trillions(current_market_cap))
        st.dataframe(selected_percentiles, use_container_width=True, hide_index=True)''',
        "ticker current market cap",
    )

    source = _replace_once(
        source,
        '''            st.subheader("ATM-model sensitivity by approach")
            st.caption("This comparison tab remains an ATM correlation/shock sensitivity. The main Overview result uses the probability model selected in the sidebar.")
            selected_compare_ticker = st.selectbox("Ticker for approach comparison", tickers, index=tickers.index("NVDA") if "NVDA" in tickers else 0, key="compare_ticker")
            ticker_comparison = comparison[comparison["Ticker"] == selected_compare_ticker].copy()
            pivot = ticker_comparison.pivot(index="Correlation method", columns="Shock model", values="Model probability")
            st.dataframe(pivot.map(pct), use_container_width=True)
            st.plotly_chart(px.bar(ticker_comparison, x="Correlation method", y="Model probability", color="Shock model", barmode="group", title=f"{selected_compare_ticker}: probability by correlation and shock model"), use_container_width=True, key="compare_bar")

            st.subheader("All tickers comparison")
            display = comparison.copy()
            display["Model probability"] = display["Model probability"].map(pct)
            display["Edge"] = display["Edge"].map(pct)
            display["Average rank"] = display["Average rank"].map(lambda value: f"{value:.2f}")
            st.dataframe(display, use_container_width=True, hide_index=True)''',
        '''            st.subheader("Probability model comparison")
            st.caption("The selected main model is shown beside the ATM correlation and shock alternatives for one ticker. This isolates the model-choice effect without a distracting all-ticker dump.")
            selected_compare_ticker = st.selectbox("Ticker", tickers, index=tickers.index("NVDA") if "NVDA" in tickers else 0, key="compare_ticker")
            ticker_comparison = comparison[comparison["Ticker"] == selected_compare_ticker].copy()
            ticker_comparison["Approach"] = ticker_comparison["Correlation method"] + " | " + ticker_comparison["Shock model"]
            main_probability = float(result.results.set_index("Ticker").loc[selected_compare_ticker, "Model probability"])
            main_label = sources.get("Probability model", probability_model)
            main_comparison = pd.DataFrame(
                [{
                    "Approach": f"Main model: {main_label}",
                    "Correlation method": correlation_method,
                    "Shock model": "Surface-implied marginals" if probability_model == "IV surface marginals" else shock_model,
                    "Model probability": main_probability,
                }]
            )
            approach_comparison = pd.concat(
                [main_comparison, ticker_comparison[["Approach", "Correlation method", "Shock model", "Model probability"]]],
                ignore_index=True,
            )
            display_comparison = approach_comparison.copy()
            display_comparison["Model probability"] = display_comparison["Model probability"].map(pct)
            st.dataframe(display_comparison, use_container_width=True, hide_index=True)
            comparison_chart = px.bar(
                approach_comparison,
                x="Approach",
                y="Model probability",
                color="Shock model",
                title=f"{selected_compare_ticker}: fair probability by model",
            )
            comparison_chart.update_layout(xaxis_title="", yaxis_tickformat=".0%")
            st.plotly_chart(comparison_chart, use_container_width=True, key="compare_bar")''',
        "model comparison",
    )
    return source
