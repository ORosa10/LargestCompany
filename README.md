# LargestCompany

Polymarket Ranking Engine.

This is an experimental quantitative research app for estimating fair probabilities that each company in a universe finishes with the largest market capitalization at a target date.

The objective is not to predict stock prices and not to outperform the option market. The objective is to translate observable market inputs into ranking probabilities, then compare those fair probabilities with Polymarket YES prices.

This is research software, not investment advice.

## Baseline Model

The default model is intentionally simple and auditable:

- Current market capitalization: Yahoo Finance current market cap via `yfinance`
- Annualized implied volatility: manual input
- Polymarket YES price: manual input
- Correlation matrix: EWMA historical correlation from Yahoo Finance adjusted close prices
- Shock distribution: normal shocks
- Drift: zero expected excess return, with only the lognormal convexity adjustment
- Dividends: ignored for now

The app also contains diagnostic pages for correlation sensitivity, IV sensitivity, return-distribution shape, and a Phase 2 workspace. Those are analysis views. The baseline probability output should be read through the default model above unless a different assumption set is explicitly selected.

## What This Tool Does

The engine estimates probabilities for ranking events such as:

```text
Which company will have the largest market capitalization at the target date?
```

For each simulation path, the app simulates terminal market capitalization for every company, ranks the companies, and records the winner and rank distribution.

The core outputs are:

- probability each company finishes #1
- probability each company finishes Top 2
- probability each company finishes Top 3
- average simulated rank
- model probability versus Polymarket YES price
- terminal market-cap distribution percentiles
- pairwise probability diagnostics
- conditional market-cap boundaries for target probabilities

## What This Tool Does Not Do

The current app does not include:

- stock-price forecasting
- alpha generation against option markets
- hedging logic
- option payoff heatmaps
- portfolio optimization
- full volatility skew or smile calibration
- automated Polymarket odds ingestion

Those belong to later phases.

## Phase 1: Probability Engine

The app uses correlated lognormal simulations of future market capitalizations:

```text
MC_T = MC_0 * exp((-0.5 * sigma^2) * T + sigma * sqrt(T) * Z)
```

Where:

- `MC_0` is current market capitalization
- `sigma` is annualized implied volatility
- `T = days_to_target / 365`
- `Z` is a correlated normal shock

The `-0.5 * sigma^2 * T` term is the lognormal adjustment. It is not an expected-return forecast.

## Phase 2: Conditional Probability Boundaries

Phase 2 lives in the `Phase 2` Streamlit page. Inside that page, Phase 2 modules are organized as tabs so they do not mix with the earlier Phase 1 diagnostic pages in the sidebar.

The first Phase 2 tab is `Conditional Boundaries`. It answers questions such as:

- what market cap does NVDA need for 50%, 60%, 70%, or 80% probability of finishing #1?
- how much would the selected company's market cap need to move versus today to reach that probability?
- what selected-company market cap corresponds to 50/50, 60/40, 70/30 pairwise odds versus each competitor?
- how does P(#1) change as the selected company's market cap is shocked up or down?

There are two boundary methods:

1. Full winner boundaries: solved by repeatedly rerunning the Phase 1 Monte Carlo engine while changing only the selected company's current market cap. This captures the full multi-company event where the selected ticker must beat every company in the universe.

2. Pairwise boundaries: solved with the analytic lognormal ratio formula for `P(selected terminal market cap > competitor terminal market cap)`. Pairwise boundaries are fast and useful for understanding nearest competitors, but they are not the same as full P(#1).

Phase 2 is still probability analysis only. It does not construct hedges or optimize payoffs.

## Drift And Dividends

For now, the model does not add a risk-free drift, equity risk premium, or dividend yield.

That is deliberate. This tool is comparing relative ranking probabilities using current market caps, implied volatility, and correlation assumptions. For short-dated ranking markets, drift and dividends are usually second-order compared with current market-cap gaps and volatility. They can be added later as explicit scenario inputs if needed.

## Correlation Estimation

The default correlation method is EWMA historical correlation using Yahoo Finance adjusted close prices.

Daily log returns are calculated as:

```text
r_t = log(P_t / P_{t-1})
```

EWMA covariance is estimated as:

```text
Cov_t = lambda * Cov_{t-1} + (1 - lambda) * r_t r_t'
Corr_ij = Cov_ij / sqrt(Cov_ii * Cov_jj)
```

The app also provides diagnostic alternatives:

- rolling historical correlation
- volatility-adjusted smooth correlation
- low-vol regime correlation
- high-vol regime correlation
- IV-based hard-switch regime correlation
- constant-correlation stress tests

These are useful for understanding sensitivity, not because there is one obviously perfect correlation model.

## Volatility Inputs

Manual IV is the baseline because this is a probability engine, not a full option-surface engine.

A Yahoo option-chain near-ATM IV helper exists as an MVP diagnostic source. It selects the option expiry closest to the target date, finds the strike nearest spot, and averages call/put implied volatility at that strike. This is not a full volatility surface and should not be treated as final production-quality IV ingestion.

## Polymarket Odds

Polymarket YES prices are manual for now. A later module can ingest prices from Polymarket APIs or uploaded market snapshots while still allowing manual overrides.

## Validation

The test suite covers:

- winner probabilities sum to 100%
- Top 2 / Top 3 probabilities are internally consistent
- rank distributions sum to 100% for each ticker
- bad company inputs are rejected
- correlation matrices are reindexed, symmetrized, and repaired where appropriate
- pairwise conditional boundaries hit their target probabilities
- winner probability rises when the selected company's market cap rises

Run tests with:

```bash
pytest
```

## Run

```bash
pip install -r requirements.txt
python -m streamlit run app.py
```

In GitHub Codespaces, expose the Streamlit server with:

```bash
python -m streamlit run app.py --server.address 0.0.0.0 --server.port 8501
```

## Files

```text
app.py                          Streamlit app loader
app_core.py                     Main Streamlit dashboard
model.py                        Probability engine
boundaries.py                   Conditional probability boundary calculations
market_data.py                  Yahoo current market-cap extraction
correlations.py                 Historical and volatility-adjusted correlation estimation
iv_surfaces.py                  Yahoo option-chain near-ATM IV extraction
pages/Phase_2.py                Phase 2 workspace with internal tabs
pages/Correlation_Comparison.py Correlation analysis page
pages/IV_Analysis.py            IV sensitivity page
pages/Return_Diagnostics.py     Return-shape diagnostics page
tests/test_model.py             Probability engine sanity tests
tests/test_boundaries.py        Conditional boundary tests
requirements.txt                Python dependencies
```

## Phase Roadmap

Phase 1: probability engine.

Phase 2: conditional probability boundaries.

Phase 3: option-based hedge structures.

Phase 4: payoff surfaces and portfolio construction.
