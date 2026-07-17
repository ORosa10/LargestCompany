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

The app also contains diagnostic pages for correlation sensitivity, IV sensitivity, return-distribution shape, and phase workspaces. Those are analysis views. The baseline probability output should be read through the default model above unless a different assumption set is explicitly selected.

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
- conditional win/loss boundary zones for a selected ticker
- candidate option building blocks derived from those boundaries
- scenario-level payoff and payoff profile diagnostics

## What This Tool Does Not Do

The current app does not include:

- stock-price forecasting
- alpha generation against option markets
- hedge ratio optimization
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

The main Phase 2 method uses the actual Phase 1 Monte Carlo scenarios:

1. Simulate final market caps for all companies.
2. Store the selected ticker's terminal market cap.
3. Store the selected ticker's final rank.
4. Store whether the selected ticker finished #1.
5. Sort/bin scenarios by selected ticker terminal market cap.
6. Estimate conditional probabilities inside each bin.

For each selected ticker, the app builds a conditional curve:

```text
P(selected ticker wins | selected ticker terminal market cap is around level X)
```

The default confidence levels are 80%, 90%, 95%, and 99%.

For each confidence level, Phase 2 estimates:

- lower loss boundary: highest selected-ticker market-cap bin where P(loss) is at least the confidence level
- upper win boundary: lowest selected-ticker market-cap bin where P(win) is at least the confidence level
- each boundary as a percentage of the selected ticker's current market cap

These boundaries are not deterministic truths. They are conditional probabilities from the Monte Carlo model and depend on IVs, correlations, current market caps, and target date. They are intended to become useful inputs for later option-strike and hedge-structure analysis.

## Phase 3: Option Construction Engine

Phase 3 lives in the `Phase 3` Streamlit page.

Phase 3 does not optimize anything. It only constructs natural vanilla option building blocks from Phase 2 boundaries.

Construction modes:

- selected-only hedge: short call and long put on the ticker behind the YES bet
- selected + single competitor diagnostic: adds candidate competitor call/put legs for one chosen rival
- selected + full universe competitors: adds competitor protection candidates for every non-selected ticker

Market-cap boundaries are converted to stock-price strikes with:

```text
strike = spot price * boundary market cap / current market cap
```

The output is:

- suggested option structure
- explanation table for every instrument
- theoretical Black-Scholes premium diagnostics
- standalone payoff chart for each option leg

Phase 3 does not combine legs into a portfolio, choose hedge ratios, or estimate optimal quantities. Those belong to Phase 4 and Phase 5.

## Phase 4: Payoff Profile Engine

Phase 4 lives in the `Phase 4` Streamlit page.

Phase 4 combines Polymarket event payoff and Phase 3 option legs across the same Monte Carlo scenarios. It is still not an optimizer: option quantities are editable construction-preview inputs, and Phase 5 will search quantities and strikes systematically.

It calculates:

- scenario-level Polymarket payoff
- scenario-level option payoff
- total payoff per scenario
- expected payoff
- median payoff
- probability of loss
- worst payoff
- expected shortfall for the worst 5% of scenarios
- selected-ticker payoff profile bins
- one-dimensional payoff heatmap by selected ticker terminal market-cap ratio
- probability-weighted payoff contribution by selected ticker terminal market-cap bin

The payoff bridge is:

```text
Global expected payoff = sum(bin scenario probability * average payoff in bin)
```

Boundary confidence affects Phase 4 through option construction: different confidence levels create different strikes and theoretical premiums. If every option quantity is zero, the expected payoff becomes only the Polymarket payoff and boundary confidence has no payoff effect.

## Phase 7: Risk Assessment And Sensitivity

Phase 7 lives in the `Phase 7` Streamlit page and the `phase7.py` engine.

Phase 7 changes no model. It stresses the inputs of the existing engine and
quantifies how fragile the outputs are, so an edge can be classified as a robust
signal or an artifact of assumptions before it is traded. Every stress reuses
the Phase 1 probability engine and the Phase 4 payoff surface, so Phase 7
measures exactly the numbers the rest of the app produces.

1. Monte Carlo error. Means and ranking probabilities converge quickly, but tail
   metrics (expected shortfall, worst case) rest on few scenarios and converge
   slowly. Phase 7 reruns across seeds and reports the cross-seed standard
   deviation as a plus/minus error. Large relative dispersion on a tail metric
   is a signal to raise the simulation count.

2. Tail-dependence stress. Marginals and dependence are separated. The Gaussian
   copula has zero tail dependence; the Student-t copula (df=5, shared
   chi-square shock) makes extremes arrive jointly. Only the dependence family
   changes. Bounded-loss spreads keep the worst case fixed by construction, so
   the effect shows up in expected payoff and edge.

3. Gap vs randomness. Scaling every IV by k isolates randomness; widening or
   compressing the market-cap gaps at fixed volatility isolates structure. The
   scan that moves P(#1) more is the dominant lever.

5. Model robustness. P(#1) and edge are recomputed across a grid of correlation
   variants and shock models. An edge that keeps its sign across the grid is
   tradeable; one that flips is model-dependent.

Test 4 (out-of-sample optimizer validation) is intentionally omitted while the
workflow uses a manual portfolio rather than an automatic optimizer.

## Drift And Dividends

The model adds no equity risk premium. It does support an optional forward
carry: when a "Forward / spot" column is supplied (a put-call parity implied
forward, produced by `implied_forwards.py`), the base engine centres each
company on its forward, `E[MC_T] = MC_0 * forward / spot`, instead of pure
zero-drift. With no forward column the engine reduces to the original lognormal
convexity adjustment, so the baseline behaviour is unchanged.

Beyond that, the tool compares relative ranking probabilities using current
market caps, implied volatility, and correlation assumptions. For short-dated
ranking markets, drift and dividends are usually second-order compared with
current market-cap gaps and volatility.

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
- scenario-based conditional win curves and all-ticker boundary summaries
- winner probability rises when the selected company's market cap rises
- option-strike construction from market-cap boundaries
- standalone option payoff signs
- Phase 3 construction modes
- theoretical option premiums
- Phase 4 Polymarket payoff and scenario payoff aggregation
- selected-ticker payoff profile binning and expected-payoff contributions

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
option_construction.py          Phase 3 option construction engine
payoff_surface.py               Phase 4 payoff surface/profile engine
market_data.py                  Yahoo market-cap and spot-price extraction
correlations.py                 Historical and volatility-adjusted correlation estimation
iv_surfaces.py                  Yahoo option-chain near-ATM IV extraction
pages/Phase_2.py                Phase 2 workspace with internal tabs
pages/Phase_3.py                Phase 3 option construction workspace
pages/Phase_4.py                Phase 4 payoff profile workspace
pages/Phase_7.py                Phase 7 risk assessment and sensitivity workspace
pages/Correlation_Comparison.py Correlation analysis page
pages/IV_Analysis.py            IV sensitivity page
pages/Return_Diagnostics.py     Return-shape diagnostics page
tests/test_model.py             Probability engine sanity tests
tests/test_boundaries.py        Conditional boundary tests
tests/test_option_construction.py Option construction tests
tests/test_payoff_surface.py    Payoff surface/profile tests
tests/test_phase7.py            Phase 7 sensitivity and robustness tests
requirements.txt                Python dependencies
```

## Phase Roadmap

Phase 1: probability engine.

Phase 2: conditional probability boundaries.

Phase 3: option construction engine.

Phase 4: payoff profile engine.

Phase 5: optimization engine.

Phase 6: robustness engine.

Phase 7: risk assessment and sensitivity engine.
