# LargestCompany

Phase 1 of the Polymarket Ranking Engine.

This is an experimental quantitative research app for estimating fair probabilities that each company in a universe finishes with the largest market capitalization at a target date.

The goal is not to predict stock prices and not to outperform the option market. The goal is to translate current market capitalization, option-implied volatility, target-date horizon, and correlation assumptions into ranking probabilities, then compare those probabilities with Polymarket YES prices.

## Prototype Status

Current data sources:

- Current market capitalization: Yahoo Finance current market cap via `yfinance`, or manual input
- Annualized implied volatility: manual input or Yahoo Finance option-chain near-ATM IV via `yfinance`
- Polymarket YES price: manual input
- Correlation matrix: Yahoo Finance historical adjusted close prices via `yfinance`, or manual input
- Target date / maturity: user-selected date

The next important product step is to replace manual Polymarket placeholders with an explicit Polymarket price pipeline or uploaded market snapshot.

## Phase 1 Scope

This phase builds the probability engine, current market-cap ingestion, historical correlation estimation, and an MVP implied-volatility source.

It does not include:

- full volatility skew or smile calibration
- hedging logic
- option payoff heatmaps
- portfolio optimization

## Default Universe

- NVDA
- AAPL
- MSFT
- GOOGL
- AMZN
- META
- AVGO
- TSLA
- BRK.B
- LLY

The universe is editable in the app under `Inputs & Data`.

## Market Capitalization Source

The app supports two market-cap modes:

1. Yahoo Finance current market cap, default
2. Manual market cap inputs

Yahoo market-cap mode uses `yfinance`:

- map app tickers to Yahoo symbols, e.g. `BRK.B` to `BRK-B`
- fetch `fast_info.market_cap` when available
- fall back to `info["marketCap"]` if needed
- replace the manual input table values before running the Monte Carlo simulation

This is suitable for the MVP research dashboard, but it should still be treated as a data input, not as audited fundamental data. If Yahoo is unavailable or stale for a ticker, switch to manual market-cap inputs.

## Model

The app uses correlated lognormal simulations of future market capitalizations:

```text
MC_T = MC_0 * exp((-0.5 * sigma^2) * T + sigma * sqrt(T) * Z)
```

Where:

- `MC_0` is current market capitalization
- `sigma` is annualized implied volatility
- `T = days_to_target / 365`
- `days_to_target = target_date - today`
- `Z` is a correlated normal shock

## Implied Volatility Source

The app supports two IV modes:

1. Manual IV inputs
2. Yahoo option-chain near-ATM IV

Yahoo IV mode uses `yfinance` option chains:

- map app tickers to Yahoo symbols, e.g. `BRK.B` to `BRK-B`
- fetch available option expiries
- select the expiry closest to the target date
- fetch the option chain for that expiry
- find the strike nearest current spot
- read call and put implied volatility at the near-ATM strike
- use the average of call IV and put IV as the annualized IV input

This is an MVP near-ATM estimate. It is not a full volatility smile/surface calibration. Future versions should ingest full option chains, clean bid/ask quotes, fit an IV surface, and derive terminal distributions from the surface.

## Correlation Estimation

The app supports three correlation modes:

1. EWMA historical correlation, default
2. Rolling historical correlation
3. Manual correlation matrix

Historical methods use adjusted close prices from Yahoo Finance through `yfinance`.

Log returns:

```text
r_t = log(P_t / P_{t-1})
```

Rolling historical correlation:

```text
Corr = PearsonCorr(log returns over trailing lookback window)
```

EWMA covariance and correlation:

```text
Cov_t = lambda * Cov_{t-1} + (1 - lambda) * r_t r_t'
Corr_ij = Cov_ij / sqrt(Cov_ii * Cov_jj)
```

Supported controls:

- price history period: 2y, 5y, 10y
- rolling lookback: 63, 126, 252, 504, 756 trading days
- EWMA lambda: 0.94 or 0.97

The final correlation matrix is symmetrized, clipped to [-1, 1], forced to diagonal 1.0, and repaired for small numerical positive-semidefinite issues before simulation.

## Outputs

The results table focuses on statistical ranking analysis:

- ticker
- current market cap
- implied volatility
- Polymarket YES price
- model probability of finishing #1
- model probability minus Polymarket price
- average simulated rank
- probability of finishing Top 2
- probability of finishing Top 3

The app also includes:

- ticker drilldown directly on the main Results tab
- selected ticker rank distribution
- interactive company comparison box plot
- terminal market-cap distribution percentiles for every company
- exact rank probability matrix, showing probability of Rank 1, Rank 2, Rank 3, etc.

The simulation diagnostics include terminal market-cap distribution statistics for every company:

- mean
- standard deviation
- 1st percentile
- 5th percentile
- 25th percentile
- 50th percentile / median
- 75th percentile
- 95th percentile
- 99th percentile

## Polymarket Odds

Polymarket YES prices remain manual for now. A later module can ingest market prices from Polymarket APIs and still allow manual overrides.

## Run

```bash
pip install -r requirements.txt
python -m streamlit run app.py
```

## Files

```text
app.py            Streamlit dashboard
model.py          Probability engine
market_data.py    Yahoo current market-cap extraction
correlations.py   Historical correlation estimation
iv_surfaces.py    Yahoo option-chain near-ATM IV extraction
requirements.txt  Python dependencies
```

This is research software, not investment advice.
