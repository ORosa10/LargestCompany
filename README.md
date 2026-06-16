# LargestCompany

Phase 1 of the Polymarket Ranking Engine.

This is an experimental quantitative research app for estimating fair probabilities that each company in a universe finishes with the largest market capitalization at a target date.

The goal is not to predict stock prices and not to outperform the option market. The goal is to translate current market capitalization, option-implied volatility, target-date horizon, and correlation assumptions into ranking probabilities, then compare those probabilities with Polymarket YES prices.

## Prototype Status

Current data sources:

- Current market capitalization: manual placeholder input
- Annualized implied volatility: manual placeholder input
- Polymarket YES price: manual placeholder input
- Correlation matrix: Yahoo Finance historical adjusted close prices via `yfinance`, or manual input
- Target date / maturity: user-selected date

The next important product step is to replace manual market cap, IV, and Polymarket placeholders with explicit data pipelines or uploaded snapshots.

## Phase 1 Scope

This phase builds the probability engine and historical correlation estimation.

It does not include:

- live market-cap ingestion
- live option-surface ingestion
- volatility skew or smile calibration
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

## IV Surface Roadmap

Volatility skew/smile is not modeled yet. The current engine uses one flat annualized implied volatility per company.

Possible live or near-live IV sources for a future module:

- Polygon.io options snapshots and Greeks
- Tradier options chains
- Interactive Brokers market data, if an IBKR account/data subscription is available
- Cboe DataShop, usually better for paid/institutional workflows
- Nasdaq Data Link / ORATS, usually paid but more research-friendly
- Yahoo Finance option chains via `yfinance`, useful for MVP experiments but not ideal as a robust production source

The future IV module should select an option expiry near the Polymarket target date, ingest option chains, clean bid/ask quotes, infer or read implied vols, and calibrate a terminal distribution instead of using one flat IV input.

## Polymarket Odds

Polymarket YES prices can remain manual for now. A later module can ingest market prices from Polymarket APIs and still allow manual overrides.

## Run

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Files

```text
app.py            Streamlit dashboard
model.py          Probability engine
correlations.py   Historical correlation estimation
requirements.txt  Python dependencies
```

This is research software, not investment advice.
