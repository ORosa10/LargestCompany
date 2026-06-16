# LargestCompany

Phase 1 of the Polymarket Ranking Engine.

This is an experimental quantitative research app for estimating fair probabilities that each company in a universe finishes with the largest market capitalization at a target date.

The goal is not to predict stock prices and not to outperform the option market. The goal is to translate current market capitalization, option-implied volatility, and correlation assumptions into ranking probabilities, then compare those probabilities with Polymarket YES prices.

## Phase 1 Scope

This phase only builds the probability engine.

It does not include:

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
- `Z` is a correlated normal shock

The correlation matrix is validated, symmetrized if needed, and passed through Cholesky decomposition. Small diagonal jitter is used only when needed for numerical stability.

## Outputs

The results table includes:

- ticker
- current market cap
- implied volatility
- Polymarket YES price
- model probability
- edge
- expected value
- ROI
- average rank
- probability Top 2
- probability Top 3

The dashboard also shows:

- model probability vs Polymarket probability
- edge by ticker
- correlation matrix heatmap
- rank distribution for selected ticker
- simulated market capitalization distribution for selected ticker

## Run

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Files

```text
app.py            Streamlit dashboard
model.py          Probability engine
requirements.txt  Python dependencies
```

This is research software, not investment advice.
