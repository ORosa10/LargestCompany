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

This phase builds the probability engine, current market-cap ingestion, historical correlation estimation, volatility-adjusted correlation sensitivity, and an MVP implied-volatility source.

It does not include:

- full volatility skew or smile calibration
- hedging logic
- option payoff heatmaps
- portfolio optimization

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

## Implied Volatility Source

The app supports two IV modes:

1. Manual IV inputs
2. Yahoo option-chain near-ATM IV

Yahoo IV mode uses `yfinance` option chains, selects the expiry closest to the target date, finds the strike nearest spot, and averages call/put implied volatility at that strike.

This is an MVP near-ATM estimate. It is not a full volatility smile/surface calibration.

## Correlation Estimation

The app supports these correlation modes:

1. EWMA historical correlation, default
2. Vol-adjusted smooth correlation
3. Rolling historical correlation
4. Low-vol regime correlation
5. High-vol regime correlation
6. IV-based hard-switch regime correlation
7. Manual correlation matrix

Historical methods use adjusted close prices from Yahoo Finance through `yfinance`.

### EWMA

```text
r_t = log(P_t / P_{t-1})
Cov_t = lambda * Cov_{t-1} + (1 - lambda) * r_t r_t'
Corr_ij = Cov_ij / sqrt(Cov_ii * Cov_jj)
```

### Vol-Adjusted Smooth Correlation

This is the preferred volatility-regime sensitivity mode.

For each pair of companies:

```text
realized_vol_i,t = rolling_std(return_i, window) * sqrt(252)
pair_realized_vol_ij,t = average(realized_vol_i,t, realized_vol_j,t)
avg_current_IV_ij = average(IV_i, IV_j)
```

The current average pair IV is mapped into the historical distribution of pair realized volatility:

```text
w_ij = percentile_rank(avg_current_IV_ij, historical_pair_realized_vol_ij)
```

Then the pair correlation is blended smoothly:

```text
Corr_ij = (1 - w_ij) * Corr_low_ij + w_ij * Corr_high_ij
```

Where:

- `Corr_low_ij` is the pair correlation on low-vol historical days, default bottom 40% of pair realized-vol observations
- `Corr_high_ij` is the pair correlation on high-vol historical days, default top 40% of pair realized-vol observations
- `w_ij` is calculated from data, not manually chosen

This avoids a hard arbitrary switch such as 49.9% IV = low regime and 50.1% IV = high regime.

### Hard-Switch Regime Correlation

The older hard-switch mode remains available as a diagnostic:

```text
if avg_current_IV_ij >= threshold:
    use high-vol historical correlation for pair i,j
else:
    use low-vol historical correlation for pair i,j
```

This is less smooth and more threshold-sensitive than the preferred vol-adjusted smooth mode.

## Outputs

The app includes:

- statistical ranking probabilities
- ticker drilldown directly on the main Results tab
- selected ticker rank distribution
- pairwise probability audit
- volatility-adjusted correlation diagnostics
- interactive company comparison box plot
- terminal market-cap distribution percentiles for every company
- exact rank probability matrix

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
correlations.py   Historical and volatility-adjusted correlation estimation
iv_surfaces.py    Yahoo option-chain near-ATM IV extraction
requirements.txt  Python dependencies
```

This is research software, not investment advice.
