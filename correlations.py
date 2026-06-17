"""Historical correlation estimation from adjusted close prices."""

from __future__ import annotations

import numpy as np
import pandas as pd
import yfinance as yf


YAHOO_TICKER_OVERRIDES = {
    "BRK.B": "BRK-B",
}


def yahoo_symbol(ticker: str) -> str:
    """Map app ticker symbols to Yahoo Finance symbols."""

    clean = ticker.strip().upper()
    return YAHOO_TICKER_OVERRIDES.get(clean, clean.replace(".", "-"))


def fetch_adjusted_close(tickers: list[str], *, period: str = "5y") -> pd.DataFrame:
    """Fetch daily adjusted close prices from Yahoo Finance via yfinance.

    Returned columns use the original app tickers, not Yahoo's substituted symbols.
    """

    if len(tickers) < 2:
        raise ValueError("At least two tickers are required for correlation estimation.")

    yahoo_to_original = {yahoo_symbol(ticker): ticker for ticker in tickers}
    yahoo_tickers = list(yahoo_to_original.keys())

    raw = yf.download(
        yahoo_tickers,
        period=period,
        auto_adjust=False,
        progress=False,
        threads=True,
    )
    if raw.empty:
        raise ValueError("No historical price data returned from Yahoo Finance.")

    if isinstance(raw.columns, pd.MultiIndex):
        price_field = "Adj Close" if "Adj Close" in raw.columns.get_level_values(0) else "Close"
        prices = raw[price_field]
    else:
        price_field = "Adj Close" if "Adj Close" in raw.columns else "Close"
        prices = raw[[price_field]].copy()
        prices.columns = yahoo_tickers

    prices = prices.rename(columns=yahoo_to_original)
    prices = prices.reindex(columns=tickers)
    prices = prices.sort_index().ffill().dropna(how="any")

    if prices.empty:
        raise ValueError("Historical prices contain no complete rows after cleaning.")
    return prices


def calculate_log_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """Calculate daily log returns r_t = log(P_t / P_{t-1})."""

    numeric_prices = prices.apply(pd.to_numeric, errors="coerce").sort_index()
    if (numeric_prices <= 0).any().any():
        raise ValueError("Prices must be positive to calculate log returns.")
    returns = np.log(numeric_prices / numeric_prices.shift(1))
    return returns.replace([np.inf, -np.inf], np.nan).dropna(how="any")


def rolling_correlation(prices: pd.DataFrame, lookback_days: int) -> pd.DataFrame:
    """Estimate Pearson correlation of log returns over a trailing lookback window."""

    returns = calculate_log_returns(prices)
    if lookback_days <= 1:
        raise ValueError("lookback_days must be greater than 1.")
    if len(returns) < lookback_days:
        raise ValueError(f"Only {len(returns)} return observations available, but {lookback_days} requested.")
    corr = returns.tail(lookback_days).corr(method="pearson")
    return validate_correlation_matrix(corr)


def ewma_covariance(returns: pd.DataFrame, lambda_value: float) -> pd.DataFrame:
    """Estimate EWMA covariance matrix from demeaned daily log returns."""

    if not 0.0 < lambda_value < 1.0:
        raise ValueError("lambda_value must be between 0 and 1.")
    if len(returns) < 2:
        raise ValueError("At least two return observations are required.")

    demeaned = returns - returns.mean(axis=0)
    values = demeaned.to_numpy(dtype=float, copy=True)

    cov = np.cov(values, rowvar=False)
    if cov.ndim == 0:
        cov = np.array([[float(cov)]])

    for row in values:
        outer = np.outer(row, row)
        cov = lambda_value * cov + (1.0 - lambda_value) * outer

    cov = 0.5 * (cov + cov.T)
    return pd.DataFrame(cov, index=returns.columns, columns=returns.columns)


def covariance_to_correlation(cov: pd.DataFrame) -> pd.DataFrame:
    """Convert covariance matrix to correlation matrix."""

    values = cov.to_numpy(dtype=float, copy=True)
    variances = np.diag(values)
    if np.any(variances <= 0):
        raise ValueError("Covariance matrix has non-positive variances.")

    std = np.sqrt(variances)
    corr_values = values / np.outer(std, std)
    corr = pd.DataFrame(corr_values, index=cov.index, columns=cov.columns)
    return validate_correlation_matrix(corr)


def ewma_correlation(prices: pd.DataFrame, lambda_value: float) -> pd.DataFrame:
    """Estimate EWMA correlation matrix from historical adjusted close prices."""

    returns = calculate_log_returns(prices)
    cov = ewma_covariance(returns, lambda_value)
    return covariance_to_correlation(cov)


def rolling_realized_volatility(
    returns: pd.DataFrame,
    window: int,
    *,
    annualization: int = 252,
) -> pd.DataFrame:
    """Calculate rolling annualized realized volatility from daily log returns."""

    if window <= 1:
        raise ValueError("volatility window must be greater than 1.")
    if len(returns) < window:
        raise ValueError(f"Only {len(returns)} return observations available, but volatility window {window} requested.")
    return returns.rolling(window=window).std() * np.sqrt(annualization)


def _pairwise_regime_correlation(
    returns: pd.DataFrame,
    realized_vol: pd.DataFrame,
    threshold: float,
    regime: str,
    *,
    min_observations: int = 30,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    tickers = returns.columns.tolist()
    corr = pd.DataFrame(np.eye(len(tickers)), index=tickers, columns=tickers)
    counts = pd.DataFrame(np.zeros((len(tickers), len(tickers)), dtype=int), index=tickers, columns=tickers)

    if regime not in {"low", "high"}:
        raise ValueError("regime must be 'low' or 'high'.")

    for i, ticker_i in enumerate(tickers):
        for ticker_j in tickers[i + 1 :]:
            pair_vol = 0.5 * (realized_vol[ticker_i] + realized_vol[ticker_j])
            mask = pair_vol >= threshold if regime == "high" else pair_vol < threshold
            pair_returns = returns.loc[mask, [ticker_i, ticker_j]].dropna()
            count = len(pair_returns)
            if count < min_observations:
                pair_returns = returns[[ticker_i, ticker_j]].dropna()
                count = len(pair_returns)
            rho = pair_returns[ticker_i].corr(pair_returns[ticker_j])
            if not np.isfinite(rho):
                rho = 0.0
            corr.loc[ticker_i, ticker_j] = rho
            corr.loc[ticker_j, ticker_i] = rho
            counts.loc[ticker_i, ticker_j] = count
            counts.loc[ticker_j, ticker_i] = count

    for ticker in tickers:
        counts.loc[ticker, ticker] = len(returns)
    return validate_correlation_matrix(corr), counts


def volatility_regime_correlation(
    prices: pd.DataFrame,
    *,
    vol_window: int,
    vol_threshold: float,
    regime: str,
    min_observations: int = 30,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Estimate pairwise correlations conditional on low or high realized-vol regimes."""

    returns = calculate_log_returns(prices)
    realized_vol = rolling_realized_volatility(returns, vol_window).reindex(index=returns.index)
    return _pairwise_regime_correlation(
        returns,
        realized_vol,
        vol_threshold,
        regime,
        min_observations=min_observations,
    )


def iv_based_regime_correlation(
    prices: pd.DataFrame,
    current_ivs: pd.Series,
    *,
    vol_window: int,
    vol_threshold: float,
    min_observations: int = 30,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Select low/high historical pair correlation using current average pair IV."""

    returns = calculate_log_returns(prices)
    realized_vol = rolling_realized_volatility(returns, vol_window).reindex(index=returns.index)
    low_corr, low_counts = _pairwise_regime_correlation(
        returns,
        realized_vol,
        vol_threshold,
        "low",
        min_observations=min_observations,
    )
    high_corr, high_counts = _pairwise_regime_correlation(
        returns,
        realized_vol,
        vol_threshold,
        "high",
        min_observations=min_observations,
    )

    tickers = returns.columns.tolist()
    corr = pd.DataFrame(np.eye(len(tickers)), index=tickers, columns=tickers)
    diagnostics = []
    for i, ticker_i in enumerate(tickers):
        for ticker_j in tickers[i + 1 :]:
            pair_iv = 0.5 * (float(current_ivs.loc[ticker_i]) + float(current_ivs.loc[ticker_j]))
            use_high = pair_iv >= vol_threshold
            selected_corr = high_corr.loc[ticker_i, ticker_j] if use_high else low_corr.loc[ticker_i, ticker_j]
            selected_count = high_counts.loc[ticker_i, ticker_j] if use_high else low_counts.loc[ticker_i, ticker_j]
            corr.loc[ticker_i, ticker_j] = selected_corr
            corr.loc[ticker_j, ticker_i] = selected_corr
            diagnostics.append(
                {
                    "Ticker 1": ticker_i,
                    "Ticker 2": ticker_j,
                    "Average current IV": pair_iv,
                    "Selected regime": "high" if use_high else "low",
                    "Selected correlation": selected_corr,
                    "Historical observations": int(selected_count),
                }
            )

    return validate_correlation_matrix(corr), pd.DataFrame(diagnostics), pd.DataFrame({"Ticker": tickers})


def _percentile_rank(values: pd.Series, point: float) -> float:
    clean = values.dropna().astype(float)
    if clean.empty:
        return 0.5
    return float((clean <= point).mean())


def smooth_vol_adjusted_correlation(
    prices: pd.DataFrame,
    current_ivs: pd.Series,
    *,
    vol_window: int = 63,
    low_quantile: float = 0.40,
    high_quantile: float = 0.60,
    min_observations: int = 30,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Blend low/high historical pair correlations using current IV percentile.

    For each pair, current average pair IV is mapped into the historical
    distribution of pair realized volatility. That percentile is the blend weight:
    final_corr = (1 - w) * low_corr + w * high_corr.
    """

    if not 0.0 < low_quantile < high_quantile < 1.0:
        raise ValueError("low_quantile and high_quantile must satisfy 0 < low < high < 1.")

    returns = calculate_log_returns(prices)
    realized_vol = rolling_realized_volatility(returns, vol_window).reindex(index=returns.index)
    tickers = returns.columns.tolist()
    corr = pd.DataFrame(np.eye(len(tickers)), index=tickers, columns=tickers)
    diagnostics = []

    for i, ticker_i in enumerate(tickers):
        for ticker_j in tickers[i + 1 :]:
            pair_vol = 0.5 * (realized_vol[ticker_i] + realized_vol[ticker_j])
            pair_vol_clean = pair_vol.dropna().astype(float)
            pair_returns_all = returns[[ticker_i, ticker_j]].dropna()

            if pair_vol_clean.empty:
                low_cutoff = np.nan
                high_cutoff = np.nan
                low_returns = pair_returns_all
                high_returns = pair_returns_all
                weight = 0.5
                pair_iv = 0.5 * (float(current_ivs.loc[ticker_i]) + float(current_ivs.loc[ticker_j]))
            else:
                low_cutoff = float(pair_vol_clean.quantile(low_quantile))
                high_cutoff = float(pair_vol_clean.quantile(high_quantile))
                pair_iv = 0.5 * (float(current_ivs.loc[ticker_i]) + float(current_ivs.loc[ticker_j]))
                weight = _percentile_rank(pair_vol_clean, pair_iv)

                low_mask = pair_vol <= low_cutoff
                high_mask = pair_vol >= high_cutoff
                low_returns = returns.loc[low_mask, [ticker_i, ticker_j]].dropna()
                high_returns = returns.loc[high_mask, [ticker_i, ticker_j]].dropna()

            if len(low_returns) < min_observations:
                low_returns = pair_returns_all
            if len(high_returns) < min_observations:
                high_returns = pair_returns_all

            low_corr = low_returns[ticker_i].corr(low_returns[ticker_j])
            high_corr = high_returns[ticker_i].corr(high_returns[ticker_j])
            if not np.isfinite(low_corr):
                low_corr = 0.0
            if not np.isfinite(high_corr):
                high_corr = low_corr

            blended_corr = (1.0 - weight) * low_corr + weight * high_corr
            corr.loc[ticker_i, ticker_j] = blended_corr
            corr.loc[ticker_j, ticker_i] = blended_corr
            diagnostics.append(
                {
                    "Ticker 1": ticker_i,
                    "Ticker 2": ticker_j,
                    "Average current IV": pair_iv,
                    "Historical pair-vol percentile": weight,
                    "Low-vol cutoff": low_cutoff,
                    "High-vol cutoff": high_cutoff,
                    "Low-regime correlation": low_corr,
                    "High-regime correlation": high_corr,
                    "Blend weight": weight,
                    "Selected correlation": blended_corr,
                    "Low observations": len(low_returns),
                    "High observations": len(high_returns),
                }
            )

    return validate_correlation_matrix(corr), pd.DataFrame(diagnostics)


def validate_correlation_matrix(corr: pd.DataFrame, *, tolerance: float = 1e-8) -> pd.DataFrame:
    """Validate and gently repair numerical issues in a correlation matrix."""

    matrix = corr.copy().astype(float)
    values = matrix.to_numpy(dtype=float, copy=True)

    if values.shape[0] != values.shape[1]:
        raise ValueError("Correlation matrix must be square.")
    if not np.isfinite(values).all():
        raise ValueError("Correlation matrix contains missing or non-finite values.")

    values = 0.5 * (values + values.T)
    values = np.clip(values, -1.0, 1.0)
    np.fill_diagonal(values, 1.0)

    eigenvalues = np.linalg.eigvalsh(values)
    min_eigenvalue = float(eigenvalues.min())
    if min_eigenvalue < -tolerance:
        values = values + np.eye(values.shape[0]) * (abs(min_eigenvalue) + tolerance)
        diagonal = np.sqrt(np.diag(values))
        values = values / np.outer(diagonal, diagonal)
        values = np.clip(values, -1.0, 1.0)
        np.fill_diagonal(values, 1.0)

    return pd.DataFrame(values, index=matrix.index, columns=matrix.columns)
