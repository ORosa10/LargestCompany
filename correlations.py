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
        raise ValueError(
            f"Only {len(returns)} return observations available, but {lookback_days} requested."
        )
    corr = returns.tail(lookback_days).corr(method="pearson")
    return validate_correlation_matrix(corr)


def ewma_covariance(returns: pd.DataFrame, lambda_value: float) -> pd.DataFrame:
    """Estimate EWMA covariance matrix from demeaned daily log returns."""

    if not 0.0 < lambda_value < 1.0:
        raise ValueError("lambda_value must be between 0 and 1.")
    if len(returns) < 2:
        raise ValueError("At least two return observations are required.")

    demeaned = returns - returns.mean(axis=0)
    values = demeaned.to_numpy(dtype=float)

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

    values = cov.to_numpy(dtype=float)
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


def validate_correlation_matrix(corr: pd.DataFrame, *, tolerance: float = 1e-8) -> pd.DataFrame:
    """Validate and gently repair numerical issues in a correlation matrix."""

    matrix = corr.copy().astype(float)
    values = matrix.to_numpy(dtype=float)

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
