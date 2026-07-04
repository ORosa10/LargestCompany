from __future__ import annotations

import pandas as pd

from boundaries import calculate_conditional_win_curve, find_probability_boundaries


DEFAULT_CONFIDENCE_LEVELS = [0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.99]


def calculate_all_conditional_curves(
    result,
    current_caps: pd.Series,
    *,
    n_bins: int,
) -> dict[str, pd.DataFrame]:
    """Calculate the full conditional win curve for every simulated ticker."""
    curves: dict[str, pd.DataFrame] = {}
    for ticker in result.terminal_market_caps.columns:
        curves[str(ticker)] = calculate_conditional_win_curve(
            result.terminal_market_caps,
            str(ticker),
            ranks=result.ranks,
            current_market_cap=float(current_caps.loc[ticker]),
            n_bins=int(n_bins),
        )
    return curves


def boundaries_from_curves(
    curves: dict[str, pd.DataFrame],
    current_caps: pd.Series,
    confidence_levels: list[float],
) -> pd.DataFrame:
    """Query any confidence levels without rerunning the Phase 1 simulation."""
    tables = []
    for ticker, curve in curves.items():
        tables.append(
            find_probability_boundaries(
                curve,
                [float(value) for value in confidence_levels],
                current_market_cap=float(current_caps.loc[ticker]),
                ticker=ticker,
            )
        )
    return pd.concat(tables, ignore_index=True) if tables else pd.DataFrame()
