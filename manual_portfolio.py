"""Manual option-portfolio builder for the Phase 5 dashboard."""

from __future__ import annotations

import numpy as np
import pandas as pd

from option_construction import black_scholes_price, option_payoff


STRIKE_SOURCES = ["Manual strike", "Win boundary", "Loss boundary"]
BOUNDARY_TYPES = ["Win boundary", "Loss boundary"]
OPTION_TYPES = ["Call", "Put"]
POSITIONS = ["Long", "Short"]
BOUNDARY_CONFIDENCES = [80, 90, 95, 99]

EDITOR_COLUMNS = [
    "Active", "Ticker", "Option type", "Position", "Quantity",
    "Strike source", "Boundary confidence (%)", "Manual strike", "Pricing IV",
]
BOUNDARY_EDITOR_COLUMNS = [
    "Active", "Ticker", "Option type", "Position", "Quantity",
    "Boundary type", "Boundary confidence (%)", "Pricing IV",
]
MANUAL_STRIKE_EDITOR_COLUMNS = [
    "Active", "Ticker", "Option type", "Position", "Quantity", "Strike", "Pricing IV",
]
RESOLVED_COLUMNS = [
    "Instrument", "Ticker", "Option type", "Position", "Quantity", "Strike",
    "Strike / spot", "Strike source", "Boundary used", "Spot", "Model IV",
    "Risk-free rate", "Time to expiry", "Theoretical premium",
]
ANALYTICS_COLUMNS = [
    "Instrument", "Expected option payoff", "Option payoff SD", "P(option loss)",
    "Expected shortfall 5%", "Worst option payoff", "Initial premium cashflow",
]


def default_boundary_portfolio(ticker: str, implied_volatility: float) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "Active": True, "Ticker": ticker, "Option type": "Put", "Position": "Long",
                "Quantity": 0.10, "Boundary type": "Loss boundary",
                "Boundary confidence (%)": 80, "Pricing IV": implied_volatility,
            },
            {
                "Active": True, "Ticker": ticker, "Option type": "Call", "Position": "Short",
                "Quantity": 0.10, "Boundary type": "Win boundary",
                "Boundary confidence (%)": 80, "Pricing IV": implied_volatility,
            },
        ],
        columns=BOUNDARY_EDITOR_COLUMNS,
    )


def empty_manual_strike_portfolio() -> pd.DataFrame:
    return pd.DataFrame(columns=MANUAL_STRIKE_EDITOR_COLUMNS)


def default_manual_portfolio(ticker: str, implied_volatility: float) -> pd.DataFrame:
    """Backward-compatible unified default table."""
    return combine_portfolio_inputs(default_boundary_portfolio(ticker, implied_volatility), empty_manual_strike_portfolio())


def combine_portfolio_inputs(boundary_table: pd.DataFrame, manual_strike_table: pd.DataFrame) -> pd.DataFrame:
    """Combine mutually exclusive boundary and manual-strike inputs."""
    boundary_rows = pd.DataFrame(columns=EDITOR_COLUMNS)
    if not boundary_table.empty:
        boundary_rows = pd.DataFrame({
            "Active": boundary_table["Active"],
            "Ticker": boundary_table["Ticker"],
            "Option type": boundary_table["Option type"],
            "Position": boundary_table["Position"],
            "Quantity": boundary_table["Quantity"],
            "Strike source": boundary_table["Boundary type"],
            "Boundary confidence (%)": boundary_table["Boundary confidence (%)"],
            "Manual strike": np.nan,
            "Pricing IV": boundary_table["Pricing IV"],
        })
    manual_rows = pd.DataFrame(columns=EDITOR_COLUMNS)
    if not manual_strike_table.empty:
        manual_rows = pd.DataFrame({
            "Active": manual_strike_table["Active"],
            "Ticker": manual_strike_table["Ticker"],
            "Option type": manual_strike_table["Option type"],
            "Position": manual_strike_table["Position"],
            "Quantity": manual_strike_table["Quantity"],
            "Strike source": "Manual strike",
            "Boundary confidence (%)": 80,
            "Manual strike": manual_strike_table["Strike"],
            "Pricing IV": manual_strike_table["Pricing IV"],
        })
    return pd.concat([boundary_rows, manual_rows], ignore_index=True)[EDITOR_COLUMNS]


def _boundary_ratio(boundaries: pd.DataFrame, ticker: str, confidence: float, source: str) -> float:
    rows = boundaries[
        (boundaries["Ticker"].astype(str) == ticker)
        & np.isclose(boundaries["Confidence level"].astype(float), confidence)
    ]
    if rows.empty:
        return np.nan
    column = "Upper win boundary / current" if source == "Win boundary" else "Lower loss boundary / current"
    return float(rows.iloc[0][column])


def resolve_manual_option_legs(
    editor_table: pd.DataFrame,
    boundaries: pd.DataFrame,
    *,
    time_to_expiry: float,
    risk_free_rate: float,
    normalized_spot: float = 100.0,
) -> pd.DataFrame:
    """Resolve manual/boundary strike instructions into priced option legs."""
    missing = set(EDITOR_COLUMNS) - set(editor_table.columns)
    if missing:
        raise ValueError(f"Manual portfolio is missing columns: {sorted(missing)}")

    rows = []
    for row_number, row in editor_table.reset_index(drop=True).iterrows():
        if not bool(row["Active"]):
            continue
        ticker = str(row["Ticker"]).strip()
        option_type = str(row["Option type"])
        position = str(row["Position"])
        source = str(row["Strike source"])
        quantity = float(row["Quantity"])
        confidence_raw = float(row["Boundary confidence (%)"])
        confidence = confidence_raw / 100.0 if confidence_raw > 1.0 else confidence_raw
        pricing_iv = float(row["Pricing IV"])

        if not ticker:
            raise ValueError(f"Row {row_number + 1}: ticker is required.")
        if option_type not in OPTION_TYPES or position not in POSITIONS:
            raise ValueError(f"Row {row_number + 1}: invalid option type or position.")
        if source not in STRIKE_SOURCES:
            raise ValueError(f"Row {row_number + 1}: invalid strike source.")
        if quantity < 0:
            raise ValueError(f"Row {row_number + 1}: quantity must be non-negative; use Position for Long/Short.")
        if pricing_iv <= 0:
            raise ValueError(f"Row {row_number + 1}: Pricing IV must be positive.")

        if source == "Manual strike":
            strike = float(row["Manual strike"])
            boundary_label = "Manual"
        else:
            boundary_ratio = _boundary_ratio(boundaries, ticker, confidence, source)
            strike = normalized_spot * boundary_ratio if np.isfinite(boundary_ratio) else np.nan
            boundary_label = f"{confidence:.0%} {'win' if source == 'Win boundary' else 'loss'}"
        if not np.isfinite(strike) or strike <= 0:
            raise ValueError(f"Row {row_number + 1}: no valid strike for {ticker} {boundary_label} boundary.")

        premium = black_scholes_price(
            spot=normalized_spot,
            strike=strike,
            time_to_expiry=time_to_expiry,
            volatility=pricing_iv,
            risk_free_rate=risk_free_rate,
            option_type=option_type,
        )
        rows.append({
            "Instrument": f"{position} {ticker} {option_type} {strike:.2f}",
            "Ticker": ticker,
            "Option type": option_type,
            "Position": position,
            "Quantity": quantity,
            "Strike": strike,
            "Strike / spot": strike / normalized_spot,
            "Strike source": source,
            "Boundary used": boundary_label,
            "Spot": normalized_spot,
            "Model IV": pricing_iv,
            "Risk-free rate": risk_free_rate,
            "Time to expiry": time_to_expiry,
            "Theoretical premium": premium,
        })
    return pd.DataFrame(rows, columns=RESOLVED_COLUMNS)


def manual_option_payoffs_and_analytics(
    legs: pd.DataFrame,
    normalized_terminal_prices: pd.DataFrame,
    *,
    contract_multiplier: float = 1.0,
    include_premiums: bool = True,
) -> tuple[np.ndarray, pd.DataFrame]:
    """Calculate total option payoff and standalone analytics for every leg."""
    total = np.zeros(len(normalized_terminal_prices), dtype=float)
    analytics = []
    for _, leg in legs.iterrows():
        ticker = str(leg["Ticker"])
        if ticker not in normalized_terminal_prices.columns:
            raise ValueError(f"No stored terminal scenarios for {ticker}.")
        premium = float(leg["Theoretical premium"]) if include_premiums else 0.0
        quantity = float(leg["Quantity"])
        payoff = option_payoff(
            str(leg["Option type"]),
            str(leg["Position"]),
            float(leg["Strike"]),
            normalized_terminal_prices[ticker].to_numpy(dtype=float),
            premium=premium,
        ) * quantity * float(contract_multiplier)
        total += payoff
        threshold = np.quantile(payoff, 0.05)
        tail = payoff[payoff <= threshold]
        premium_cashflow = premium * quantity * float(contract_multiplier)
        if str(leg["Position"]) == "Long":
            premium_cashflow *= -1.0
        analytics.append({
            "Instrument": leg["Instrument"],
            "Expected option payoff": payoff.mean(),
            "Option payoff SD": payoff.std(ddof=0),
            "P(option loss)": (payoff < 0).mean(),
            "Expected shortfall 5%": tail.mean() if len(tail) else np.nan,
            "Worst option payoff": payoff.min(),
            "Initial premium cashflow": premium_cashflow,
        })
    return total, pd.DataFrame(analytics, columns=ANALYTICS_COLUMNS)
