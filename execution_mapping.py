"""Phase 6 helpers for mapping normalized research legs to executable terms."""

from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import pandas as pd
import yfinance as yf

from correlations import yahoo_symbol
from option_construction import black_scholes_price

PHASE6_LISTED_CONTRACT_MULTIPLIER = 100.0


def default_strike_step(spot: float) -> float:
    """Provide an editable display grid until listed strikes arrive in Phase 7."""
    if spot < 25:
        return 0.5
    if spot < 100:
        return 1.0
    if spot < 250:
        return 2.5
    if spot < 500:
        return 5.0
    return 10.0


def round_to_strike_grid(strike: float, step: float) -> float:
    if strike <= 0 or step <= 0:
        raise ValueError("strike and strike step must be positive.")
    return float(max(step, np.round(strike / step) * step))


def fetch_option_expirations(tickers: list[str]) -> dict[str, list[date]]:
    """Fetch currently listed expiration dates without downloading option quotes."""
    output: dict[str, list[date]] = {}
    for ticker in tickers:
        values = yf.Ticker(yahoo_symbol(ticker)).options
        output[ticker] = sorted({pd.Timestamp(value).date() for value in values})
    return output


def _clean_chain_side(options: pd.DataFrame, option_type: str) -> pd.DataFrame:
    if options is None or options.empty:
        return pd.DataFrame()
    clean = options.copy()
    clean["Option type"] = option_type
    for column in ["strike", "bid", "ask", "lastPrice", "impliedVolatility", "volume", "openInterest"]:
        if column in clean.columns:
            clean[column] = pd.to_numeric(clean[column], errors="coerce")
    clean = clean.dropna(subset=["strike"])
    clean = clean[clean["strike"] > 0]
    return clean


def fetch_option_chain_quotes(ticker: str, expiry: date | str) -> pd.DataFrame:
    """Fetch listed option quotes for one ticker and expiry from Yahoo Finance."""
    expiry_text = pd.Timestamp(expiry).date().isoformat()
    chain = yf.Ticker(yahoo_symbol(ticker)).option_chain(expiry_text)
    calls = _clean_chain_side(chain.calls, "Call")
    puts = _clean_chain_side(chain.puts, "Put")
    quotes = pd.concat([calls, puts], ignore_index=True)
    if quotes.empty:
        raise ValueError(f"No listed option quotes returned for {ticker} expiry {expiry_text}.")
    quotes["Ticker"] = ticker
    quotes["Expiry"] = expiry_text
    keep = [
        "Ticker", "Expiry", "Option type", "contractSymbol", "strike", "bid", "ask",
        "lastPrice", "impliedVolatility", "volume", "openInterest",
    ]
    return quotes[[column for column in keep if column in quotes.columns]]


def fetch_option_chain_quotes_for_expiries(expiry_by_ticker: dict[str, date | str | None]) -> dict[str, pd.DataFrame]:
    """Fetch listed option quote chains keyed by ticker for selected expiries."""
    output: dict[str, pd.DataFrame] = {}
    for ticker, expiry in expiry_by_ticker.items():
        if expiry is None or pd.isna(expiry):
            continue
        output[ticker] = fetch_option_chain_quotes(ticker, expiry)
    return output


def infer_listed_strike_step(strikes: pd.Series, near: float | None = None, window: int = 12) -> float | None:
    """Infer the local listed strike spacing from an actual chain."""
    clean = np.sort(pd.to_numeric(strikes, errors="coerce").dropna().unique())
    clean = clean[clean > 0]
    if len(clean) < 2:
        return None
    if near is not None and np.isfinite(near):
        order = np.argsort(np.abs(clean - float(near)))
        clean = np.sort(clean[order[: max(2, min(len(clean), window))]])
    diffs = np.diff(clean)
    diffs = diffs[diffs > 1e-9]
    if len(diffs) == 0:
        return None
    return float(pd.Series(diffs).round(4).mode().iloc[0])


def nearest_listed_option(chain: pd.DataFrame, option_type: str, target_strike: float) -> dict:
    """Return the listed option row nearest to the desired strike."""
    if chain is None or chain.empty:
        return {}
    side = chain[chain["Option type"].astype(str).str.lower() == option_type.lower()].copy()
    if side.empty:
        return {}
    side["Distance"] = (side["strike"].astype(float) - float(target_strike)).abs()
    row = side.sort_values(["Distance", "strike"]).iloc[0]
    bid = float(row.get("bid", np.nan))
    ask = float(row.get("ask", np.nan))
    last = float(row.get("lastPrice", np.nan))
    mid = np.nan
    if np.isfinite(bid) and np.isfinite(ask) and bid > 0 and ask > 0 and ask >= bid:
        mid = 0.5 * (bid + ask)
    elif np.isfinite(last) and last > 0:
        mid = last
    spread = ask - bid if np.isfinite(bid) and np.isfinite(ask) else np.nan
    return {
        "Listed strike": float(row["strike"]),
        "Contract symbol": row.get("contractSymbol", ""),
        "Bid": bid,
        "Ask": ask,
        "Mid": float(mid) if np.isfinite(mid) else np.nan,
        "Last": last,
        "Market IV": float(row.get("impliedVolatility", np.nan)),
        "Volume": float(row.get("volume", np.nan)),
        "Open interest": float(row.get("openInterest", np.nan)),
        "Bid/ask spread": float(spread) if np.isfinite(spread) else np.nan,
    }


def last_friday_before_month_end(target_date: date) -> date:
    month_end = (pd.Timestamp(target_date) + pd.offsets.MonthEnd(0)).date()
    days_back = (month_end.weekday() - 4) % 7
    return month_end - timedelta(days=days_back)


def choose_expiration(
    expirations: list[date],
    target_date: date,
    policy: str,
) -> date | None:
    if not expirations:
        return None
    ordered = sorted(expirations)
    if policy == "Last Friday before month end":
        anchor = last_friday_before_month_end(target_date)
        candidates = [value for value in ordered if value <= anchor]
        return candidates[-1] if candidates else ordered[0]
    if policy == "First expiry on/after target":
        candidates = [value for value in ordered if value >= target_date]
        return candidates[0] if candidates else ordered[-1]
    if policy == "Last expiry on/before target":
        candidates = [value for value in ordered if value <= target_date]
        return candidates[-1] if candidates else ordered[0]
    if policy == "Nearest listed expiry":
        return min(ordered, key=lambda value: abs((value - target_date).days))
    raise ValueError(f"Unknown expiration policy: {policy}")


def phase5_share_equivalent_to_contracts(quantity: float) -> float:
    """Convert Phase 5 share-equivalent quantity to listed contract-equivalent quantity.

    Phase 5 quantities are intentionally share-equivalent units. Phase 6 still
    multiplies option payoff by 100 because listed contracts are 100 shares, so
    the mapped quantity must be divided by 100 to preserve the Phase 5 sizing.
    """
    return float(quantity) / PHASE6_LISTED_CONTRACT_MULTIPLIER


def map_normalized_legs(
    legs: pd.DataFrame,
    spot_by_ticker: pd.Series,
    strike_step_by_ticker: pd.Series,
    option_chains: dict[str, pd.DataFrame] | None = None,
) -> pd.DataFrame:
    """Map Phase 5 strikes (spot=100) into current-dollar strike grids."""
    rows = []
    for index, leg in legs.reset_index(drop=True).iterrows():
        ticker = str(leg["Ticker"])
        spot = float(spot_by_ticker.loc[ticker])
        step = float(strike_step_by_ticker.loc[ticker])
        normalized_strike = float(leg["Strike"])
        phase5_quantity = float(leg["Quantity"])
        mapped_quantity = phase5_share_equivalent_to_contracts(phase5_quantity)
        raw_strike = spot * normalized_strike / 100.0
        executable_strike = round_to_strike_grid(raw_strike, step)
        quote = nearest_listed_option((option_chains or {}).get(ticker), str(leg["Option type"]), raw_strike)
        if quote:
            executable_strike = float(quote["Listed strike"])
        mapped_normalized = executable_strike / spot * 100.0
        row = {
            "Leg": index + 1,
            "Use": True,
            "Ticker": ticker,
            "Option type": str(leg["Option type"]),
            "Position": str(leg["Position"]),
            "Quantity": mapped_quantity,
            "Phase 5 share-equivalent quantity": phase5_quantity,
            "Phase 5 normalized strike": normalized_strike,
            "Current spot": spot,
            "Raw real strike": raw_strike,
            "Strike step": step,
            "Executable strike": executable_strike,
            "Mapped normalized strike": mapped_normalized,
            "Strike mapping error": mapped_normalized - normalized_strike,
            "Model IV": float(leg.get("Model IV", np.nan)),
        }
        row.update(quote)
        if "Mid" in row and np.isfinite(float(row["Mid"])):
            row["Market premium normalized"] = float(row["Mid"]) / spot * 100.0
        conservative = np.nan
        position = str(leg["Position"]).lower()
        if position == "long" and "Ask" in row and np.isfinite(float(row["Ask"])) and float(row["Ask"]) > 0:
            conservative = float(row["Ask"])
        elif position == "short" and "Bid" in row and np.isfinite(float(row["Bid"])) and float(row["Bid"]) >= 0:
            conservative = float(row["Bid"])
        elif "Mid" in row and np.isfinite(float(row["Mid"])):
            conservative = float(row["Mid"])
        if np.isfinite(conservative):
            row["Conservative premium normalized"] = conservative / spot * 100.0
        rows.append(row)
    return pd.DataFrame(rows)


def rebuild_normalized_legs(
    original_legs: pd.DataFrame,
    mapping: pd.DataFrame,
    *,
    time_to_expiry: float,
    risk_free_rate: float,
) -> pd.DataFrame:
    """Rebuild Phase 5-compatible legs after real-strike editing and rounding."""
    rows = []
    original = original_legs.reset_index(drop=True)
    for _, mapped in mapping[mapping["Use"]].iterrows():
        source = original.iloc[int(mapped["Leg"]) - 1]
        strike = float(mapped["Executable strike"]) / float(mapped["Current spot"]) * 100.0
        volatility = float(mapped["Model IV"])
        market_premium = pd.to_numeric(pd.Series([mapped.get("Market premium normalized", np.nan)]), errors="coerce").iloc[0]
        model_premium = black_scholes_price(
            spot=100.0,
            strike=strike,
            time_to_expiry=time_to_expiry,
            volatility=volatility,
            risk_free_rate=risk_free_rate,
            option_type=str(mapped["Option type"]),
        )
        premium = float(market_premium) if np.isfinite(market_premium) and market_premium >= 0 else model_premium
        rows.append(
            {
                "Instrument": f"{mapped['Position']} {mapped['Ticker']} {mapped['Option type']} {mapped['Executable strike']:.2f}",
                "Ticker": str(mapped["Ticker"]),
                "Option type": str(mapped["Option type"]),
                "Position": str(mapped["Position"]),
                "Quantity": float(mapped["Quantity"]),
                "Strike": strike,
                "Strike / spot": strike / 100.0,
                "Strike source": "Phase 6 real-strike mapping",
                "Boundary used": source.get("Boundary used", "Mapped from Phase 5"),
                "Spot": 100.0,
                "Model IV": volatility,
                "Risk-free rate": risk_free_rate,
                "Time to expiry": time_to_expiry,
                "Theoretical premium": premium,
                "Phase 6 model premium": model_premium,
                "Phase 6 premium source": "Yahoo bid/ask mid" if premium != model_premium else "Model fallback",
            }
        )
    return pd.DataFrame(rows)
