"""Phase 8: risk-management and position-sizing engine.

Phase 8 turns the payoff distribution (Phase 4 scenarios) and the real execution
costs (Phase 6) into the decision metrics a trader actually sizes against: how
much capital goes in, how much can be lost, the return on that capital and on
the capital genuinely at risk, the probability of profit and of ruin, and a
Kelly-based position size for a given bankroll.

Sign convention: the Phase 4 payoff surface already reports *net* profit and
loss per scenario (option premiums and the Polymarket entry cost are baked in).
So "Total payoff" is net P&L, the worst scenario payoff is the maximum loss, and
expected payoff is expected net profit. Phase 8 never re-derives payoffs; it
reads that distribution and adds the money view on top.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from phase7 import PortfolioSpec, portfolio_scenarios


# ---------------------------------------------------------------------------
# Capital: cash outlay and gross premium
# ---------------------------------------------------------------------------
def capital_metrics(portfolio: PortfolioSpec) -> pd.Series:
    """Up-front capital for the portfolio, independent of scenarios.

    * Polymarket cost: shares are fully collateralized, so cost = shares * entry.
    * Option premiums: long legs are a debit (cash paid), short legs a credit
      (cash received), each scaled by quantity and the contract multiplier.

    Returns the Polymarket cost, the long/short/net option premium, the net cash
    outlay (Polymarket + net option debit; negative means a net credit is
    received), and the gross premium paid (ignoring credits) as the most
    conservative "money in".
    """

    legs = portfolio.option_legs.copy()
    if "Quantity" not in legs.columns:
        legs["Quantity"] = 0.0
    multiplier = float(portfolio.contract_multiplier)

    long_premium = 0.0
    short_premium = 0.0
    for _, leg in legs.iterrows():
        quantity = float(pd.to_numeric(leg.get("Quantity", 0.0), errors="coerce") or 0.0)
        if quantity == 0.0:
            continue
        premium = float(pd.to_numeric(leg.get("Theoretical premium", 0.0), errors="coerce") or 0.0)
        cash = premium * quantity * multiplier
        if str(leg.get("Position", "")).lower() == "long":
            long_premium += cash
        else:
            short_premium += cash

    polymarket_cost = float(portfolio.polymarket_quantity) * float(portfolio.polymarket_entry_price)
    net_option_debit = long_premium - short_premium
    net_cash_outlay = polymarket_cost + net_option_debit
    gross_premium_paid = polymarket_cost + long_premium
    return pd.Series(
        {
            "Polymarket cost": polymarket_cost,
            "Long option premium": long_premium,
            "Short option premium (credit)": short_premium,
            "Net option debit": net_option_debit,
            "Net cash outlay": net_cash_outlay,
            "Gross premium paid": gross_premium_paid,
        }
    )


# ---------------------------------------------------------------------------
# Core risk-management metrics from the scenario payoff distribution
# ---------------------------------------------------------------------------
def risk_metrics(
    result,
    portfolio: PortfolioSpec,
    *,
    shortfall_probability: float = 0.05,
    ruin_fraction: float = 0.9,
) -> pd.Series:
    """Money-view metrics for the portfolio on the saved scenarios.

    ``ruin_fraction`` defines a ruin event as losing at least that share of the
    capital at risk (default 90%).
    """

    scenario = portfolio_scenarios(result, portfolio)
    payoff = scenario["Total payoff"].astype(float)

    expected = float(payoff.mean())
    worst = float(payoff.min())
    best = float(payoff.max())
    max_loss = max(-worst, 0.0)  # capital genuinely at risk (bounded by construction)
    threshold = float(payoff.quantile(shortfall_probability))
    tail = payoff[payoff <= threshold]
    expected_shortfall = float(tail.mean()) if not tail.empty else np.nan

    capital = capital_metrics(portfolio)
    cash_outlay = float(capital["Net cash outlay"])
    # Denominator for return on capital: real cash if positive, otherwise fall
    # back to the capital genuinely at risk (a credit structure still risks the
    # worst-case loss).
    capital_deployed = cash_outlay if cash_outlay > 0 else max_loss

    return_on_capital = expected / capital_deployed if capital_deployed > 0 else np.nan
    return_on_capital_at_risk = expected / max_loss if max_loss > 0 else np.nan

    probability_of_profit = float((payoff > 0).mean())
    probability_of_loss = float((payoff < 0).mean())
    risk_of_ruin = float((payoff <= -ruin_fraction * max_loss).mean()) if max_loss > 0 else 0.0

    return pd.Series(
        {
            "Capital deployed (cash)": capital_deployed,
            "Net cash outlay": cash_outlay,
            "Gross premium paid": float(capital["Gross premium paid"]),
            "Max loss (capital at risk)": max_loss,
            "Expected profit": expected,
            "Best case profit": best,
            "Return on capital": return_on_capital,
            "Return on capital-at-risk": return_on_capital_at_risk,
            "Expected shortfall": expected_shortfall,
            "Expected shortfall / capital-at-risk": (expected_shortfall / max_loss) if max_loss > 0 else np.nan,
            "Probability of profit": probability_of_profit,
            "Probability of loss": probability_of_loss,
            f"Risk of ruin (lose >= {ruin_fraction:.0%})": risk_of_ruin,
        }
    )


def value_at_risk(result, portfolio: PortfolioSpec, probability: float = 0.05) -> float:
    """Value at Risk: the loss (positive number) at the given tail probability."""

    payoff = portfolio_scenarios(result, portfolio)["Total payoff"].astype(float)
    loss = -float(payoff.quantile(probability))
    return loss if np.isfinite(loss) else np.nan


def capital_return_table(result, portfolio: PortfolioSpec) -> pd.DataFrame:
    """Return on capital under several capital-at-risk definitions.

    The capital you must reserve depends on how conservative you are: the cash
    you actually pay, the VaR at 95%/99% (the loss you would not exceed at that
    confidence), the expected shortfall (mean of the worst 5%), or the absolute
    worst simulated loss. Return on capital = expected profit / that capital.
    """

    scenario = portfolio_scenarios(result, portfolio)
    payoff = scenario["Total payoff"].astype(float)
    expected = float(payoff.mean())
    max_loss = max(-float(payoff.min()), 0.0)
    cash = float(capital_metrics(portfolio)["Net cash outlay"])
    cash = cash if cash > 0 else max_loss

    def _var(probability: float) -> float:
        loss = -float(payoff.quantile(probability))
        return loss if loss > 0 else np.nan

    tail = payoff[payoff <= payoff.quantile(0.05)]
    expected_shortfall = -float(tail.mean()) if not tail.empty else np.nan

    bases = [
        ("Initial cash outlay", cash),
        ("VaR 5% (95% worst loss)", _var(0.05)),
        ("VaR 1% (99% worst loss)", _var(0.01)),
        ("Expected shortfall 5% (CVaR)", expected_shortfall),
        ("Max loss (worst case)", max_loss),
    ]
    rows = []
    for name, amount in bases:
        roc = expected / amount if (np.isfinite(amount) and amount > 0) else np.nan
        rows.append({"Capital basis": name, "Capital needed ($)": amount, "Return on capital": roc})
    return pd.DataFrame(rows)


def return_distribution(
    result,
    portfolio: PortfolioSpec,
    *,
    quantiles=(0.01, 0.05, 0.25, 0.50, 0.75, 0.95, 0.99),
) -> pd.DataFrame:
    """Percentiles of profit, and return vs both max loss and initial cash."""

    scenario = portfolio_scenarios(result, portfolio)
    payoff = scenario["Total payoff"].astype(float)
    max_loss = max(-float(payoff.min()), 0.0)
    cash = float(capital_metrics(portfolio)["Net cash outlay"])
    cash = cash if cash > 0 else max_loss
    rows = []
    for q in quantiles:
        profit = float(payoff.quantile(q))
        rows.append(
            {
                "Percentile": f"P{int(round(q * 100))}",
                "Profit": profit,
                "Return on initial cash": (profit / cash) if cash > 0 else np.nan,
                "Return on capital-at-risk": (profit / max_loss) if max_loss > 0 else np.nan,
            }
        )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Breakeven bands on the selected ticker
# ---------------------------------------------------------------------------
def breakeven_bands(
    result,
    portfolio: PortfolioSpec,
    *,
    bins: int = 25,
) -> pd.DataFrame:
    """Where expected payoff crosses zero along the selected ticker's terminal ratio.

    Uses the Phase 4 selected-ratio profile: each row is a sign change between
    adjacent bins, reported as the terminal market-cap ratio (and stock-price
    level) at the crossover.
    """

    from payoff_surface import selected_payoff_profile_bins

    scenario = portfolio_scenarios(result, portfolio)
    profile = selected_payoff_profile_bins(
        scenario,
        result.terminal_market_caps,
        portfolio.current_market_caps,
        selected_ticker=portfolio.selected_ticker,
        bins=int(bins),
    ).sort_values("selected_ratio").reset_index(drop=True)

    crossings = []
    payoff = profile["expected_payoff"].to_numpy(dtype=float)
    ratio = profile["selected_ratio"].to_numpy(dtype=float)
    price = profile["selected_stock_price"].to_numpy(dtype=float)
    for i in range(1, len(profile)):
        if payoff[i - 1] == 0.0 or (payoff[i - 1] < 0.0) != (payoff[i] < 0.0):
            # linear interpolation of the zero crossing in ratio space
            denom = payoff[i] - payoff[i - 1]
            weight = 0.0 if denom == 0 else -payoff[i - 1] / denom
            cross_ratio = ratio[i - 1] + weight * (ratio[i] - ratio[i - 1])
            cross_price = price[i - 1] + weight * (price[i] - price[i - 1])
            direction = "loss -> profit" if payoff[i] >= 0 else "profit -> loss"
            crossings.append(
                {
                    "Breakeven selected ratio": cross_ratio,
                    "Breakeven stock price": cross_price,
                    "Direction": direction,
                }
            )
    return pd.DataFrame(crossings)


# ---------------------------------------------------------------------------
# Budget scaling and Kelly sizing
# ---------------------------------------------------------------------------
def budget_scaling(
    metrics: pd.Series,
    budget: float,
    *,
    basis: str = "capital-at-risk",
) -> pd.Series:
    """Linearly scale a fixed structure to a capital budget.

    ``basis`` = "capital-at-risk" caps the *maximum loss* at the budget;
    "cash" deploys exactly the budget as cash outlay. Because the structure is
    fixed, every payoff scales by the same factor.
    """

    if basis == "capital-at-risk":
        base = float(metrics["Max loss (capital at risk)"])
    elif basis == "cash":
        base = float(metrics["Capital deployed (cash)"])
    else:
        raise ValueError("basis must be 'capital-at-risk' or 'cash'.")
    if base <= 0:
        raise ValueError("Base capital must be positive to scale.")
    factor = float(budget) / base
    return pd.Series(
        {
            "Budget": float(budget),
            "Scale factor": factor,
            "Scaled expected profit": float(metrics["Expected profit"]) * factor,
            "Scaled max loss": float(metrics["Max loss (capital at risk)"]) * factor,
            "Scaled cash outlay": float(metrics["Capital deployed (cash)"]) * factor,
        }
    )


def _log_growth(fraction: float, returns: np.ndarray) -> float:
    values = 1.0 + fraction * returns
    if np.any(values <= 0.0):
        return -np.inf
    return float(np.mean(np.log(values)))


def kelly_sizing(
    result,
    portfolio: PortfolioSpec,
    *,
    fractions=(1.0, 0.5, 0.25),
    grid: int = 400,
) -> pd.Series:
    """Kelly-optimal fraction of bankroll to risk, from the scenario returns.

    Returns are expressed per unit of capital at risk, so the worst scenario
    maps to a return of -1 (a full unit lost). The optimal fraction f* maximizes
    the mean log growth E[log(1 + f * r)] over scenarios; f is bounded to
    (0, 1] so a scenario can never wipe out more than the bankroll. When the
    expected return is not positive, Kelly recommends f* = 0 (no bet).
    """

    scenario = portfolio_scenarios(result, portfolio)
    payoff = scenario["Total payoff"].astype(float).to_numpy()
    max_loss = max(-float(payoff.min()), 0.0)
    if max_loss <= 0:
        raise ValueError("Portfolio has no downside scenario; Kelly sizing is undefined.")
    returns = payoff / max_loss  # worst scenario == -1

    if returns.mean() <= 0.0:
        full_kelly = 0.0
        growth = 0.0
    else:
        candidates = np.linspace(1e-4, 1.0, int(grid))
        growths = np.array([_log_growth(f, returns) for f in candidates])
        best_index = int(np.argmax(growths))
        # local parabolic refine around the grid maximum
        lo = candidates[max(best_index - 1, 0)]
        hi = candidates[min(best_index + 1, len(candidates) - 1)]
        refine = np.linspace(lo, hi, 101)
        refine_growth = np.array([_log_growth(f, returns) for f in refine])
        full_kelly = float(refine[int(np.argmax(refine_growth))])
        growth = float(np.max(refine_growth))

    out = {"Full Kelly fraction": full_kelly, "Full Kelly log-growth": growth}
    for fraction in fractions:
        if float(fraction) == 1.0:
            continue
        out[f"{fraction:g}x Kelly fraction"] = full_kelly * float(fraction)
    out["Capital at risk (1x portfolio)"] = max_loss
    out["Expected return on capital-at-risk"] = float(returns.mean())
    return pd.Series(out)
