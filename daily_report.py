"""Headless daily pipeline: fetch data, run Phases 1-8, write a verdict report.

Designed to run in GitHub Actions (which has open internet, so it fetches market
caps and spot prices from Yahoo just like Phase 0). Polymarket prices and the
traded structure come from ``daily_inputs.json`` (Polymarket cannot be fetched
and is provided manually). If a live fetch fails, the values in
``daily_inputs.json`` are used as a fallback so the run still completes.

No Streamlit. Writes ``reports/<date>.md`` and prints the report.
"""

from __future__ import annotations

import json
from datetime import date
from math import erf, exp, log, sqrt
from pathlib import Path

import pandas as pd

from iv_surface_model import (
    SURFACE_EXPIRY,
    apply_surface_atm_ivs,
    default_surface_nodes,
    run_surface_probability_engine,
)
from model import default_correlation_matrix
import phase7 as p7
import phase8 as p8

REPO = Path(__file__).resolve().parent
INPUTS_PATH = REPO / "daily_inputs.json"
REPORTS_DIR = REPO / "reports"

# Fixed hedge: normalized strikes (spot=100) and share-equivalent weights.
HEDGE_LEGS = [
    {"Option type": "Put", "Position": "Long", "ratio": 0.80, "weight": 2.0},
    {"Option type": "Put", "Position": "Short", "ratio": 1.00, "weight": 2.0},
    {"Option type": "Call", "Position": "Long", "ratio": 0.90, "weight": 3.0},
    {"Option type": "Call", "Position": "Short", "ratio": 1.05, "weight": 3.0},
]


def _bs_premium(spot, strike, years, iv, rate, kind):
    if years <= 0 or iv <= 0:
        return float(max(spot - strike, 0.0) if kind == "Call" else max(strike - spot, 0.0))
    d1 = (log(spot / strike) + (rate + 0.5 * iv * iv) * years) / (iv * sqrt(years))
    d2 = d1 - iv * sqrt(years)
    ncdf = lambda x: 0.5 * (1.0 + erf(x / sqrt(2.0)))
    if kind == "Call":
        return float(spot * ncdf(d1) - strike * exp(-rate * years) * ncdf(d2))
    return float(strike * exp(-rate * years) * ncdf(-d2) - spot * ncdf(-d1))


def fetch_market_data(tickers, fallback_caps, fallback_spots):
    """Return (caps, spots, source). Try Yahoo; fall back to provided values."""
    caps, spots, notes = {}, {}, []
    try:
        from market_data import fetch_market_caps
        cap_df = fetch_market_caps(list(tickers))
        caps = {str(r["ticker"]): float(r["market_cap"]) for _, r in cap_df.iterrows()}
        notes.append("caps: Yahoo live")
    except Exception as exc:  # noqa: BLE001
        caps = {t: float(fallback_caps[t]) for t in tickers}
        notes.append(f"caps: fallback ({type(exc).__name__})")
    try:
        from market_data import fetch_spot_prices
        spot_df = fetch_spot_prices(list(tickers))
        spot_col = next((c for c in ["spot_price", "spot"] if c in spot_df.columns), None)
        if spot_col is None:
            raise KeyError("no spot column in fetch_spot_prices output")
        spots = {str(r["ticker"]): float(r[spot_col]) for _, r in spot_df.iterrows()}
        notes.append("spots: Yahoo live")
    except Exception as exc:  # noqa: BLE001
        spots = {t: float(fallback_spots[t]) for t in tickers}
        notes.append(f"spots: fallback ({type(exc).__name__})")
    for t in tickers:
        caps.setdefault(t, float(fallback_caps[t]))
        spots.setdefault(t, float(fallback_spots[t]))
    return caps, spots, "; ".join(notes)


def build_legs(ticker, spot, iv, years, rate):
    rows = []
    for leg in HEDGE_LEGS:
        strike = round(spot * leg["ratio"], 2)
        # Phase 5/6 sizing: share-equivalent weight -> listed contracts = weight / spot.
        quantity = leg["weight"] / spot
        rows.append({
            "Instrument": f"{leg['Position']} {ticker} {leg['Option type']} {strike:.2f}",
            "Ticker": ticker, "Option type": leg["Option type"], "Position": leg["Position"],
            "Quantity": quantity, "Strike": strike, "Spot": spot,
            "Theoretical premium": _bs_premium(spot, strike, years, iv, rate, leg["Option type"]),
        })
    return pd.DataFrame(rows)


def run(inputs: dict) -> str:
    target = date.fromisoformat(inputs["target_date"])
    as_of = date.fromisoformat(inputs.get("as_of", date.today().isoformat()))
    days = max((target - as_of).days, 1)
    years = days / 365.0
    rate = float(inputs.get("risk_free_rate", 0.04))
    sims = int(inputs.get("simulations", 40000))
    seed = int(inputs.get("seed", 42))

    universe_rows = inputs["universe"]
    tickers = [r["Ticker"] for r in universe_rows]
    fallback_caps = {r["Ticker"]: r["Current market cap"] for r in universe_rows}
    yes_prices = {r["Ticker"]: r["Polymarket YES price"] for r in universe_rows}
    fallback_spots = inputs["spots"]

    caps, spots, data_source = fetch_market_data(tickers, fallback_caps, fallback_spots)

    universe = pd.DataFrame([
        {"Ticker": t, "Current market cap": caps[t], "Implied volatility": 0.30, "Polymarket YES price": yes_prices[t]}
        for t in tickers
    ])
    # traded name = the market favourite (highest YES) unless overridden
    traded = inputs.get("traded_ticker") or max(yes_prices, key=yes_prices.get)
    spot = spots[traded]

    corr = default_correlation_matrix(tickers)
    surf_inputs = apply_surface_atm_ivs(universe.copy())
    result, _ = run_surface_probability_engine(
        surf_inputs, corr, days_to_target=days, simulations=sims, seed=seed,
        surface_nodes=default_surface_nodes(), risk_free_rate=rate,
    )
    probs = result.results.set_index("Ticker")
    iv_atm = float(surf_inputs.set_index("Ticker").loc[traded, "Implied volatility"])

    legs = build_legs(traded, spot, iv_atm, years, rate)
    caps_series = universe.set_index("Ticker")["Current market cap"].astype(float)
    portfolio = p7.PortfolioSpec(
        option_legs=legs, current_market_caps=caps_series, spot_prices=pd.Series({traded: spot}),
        selected_ticker=traded, polymarket_side=inputs["side"],
        polymarket_entry_price=float(inputs["entry"]), polymarket_quantity=float(inputs.get("shares", 100.0)),
        contract_multiplier=100.0, include_option_premiums=True,
    )

    seeds = list(range(seed, seed + 5))
    disp = p7.dispersion_summary(p7.multi_seed_dispersion(surf_inputs, corr, portfolio, days_to_target=days, simulations=sims, seeds=seeds))
    cop = p7.copula_tail_stress(surf_inputs, corr, portfolio, days_to_target=days, simulations=sims, seeds=seeds)
    ivs = p7.iv_scaling_scan(surf_inputs, corr, selected_ticker=traded, days_to_target=days, simulations=sims, seed=seed, factors=[0.5, 0.75, 1.0, 1.25, 1.5, 2.0])
    gps = p7.gap_scaling_scan(surf_inputs, corr, selected_ticker=traded, days_to_target=days, simulations=sims, seed=seed, factors=[0.5, 0.75, 1.0, 1.25, 1.5, 2.0])
    gv = p7.gap_vs_randomness(ivs, gps)
    variants = {"Saved": corr, "Independent": p7.constant_correlation(tickers, 0.0), "High 0.8": p7.constant_correlation(tickers, 0.8)}
    grid = p7.model_robustness(surf_inputs, variants, selected_ticker=traded, days_to_target=days, simulations=sims, seed=seed, shock_models=["Normal shocks", "Student-t copula df=5"])
    assessment = p7.assessment(disp, cop, gv, p7.robustness_summary(grid), selected_ticker=traded)

    rm = p8.risk_metrics(result, portfolio)
    var5 = p8.value_at_risk(result, portfolio, 0.05)
    var1 = p8.value_at_risk(result, portfolio, 0.01)

    model_p = float(probs.loc[traded, "Model probability"])
    yes_price = float(probs.loc[traded, "Polymarket YES price"])
    expected = float(rm["Expected profit"])
    max_loss = float(rm["Max loss (capital at risk)"])
    rocar = float(rm["Return on capital-at-risk"])
    estimate_robust = "robust" in assessment["headline"].lower() and "not fully" not in assessment["headline"].lower()

    # Edge on the side actually traded: model fair probability for that side minus its price.
    side = portfolio.polymarket_side.upper()
    model_side_prob = (1.0 - model_p) if side == "NO" else model_p
    side_edge = model_side_prob - float(portfolio.polymarket_entry_price)

    verdict = "FAVORABLE" if expected > 0 else ("BREAKEVEN" if abs(expected) < 1e-6 else "UNFAVORABLE")

    L = []
    L.append(f"# LargestCompany daily report - {as_of.isoformat()}")
    L.append("")
    L.append(f"Target {target.isoformat()} ({days} days) | traded {traded} {side} @ {portfolio.polymarket_entry_price:.2f} | {sims:,} sims")
    L.append(f"Data: {data_source}")
    L.append("")
    L.append(f"## Verdict: {verdict}")
    L.append(f"- Expected profit ${expected:,.2f} on ${max_loss:,.2f} capital at risk (RoCaR {rocar:.1%}).")
    L.append(f"- Your {side} edge: model P({traded} {'NOT #1' if side=='NO' else '#1'}) {model_side_prob:.1%} vs {side} price {float(portfolio.polymarket_entry_price):.0%} -> {side_edge:+.1%}.")
    L.append(f"- Probability estimate is {'robust' if estimate_robust else 'NOT fully robust'} across the Phase 7 tests (this is about the estimate's stability, not trade direction).")
    L.append("")
    L.append("## Probability & edge")
    L.append(f"- Model P({traded} #1) {model_p:.1%} vs Polymarket YES {yes_price:.1%}. A {side} bet wins when {traded} does {'NOT ' if side=='NO' else ''}finish #1.")
    for t in tickers:
        L.append(f"  - {t}: model {float(probs.loc[t,'Model probability']):.1%} | market YES {float(probs.loc[t,'Polymarket YES price']):.1%}")
    L.append("")
    L.append("## Money view")
    L.append(f"- Expected profit: ${expected:,.2f}")
    L.append(f"- Capital at risk (max loss): ${max_loss:,.2f} | net cash ${float(rm['Net cash outlay']):,.2f}")
    L.append(f"- Return on capital-at-risk: {rocar:.1%}")
    L.append(f"- Loss ladder: VaR5% ${var5:,.2f} | VaR1% ${var1:,.2f} | worst ${max_loss:,.2f}")
    L.append(f"- P(profit) {float(rm['Probability of profit']):.1%} | P(loss) {float(rm['Probability of loss']):.1%}")
    L.append("")
    L.append("## Sensitivity")
    for _, row in assessment["findings"].iterrows():
        L.append(f"- {row['Area']}: {row['Verdict']}")
    L.append("")
    L.append("## Watch-outs")
    for w in assessment["watch_outs"]:
        L.append(f"- {w}")
    return "\n".join(L)


def main() -> str:
    inputs = json.loads(INPUTS_PATH.read_text())
    inputs["as_of"] = date.today().isoformat()
    report = run(inputs)
    REPORTS_DIR.mkdir(exist_ok=True)
    (REPORTS_DIR / f"{inputs['as_of']}.md").write_text(report)
    (REPORTS_DIR / "latest.md").write_text(report)
    return report


if __name__ == "__main__":
    print(main())
