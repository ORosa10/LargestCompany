"""Headless daily pipeline: fetch data, run Phases 1-8, write a verdict report.

Runs in GitHub Actions (open internet -> live Yahoo data). Polymarket YES/NO
prices come from daily_inputs.json (provided manually). The pipeline:

1. Fetch market caps + spots (live). If missing and no manual value -> report says
   data is unavailable and asks for it by hand (never fabricates from stale data).
2. Run the IV-surface probability engine -> model P(ticker #1).
3. Pick the Polymarket side automatically: the naked bet (YES vs NO) with the
   higher expected value on the model. If YES wins, the option hedge is mirrored
   around spot (Put<->Call, strike ratio -> 2 - ratio; spreads reverse and skew
   lightly above spot).
4. Sweep 4 weight variants and pick the best by an equal-weight composite of
   EV/SD, return on capital-at-risk, return on ES5% capital, and P(win).
5. Run the Phase 7 assessment and Phase 8 risk metrics on the winner.

Writes reports/<date>.md and reports/latest.md.
"""

from __future__ import annotations

import json
from datetime import date
from math import erf, exp, log, sqrt
from pathlib import Path

import numpy as np
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

# Hedge template for a NO bet: normalized strikes (spot=100). For a YES bet the
# structure is mirrored around spot. Put legs share put_weight, calls call_weight.
HEDGE_TEMPLATE = [
    {"Option type": "Put", "Position": "Long", "ratio": 0.80, "kind": "put"},
    {"Option type": "Put", "Position": "Short", "ratio": 1.00, "kind": "put"},
    {"Option type": "Call", "Position": "Long", "ratio": 0.90, "kind": "call"},
    {"Option type": "Call", "Position": "Short", "ratio": 1.05, "kind": "call"},
]
WEIGHT_VARIANTS = [(1, 3), (2, 3), (1, 4), (2, 4)]  # (put_weight, call_weight)
RATING_METRICS = ["ev_sd", "rocar", "roc_es5", "p_win"]  # equal weight, higher = better


def _bs_premium(spot, strike, years, iv, rate, kind):
    if years <= 0 or iv <= 0:
        return float(max(spot - strike, 0.0) if kind == "Call" else max(strike - spot, 0.0))
    d1 = (log(spot / strike) + (rate + 0.5 * iv * iv) * years) / (iv * sqrt(years))
    d2 = d1 - iv * sqrt(years)
    ncdf = lambda x: 0.5 * (1.0 + erf(x / sqrt(2.0)))
    if kind == "Call":
        return float(spot * ncdf(d1) - strike * exp(-rate * years) * ncdf(d2))
    return float(strike * exp(-rate * years) * ncdf(-d2) - spot * ncdf(-d1))


def build_legs(ticker, spot, iv, years, rate, put_weight, call_weight, side="NO"):
    """Build the 4 option legs. For a YES bet the structure is mirrored around
    spot: Put<->Call and strike ratio -> 2 - ratio (weights follow the template
    slot so the payoff is a true reflection)."""
    rows = []
    for leg in HEDGE_TEMPLATE:
        weight = put_weight if leg["kind"] == "put" else call_weight
        otype, ratio = leg["Option type"], leg["ratio"]
        if str(side).upper() == "YES":
            otype = "Call" if otype == "Put" else "Put"
            ratio = 2.0 - ratio
        strike = round(spot * ratio, 2)
        quantity = weight / spot  # Phase 5/6 share-equivalent -> contracts
        rows.append({
            "Instrument": f"{leg['Position']} {ticker} {otype} {strike:.2f}",
            "Ticker": ticker, "Option type": otype, "Position": leg["Position"],
            "Quantity": quantity, "Strike": strike, "Spot": spot,
            "Theoretical premium": _bs_premium(spot, strike, years, iv, rate, otype),
        })
    return pd.DataFrame(rows)


def _make_portfolio(traded, spot, iv, years, rate, caps_series, side, entry, shares, put_weight, call_weight):
    legs = build_legs(traded, spot, iv, years, rate, put_weight, call_weight, side)
    return p7.PortfolioSpec(
        option_legs=legs, current_market_caps=caps_series, spot_prices=pd.Series({traded: spot}),
        selected_ticker=traded, polymarket_side=side, polymarket_entry_price=float(entry),
        polymarket_quantity=float(shares), contract_multiplier=100.0, include_option_premiums=True,
    )


def _variant_stats(result, portfolio):
    pay = p7.portfolio_scenarios(result, portfolio)["Total payoff"].astype(float)
    expected = float(pay.mean())
    max_loss = max(-float(pay.min()), 0.0)
    sd = float(pay.std(ddof=0))
    threshold = float(pay.quantile(0.05))
    tail = pay[pay <= threshold]
    es5 = -float(tail.mean()) if not tail.empty else np.nan  # positive loss magnitude
    return {
        "expected": expected, "max_loss": max_loss, "sd": sd,
        "p_loss": float((pay < 0).mean()), "p_win": float((pay > 0).mean()),
        "es5": es5,
        "rocar": (expected / max_loss) if max_loss > 0 else np.nan,
        "ev_sd": (expected / sd) if sd > 0 else np.nan,
        "roc_es5": (expected / es5) if (np.isfinite(es5) and es5 > 0) else np.nan,
    }


def rate_variants(variants):
    """Equal-weight composite score in [0,1] over the rating metrics (higher
    better), min-max normalized across the variants."""
    for metric in RATING_METRICS:
        vals = np.array([v[metric] if np.isfinite(v[metric]) else np.nan for v in variants], dtype=float)
        finite = vals[np.isfinite(vals)]
        lo, hi = (finite.min(), finite.max()) if finite.size else (0.0, 0.0)
        for v in variants:
            x = v[metric]
            if not np.isfinite(x):
                v[f"n_{metric}"] = 0.0
            elif hi > lo:
                v[f"n_{metric}"] = (x - lo) / (hi - lo)
            else:
                v[f"n_{metric}"] = 0.5
    for v in variants:
        v["score"] = float(np.mean([v[f"n_{m}"] for m in RATING_METRICS]))
    return max(variants, key=lambda v: v["score"])


def fetch_market_data(tickers, manual_caps, manual_spots):
    """(caps, spots, source, missing). Live Yahoo; only explicit manual values
    otherwise; never stale defaults. Missing tickers are reported."""
    notes, missing = [], []
    live_caps, live_spots = {}, {}
    try:
        from market_data import fetch_market_caps
        df = fetch_market_caps(list(tickers))
        live_caps = {str(r["ticker"]): float(r["market_cap"]) for _, r in df.iterrows()}
    except Exception as exc:  # noqa: BLE001
        notes.append(f"caps fetch failed ({type(exc).__name__})")
    try:
        from market_data import fetch_spot_prices
        df = fetch_spot_prices(list(tickers))
        col = next((c for c in ["spot_price", "spot"] if c in df.columns), None)
        if col is None:
            raise KeyError("no spot column")
        live_spots = {str(r["ticker"]): float(r[col]) for _, r in df.iterrows()}
    except Exception as exc:  # noqa: BLE001
        notes.append(f"spots fetch failed ({type(exc).__name__})")
    caps, spots = {}, {}
    for t in tickers:
        if live_caps.get(t, 0) > 0:
            caps[t] = live_caps[t]
        elif t in (manual_caps or {}):
            caps[t] = float(manual_caps[t])
        else:
            missing.append(f"{t} market cap")
        if live_spots.get(t, 0) > 0:
            spots[t] = live_spots[t]
        elif t in (manual_spots or {}):
            spots[t] = float(manual_spots[t])
        else:
            missing.append(f"{t} spot")
    source = "Yahoo live" if not notes else "; ".join(notes) + "; manual where provided"
    return caps, spots, source, missing


def run(inputs: dict) -> str:
    target = date.fromisoformat(inputs["target_date"])
    as_of = date.fromisoformat(inputs.get("as_of", date.today().isoformat()))
    days = max((target - as_of).days, 1)
    years = days / 365.0
    rate = float(inputs.get("risk_free_rate", 0.04))
    sims = int(inputs.get("simulations", 40000))
    seed = int(inputs.get("seed", 42))
    shares = float(inputs.get("shares", 100.0))

    yes_prices = dict(inputs.get("polymarket_yes") or {r["Ticker"]: r["Polymarket YES price"] for r in inputs.get("universe", [])})
    no_prices = dict(inputs.get("polymarket_no") or {})
    tickers = list(yes_prices.keys())
    caps, spots, data_source, missing = fetch_market_data(tickers, inputs.get("manual_market_caps") or {}, inputs.get("manual_spots") or {})

    if missing:
        return (
            f"# LargestCompany daily report - {as_of.isoformat()}\n\n## Data unavailable\n"
            f"Could not get live data from Yahoo and no manual values were provided for: {', '.join(missing)}.\n\n"
            f"Fetch status: {data_source}.\n\nNo numbers were produced (nothing stale is used). "
            f"To run today, provide the missing values manually: edit `daily_inputs.json` -> `manual_market_caps` "
            f"and/or `manual_spots`, commit, and re-run - or send them to Claude."
        )

    universe = pd.DataFrame([
        {"Ticker": t, "Current market cap": caps[t], "Implied volatility": 0.30, "Polymarket YES price": yes_prices[t]}
        for t in tickers
    ])
    traded = inputs.get("traded_ticker") or max(yes_prices, key=yes_prices.get)
    spot = spots[traded]

    corr = default_correlation_matrix(tickers)
    surf_inputs = apply_surface_atm_ivs(universe.copy())
    result, _ = run_surface_probability_engine(
        surf_inputs, corr, days_to_target=days, simulations=sims, seed=seed,
        surface_nodes=default_surface_nodes(), risk_free_rate=rate,
    )
    probs = result.results.set_index("Ticker")
    model_p = float(probs.loc[traded, "Model probability"])
    iv_atm = float(surf_inputs.set_index("Ticker").loc[traded, "Implied volatility"])
    caps_series = universe.set_index("Ticker")["Current market cap"].astype(float)

    # --- Side selection: naked bet with higher expected value on the model ---
    yes_price = float(yes_prices[traded])
    no_price = float(no_prices.get(traded, round(1.0 - yes_price, 4)))
    yes_ev = model_p - yes_price
    no_ev = (1.0 - model_p) - no_price
    side = str(inputs.get("force_side") or ("YES" if yes_ev >= no_ev else "NO")).upper()
    entry = yes_price if side == "YES" else no_price

    # --- Sweep weight variants, rate, pick winner ---
    variants = []
    for pw, cw in WEIGHT_VARIANTS:
        pf = _make_portfolio(traded, spot, iv_atm, years, rate, caps_series, side, entry, shares, pw, cw)
        variants.append({"label": f"{pw}/{pw}/{cw}/{cw}", "portfolio": pf, **_variant_stats(result, pf)})
    best = rate_variants(variants)
    portfolio = best["portfolio"]

    seeds = list(range(seed, seed + 5))
    disp = p7.dispersion_summary(p7.multi_seed_dispersion(surf_inputs, corr, portfolio, days_to_target=days, simulations=sims, seeds=seeds))
    cop = p7.copula_tail_stress(surf_inputs, corr, portfolio, days_to_target=days, simulations=sims, seeds=seeds)
    ivs = p7.iv_scaling_scan(surf_inputs, corr, selected_ticker=traded, days_to_target=days, simulations=sims, seed=seed, factors=[0.5, 0.75, 1.0, 1.25, 1.5, 2.0])
    gps = p7.gap_scaling_scan(surf_inputs, corr, selected_ticker=traded, days_to_target=days, simulations=sims, seed=seed, factors=[0.5, 0.75, 1.0, 1.25, 1.5, 2.0])
    gv = p7.gap_vs_randomness(ivs, gps)
    variants_corr = {"Saved": corr, "Independent": p7.constant_correlation(tickers, 0.0), "High 0.8": p7.constant_correlation(tickers, 0.8)}
    grid = p7.model_robustness(surf_inputs, variants_corr, selected_ticker=traded, days_to_target=days, simulations=sims, seed=seed, shock_models=["Normal shocks", "Student-t copula df=5"])
    assessment = p7.assessment(disp, cop, gv, p7.robustness_summary(grid), selected_ticker=traded)

    rm = p8.risk_metrics(result, portfolio)
    var5 = p8.value_at_risk(result, portfolio, 0.05)
    var1 = p8.value_at_risk(result, portfolio, 0.01)
    expected = float(rm["Expected profit"])
    max_loss = float(rm["Max loss (capital at risk)"])
    rocar = float(rm["Return on capital-at-risk"])
    estimate_robust = "robust" in assessment["headline"].lower() and "not fully" not in assessment["headline"].lower()
    verdict = "FAVORABLE" if expected > 0 else ("BREAKEVEN" if abs(expected) < 1e-6 else "UNFAVORABLE")
    model_side_prob = (1.0 - model_p) if side == "NO" else model_p

    L = []
    L.append(f"# LargestCompany daily report - {as_of.isoformat()}")
    L.append("")
    L.append(f"Target {target.isoformat()} ({days} days) | traded {traded} | side {side} @ {entry:.2f} | best structure {best['label']} | {sims:,} sims")
    L.append(f"Data: {data_source}")
    L.append("")
    L.append(f"## Verdict: {verdict}")
    L.append(f"- Expected profit ${expected:,.2f} on ${max_loss:,.2f} capital at risk (RoCaR {rocar:.1%}).")
    L.append(f"- Side auto-picked {side}: naked YES EV {yes_ev:+.1%} vs naked NO EV {no_ev:+.1%}.")
    L.append(f"- Your {side} edge: model P({traded} {'#1' if side=='YES' else 'NOT #1'}) {model_side_prob:.1%} vs {side} price {entry:.0%} -> {model_side_prob - entry:+.1%}.")
    L.append(f"- Probability estimate is {'robust' if estimate_robust else 'NOT fully robust'} (stability of the estimate, not trade direction).")
    L.append("")
    L.append("## Probability & edge")
    L.append(f"- Model P({traded} #1) {model_p:.1%} vs Polymarket YES {yes_price:.1%}.")
    for t in tickers:
        L.append(f"  - {t}: model {float(probs.loc[t,'Model probability']):.1%} | market YES {float(probs.loc[t,'Polymarket YES price']):.1%}")
    L.append("")
    L.append("## Money view")
    L.append(f"- Expected profit: ${expected:,.2f}")
    L.append(f"- Capital at risk (max loss): ${max_loss:,.2f} | net cash ${float(rm['Net cash outlay']):,.2f}")
    L.append(f"- Return on capital-at-risk: {rocar:.1%}")
    L.append(f"- Loss ladder: VaR5% ${var5:,.2f} | VaR1% ${var1:,.2f} | worst ${max_loss:,.2f}")
    L.append(f"- P(win) {float(rm['Probability of profit']):.1%} | P(loss) {float(rm['Probability of loss']):.1%}")
    L.append("")
    L.append("## Structure selection (put/put/call/call weights)")
    L.append("Equal-weight composite score of EV/SD, RoCaR, RoC/ES5%, P(win) (0-1, higher=better). Options are ~fairly priced, so weights reshape risk more than edge.")
    for v in sorted(variants, key=lambda x: x["score"], reverse=True):
        mark = " <- best" if v is best else ""
        L.append(
            f"- {v['label']}: score {v['score']:.2f} | EV/SD {v['ev_sd']:+.2f} | RoCaR {v['rocar']:+.1%} | "
            f"RoC/ES5% {v['roc_es5']:+.1%} | P(win) {v['p_win']:.0%} | exp ${v['expected']:,.2f} | maxloss ${v['max_loss']:,.2f}{mark}"
        )
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
