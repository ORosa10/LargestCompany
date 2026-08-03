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
import pickle
from datetime import date
from math import erf, exp, log, sqrt
from pathlib import Path

import numpy as np
import pandas as pd

from iv_surface_model import (
    SURFACE_EXPIRY,
    _interpolated_iv,
    apply_surface_atm_ivs,
    default_surface_nodes,
    market_for,
    normal_cdf_approx,
    run_surface_probability_engine,
)
from model import default_correlation_matrix, run_probability_engine
import phase7 as p7
import phase8 as p8

REPO = Path(__file__).resolve().parent
INPUTS_PATH = REPO / "daily_inputs.json"
REPORTS_DIR = REPO / "reports"
SAVED_STATE = REPO / "saved_state"  # repo-persistence dir the app restores from

# Hedge template for a NO bet: normalized strikes (spot=100). For a YES bet the
# structure is mirrored around spot. Put legs share put_weight, calls call_weight.
HEDGE_TEMPLATE = [
    {"Option type": "Put", "Position": "Long", "ratio": 0.80, "kind": "put"},
    {"Option type": "Put", "Position": "Short", "ratio": 1.00, "kind": "put"},
    {"Option type": "Call", "Position": "Long", "ratio": 0.90, "kind": "call"},
    {"Option type": "Call", "Position": "Short", "ratio": 1.05, "kind": "call"},
]
WEIGHT_VARIANTS = [(1, 3), (2, 3), (1, 4), (2, 4)]  # (put_weight, call_weight)
# Four selection criteria, equal weight. "+" = higher better, "-" = lower better.
RATING_METRICS = [("p_win", "+"), ("ev_sd", "+"), ("es5", "-"), ("roc_var5", "+")]


def _bs_premium(spot, strike, years, iv, rate, kind):
    if years <= 0 or iv <= 0:
        return float(max(spot - strike, 0.0) if kind == "Call" else max(strike - spot, 0.0))
    d1 = (log(spot / strike) + (rate + 0.5 * iv * iv) * years) / (iv * sqrt(years))
    d2 = d1 - iv * sqrt(years)
    ncdf = lambda x: 0.5 * (1.0 + erf(x / sqrt(2.0)))
    if kind == "Call":
        return float(spot * ncdf(d1) - strike * exp(-rate * years) * ncdf(d2))
    return float(strike * exp(-rate * years) * ncdf(-d2) - spot * ncdf(-d1))


def _surface_premium(ticker_nodes, spot, strike, years, rate, forward_ratio, kind):
    """Premium priced off the IV surface: skew IV at the leg strike, with the
    same forward and discount build_surface_marginal uses - so the option is
    valued at the model's own risk-neutral price and adds ~0 EV in the sim."""
    if years <= 0:
        return float(max(spot - strike, 0.0) if kind == "Call" else max(strike - spot, 0.0))
    m = strike / spot
    iv = float(_interpolated_iv(ticker_nodes, np.array([m], dtype=float))[0])
    if iv <= 0:
        return float(max(spot - strike, 0.0) if kind == "Call" else max(strike - spot, 0.0))
    disc = exp(-rate * years)
    rt = sqrt(years)
    F = float(forward_ratio)
    d1 = (log(F / m) + 0.5 * iv * iv * years) / (iv * rt)
    d2 = d1 - iv * rt
    N = lambda z: float(normal_cdf_approx(np.array([z], dtype=float))[0])
    if kind == "Call":
        ratio_price = disc * (F * N(d1) - m * N(d2))
    else:
        ratio_price = disc * (m * N(-d2) - F * N(-d1))
    return float(max(ratio_price, 0.0) * spot)


def _sample_premium(price_samples, strike, kind):
    """Model-fair premium = mean intrinsic value over the simulated terminal
    prices (the surface-implied distribution). Premium nets undiscounted vs
    the same terminal payoff, so each leg contributes ~0 EV by construction -
    options only reshape risk, never manufacture edge."""
    s = np.asarray(price_samples, dtype=float)
    intrinsic = np.maximum(s - strike, 0.0) if kind == "Call" else np.maximum(strike - s, 0.0)
    return float(intrinsic.mean())


def build_legs(ticker, spot, iv, years, rate, put_weight, call_weight, side="NO", ticker_nodes=None, forward_ratio=1.0, price_samples=None):
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
        if price_samples is not None:
            prem = _sample_premium(price_samples, strike, otype)
        elif ticker_nodes is not None and len(ticker_nodes) > 0:
            prem = _surface_premium(ticker_nodes, spot, strike, years, rate, forward_ratio, otype)
        else:
            prem = _bs_premium(spot, strike, years, iv, rate, otype)
        rows.append({
            "Instrument": f"{leg['Position']} {ticker} {otype} {strike:.2f}",
            "Ticker": ticker, "Option type": otype, "Position": leg["Position"],
            "Quantity": quantity, "Strike": strike, "Spot": spot,
            "Theoretical premium": prem,
        })
    return pd.DataFrame(rows)


def _make_portfolio(traded, spot, iv, years, rate, caps_series, side, entry, shares, put_weight, call_weight, ticker_nodes=None, forward_ratio=1.0, price_samples=None):
    legs = build_legs(traded, spot, iv, years, rate, put_weight, call_weight, side, ticker_nodes, forward_ratio, price_samples)
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
    es5 = -float(tail.mean()) if not tail.empty else np.nan  # CVaR 5% (positive loss)
    var5 = -float(pay.quantile(0.05))                        # VaR 5% (positive loss)
    return {
        "expected": expected, "max_loss": max_loss, "sd": sd,
        "p_loss": float((pay < 0).mean()), "p_win": float((pay > 0).mean()),
        "es5": es5, "var5": var5,
        "rocar": (expected / max_loss) if max_loss > 0 else np.nan,
        "ev_sd": (expected / sd) if sd > 0 else np.nan,
        "roc_es5": (expected / es5) if (np.isfinite(es5) and es5 > 0) else np.nan,
        "roc_var5": (expected / var5) if (np.isfinite(var5) and var5 > 0) else np.nan,
    }


def rate_variants(variants):
    """Equal-weight composite in [0,1] over P(win), EV/SD, CVaR5% (lower better),
    and return-on-VaR5%. Each min-max normalized across the variants."""
    for metric, direction in RATING_METRICS:
        vals = np.array([v[metric] if np.isfinite(v[metric]) else np.nan for v in variants], dtype=float)
        finite = vals[np.isfinite(vals)]
        lo, hi = (finite.min(), finite.max()) if finite.size else (0.0, 0.0)
        for v in variants:
            x = v[metric]
            if not np.isfinite(x):
                n = 0.0
            elif hi > lo:
                n = (x - lo) / (hi - lo) if direction == "+" else (hi - x) / (hi - lo)
            else:
                n = 0.5
            v[f"n_{metric}"] = n
    for v in variants:
        v["score"] = float(np.mean([v[f"n_{m}"] for m, _ in RATING_METRICS]))
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


def upcoming_earnings(tickers, as_of, resolution):
    """Best-effort earnings dates for each ticker between as_of and resolution.
    Network-dependent (Yahoo). Returns {ticker: [date, ...]} and NEVER raises -
    on any failure it returns {} so the report is never blocked."""
    out = {}
    try:
        import yfinance as yf
        from market_data import yahoo_symbol
    except Exception:
        return out
    for t in tickers:
        found = []
        try:
            yft = yf.Ticker(yahoo_symbol(t))
            try:
                df = yft.get_earnings_dates(limit=16)
            except Exception:
                df = None
            if df is not None and len(df):
                for ts in df.index:
                    try:
                        dd = ts.date()
                    except Exception:
                        continue
                    if as_of <= dd <= resolution:
                        found.append(dd)
            if not found:
                cal = getattr(yft, "calendar", None)
                ed = cal.get("Earnings Date") if isinstance(cal, dict) else None
                if ed:
                    for d0 in (ed if isinstance(ed, (list, tuple)) else [ed]):
                        dd = d0.date() if hasattr(d0, "date") else d0
                        if isinstance(dd, date) and as_of <= dd <= resolution:
                            found.append(dd)
        except Exception:
            pass
        if found:
            out[t] = sorted(set(found))
    return out


def run(inputs: dict) -> str:
    resolution_iso = inputs["target_date"]
    target = date.fromisoformat(resolution_iso)
    as_of = date.fromisoformat(inputs.get("as_of", date.today().isoformat()))
    market = market_for(resolution_iso)
    option_expiry = date.fromisoformat(market["option_expiry"])
    # Horizon runs to the option expiry (where the IV lives and options settle);
    # the Polymarket resolution can be a few days later - a small basis.
    days = max((option_expiry - as_of).days, 1)
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

    # Save the morning market snapshot so the Streamlit app can offer "morning
    # Yahoo data" as an alternative to a fresh live fetch.
    try:
        REPORTS_DIR.mkdir(exist_ok=True)
        (REPORTS_DIR / "market_snapshot.json").write_text(json.dumps({
            "as_of": as_of.isoformat(),
            "resolution": resolution_iso,
            "option_expiry": market["option_expiry"],
            "source": data_source,
            "market_caps": caps,
            "spots": spots,
            "polymarket_yes": yes_prices,
            "polymarket_no": no_prices,
        }, indent=2))
    except Exception:  # noqa: BLE001
        pass

    corr = default_correlation_matrix(tickers)
    surf_inputs = apply_surface_atm_ivs(universe.copy(), resolution=resolution_iso)
    result, _ = run_surface_probability_engine(
        surf_inputs, corr, days_to_target=days, simulations=sims, seed=seed,
        surface_nodes=default_surface_nodes(resolution=resolution_iso), risk_free_rate=rate,
    )
    probs = result.results.set_index("Ticker")
    caps_series = universe.set_index("Ticker")["Current market cap"].astype(float)

    # --- Candidates: YES and NO edge for every ticker (edge = model fair - price) ---
    candidates = []
    for t in tickers:
        mp = float(probs.loc[t, "Model probability"])
        yp = float(yes_prices[t])
        np_ = float(no_prices.get(t, round(1.0 - yp, 4)))
        candidates.append({"ticker": t, "side": "YES", "model_side": mp, "price": yp, "edge": mp - yp})
        candidates.append({"ticker": t, "side": "NO", "model_side": 1.0 - mp, "price": np_, "edge": (1.0 - mp) - np_})
    best_yes = max((c for c in candidates if c["side"] == "YES"), key=lambda c: c["edge"])
    best_no = max((c for c in candidates if c["side"] == "NO"), key=lambda c: c["edge"])

    # Primary trade is chosen AFTER both sides are analyzed - see below (max composite).
    _surface_nodes_all = default_surface_nodes(resolution=resolution_iso)

    def d(x):
        return f"${x:,.2f}"

    iv_by_ticker = surf_inputs.set_index("Ticker")["Implied volatility"].astype(float)

    def _analyze(cand):
        """Full weight sweep + Phase 7/8 for one candidate (ticker+side)."""
        tk, sd, ent = cand["ticker"], cand["side"], float(cand["price"])
        sp = float(spots[tk])
        mp = float(probs.loc[tk, "Model probability"])
        iv_c = float(iv_by_ticker.loc[tk])
        tk_nodes = _surface_nodes_all[_surface_nodes_all["Ticker"] == tk]
        fr = float(surf_inputs.set_index("Ticker").loc[tk, "Forward / spot"]) if "Forward / spot" in surf_inputs.columns else 1.0
        vlist = []
        for pw, cw in WEIGHT_VARIANTS:
            pf = _make_portfolio(tk, sp, iv_c, years, rate, caps_series, sd, ent, shares, pw, cw, ticker_nodes=tk_nodes, forward_ratio=fr)
            vlist.append({"label": f"{pw}/{pw}/{cw}/{cw}", "portfolio": pf, **_variant_stats(result, pf)})
        bestv = rate_variants(vlist)
        pf = bestv["portfolio"]
        seeds = list(range(seed, seed + 5))
        disp = p7.dispersion_summary(p7.multi_seed_dispersion(surf_inputs, corr, pf, days_to_target=days, simulations=sims, seeds=seeds))
        cop = p7.copula_tail_stress(surf_inputs, corr, pf, days_to_target=days, simulations=sims, seeds=seeds)
        ivs = p7.iv_scaling_scan(surf_inputs, corr, selected_ticker=tk, days_to_target=days, simulations=sims, seed=seed, factors=[0.5, 0.75, 1.0, 1.25, 1.5, 2.0])
        gps = p7.gap_scaling_scan(surf_inputs, corr, selected_ticker=tk, days_to_target=days, simulations=sims, seed=seed, factors=[0.5, 0.75, 1.0, 1.25, 1.5, 2.0])
        gv = p7.gap_vs_randomness(ivs, gps)
        variants_corr = {"Saved": corr, "Independent": p7.constant_correlation(tickers, 0.0), "High 0.8": p7.constant_correlation(tickers, 0.8)}
        grid = p7.model_robustness(surf_inputs, variants_corr, selected_ticker=tk, days_to_target=days, simulations=sims, seed=seed, shock_models=["Normal shocks", "Student-t copula df=5"])
        asmt = p7.assessment(disp, cop, gv, p7.robustness_summary(grid), selected_ticker=tk)
        rm = p8.risk_metrics(result, pf)
        v5 = p8.value_at_risk(result, pf, 0.05)
        v1 = p8.value_at_risk(result, pf, 0.01)
        exp = float(rm["Expected profit"])
        mloss = float(rm["Max loss (capital at risk)"])
        rcar = float(rm["Return on capital-at-risk"])
        msp = (1.0 - mp) if sd == "NO" else mp
        edg = msp - ent
        vd = "UNFAVORABLE" if edg <= 0.0 else ("MARGINAL" if edg <= 0.05 else "FAVORABLE")
        ftext = " ".join(asmt["findings"]["Verdict"].astype(str)).lower()
        if "flips sign" in ftext:
            frag = "HIGH - edge is model-dependent (flips sign across models)"
        elif ("not converged" in ftext) or ("sensitive to tail" in ftext):
            frag = "MEDIUM - some assumptions move it (convergence / tail dependence)"
        else:
            frag = "LOW - holds across the robustness tests"

        def _pt(res):
            return float(res.results.set_index("Ticker").loc[tk, "Model probability"])
        mprobs = {"IV surface + Gaussian copula": mp}
        try:
            mprobs["ATM lognormal + Normal"] = _pt(run_probability_engine(surf_inputs, corr, days_to_target=days, simulations=sims, seed=seed, shock_model="Normal shocks"))
            mprobs["ATM lognormal + Student-t copula df=5"] = _pt(run_probability_engine(surf_inputs, corr, days_to_target=days, simulations=sims, seed=seed, shock_model="Student-t copula df=5"))
            mprobs["ATM lognormal + Student-t df=6 (fat marginals)"] = _pt(run_probability_engine(surf_inputs, corr, days_to_target=days, simulations=sims, seed=seed, shock_model="Student-t df=6"))
        except Exception:  # noqa: BLE001
            pass
        return {
            "ticker": tk, "side": sd, "entry": ent, "spot": sp, "model_p": mp,
            "model_side_prob": msp, "edge": edg, "verdict": vd, "fragility": frag,
            "best": bestv, "variants": vlist, "portfolio": pf, "assessment": asmt,
            "rm": rm, "var5": v5, "var1": v1, "expected": exp, "max_loss": mloss,
            "rocar": rcar, "model_probs": mprobs,
            "prob_lo": min(mprobs.values()), "prob_hi": max(mprobs.values()),
        }

    _cache = {}

    def analyze(cand):
        key = (cand["ticker"], cand["side"])
        if key not in _cache:
            _cache[key] = _analyze(cand)
        return _cache[key]

    yes_A = analyze(best_yes)
    no_A = analyze(best_no)

    # Trade-level composite over each side's BEST structure, using the same four
    # criteria as the structure sweep: P(win), EV/SD, CVaR5% (lower better) and
    # RoC/VaR5%. Computed always (shown in the report); it drives the default pick.
    _sides = [yes_A, no_A]
    for metric, direction in RATING_METRICS:
        vals = np.array([A["best"][metric] for A in _sides], dtype=float)
        finite = vals[np.isfinite(vals)]
        lo, hi = (finite.min(), finite.max()) if finite.size else (0.0, 0.0)
        for A in _sides:
            x = A["best"][metric]
            if not np.isfinite(x):
                n = 0.0
            elif hi > lo:
                n = (x - lo) / (hi - lo) if direction == "+" else (hi - x) / (hi - lo)
            else:
                n = 0.5
            A.setdefault("_ncomp", {})[metric] = n
    for A in _sides:
        A["trade_composite"] = float(np.mean([A["_ncomp"][m] for m, _ in RATING_METRICS]))

    # --- Primary trade selection: overrides win, else higher composite ---
    if inputs.get("traded_ticker") and inputs.get("force_side"):
        primary = next(c for c in candidates if c["ticker"] == inputs["traded_ticker"] and c["side"] == str(inputs["force_side"]).upper())
    elif inputs.get("traded_ticker"):
        primary = max((c for c in candidates if c["ticker"] == inputs["traded_ticker"]), key=lambda c: c["edge"])
    else:
        primary = best_yes if yes_A["trade_composite"] >= no_A["trade_composite"] else best_no
    primary_A = analyze(primary)

    # Save the primary (max-edge) structure as the app preset - unchanged.
    try:
        SAVED_STATE.mkdir(exist_ok=True)
        candidate = {
            "mapped_legs": primary_A["portfolio"].option_legs.copy(),
            "polymarket": {"selected_ticker": primary_A["ticker"], "side": primary_A["side"], "entry": float(primary_A["entry"]), "shares": shares},
            "spots": {primary_A["ticker"]: float(primary_A["spot"])},
            "contract_multiplier": 100.0,
            "run_metadata": {"target_date": resolution_iso, "option_expiry": market["option_expiry"], "source": "daily_report"},
            "saved_at": as_of.isoformat(),
            "weights": primary_A["best"]["label"],
        }
        with (SAVED_STATE / "phase6_execution_candidate.pkl").open("wb") as fh:
            pickle.dump(candidate, fh)
    except Exception:  # noqa: BLE001
        pass

    # Top of report keyed to the primary (traded) trade.
    traded = primary_A["ticker"]; side = primary_A["side"]; entry = primary_A["entry"]
    model_p = primary_A["model_p"]; model_side_prob = primary_A["model_side_prob"]
    edge = primary_A["edge"]; verdict = primary_A["verdict"]; fragility = primary_A["fragility"]
    expected = primary_A["expected"]; max_loss = primary_A["max_loss"]; rocar = primary_A["rocar"]
    side_word = "#1" if side == "YES" else "NOT #1"
    traded_yes = next(c for c in candidates if c["ticker"] == traded and c["side"] == "YES")
    traded_no = next(c for c in candidates if c["ticker"] == traded and c["side"] == "NO")
    naked_yes, naked_no = traded_yes["edge"], traded_no["edge"]
    if fragility.startswith("LOW"):
        rob_note = "estimate is robust across the Phase 7 tests."
    elif fragility.startswith("HIGH"):
        rob_note = "edge is model-dependent (flips sign across models) - treat the direction as uncertain."
    else:
        rob_note = "probability estimate is NOT fully robust (stability of the estimate, not the trade direction)."

    earnings = upcoming_earnings(tickers, as_of, target) if inputs.get("check_earnings", True) else {}

    L = []
    L.append(f"# LargestCompany daily report - {as_of.isoformat()}")
    L.append("")
    L.append(f"## Verdict: {verdict}  (edge {edge:+.1%})")
    L.append(f"- Resolution {target.isoformat()} | option expiry {option_expiry.isoformat()} ({days} days) | traded {traded} | side auto-picked **{side}** @ {entry:.2f}")
    L.append(f"- Fragility (Phase 7): **{fragility}**")
    L.append(f"- Data: {data_source}")
    L.append("")
    L.append("## Summary")
    L.append(f"- Expected profit **{d(expected)}** on {d(max_loss)} capital at risk (RoCaR {rocar:.1%}).")
    L.append(f"- Side auto-picked **{side}** by composite (P(win), EV/SD, CVaR5%, RoC/VaR5%); naked YES EV {naked_yes:+.1%} vs naked NO EV {naked_no:+.1%}.")
    L.append(f"- Your {side} edge: model P({traded} {side_word}) {model_side_prob:.1%} vs {side} price {entry:.0%} -> {edge:+.1%}.")
    L.append(f"- Robustness: {rob_note}")
    cap_str = ", ".join(f"{t} ${caps[t] / 1e12:.2f}T" for t in sorted(caps, key=lambda x: -caps[x]))
    L.append(f"- Current market caps (Yahoo, model ranks by these): {cap_str}.")
    L.append("")
    L.append("## Earnings before resolution (data caveat)")
    if earnings:
        for t in tickers:
            if t in earnings:
                L.append(f"- **{t}** reports: {', '.join(dd.isoformat() for dd in earnings[t])}")
        L.append("Heads-up: Yahoo spot/caps use the last **regular** close, so on/after these dates the after-hours earnings move is not yet in the ranking - treat P(#1) and edge as stale for up to a session around them.")
    else:
        L.append(f"- No scheduled NVDA/AAPL/GOOGL earnings detected between {as_of.isoformat()} and {target.isoformat()} (or the calendar was unavailable).")
    L.append("")
    L.append("## Edge: market vs simulation")
    L.append("| | Value |")
    L.append("|---|---|")
    L.append(f"| Polymarket says ({side}) | {entry:.1%} |")
    L.append(f"| Simulation says ({side} fair) | {model_side_prob:.1%} |")
    L.append(f"| Edge (fair - price) | **{edge:+.1%}** |")
    L.append(f"| Grade | **{verdict}** (>5% favorable, 0-5% marginal, <=0 unfavorable) |")
    L.append("")
    L.append("## Trade candidates (best edge per side)")
    L.append("| | Ticker | Model fair | Price | Edge | Composite |")
    L.append("|---|---|---|---|---|---|")
    yes_mark = " (traded)" if (side == "YES" and traded == best_yes["ticker"]) else ""
    no_mark = " (traded)" if (side == "NO" and traded == best_no["ticker"]) else ""
    L.append(f"| Best YES{yes_mark} | {best_yes['ticker']} | {best_yes['model_side']:.1%} | {best_yes['price']:.0%} | {best_yes['edge']:+.1%} | {yes_A['trade_composite']:.2f} |")
    L.append(f"| Best NO{no_mark} | {best_no['ticker']} | {best_no['model_side']:.1%} | {best_no['price']:.0%} | {best_no['edge']:+.1%} | {no_A['trade_composite']:.2f} |")
    L.append(f"Default = max composite: **{traded} {side}** (composite {primary_A['trade_composite']:.2f}, edge {edge:+.1%}). Composite = P(win), EV/SD, CVaR5%, RoC/VaR5% - risk-adjusted, not raw edge. Both sides get a full risk block below.")
    L.append("")
    L.append("## Probability by name (model vs market)")
    L.append("| Ticker | Model P(#1) | Market YES | Market NO | YES edge | NO edge |")
    L.append("|---|---|---|---|---|---|")
    for t in tickers:
        mp_t = float(probs.loc[t, "Model probability"])
        yp_t = float(yes_prices[t])
        np_t = no_prices.get(t)
        yes_edge_t = mp_t - yp_t
        star_t = " *" if t == traded else ""
        if np_t is None:
            L.append(f"| {t}{star_t} | {mp_t:.1%} | {yp_t:.0%} | n/a | {yes_edge_t:+.1%} | n/a |")
        else:
            no_edge_t = (1.0 - mp_t) - float(np_t)
            L.append(f"| {t}{star_t} | {mp_t:.1%} | {yp_t:.0%} | {float(np_t):.0%} | {yes_edge_t:+.1%} | {no_edge_t:+.1%} |")
    L.append("* = traded name.")
    L.append("")

    def render_block(A, label):
        is_primary = (A["ticker"] == traded and A["side"] == side)
        tag = " (traded / app preset)" if is_primary else ""
        best = A["best"]; rm = A["rm"]; variants = A["variants"]
        var5 = A["var5"]; var1 = A["var1"]; var_worst = A["max_loss"]
        exp = A["expected"]
        rv5 = exp / var5 if var5 > 0 else float("nan")
        rv1 = exp / var1 if var1 > 0 else float("nan")
        rvw = exp / var_worst if var_worst > 0 else float("nan")
        L.append("---")
        L.append(f"# {label} trade: {A['ticker']} {A['side']} @ {A['entry']:.2f}{tag}")
        L.append(f"Edge {A['edge']:+.1%} | Verdict {A['verdict']} | Fragility {A['fragility'].split(' - ')[0]}")
        L.append("")
        L.append(f"## Best structure: {best['label']} (put/put/call/call)")
        L.append("| Metric | Value |")
        L.append("|---|---|")
        L.append(f"| Expected profit | {d(exp)} |")
        L.append(f"| Payoff SD | {d(float(best['sd']))} |")
        L.append(f"| VaR 5% | {d(var5)} |")
        L.append(f"| VaR 1% | {d(var1)} |")
        L.append(f"| Worst case | {d(var_worst)} |")
        L.append(f"| Probability of profit | {float(rm['Probability of profit']):.1%} |")
        L.append(f"| Return on VaR 5% | {rv5:.1%} |")
        L.append(f"| Return on VaR 1% | {rv1:.1%} |")
        L.append(f"| Return on worst case | {rvw:.1%} |")
        L.append("")
        L.append("## Structure comparison (weights: put/put/call/call)")
        L.append("Score = equal-weight of P(win), EV/SD, CVaR5% (lower better), return-on-VaR5%.")
        L.append("| Weights | Score | P(win) | EV/SD | CVaR5% | RoC/VaR5% | Expected |")
        L.append("|---|---|---|---|---|---|---|")
        for v in sorted(variants, key=lambda x: x["score"], reverse=True):
            star = " (best)" if v is best else ""
            L.append(f"| {v['label']}{star} | {v['score']:.2f} | {v['p_win']:.0%} | {v['ev_sd']:+.2f} | {d(v['es5'])} | {v['roc_var5']:+.1%} | {d(v['expected'])} |")
        L.append("")
        fmap = {str(r["Area"]).split(".")[0].strip(): str(r["Verdict"]) for _, r in A["assessment"]["findings"].iterrows()}
        L.append("## Consistency check (across assumptions & simulation reruns)")
        L.append(f"Overall fragility: **{A['fragility']}**. P({A['ticker']} #1) ranges {A['prob_lo']:.1%}-{A['prob_hi']:.1%} across models (spread {A['prob_hi'] - A['prob_lo']:.1%}).")
        L.append("| Check | Result |")
        L.append("|---|---|")
        L.append(f"| Across simulation reruns (seeds) | {fmap.get('1', 'n/a')} |")
        L.append(f"| Across models & tails (IV surface / ATM / copula) | {fmap.get('5', 'n/a')} |")
        L.append(f"| Tail dependence (joint crashes) | {fmap.get('2', 'n/a')} |")
        L.append(f"| Dominant lever | {fmap.get('3', 'n/a')} |")
        L.append("")
        L.append(f"P({A['ticker']} #1) by model:")
        L.append("| Model | P(#1) |")
        L.append("|---|---|")
        for name, val in A["model_probs"].items():
            L.append(f"| {name} | {val:.1%} |")
        L.append("")
        L.append("## Watch-outs")
        for w in A["assessment"]["watch_outs"]:
            L.append(f"- {w}")
        L.append("")

    render_block(yes_A, "Best YES")
    render_block(no_A, "Best NO")
    return "\n".join(L).rstrip() + "\n"


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
