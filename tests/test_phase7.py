import numpy as np
import pandas as pd
import pytest

from model import default_correlation_matrix
from phase7 import (
    PortfolioSpec,
    constant_correlation,
    copula_tail_stress,
    dispersion_summary,
    gap_scaling_scan,
    gap_vs_randomness,
    iv_scaling_scan,
    model_robustness,
    multi_seed_dispersion,
    robustness_summary,
    scan_sensitivity,
)


def _universe():
    company = pd.DataFrame(
        [
            {"Ticker": "NVDA", "Current market cap": 4_300e9, "Implied volatility": 0.42, "Polymarket YES price": 0.83},
            {"Ticker": "AAPL", "Current market cap": 3_100e9, "Implied volatility": 0.24, "Polymarket YES price": 0.123},
            {"Ticker": "GOOGL", "Current market cap": 2_100e9, "Implied volatility": 0.28, "Polymarket YES price": 0.046},
        ]
    )
    corr = default_correlation_matrix(company["Ticker"].tolist())
    return company, corr


def _portfolio():
    current_caps = pd.Series({"NVDA": 4_300e9, "AAPL": 3_100e9, "GOOGL": 2_100e9})
    spots = pd.Series({"NVDA": 175.0, "AAPL": 230.0, "GOOGL": 180.0})
    legs = pd.DataFrame(
        [
            {"Instrument": "NVDA long put 160", "Ticker": "NVDA", "Option type": "Put", "Position": "Long", "Strike": 160.0, "Theoretical premium": 6.0, "Quantity": 1.0},
            {"Instrument": "NVDA short put 140", "Ticker": "NVDA", "Option type": "Put", "Position": "Short", "Strike": 140.0, "Theoretical premium": 3.0, "Quantity": 1.0},
            {"Instrument": "NVDA long call 200", "Ticker": "NVDA", "Option type": "Call", "Position": "Long", "Strike": 200.0, "Theoretical premium": 7.0, "Quantity": 1.0},
            {"Instrument": "NVDA short call 230", "Ticker": "NVDA", "Option type": "Call", "Position": "Short", "Strike": 230.0, "Theoretical premium": 3.5, "Quantity": 1.0},
        ]
    )
    return PortfolioSpec(
        option_legs=legs, current_market_caps=current_caps, spot_prices=spots,
        selected_ticker="NVDA", polymarket_side="NO", polymarket_entry_price=0.17,
        polymarket_quantity=100.0, contract_multiplier=100.0, include_option_premiums=True,
    )


def test_constant_correlation_is_symmetric_unit_diagonal():
    corr = constant_correlation(["A", "B", "C"], 0.3)
    values = corr.to_numpy()
    assert np.allclose(np.diag(values), 1.0)
    assert np.allclose(values, values.T)
    assert corr.loc["A", "B"] == pytest.approx(0.3)


def test_multi_seed_dispersion_shape_and_summary():
    company, corr = _universe()
    seeds = [1, 2, 3]
    per_seed = multi_seed_dispersion(company, corr, _portfolio(), days_to_target=90, simulations=8000, seeds=seeds)
    assert len(per_seed) == len(seeds)
    for column in ["P(#1) selected", "Edge selected", "Expected shortfall 5%", "Worst payoff", "Probability of loss"]:
        assert column in per_seed.columns
    assert per_seed["P(#1) selected"].between(0.0, 1.0).all()

    summary = dispersion_summary(per_seed)
    assert (summary["MC error (std)"] >= 0.0).all()
    assert set(summary["Metric"]) >= {"P(#1) selected", "Worst payoff"}


def test_bounded_spread_worst_case_is_stable_across_seeds():
    # Put+call debit/credit spreads bound the maximum loss by construction, so
    # the worst-case payoff must not wander across seeds (near-zero MC error).
    company, corr = _universe()
    per_seed = multi_seed_dispersion(company, corr, _portfolio(), days_to_target=90, simulations=8000, seeds=[1, 2, 3, 4])
    assert per_seed["Worst payoff"].std(ddof=0) == pytest.approx(0.0, abs=1e-6)


def test_multi_seed_without_portfolio_reports_probabilities_only():
    company, corr = _universe()
    per_seed = multi_seed_dispersion(company, corr, None, days_to_target=90, simulations=5000, seeds=[1, 2])
    assert list(per_seed.columns) == ["Seed", "P(#1) selected", "Edge selected"]


def test_copula_stress_matches_baseline_and_reports_change():
    company, corr = _universe()
    port = _portfolio()
    seeds = [1, 2, 3]
    result = copula_tail_stress(company, corr, port, days_to_target=90, simulations=8000, seeds=seeds)
    assert result.baseline_model == "Normal shocks"
    assert result.stress_model == "Student-t copula df=5"
    # The Gaussian branch must equal a direct normal multi-seed run (same seeds).
    direct = multi_seed_dispersion(company, corr, port, days_to_target=90, simulations=8000, seeds=seeds, shock_model="Normal shocks")
    row = result.comparison.loc[result.comparison["Metric"] == "P(#1) selected"].iloc[0]
    assert row["Normal shocks"] == pytest.approx(float(direct["P(#1) selected"].mean()))
    assert {"Change", "Change %"}.issubset(result.comparison.columns)


def test_iv_scaling_lowers_leader_dominance():
    # Low volatility => the largest-cap name wins almost surely; high volatility
    # spreads probability out. So P(#1) at the smallest IV scale must exceed the
    # largest scale (randomness dilutes the structural gap).
    company, corr = _universe()
    scan = iv_scaling_scan(company, corr, selected_ticker="NVDA", days_to_target=90, simulations=12000, seed=7, factors=[0.5, 1.0, 2.0])
    low = scan.loc[scan["IV scale"] == 0.5, "P(#1) selected"].iloc[0]
    high = scan.loc[scan["IV scale"] == 2.0, "P(#1) selected"].iloc[0]
    assert low > high
    # per-scenario probabilities across all tickers sum to one
    ticker_cols = [c for c in scan.columns if c.startswith("P(#1) ") and c != "P(#1) selected"]
    assert np.allclose(scan[ticker_cols].sum(axis=1).to_numpy(), 1.0, atol=1e-6)


def test_gap_scaling_widens_leader_dominance():
    company, corr = _universe()
    scan = gap_scaling_scan(company, corr, selected_ticker="NVDA", days_to_target=90, simulations=12000, seed=7, factors=[0.5, 1.0, 2.0])
    narrow = scan.loc[scan["Gap scale"] == 0.5, "P(#1) selected"].iloc[0]
    wide = scan.loc[scan["Gap scale"] == 2.0, "P(#1) selected"].iloc[0]
    assert wide > narrow


def test_gap_vs_randomness_and_scan_sensitivity():
    company, corr = _universe()
    ivs = iv_scaling_scan(company, corr, selected_ticker="NVDA", days_to_target=90, simulations=8000, seed=7, factors=[0.5, 1.0, 2.0])
    gps = gap_scaling_scan(company, corr, selected_ticker="NVDA", days_to_target=90, simulations=8000, seed=7, factors=[0.5, 1.0, 2.0])
    sens = scan_sensitivity(ivs)
    assert sens["range"] == pytest.approx(sens["max"] - sens["min"])
    verdict = gap_vs_randomness(ivs, gps)
    assert len(verdict) == 2
    assert "Verdict" in verdict.columns


def test_model_robustness_grid_and_summary():
    company, corr = _universe()
    tickers = company["Ticker"].tolist()
    variants = {"Base": corr, "Independent": constant_correlation(tickers, 0.0), "High": constant_correlation(tickers, 0.8)}
    shock_models = ["Normal shocks", "Student-t copula df=5"]
    grid = model_robustness(company, variants, selected_ticker="NVDA", days_to_target=90, simulations=8000, seed=7, shock_models=shock_models)
    assert len(grid) == len(variants) * len(shock_models)
    summary = robustness_summary(grid)
    assert summary["P(#1) spread"] >= 0.0
    assert summary["Edge spread"] >= 0.0
    assert isinstance(bool(summary["Edge sign consistent"]), bool)


def _dispersion(edge_rel=0.004, worst_rel=0.10, ploss_mean=0.50):
    return pd.DataFrame(
        [
            {"Metric": "Edge selected", "Mean": -0.42, "MC error (std)": 0.0015, "Relative dispersion": edge_rel, "Min": -0.42, "Max": -0.41, "Seeds": 6},
            {"Metric": "Worst payoff", "Mean": -74.5, "MC error (std)": 7.3, "Relative dispersion": worst_rel, "Min": -83.0, "Max": -66.0, "Seeds": 6},
            {"Metric": "Probability of loss", "Mean": ploss_mean, "MC error (std)": 0.0016, "Relative dispersion": 0.003, "Min": 0.49, "Max": 0.50, "Seeds": 6},
        ]
    )


def _copula(edge_change_pct=-0.001, worst_change_pct=-2.75):
    from phase7 import CopulaStressResult
    comparison = pd.DataFrame(
        [
            {"Metric": "Edge selected", "Normal shocks": -0.4202, "Student-t copula df=5": -0.4205, "Change": -0.0003, "Change %": edge_change_pct},
            {"Metric": "Worst payoff", "Normal shocks": -74.5, "Student-t copula df=5": -279.8, "Change": -205.0, "Change %": worst_change_pct},
        ]
    )
    empty = pd.DataFrame()
    return CopulaStressResult(comparison=comparison, baseline_per_seed=empty, stress_per_seed=empty, baseline_model="Normal shocks", stress_model="Student-t copula df=5")


def _gap_verdict(iv_range=0.064, gap_range=0.056, verdict="randomness-dominated (IV lever)"):
    return pd.DataFrame(
        [
            {"Lever": "IV scaling", "P(#1) min": 0.451, "P(#1) max": 0.515, "P(#1) range": iv_range, "Verdict": verdict},
            {"Lever": "Gap scaling", "P(#1) min": 0.456, "P(#1) max": 0.512, "P(#1) range": gap_range, "Verdict": verdict},
        ]
    )


def test_assessment_robust_case_flags_watch_outs_but_confirms_edge():
    from phase7 import assessment
    robustness = pd.Series({"Edge sign consistent": True, "P(#1) spread": 0.015, "Edge spread": 0.016})
    out = assessment(_dispersion(), _copula(), _gap_verdict(), robustness, selected_ticker="NVDA")
    assert "robust" in out["headline"].lower()
    joined = " ".join(out["watch_outs"]).lower()
    assert "worst-case payoff is noisy" in joined       # Test 1 noisy worst
    assert "deep tail is not bounded" in joined          # Test 2 tail blowup
    assert "lightly hedged" in joined                    # near-unhedged portfolio
    assert "model grid" not in joined                    # sign consistent -> no flip flag
    assert len(out["findings"]) == 4


def test_assessment_flags_model_dependence_when_sign_flips():
    from phase7 import assessment
    robustness = pd.Series({"Edge sign consistent": False, "P(#1) spread": 0.2, "Edge spread": 0.5})
    out = assessment(_dispersion(edge_rel=0.2), _copula(edge_change_pct=-0.30), _gap_verdict(), robustness, selected_ticker="NVDA")
    assert "not fully robust" in out["headline"].lower()
    joined = " ".join(out["watch_outs"]).lower()
    assert "sign somewhere in the model grid" in joined
    edge_finding = out["findings"].loc[out["findings"]["Area"] == "2. Tail dependence", "Verdict"].iloc[0]
    assert "sensitive" in edge_finding.lower()


def test_assessment_strong_iv_flag_when_iv_range_large():
    from phase7 import assessment
    robustness = pd.Series({"Edge sign consistent": True})
    out = assessment(_dispersion(worst_rel=0.01, ploss_mean=0.18), _copula(worst_change_pct=-0.05), _gap_verdict(iv_range=0.40), robustness, selected_ticker="NVDA")
    joined = " ".join(out["watch_outs"]).lower()
    assert "strongly iv-driven" in joined
    assert "lightly hedged" not in joined                # P(loss) 0.18 -> well hedged
    assert "deep tail" not in joined                     # worst change small


def _phase6_payload():
    mapped_legs = pd.DataFrame(
        [
            {"Instrument": "Short NVDA Put 140.00", "Ticker": "NVDA", "Option type": "Put", "Position": "Short", "Quantity": 2.0, "Strike": 140.0, "Spot": 175.0, "Theoretical premium": 3.0, "Execution premium": 3.0},
            {"Instrument": "Long NVDA Call 190.00", "Ticker": "NVDA", "Option type": "Call", "Position": "Long", "Quantity": 3.0, "Strike": 190.0, "Spot": 175.0, "Theoretical premium": 7.0, "Execution premium": 7.0},
        ]
    )
    return {
        "mapped_legs": mapped_legs,
        "polymarket": {"selected_ticker": "NVDA", "side": "NO", "entry": 0.17, "shares": 100.0},
        "contract_multiplier": 100.0,
        "spots": pd.Series({"NVDA": 175.0}),
    }


def _phase4_payload():
    return {
        "active_option_legs": pd.DataFrame(
            [{"Instrument": "L", "Ticker": "NVDA", "Option type": "Call", "Position": "Long", "Quantity": 1.0, "Strike": 200.0, "Spot": 175.0, "Theoretical premium": 5.0}]
        ),
        "selected_ticker": "NVDA",
        "polymarket_side": "YES",
        "polymarket_entry_price": 0.83,
        "polymarket_quantity": 100.0,
    }


def test_load_saved_portfolio_prefers_phase6():
    from phase7 import load_saved_portfolio
    caps = pd.Series({"NVDA": 4_300e9, "AAPL": 3_100e9, "GOOGL": 2_100e9})
    spec, label = load_saved_portfolio(caps, "NVDA", phase6=_phase6_payload(), phase4=_phase4_payload())
    assert label == "Phase 6 real execution"
    assert spec.polymarket_side == "NO"
    assert len(spec.option_legs) == 2
    assert spec.spot_prices["NVDA"] == pytest.approx(175.0)


def test_load_saved_portfolio_falls_back_to_phase4_then_none():
    from phase7 import load_saved_portfolio
    caps = pd.Series({"NVDA": 4_300e9})
    spec, label = load_saved_portfolio(caps, "NVDA", phase6=None, phase4=_phase4_payload())
    assert label == "Phase 4 theory"
    assert spec.polymarket_side == "YES"
    spec_none, label_none = load_saved_portfolio(caps, "NVDA", phase6=None, phase4=None)
    assert spec_none is None and label_none == "none"


def test_phase6_portfolio_scenarios_run():
    from phase7 import phase6_portfolio, portfolio_scenarios
    from model import default_correlation_matrix, run_probability_engine
    company = pd.DataFrame(
        [
            {"Ticker": "NVDA", "Current market cap": 4_300e9, "Implied volatility": 0.42, "Polymarket YES price": 0.83},
            {"Ticker": "AAPL", "Current market cap": 3_100e9, "Implied volatility": 0.24, "Polymarket YES price": 0.123},
            {"Ticker": "GOOGL", "Current market cap": 2_100e9, "Implied volatility": 0.28, "Polymarket YES price": 0.046},
        ]
    )
    corr = default_correlation_matrix(company["Ticker"].tolist())
    result = run_probability_engine(company, corr, days_to_target=90, simulations=6000, seed=3)
    caps = company.set_index("Ticker")["Current market cap"].astype(float)
    spec = phase6_portfolio(_phase6_payload(), caps)
    scenario = portfolio_scenarios(result, spec)
    assert "Total payoff" in scenario.columns
    assert len(scenario) == 6000
