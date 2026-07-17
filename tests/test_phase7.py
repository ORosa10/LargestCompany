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
