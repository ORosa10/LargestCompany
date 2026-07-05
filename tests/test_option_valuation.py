import numpy as np
import pandas as pd
import pytest

from option_valuation import (
    attach_market_consistent_premiums,
    black_scholes_price_with_carry,
    implied_dividend_yield,
    surface_iv_for_strike,
)


def sample_structure() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "Instrument": "Short NVDA Call",
                "Ticker": "NVDA",
                "Option type": "Call",
                "Position": "Short",
                "Strike": 210.0,
                "Boundary used": "80% win boundary",
                "Boundary market cap": 5.0e12,
                "Boundary / current cap": 1.08,
                "Spot": 194.42,
                "Purpose": "test",
            },
            {
                "Instrument": "Long NVDA Put",
                "Ticker": "NVDA",
                "Option type": "Put",
                "Position": "Long",
                "Strike": 170.0,
                "Boundary used": "80% loss boundary",
                "Boundary market cap": 4.2e12,
                "Boundary / current cap": 0.91,
                "Spot": 194.42,
                "Purpose": "test",
            },
        ]
    )


def test_surface_iv_is_strike_specific_for_nvda():
    call_iv = surface_iv_for_strike("NVDA", 210.0, 194.42)
    put_iv = surface_iv_for_strike("NVDA", 170.0, 194.42)

    assert call_iv == pytest.approx(0.37)
    assert put_iv == pytest.approx(0.46)
    assert call_iv != put_iv


def test_surface_iv_uses_current_moneyness_not_stale_observed_spot():
    iv_at_current_atm = surface_iv_for_strike("NVDA", 200.0, 200.0)

    assert iv_at_current_atm == pytest.approx(0.39, abs=0.01)


def test_missing_surface_ticker_returns_none():
    assert surface_iv_for_strike("MSFT", 100.0, 100.0) is None


def test_implied_dividend_yield_round_trips_forward_ratio():
    years = 0.5
    rate = 0.04
    expected_q = 0.01
    forward_ratio = np.exp((rate - expected_q) * years)

    assert implied_dividend_yield(forward_ratio, years, rate) == pytest.approx(expected_q)


def test_carry_pricing_satisfies_put_call_parity():
    spot = 100.0
    strike = 105.0
    years = 0.5
    rate = 0.04
    q = 0.01
    volatility = 0.30
    call = black_scholes_price_with_carry(
        spot=spot,
        strike=strike,
        time_to_expiry=years,
        volatility=volatility,
        risk_free_rate=rate,
        dividend_yield=q,
        option_type="Call",
    )
    put = black_scholes_price_with_carry(
        spot=spot,
        strike=strike,
        time_to_expiry=years,
        volatility=volatility,
        risk_free_rate=rate,
        dividend_yield=q,
        option_type="Put",
    )

    parity = spot * np.exp(-q * years) - strike * np.exp(-rate * years)
    assert call - put == pytest.approx(parity)


def test_attach_market_consistent_premiums_uses_iv_per_leg():
    valued = attach_market_consistent_premiums(
        sample_structure(),
        pd.Series({"NVDA": 0.39}),
        forward_ratios=pd.Series({"NVDA": 1.002}),
        time_to_expiry=26 / 365,
        risk_free_rate=0.04,
        use_surface=True,
    )

    assert valued["Model IV"].tolist() == pytest.approx([0.37, 0.46])
    assert valued["IV source"].str.contains("Calibrated smile").all()
    assert valued["Theoretical premium"].gt(0.0).all()
    assert valued["Premium direction"].tolist() == ["Credit", "Debit"]


def test_attach_market_consistent_premiums_falls_back_to_phase1_atm():
    valued = attach_market_consistent_premiums(
        sample_structure().iloc[[0]],
        pd.Series({"NVDA": 0.41}),
        forward_ratios=None,
        time_to_expiry=26 / 365,
        risk_free_rate=0.04,
        use_surface=False,
    )

    assert valued["Model IV"].iloc[0] == pytest.approx(0.41)
    assert valued["IV source"].iloc[0] == "Phase 1 ATM fallback"
    assert valued["Implied dividend yield"].iloc[0] == pytest.approx(0.0)
