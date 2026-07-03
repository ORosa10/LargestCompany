from datetime import date, timedelta
from types import SimpleNamespace

import numpy as np
import pandas as pd

from implied_forwards import _forward_for_expiry, apply_implied_forwards


class FakeTicker:
    def __init__(self, calls: pd.DataFrame, puts: pd.DataFrame):
        self._chain = SimpleNamespace(calls=calls, puts=puts)

    def option_chain(self, expiry: str):
        return self._chain


def test_forward_for_expiry_recovers_put_call_parity_forward():
    as_of = date(2026, 1, 1)
    expiry = as_of + timedelta(days=365)
    spot = 100.0
    rate = 0.04
    expected_forward = 104.0
    strikes = np.array([95.0, 100.0, 105.0])
    discounted_differences = np.exp(-rate) * (expected_forward - strikes)

    put_mids = np.array([4.0, 6.0, 9.0])
    call_mids = put_mids + discounted_differences
    calls = pd.DataFrame(
        {"strike": strikes, "bid": call_mids - 0.05, "ask": call_mids + 0.05}
    )
    puts = pd.DataFrame(
        {"strike": strikes, "bid": put_mids - 0.05, "ask": put_mids + 0.05}
    )

    result = _forward_for_expiry(
        FakeTicker(calls, puts),
        expiry.isoformat(),
        spot=spot,
        risk_free_rate=rate,
        as_of=as_of,
        strikes_each_side=3,
    )

    assert result is not None
    assert np.isclose(result["forward"], expected_forward)
    assert np.isclose(result["forward_to_spot"], 1.04)
    assert result["pair_count"] == 3


def test_apply_implied_forwards_preserves_inputs_and_maps_ratios():
    inputs = pd.DataFrame(
        {
            "Ticker": ["NVDA", "AAPL"],
            "Current market cap": [5.0e12, 4.0e12],
            "Implied volatility": [0.40, 0.25],
            "Polymarket YES price": [0.40, 0.20],
        }
    )
    forwards = pd.DataFrame(
        {"ticker": ["AAPL", "NVDA"], "forward_to_spot": [1.01, 1.03]}
    )

    result = apply_implied_forwards(inputs, forwards)

    assert result["Ticker"].tolist() == ["NVDA", "AAPL"]
    assert np.allclose(result["Forward / spot"], [1.03, 1.01])
    assert "Forward / spot" not in inputs.columns
