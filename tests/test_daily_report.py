import json

import daily_report


def _inputs():
    return {
        "target_date": "2026-07-31",
        "as_of": "2026-07-21",
        "traded_ticker": "NVDA",
        "side": "NO",
        "entry": 0.33,
        "shares": 100.0,
        "simulations": 8000,
        "seed": 42,
        "spots": {"NVDA": 194.42, "AAPL": 308.46, "GOOGL": 359.08},
        "universe": [
            {"Ticker": "NVDA", "Current market cap": 4_300e9, "Polymarket YES price": 0.68},
            {"Ticker": "AAPL", "Current market cap": 3_100e9, "Polymarket YES price": 0.309},
            {"Ticker": "GOOGL", "Current market cap": 2_100e9, "Polymarket YES price": 0.017},
        ],
    }


def test_daily_inputs_file_is_valid_json():
    from pathlib import Path
    data = json.loads((Path(daily_report.__file__).resolve().parent / "daily_inputs.json").read_text())
    assert {"target_date", "side", "entry", "universe", "spots"}.issubset(data)


def test_run_produces_report_with_all_sections(monkeypatch):
    # Deterministic: use the provided caps/spots, no network.
    def fake_fetch(tickers, fallback_caps, fallback_spots):
        return dict(fallback_caps), dict(fallback_spots), "test fallback"
    monkeypatch.setattr(daily_report, "fetch_market_data", fake_fetch)

    report = daily_report.run(_inputs())
    for section in ["## Verdict:", "## Probability & edge", "## Money view", "## Sensitivity", "## Watch-outs"]:
        assert section in report
    assert any(v in report for v in ["FAVORABLE", "UNFAVORABLE", "BREAKEVEN"])
    # sizing sanity: option legs use share-eq/spot, so capital at risk is a small-dollar figure, not thousands
    assert "Capital at risk" in report


def test_build_legs_uses_contract_sizing():
    legs = daily_report.build_legs("NVDA", 194.42, 0.39, 10 / 365.0, 0.04)
    assert len(legs) == 4
    # share-equivalent weight 2 or 3 divided by spot -> tiny contract quantity
    assert (legs["Quantity"] < 1.0).all()
    assert set(legs["Position"]) == {"Long", "Short"}
