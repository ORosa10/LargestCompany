import json
from pathlib import Path

import daily_report


def _inputs():
    return {
        "target_date": "2026-07-31",
        "as_of": "2026-07-22",
        "traded_ticker": "NVDA",
        "side": "NO",
        "entry": 0.33,
        "shares": 100.0,
        "simulations": 8000,
        "seed": 42,
        "polymarket_yes": {"NVDA": 0.68, "AAPL": 0.309, "GOOGL": 0.017},
        "manual_market_caps": {},
        "manual_spots": {},
    }


_CAPS = {"NVDA": 4_300e9, "AAPL": 3_100e9, "GOOGL": 2_100e9}
_SPOTS = {"NVDA": 194.42, "AAPL": 308.46, "GOOGL": 359.08}


def test_daily_inputs_file_is_valid_json():
    data = json.loads((Path(daily_report.__file__).resolve().parent / "daily_inputs.json").read_text())
    assert {"target_date", "polymarket_yes"}.issubset(data)


def test_run_produces_report_with_all_sections(monkeypatch):
    def fake_fetch(tickers, manual_caps, manual_spots):
        return dict(_CAPS), dict(_SPOTS), "test", []
    monkeypatch.setattr(daily_report, "fetch_market_data", fake_fetch)
    report = daily_report.run(_inputs())
    for section in ["## Verdict:", "## Probability & edge", "## Money view", "## Structure selection", "## Sensitivity", "## Watch-outs"]:
        assert section in report
    assert any(v in report for v in ["FAVORABLE", "UNFAVORABLE", "BREAKEVEN"])


def test_missing_data_produces_failure_notice_no_numbers(monkeypatch):
    def fake_fetch(tickers, manual_caps, manual_spots):
        # Yahoo failed, no manual values -> everything missing
        return {}, {}, "caps fetch failed; spots fetch failed", [f"{t} market cap" for t in tickers] + [f"{t} spot" for t in tickers]
    monkeypatch.setattr(daily_report, "fetch_market_data", fake_fetch)
    report = daily_report.run(_inputs())
    assert "Data unavailable" in report
    assert "provide the missing" in report
    # no verdict / numbers fabricated
    assert "## Verdict:" not in report


def test_build_legs_uses_contract_sizing():
    legs = daily_report.build_legs("NVDA", 194.42, 0.39, 10 / 365.0, 0.04, put_weight=2, call_weight=3)
    assert len(legs) == 4
    assert (legs["Quantity"] < 1.0).all()
    assert set(legs["Position"]) == {"Long", "Short"}


def test_weight_variants_are_the_four_requested():
    assert set(daily_report.WEIGHT_VARIANTS) == {(1, 3), (2, 3), (1, 4), (2, 4)}


def test_report_includes_structure_selection(monkeypatch):
    def fake_fetch(tickers, manual_caps, manual_spots):
        return dict(_CAPS), dict(_SPOTS), "test", []
    monkeypatch.setattr(daily_report, "fetch_market_data", fake_fetch)
    report = daily_report.run(_inputs())
    for label in ["1/1/3/3", "2/2/3/3", "1/1/4/4", "2/2/4/4"]:
        assert label in report
