import numpy as np
import pandas as pd
import pytest

from model import (
    clean_correlation_matrix,
    default_company_inputs,
    default_correlation_matrix,
    run_probability_engine,
    validate_company_inputs,
)


def test_probability_bookkeeping_sums_to_one():
    inputs = default_company_inputs().head(4)
    tickers = inputs["Ticker"].tolist()
    corr = default_correlation_matrix(tickers)

    result = run_probability_engine(
        inputs,
        corr,
        days_to_target=90,
        simulations=5_000,
        seed=123,
    )

    assert result.results["Model probability"].sum() == pytest.approx(1.0)
    assert (result.results["Probability Top 2"] >= result.results["Model probability"]).all()
    assert (result.results["Probability Top 3"] >= result.results["Probability Top 2"]).all()

    rank_sums = result.rank_distribution.groupby("Ticker")["Probability"].sum()
    assert (rank_sums == pytest.approx(1.0)).all()


def test_two_company_top_two_is_certain():
    inputs = pd.DataFrame(
        [
            {"Ticker": "AAA", "Current market cap": 100.0, "Implied volatility": 0.25, "Polymarket YES price": 0.50},
            {"Ticker": "BBB", "Current market cap": 95.0, "Implied volatility": 0.25, "Polymarket YES price": 0.50},
        ]
    )
    corr = pd.DataFrame(np.eye(2), index=["AAA", "BBB"], columns=["AAA", "BBB"])

    result = run_probability_engine(
        inputs,
        corr,
        days_to_target=30,
        simulations=2_000,
        seed=7,
    )

    assert result.results["Model probability"].sum() == pytest.approx(1.0)
    assert (result.results["Probability Top 2"] == pytest.approx(1.0)).all()


def test_input_validation_rejects_bad_rows():
    duplicate_inputs = pd.DataFrame(
        [
            {"Ticker": "AAA", "Current market cap": 100.0, "Implied volatility": 0.25, "Polymarket YES price": 0.50},
            {"Ticker": "AAA", "Current market cap": 95.0, "Implied volatility": 0.25, "Polymarket YES price": 0.50},
        ]
    )
    with pytest.raises(ValueError, match="Duplicate tickers"):
        validate_company_inputs(duplicate_inputs)

    bad_price_inputs = duplicate_inputs.copy()
    bad_price_inputs.loc[1, "Ticker"] = "BBB"
    bad_price_inputs.loc[0, "Polymarket YES price"] = 1.2
    with pytest.raises(ValueError, match="between 0 and 1"):
        validate_company_inputs(bad_price_inputs)


def test_correlation_cleaning_repairs_order_symmetry_and_diagonal():
    tickers = ["AAA", "BBB"]
    corr = pd.DataFrame(
        [[0.8, 0.20], [0.40, 1.2]],
        index=["BBB", "AAA"],
        columns=["BBB", "AAA"],
    )

    cleaned, warnings = clean_correlation_matrix(corr, tickers)

    assert list(cleaned.index) == tickers
    assert list(cleaned.columns) == tickers
    assert np.diag(cleaned.to_numpy()).tolist() == pytest.approx([1.0, 1.0])
    assert cleaned.loc["AAA", "BBB"] == pytest.approx(cleaned.loc["BBB", "AAA"])
    assert warnings
