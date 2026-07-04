import numpy as np
import pandas as pd

from iv_surface_model import (
    apply_surface_atm_ivs,
    default_surface_nodes,
    normal_cdf_approx,
    run_surface_probability_engine,
    sample_surface_marginal,
)


def test_normal_cdf_approx_is_symmetric_and_centered():
    values = normal_cdf_approx(np.array([-2.0, 0.0, 2.0]))
    assert np.isclose(values[1], 0.5, atol=1e-7)
    assert np.isclose(values[0] + values[2], 1.0, atol=1e-7)
    assert np.all(np.diff(values) > 0)


def test_surface_sample_reanchors_mean_to_forward():
    nodes = default_surface_nodes()
    nvda = nodes[nodes["Ticker"] == "NVDA"]
    uniforms = np.linspace(0.0001, 0.9999, 20_000)
    samples, diagnostics = sample_surface_marginal(
        uniforms,
        nvda,
        forward_ratio=1.003,
        horizon_years=27 / 365,
        risk_free_rate=0.04,
    )

    assert np.all(samples > 0)
    assert np.isclose(samples.mean(), 1.003, atol=1e-10)
    assert diagnostics["atm_iv"] > 0


def test_surface_engine_returns_one_winner_per_path():
    inputs = pd.DataFrame(
        {
            "Ticker": ["NVDA", "AAPL", "GOOGL", "MSFT"],
            "Current market cap": [4.6e12, 3.8e12, 3.7e12, 3.6e12],
            "Implied volatility": [0.42, 0.24, 0.28, 0.25],
            "Polymarket YES price": [0.83, 0.123, 0.046, 0.001],
            "Forward / spot": [1.002, 1.001, 1.002, 1.002],
        }
    )
    inputs = apply_surface_atm_ivs(inputs)
    correlation = pd.DataFrame(
        np.array(
            [
                [1.0, 0.45, 0.50, 0.48],
                [0.45, 1.0, 0.52, 0.58],
                [0.50, 0.52, 1.0, 0.60],
                [0.48, 0.58, 0.60, 1.0],
            ]
        ),
        index=inputs["Ticker"],
        columns=inputs["Ticker"],
    )

    result, diagnostics = run_surface_probability_engine(
        inputs,
        correlation,
        days_to_target=27,
        simulations=10_000,
        seed=42,
    )

    assert np.isclose(result.results["Model probability"].sum(), 1.0)
    assert (result.ranks.eq(1).sum(axis=1) == 1).all()
    assert set(diagnostics["Marginal model"]) == {
        "IV surface risk-neutral CDF",
        "ATM lognormal fallback",
    }
    assert diagnostics.set_index("Ticker").loc["NVDA", "ATM IV"] > 0
