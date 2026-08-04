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


def test_august_market_registered_and_distinct():
    import pandas as pd
    from iv_surface_model import MARKETS, market_for, default_surface_nodes, apply_surface_atm_ivs
    assert "2026-08-31" in MARKETS
    aug = market_for("2026-08-31")
    assert aug["option_expiry"] == "2026-08-28"  # Friday, not the Monday month-end
    nodes = default_surface_nodes("2026-08-31")
    assert set(nodes["Ticker"]) == {"AAPL", "NVDA", "GOOGL"}
    assert (nodes["Expiry"] == "2026-08-28").all()
    # ATM IVs applied per the August market
    inp = pd.DataFrame([{"Ticker": t, "Current market cap": 1e12, "Implied volatility": 0.30, "Polymarket YES price": 0.3}
                        for t in ["AAPL", "NVDA", "GOOGL"]])
    out = apply_surface_atm_ivs(inp, resolution="2026-08-31").set_index("Ticker")["Implied volatility"]
    assert abs(out["NVDA"] - 0.47) < 1e-9 and abs(out["AAPL"] - 0.29) < 1e-9


def test_july_still_default_and_unchanged():
    from iv_surface_model import default_surface_nodes, SURFACE_EXPIRY
    assert SURFACE_EXPIRY == "2026-07-31"
    nodes = default_surface_nodes()  # no arg -> July default
    assert (nodes["Expiry"] == "2026-07-31").all()


def test_tail_df_preserves_margins_and_shifts_probability():
    """Student-t copula (tail_df) keeps unit-sum surface marginals but moves P(#1)
    versus the Gaussian copula (tail dependence changes the ranking odds)."""
    import pandas as pd
    from iv_surface_model import (
        apply_surface_atm_ivs,
        default_surface_nodes,
        run_surface_probability_engine,
    )
    from model import default_correlation_matrix

    tickers = ["NVDA", "AAPL", "GOOGL"]
    caps = {"NVDA": 4.70e12, "AAPL": 4.89e12, "GOOGL": 4.05e12}
    uni = pd.DataFrame([
        {"Ticker": t, "Current market cap": caps[t], "Implied volatility": 0.30,
         "Polymarket YES price": 0.33} for t in tickers
    ])
    surf = apply_surface_atm_ivs(uni.copy(), resolution="2026-08-31")
    corr = default_correlation_matrix(tickers)
    nodes = default_surface_nodes(resolution="2026-08-31")
    kw = dict(days_to_target=24, simulations=20000, seed=42, surface_nodes=nodes)

    g, _ = run_surface_probability_engine(surf, corr, tail_df=None, **kw)
    t, _ = run_surface_probability_engine(surf, corr, tail_df=5, **kw)
    pg = g.results.set_index("Ticker")["Model probability"]
    pt = t.results.set_index("Ticker")["Model probability"]

    assert abs(float(pg.sum()) - 1.0) < 1e-6
    assert abs(float(pt.sum()) - 1.0) < 1e-6
    # tail dependence should move at least one probability meaningfully
    assert max(abs(float(pt[k]) - float(pg[k])) for k in tickers) > 0.002
