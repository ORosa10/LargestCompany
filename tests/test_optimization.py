import numpy as np
import pandas as pd
import pytest

from optimization import (
    build_candidate_option_universe,
    long_option_payoff_matrix,
    optimize_option_portfolio,
    payoff_metrics,
    selected_quantities_to_legs,
)


def test_candidate_option_universe_builds_calls_and_puts():
    candidates = build_candidate_option_universe(
        ticker="AAA",
        spot=100.0,
        volatility=0.30,
        time_to_expiry=1.0,
        risk_free_rate=0.04,
        strike_multipliers=[0.8, 1.0, 1.2],
    )

    assert len(candidates) == 6
    assert set(candidates["Option type"]) == {"Call", "Put"}
    assert candidates["Theoretical premium"].gt(0).all()
    assert candidates["Strike"].tolist() == pytest.approx([80.0, 80.0, 100.0, 100.0, 120.0, 120.0])


def test_long_option_payoff_matrix_includes_intrinsic_and_premium():
    candidates = pd.DataFrame(
        [
            {"Option type": "Call", "Strike": 100.0, "Theoretical premium": 5.0},
            {"Option type": "Put", "Strike": 100.0, "Theoretical premium": 4.0},
        ]
    )

    matrix = long_option_payoff_matrix([80.0, 120.0], candidates, contract_multiplier=1.0)

    assert matrix[:, 0].tolist() == pytest.approx([-5.0, 15.0])
    assert matrix[:, 1].tolist() == pytest.approx([16.0, -4.0])


def test_optimizer_can_reduce_sd_with_a_negatively_correlated_leg():
    base = np.array([-1.0, 1.0, -1.0, 1.0])
    option_matrix = np.array([[1.0], [-1.0], [1.0], [-1.0]])
    candidates = pd.DataFrame(
        [
            {
                "Instrument": "AAA Put",
                "Ticker": "AAA",
                "Option type": "Put",
                "Position": "Long",
                "Strike": 90.0,
                "Strike / spot": 0.9,
                "Spot": 100.0,
                "Model IV": 0.3,
                "Risk-free rate": 0.04,
                "Time to expiry": 1.0,
                "Theoretical premium": 2.0,
                "Quantity": 0.0,
            }
        ]
    )

    result = optimize_option_portfolio(
        base,
        option_matrix,
        candidates,
        quantity_min=0.0,
        quantity_max=1.0,
        quantity_step=1.0,
        max_legs=1,
        max_total_absolute_quantity=1.0,
        objective="Minimum SD with baseline EV floor",
        optimization_scenarios=4,
        seed=1,
    )

    assert result.quantities.tolist() == pytest.approx([1.0])
    assert result.optimized_metrics["Payoff standard deviation"] == pytest.approx(0.0)
    assert result.optimized_metrics["Expected payoff"] == pytest.approx(0.0)


def test_signed_quantities_are_converted_to_long_and_short_legs():
    candidates = pd.DataFrame(
        [
            {"Instrument": "Call", "Ticker": "AAA", "Option type": "Call", "Position": "Long", "Strike": 110.0},
            {"Instrument": "Put", "Ticker": "AAA", "Option type": "Put", "Position": "Long", "Strike": 90.0},
        ]
    )

    selected = selected_quantities_to_legs(candidates, np.array([-0.2, 0.1]))

    assert selected["Position"].tolist() == ["Short", "Long"]
    assert selected["Quantity"].tolist() == pytest.approx([0.2, 0.1])


def test_payoff_metrics_reports_probability_weighted_distribution():
    metrics = payoff_metrics(np.array([-10.0, 0.0, 10.0, 20.0]))

    assert metrics["Expected payoff"] == pytest.approx(5.0)
    assert metrics["Payoff standard deviation"] == pytest.approx(np.std([-10.0, 0.0, 10.0, 20.0]))
    assert metrics["Probability of loss"] == pytest.approx(0.25)
