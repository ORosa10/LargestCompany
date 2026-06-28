import numpy as np
import pandas as pd
import pytest

from payoff_surface import (
    calculate_scenario_payoffs,
    payoff_summary,
    payoff_surface_bins,
    polymarket_payoff,
    selected_payoff_profile_bins,
    winner_from_ranks,
)


def test_polymarket_yes_and_no_payoffs_are_net_of_entry_price():
    winners = pd.Series(["AAA", "BBB", "AAA"])

    yes = polymarket_payoff(winners, selected_ticker="AAA", side="YES", entry_price=0.40, quantity=10.0)
    no = polymarket_payoff(winners, selected_ticker="AAA", side="NO", entry_price=0.60, quantity=10.0)

    assert yes.tolist() == pytest.approx([6.0, -4.0, 6.0])
    assert no.tolist() == pytest.approx([-6.0, 4.0, -6.0])


def test_winner_from_ranks_returns_rank_one_ticker():
    ranks = pd.DataFrame({"AAA": [1, 2], "BBB": [2, 1]})

    assert winner_from_ranks(ranks).tolist() == ["AAA", "BBB"]


def test_calculate_scenario_payoffs_combines_polymarket_and_options():
    terminal_caps = pd.DataFrame({"AAA": [120.0, 80.0], "BBB": [90.0, 130.0]})
    ranks = pd.DataFrame({"AAA": [1, 2], "BBB": [2, 1]})
    current_caps = pd.Series({"AAA": 100.0, "BBB": 100.0})
    spots = pd.Series({"AAA": 50.0, "BBB": 40.0})
    option_legs = pd.DataFrame(
        [
            {
                "Instrument": "Long AAA Put",
                "Ticker": "AAA",
                "Option type": "Put",
                "Position": "Long",
                "Strike": 45.0,
                "Spot": 50.0,
                "Theoretical premium": 1.0,
                "Quantity": 1.0,
            }
        ]
    )

    scenario = calculate_scenario_payoffs(
        terminal_caps,
        ranks,
        current_caps,
        spots,
        option_legs,
        selected_ticker="AAA",
        polymarket_side="YES",
        polymarket_entry_price=0.4,
        polymarket_quantity=10.0,
        contract_multiplier=1.0,
        include_option_premiums=True,
    )

    assert scenario["Polymarket payoff"].tolist() == pytest.approx([6.0, -4.0])
    assert scenario["Option payoff"].tolist() == pytest.approx([-1.0, 4.0])
    assert scenario["Total payoff"].tolist() == pytest.approx([5.0, 0.0])


def test_calculate_scenario_payoffs_rejects_quantity_on_missing_boundary_strike():
    terminal_caps = pd.DataFrame({"AAA": [120.0, 80.0], "BBB": [90.0, 130.0]})
    ranks = pd.DataFrame({"AAA": [1, 2], "BBB": [2, 1]})
    current_caps = pd.Series({"AAA": 100.0, "BBB": 100.0})
    spots = pd.Series({"AAA": 50.0, "BBB": 40.0})
    option_legs = pd.DataFrame(
        [
            {
                "Instrument": "Long AAA Put",
                "Ticker": "AAA",
                "Option type": "Put",
                "Position": "Long",
                "Strike": np.nan,
                "Spot": 50.0,
                "Theoretical premium": 1.0,
                "Quantity": 1.0,
            }
        ]
    )

    with pytest.raises(ValueError, match="no valid strike"):
        calculate_scenario_payoffs(
            terminal_caps,
            ranks,
            current_caps,
            spots,
            option_legs,
            selected_ticker="AAA",
            polymarket_side="YES",
            polymarket_entry_price=0.4,
            polymarket_quantity=10.0,
            contract_multiplier=1.0,
            include_option_premiums=True,
        )


def test_payoff_summary_reports_expected_payoff_and_loss_probability():
    scenario = pd.DataFrame({"Total payoff": [10.0, -5.0, 0.0, 5.0]})

    summary = payoff_summary(scenario)

    assert summary["Expected payoff"] == pytest.approx(2.5)
    assert summary["Probability of loss"] == pytest.approx(0.25)
    assert summary["Worst payoff"] == pytest.approx(-5.0)


def test_selected_payoff_profile_bins_returns_expected_payoff_bridge():
    terminal_caps = pd.DataFrame({"AAA": [100.0, 110.0, 120.0, 130.0], "BBB": [90.0, 95.0, 105.0, 115.0]})
    current_caps = pd.Series({"AAA": 100.0, "BBB": 100.0})
    scenario = pd.DataFrame(
        {
            "Winner": ["AAA", "BBB", "AAA", "AAA"],
            "Selected terminal stock price": [50.0, 55.0, 60.0, 65.0],
            "Polymarket payoff": [1.0, -1.0, 1.0, 1.0],
            "Option payoff": [0.0, 1.0, 2.0, 3.0],
            "Total payoff": [1.0, 0.0, 3.0, 4.0],
        }
    )

    profile = selected_payoff_profile_bins(
        scenario,
        terminal_caps,
        current_caps,
        selected_ticker="AAA",
        bins=2,
    )

    assert not profile.empty
    assert profile["scenario_probability"].sum() == pytest.approx(1.0)
    assert profile["weighted_payoff_contribution"].sum() == pytest.approx(scenario["Total payoff"].mean())


def test_payoff_surface_bins_returns_weighted_contributions():
    terminal_caps = pd.DataFrame({"AAA": [100.0, 110.0, 120.0, 130.0], "BBB": [90.0, 95.0, 105.0, 115.0]})
    current_caps = pd.Series({"AAA": 100.0, "BBB": 100.0})
    scenario = pd.DataFrame({"Total payoff": [1.0, 2.0, 3.0, 4.0]})

    surface = payoff_surface_bins(
        scenario,
        terminal_caps,
        current_caps,
        selected_ticker="AAA",
        competitor_ticker="BBB",
        x_bins=2,
        y_bins=2,
    )

    assert not surface.empty
    assert surface["scenario_probability"].sum() == pytest.approx(1.0)
    assert surface["weighted_payoff_contribution"].sum() == pytest.approx(scenario["Total payoff"].mean())
