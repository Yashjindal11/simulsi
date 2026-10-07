"""Built-in models against theory and limiting cases."""

from __future__ import annotations

import math

import numpy as np
import pytest
from scipy.optimize import brentq
from scipy.stats import norm

from simulsi.models import BUILTIN_MODELS
from simulsi.randomness.stream import derive_seed


def final_size(r0: float) -> float:
    """Kermack-McKendrick final size: z = 1 - exp(-R0 z) (0 when R0 <= 1)."""
    if r0 <= 1:
        return 0.0
    return float(brentq(lambda z: z - 1 + math.exp(-r0 * z), 1e-9, 1.0))


@pytest.mark.parametrize(("r0", "vaccination"), [(2.5, 0.0), (1.8, 0.0), (2.5, 0.3)])
def test_epidemic_matches_final_size_equation(r0: float, vaccination: float) -> None:
    m = BUILTIN_MODELS["epidemic"]
    params = {
        "r0": r0,
        "vaccination": vaccination,
        "lockdown_trigger": 2.0,
        "population": 20_000,
        "initial_infected": 20,
    }
    rates = [
        m.simulate(params, seed=derive_seed(7, "replication", i)).metrics["attack_rate"]
        for i in range(8)
    ]
    major = [a for a in rates if a > 0.05]
    assert len(major) >= 6
    susceptible = 1 - vaccination
    expected = susceptible * final_size(r0 * susceptible)
    assert np.mean(major) == pytest.approx(expected, abs=0.02)


def test_epidemic_below_threshold_dies_out() -> None:
    m = BUILTIN_MODELS["epidemic"]
    rates = [
        m.simulate({"r0": 0.8, "lockdown_trigger": 2.0}, seed=s).metrics["attack_rate"]
        for s in range(5)
    ]
    assert max(rates) < 0.02


def test_airline_without_disruptions_is_on_time() -> None:
    r = (
        BUILTIN_MODELS["airline"]
        .simulate(
            {"disruption_prob": 0.0, "crew_swap_prob": 0.0, "gates": 50, "runway_rate": 200.0},
            seed=1,
        )
        .metrics
    )
    assert r["otp"] > 0.9  # block-time noise alone makes a few flights late
    assert r["flights.cancelled"] == 0 and r["spare_swaps"] == 0


def test_supply_chain_without_noise_has_no_backorders() -> None:
    r = BUILTIN_MODELS["supply_chain"].simulate({"demand_cv": 0.0}, seed=1).metrics
    assert r["fill_rate"] == pytest.approx(1.0)
    assert r["backlog.retailer.max"] == 0
    assert "bullwhip.retailer" not in r  # zero demand variance: the ratio is undefined


def test_ride_hailing_huge_fleet_serves_everyone() -> None:
    r = BUILTIN_MODELS["ride_hailing"].simulate({"drivers": 1000}, seed=1).metrics
    assert r["service_level"] > 0.99 and r["pickup_minutes.mean"] < 3


def test_cloud_without_queueing_matches_service_time_tail() -> None:
    m = BUILTIN_MODELS["cloud_autoscaling"].with_options(duration=900.0, warmup=60.0)
    r = m.simulate({"min_servers": 40, "max_servers": 40}, seed=1).metrics
    assert r["error_rate"] == 0 and r["resource.server.wait.max"] == 0
    # latency = service time ~ LogNormal(mean 0.2, sd 0.2): P(latency <= 0.5)
    sigma = math.sqrt(math.log(2.0))
    mu = math.log(0.2) - sigma**2 / 2
    expected = float(norm.cdf((math.log(0.5) - mu) / sigma))
    assert r["slo_attainment"] == pytest.approx(expected, abs=0.01)


def test_turnaround_with_unlimited_crews_follows_the_task_network() -> None:
    m = BUILTIN_MODELS["airport_turnaround"]
    crews = {
        k: 100
        for k in ("cleaning_crews", "catering_trucks", "fuel_trucks", "baggage_teams", "tugs")
    }
    r = m.simulate({**crews, "late_arrival_prob": 0.0, "scheduled_turn": 80.0}, seed=1).metrics
    assert r["otp"] == pytest.approx(1.0)
    assert r["resource.fuel_truck.wait.mean"] == 0 and r["resource.baggage_team.wait.mean"] == 0


def test_disruption_recovery_without_storm_is_on_time() -> None:
    r = BUILTIN_MODELS["disruption_recovery"].simulate({"closure_minutes": 0.0}, seed=1).metrics
    assert r["otp"] > 0.95 and r["recovery_minutes"] == 0
