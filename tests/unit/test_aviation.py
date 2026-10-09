from __future__ import annotations

import math

import pytest

from simulsi.aviation import (
    Cancel,
    Connection,
    DelayModel,
    Flight,
    OpsConfig,
    OpsState,
    Retime,
    Schedule,
    Swap,
    WeatherEvent,
    apply_actions,
    closure,
    eu261_compensation,
    format_time,
    network_model,
    parse_action,
    parse_time,
    simulate_day,
)
from simulsi.aviation.state import FlightStatus
from simulsi.errors import ConfigError

CALM = DelayModel(prob=0.0, block_cv=0.0)


def calm(**kw: object) -> OpsConfig:
    return OpsConfig(delays=CALM, **kw)  # type: ignore[arg-type]


def two_legs(gap: float = 60.0, crew: str = "") -> Schedule:
    return Schedule(
        [
            Flight("A1", "T1", "HUB", "AAA", 360, 420, crew=crew),
            Flight("A2", "T1", "AAA", "HUB", 420 + gap, 480 + gap, crew=crew),
        ]
    )


def test_time_parsing_and_formatting() -> None:
    assert parse_time("06:30") == 390
    assert parse_time("25:10") == 1510
    assert parse_time(95) == 95
    assert parse_time("2026-10-08T06:30", None) == 390
    assert format_time(390) == "06:30" and format_time(1510) == "25:10"
    with pytest.raises(ConfigError):
        parse_time("soon")


def test_schedule_from_records_aliases_and_overnight() -> None:
    s = Schedule.from_records(
        [
            {
                "Flight_No": "X1",
                "Registration": "R1",
                "From": "lhr",
                "To": "jfk",
                "Departure": "22:00",
                "Arrival": "01:00",
            },
        ]
    )
    f = s.by_id["X1"]
    assert (f.origin, f.dest, f.std, f.sta) == ("LHR", "JFK", 1320, 1500)
    with pytest.raises(ConfigError, match="missing columns"):
        Schedule.from_records([{"flight": "X"}])


def test_validate_catches_broken_rotations() -> None:
    s = Schedule(
        [Flight("A1", "T1", "HUB", "AAA", 360, 420), Flight("A2", "T1", "BBB", "HUB", 500, 560)],
        [Connection("A1", "ZZ", 5)],
    )
    issues = " ".join(s.validate())
    assert "arrives at AAA but A2 departs BBB" in issues and "unknown flight" in issues
    with pytest.raises(ConfigError):
        s.check()
    with pytest.raises(ConfigError, match="duplicate"):
        Schedule([Flight("A1", "T1", "HUB", "AAA", 360, 420)] * 2)


def test_synthetic_schedule_is_consistent() -> None:
    s = Schedule.synthetic(seed=3)
    assert s.validate() == []
    assert len(s.rotations) == 12 and len(s.connections) > 0
    # some crews change aircraft at the hub
    assert any(len({f.tail for f in legs}) > 1 for legs in s.pairings.values())
    assert Schedule.synthetic(banks=False).validate() == []


def test_calm_day_is_on_time() -> None:
    day = simulate_day(two_legs(), calm(min_turn=30), seed=1)
    assert day.metrics["otp"] == 1.0 and day.metrics["cancelled"] == 0
    for f in two_legs():
        assert day.flights[f.id].dep == pytest.approx(f.std)


def test_reactionary_delay_propagates_through_the_aircraft() -> None:
    state = OpsState(0.0, {"A1": FlightStatus(etd=460.0)})
    day = simulate_day(two_legs(gap=45), calm(min_turn=30), seed=1, state=state)
    second = day.flights["A2"]
    assert day.flights["A1"].dep == pytest.approx(460)
    assert second.dep_delay > 50 and second.reactionary_aircraft > 50
    assert second.primary == 0


def test_crew_connection_delays_another_aircraft() -> None:
    s = Schedule(
        [
            Flight("A1", "T1", "AAA", "HUB", 360, 420, crew="C1"),
            Flight("B1", "T2", "HUB", "BBB", 480, 540, crew="C1"),
        ]
    )
    state = OpsState(0.0, {"A1": FlightStatus(etd=450.0)})
    day = simulate_day(s, calm(min_crew_connect=30), seed=1, state=state)
    b1 = day.flights["B1"]
    assert b1.dep == pytest.approx(450 + 60 + 30)
    assert b1.reactionary_crew == pytest.approx(60) and b1.reactionary_aircraft == 0


def test_standby_crew_replaces_a_late_crew() -> None:
    s = Schedule(
        [
            Flight("A1", "T1", "AAA", "HUB", 360, 420, crew="C1"),
            Flight("B1", "T2", "HUB", "BBB", 480, 540, crew="C1"),
        ]
    )
    state = OpsState(0.0, {"A1": FlightStatus(etd=560.0)})
    day = simulate_day(s, calm(standby_crews={"HUB": 1}, swap_threshold=60), seed=1, state=state)
    assert day.flights["B1"].standby and day.flights["B1"].dep_delay < 30


def test_closure_holds_departures_until_it_ends() -> None:
    s = Schedule([Flight("A1", "T1", "HUB", "AAA", 600, 660)])
    day = simulate_day(s, calm(), seed=1, weather=[closure("HUB", "09:30", 90)])
    assert day.flights["A1"].dep == pytest.approx(660)
    assert day.flights["A1"].runway == pytest.approx(60)
    assert day.weather == ["closure HUB"]


def test_weather_probability_zero_has_no_effect() -> None:
    s = Schedule([Flight("A1", "T1", "HUB", "AAA", 600, 660)])
    wx = WeatherEvent("HUB", 570, 720, 0.0, probability=0.0)
    assert simulate_day(s, calm(), seed=1, weather=[wx]).flights["A1"].dep == pytest.approx(600)
    assert WeatherEvent.parse("hub 15:00-18:00 0.3 p=0.5") == WeatherEvent(
        "HUB", 900, 1080, 0.3, 0.5, "weather HUB"
    )


def test_curfew_cancels_the_rest_of_the_rotation() -> None:
    state = OpsState(0.0, {"A1": FlightStatus(etd=500.0)})
    cfg = calm(min_turn=30, curfew={"HUB": parse_time("09:00")}, cancel_threshold=1e9)
    day = simulate_day(two_legs(), cfg, seed=1, state=state)
    assert day.flights["A2"].cancelled and day.flights["A2"].reason == "curfew"
    assert day.metrics["cancelled.curfew"] == 1


def test_long_delay_cancels_a_round_trip() -> None:
    s = Schedule(
        [
            Flight("A1", "T1", "HUB", "AAA", 360, 420),
            Flight("A2", "T1", "AAA", "HUB", 470, 530),
            Flight("A3", "T1", "HUB", "BBB", 580, 640),
            Flight("A4", "T1", "BBB", "HUB", 690, 750),
        ]
    )
    state = OpsState(0.0, {"A1": FlightStatus(etd=700.0)})
    day = simulate_day(s, calm(min_turn=30, cancel_threshold=120), seed=1, state=state)
    # A1 itself is known late (etd), so it and its return are cancelled; the aircraft stays at HUB
    assert day.flights["A1"].cancelled and day.flights["A2"].cancelled
    assert not day.flights["A3"].cancelled and day.flights["A3"].dep == pytest.approx(580)


def test_spare_aircraft_takes_over_a_late_rotation() -> None:
    s = Schedule(
        [
            Flight("A1", "T1", "AAA", "HUB", 360, 420),
            Flight("A2", "T1", "HUB", "BBB", 480, 540),
        ]
    )
    state = OpsState(0.0, {"A1": FlightStatus(etd=500.0)})
    base = simulate_day(s, calm(), seed=1, state=state)
    spare = simulate_day(s, calm(spares={"HUB": 1}), seed=1, state=state)
    assert not base.flights["A2"].spare and base.flights["A2"].dep_delay > 100
    # the swap is decided when A1 finally leaves (08:20); the spare needs 20 minutes to get ready
    assert spare.flights["A2"].spare and spare.flights["A2"].dep == pytest.approx(520)
    assert spare.metrics["spare_swaps"] == 1


def test_duty_limit_without_standby_cancels() -> None:
    s = two_legs(crew="C1")
    state = OpsState(0.0, {"A1": FlightStatus(etd=500.0)})
    cfg = calm(min_turn=30, duty_limit=4 * 60, cancel_threshold=1e9)
    day = simulate_day(s, cfg, seed=1, state=state)
    assert day.flights["A2"].cancelled and day.flights["A2"].reason == "crew"


def test_gates_make_arrivals_wait() -> None:
    s = Schedule(
        [
            Flight("A1", "T1", "AAA", "HUB", 360, 420),
            Flight("B1", "T2", "BBB", "HUB", 360, 420),
            Flight("A2", "T1", "HUB", "AAA", 480, 540),
            Flight("B2", "T2", "HUB", "BBB", 600, 660),
        ]
    )
    day = simulate_day(s, calm(gates={"HUB": 1}, min_turn=30), seed=1)
    waits = sorted(day.flights[f].gate_wait for f in ("A1", "B1"))
    assert waits[0] == 0 and waits[1] > 0
    assert day.metrics["gate_waits"] == 1


def test_misconnections_and_eu261() -> None:
    s = Schedule(
        [
            Flight("IN", "T1", "AAA", "HUB", 360, 420, pax=100),
            Flight("OUT", "T2", "HUB", "BBB", 470, 530, pax=100),
        ],
        [Connection("IN", "OUT", 20)],
    )
    state = OpsState(0.0, {"IN": FlightStatus(etd=400.0)})
    day = simulate_day(s, calm(mct=35, eu261=True), seed=1, state=state)
    assert day.misconnected == {("IN", "OUT"): 20}
    assert day.metrics["misconnect_rate"] == 1.0
    assert eu261_compensation(s.by_id["IN"], 200, False) == 100 * 250
    assert eu261_compensation(s.by_id["IN"], 100, False) == 0
    assert eu261_compensation(s.by_id["IN"], math.nan, True) == 100 * 250
    assert day.metrics["eu261_compensation"] == 0


def test_live_state_replays_known_times() -> None:
    s = two_legs()
    state = OpsState(450.0, {"A1": FlightStatus(atd=370.0, ata=433.0)})
    day = simulate_day(s, calm(min_turn=30), seed=1, state=state)
    assert day.flights["A1"].dep == 370 and day.flights["A1"].arr == 433
    assert day.flights["A2"].dep >= 480
    with pytest.raises(ConfigError, match="after now"):
        OpsState(300.0, {"A1": FlightStatus(atd=370.0)}).check(s)
    with pytest.raises(ConfigError, match="unknown flights"):
        OpsState(300.0, {"ZZ": FlightStatus()}).check(s)


def test_actions_and_common_random_numbers() -> None:
    s = Schedule.synthetic(tails=4)
    plan, cancelled = apply_actions(s, [Cancel("F100", "F101"), Retime("F102", 10)])
    assert cancelled == {"F100", "F101"} and plan.by_id["F102"].std == s.by_id["F102"].std + 10
    a = simulate_day(s, seed=7)
    b = simulate_day(s, seed=7, actions=[Retime("F102", 0)])
    assert a.metrics == b.metrics
    with pytest.raises(ConfigError, match="is at"):
        apply_actions(s, [Swap("T01", "T02", 400)])
    assert parse_action("swap T1 T2 12:00") == Swap("T1", "T2", 720)
    assert parse_action("retime F1 30") == Retime("F1", 30.0)
    assert parse_action("cancel F1, F2") == Cancel("F1", "F2")
    with pytest.raises(ConfigError):
        parse_action("divert F1")


def test_ops_config_round_trip(tmp_path) -> None:  # type: ignore[no-untyped-def]
    cfg = OpsConfig(
        curfew={"HUB": 1410}, spares={"HUB": 2}, delays=DelayModel(table={"HUB": (0.3, 20.0)})
    )
    path = tmp_path / "ops.yaml"
    cfg.to_yaml(path)
    back = OpsConfig.from_yaml(path)
    assert back.curfew == {"HUB": 1410} and back.delays.table == {"HUB": (0.3, 20.0)}
    with pytest.raises(ConfigError, match="unknown ops setting"):
        OpsConfig.from_dict({"spare_planes": 3})


def test_delay_predictor_hook_drives_primary_delay() -> None:
    s = two_legs()
    cfg = OpsConfig(
        delays=DelayModel(block_cv=0.0, predictor=lambda f: 120.0 if f.id == "A1" else 0.0)
    )
    days = [simulate_day(s, cfg, seed=k) for k in range(30)]
    mean_primary = sum(d.flights["A1"].primary for d in days) / len(days)
    assert 70 < mean_primary < 180
    assert all(d.flights["A2"].primary == 0 for d in days)


def test_network_model_plugs_into_simulsi() -> None:
    model = network_model(Schedule.synthetic(tails=4), OpsConfig(runway_rate={"HUB": 30}))
    res = model.simulate({"spares": 1, "delay_scale": 0.5}, seed=1)
    assert 0 <= res.metrics["otp"] <= 1
    assert model.evaluate({"delay_scale": 0.0}, metric="otp", replications=2) >= model.evaluate(
        {"delay_scale": 2.0}, metric="otp", replications=2
    )


def test_rebooking_reaccommodates_on_later_flights() -> None:
    s = Schedule(
        [
            Flight("IN", "T1", "AAA", "HUB", 360, 420, pax=100),
            Flight("OUT1", "T2", "HUB", "BBB", 450, 510, pax=150, seats=180),
            Flight("OUT2", "T3", "HUB", "BBB", 600, 660, pax=150, seats=180),
        ],
        [Connection("IN", "OUT1", 40)],
    )
    state = OpsState(0.0, {"IN": FlightStatus(etd=420.0)})
    day = simulate_day(s, calm(mct=35), seed=1, state=state)
    assert day.misconnected == {("IN", "OUT1"): 40}
    # 30 free seats on OUT2: 30 rebooked 150 minutes late, 10 stranded overnight
    assert day.metrics["rebooked_pax"] == 30 and day.metrics["stranded_pax"] == 10
    assert day.metrics["disrupted_pax_delay_hours"] == pytest.approx((30 * 150 + 10 * 18 * 60) / 60)


def test_spare_can_be_ferried_from_another_airport() -> None:
    s = Schedule(
        [
            Flight("A1", "T1", "AAA", "HUB", 360, 600),
            Flight("A2", "T1", "HUB", "BBB", 660, 720),
        ]
    )
    state = OpsState(0.0, {"A1": FlightStatus(etd=510.0)})
    local_only = simulate_day(s, calm(spares={"CCC": 1}), seed=1, state=state)
    ferried = simulate_day(s, calm(spares={"CCC": 1}, spare_ferry_minutes=45), seed=1, state=state)
    assert not local_only.flights["A2"].spare
    # decided an hour before departure: 20 min to get ready plus a 45 min ferry
    assert ferried.flights["A2"].spare and ferried.flights["A2"].dep == pytest.approx(665)


def test_storm_cancellations_and_cancel_days() -> None:
    s = two_legs()
    storm = WeatherEvent("HUB", 300, 500, 1.0, 1.0, "storm", cancel=1.0)
    day = simulate_day(s, calm(), seed=1, weather=[storm])
    assert day.flights["A1"].cancelled and day.flights["A1"].reason == "weather"
    assert day.metrics["cancelled.weather"] == 2
    assert WeatherEvent.parse("HUB 06:00-12:00 0.5 c=0.3").cancel == 0.3
    sure = OpsConfig(delays=DelayModel(prob=0.0, block_cv=0.0, cancel_days={"*": [1.0, 1.0]}))
    assert simulate_day(s, sure, seed=1).metrics["cancelled.other"] == 2
    back = DelayModel.from_dict(sure.delays.to_dict())
    assert back.cancel_days == {"*": [1.0, 1.0]}


def test_late_turn_compression_speeds_up_late_turns() -> None:
    state = OpsState(0.0, {"A1": FlightStatus(etd=450.0)})
    slow = simulate_day(two_legs(gap=30), calm(min_turn=60), seed=3, state=state)
    fast = simulate_day(
        two_legs(gap=30), calm(min_turn=60, late_turn_compression=0.0), seed=3, state=state
    )
    assert fast.flights["A2"].dep == pytest.approx(450 + 60 + 54)
    assert fast.flights["A2"].dep <= slow.flights["A2"].dep


def test_live_feed_loader(tmp_path) -> None:  # type: ignore[no-untyped-def]
    import json

    feed = tmp_path / "feed.json"
    feed.write_text(json.dumps({"now": "08:00", "flights": [{"flight": "A1", "atd": "06:05"}]}))
    st = OpsState.load(str(feed))
    assert st.now == 480 and st.flights["A1"].atd == 365
    csv_feed = tmp_path / "feed.csv"
    csv_feed.write_text("flight,atd,ata,etd,status\nA1,06:05,07:02,,\n")
    assert OpsState.load(str(csv_feed), "09:00").flights["A1"].ata == 422
    with pytest.raises(ConfigError, match="scheme"):
        OpsState.load("ftp://example.com/feed.csv", "09:00")
