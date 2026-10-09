from __future__ import annotations

import json
from pathlib import Path

import pytest

from simulsi.aviation import (
    Cancel,
    DelayModel,
    History,
    OpsConfig,
    Retime,
    Schedule,
    TurnaroundTask,
    backtest,
    calibrate,
    checkin_staffing,
    closure,
    compare_plans,
    fit_delay_model,
    forecast,
    optimize_buffers,
    overbooking,
    plan_reserves,
    recommend_mct,
    recover,
    reliability_table,
    rolling_forecast,
    schedule_impact,
    simulate_day,
    state_at,
    thunderstorm,
    turnaround,
)
from simulsi.cli import main
from simulsi.errors import ConfigError

HUB = OpsConfig(runway_rate={"HUB": 30}, gates={"HUB": 10}, curfew={"HUB": 1410})


@pytest.fixture(scope="module")
def schedule() -> Schedule:
    return Schedule.synthetic()


def test_forecast_views(schedule: Schedule) -> None:
    fc = forecast(schedule, HUB, replications=30, weather=[thunderstorm("HUB", "15:00", 3, 0.5)])
    assert set(fc.flights) == set(schedule.by_id)
    for f in fc.flights.values():
        assert 0 <= f.p_on_time <= 1 and 0 <= f.p_cancel <= 1
    s = fc.summary()
    assert s["otp"]["p10"] <= s["otp"]["mean"] <= s["otp"]["p90"]
    assert len(fc.gantt()) == len(schedule)
    assert fc.rotations()[0]["risk_score"] >= fc.rotations()[-1]["risk_score"]
    assert {a["airport"] for a in fc.airports()} == set(schedule.airports)
    assert any(a.kind == "weather" for a in fc.alerts())
    data = json.loads(fc.to_json())
    assert data["replications"] == 30 and len(data["flights"]) == len(schedule)
    assert fc.to_csv().startswith("id,tail,origin")
    assert "Forecast from 30 simulated days" in fc.format()


def test_closure_forecast_is_worse_and_raises_alerts(schedule: Schedule) -> None:
    calm = forecast(schedule, HUB, replications=20)
    storm = forecast(schedule, HUB, replications=20, weather=[closure("HUB", "09:00", 150)])
    assert storm.summary()["otp"]["mean"] < calm.summary()["otp"]["mean"] - 0.1
    assert any(a.kind in {"delay", "cancellation"} for a in storm.alerts())


def test_forecast_from_live_state_marks_flown_flights(schedule: Schedule) -> None:
    truth = simulate_day(schedule, HUB, seed=99)
    state = state_at(schedule, truth, 720)
    fc = forecast(schedule, HUB, replications=20, state=state)
    flown = [f for f in fc.flights.values() if f.status in {"landed", "airborne"}]
    assert flown and all(f.status != "planned" for f in flown)
    landed = next(f for f in flown if f.status == "landed")
    assert landed.p_cancel == 0
    assert landed.dep_delay_p50 == pytest.approx(truth.flights[landed.id].dep_delay)
    rows = rolling_forecast(schedule, HUB, truth=truth, times=[360, 1200], replications=20)
    assert rows[0]["remaining_flights"] > rows[1]["remaining_flights"]
    # late in the day almost everything is known
    assert abs(rows[1]["predicted_otp"] - rows[1]["actual_otp"]) < 0.05


def test_compare_plans_uses_common_random_numbers(schedule: Schedule) -> None:
    cmp = compare_plans(
        schedule,
        {"same": [Retime("F102", 0)], "cancel": [Cancel("F100", "F101")]},
        HUB,
        replications=15,
    )
    assert cmp.names == ["as planned", "same", "cancel"]
    rows = {r["option"]: r for r in cmp.table()}
    assert rows["same"]["cost_vs_as planned"] == 0
    assert rows["cancel"]["cancelled"] >= rows["as planned"]["cancelled"] + 2
    assert "lowest expected cost" in cmp.format()


def test_recover_never_recommends_a_costlier_plan(schedule: Schedule) -> None:
    res = recover(
        schedule, HUB, weather=[closure("HUB", "09:00", 150)], replications=15, max_actions=2
    )
    assert res.after["cost"] <= res.before["cost"] + 1e-6
    assert len(res.actions) <= 2 and res.candidates_tried > 0
    assert "plan" in res.format()


def test_reserves_buffers_and_impact(schedule: Schedule) -> None:
    cfg = HUB.replace(delays=DelayModel(prob=0.25, mean=60))
    plan = plan_reserves(schedule, cfg, spares=(0, 1), standby_crews=(0, 1), replications=10)
    assert len(plan.rows) == 4 and plan.front and plan.recommended in plan.rows
    buf = optimize_buffers(schedule, cfg, budget=30, replications=15)
    assert sum(buf.added.values()) <= 30
    assert buf.schedule.validate() == []
    assert buf.after["delay_minutes"] <= buf.before["delay_minutes"] * 1.02
    impact = schedule_impact(schedule, buf.schedule, cfg, replications=10)
    assert {r["metric"] for r in impact.rows} >= {"otp", "cost"}


def test_calibration_recovers_the_delay_process(schedule: Schedule) -> None:
    truth = HUB.replace(delays=DelayModel(prob=0.3, mean=35, block_cv=0.08))
    history = History.simulated(schedule, truth, days=20, seed=1)
    assert history.summary()["days"] == 20
    quick = fit_delay_model(history)
    assert 0.03 < quick.block_cv < 0.13
    model = calibrate(history, HUB, iterations=4, replications=5, max_days=4)
    assert 0.2 < model.prob < 0.42 and 22 < model.mean < 50
    result = backtest(history, HUB.replace(delays=model), replications=20, dates=history.dates[:3])
    assert 0.6 < result.coverage["p80"] < 0.95
    assert result.otp_mae < 0.2 and "reliability" in result.format()


def test_history_reads_bts_columns(tmp_path: Path) -> None:
    path = tmp_path / "bts.csv"
    path.write_text(
        "FL_DATE,OP_UNIQUE_CARRIER,OP_CARRIER_FL_NUM,TAIL_NUM,ORIGIN,DEST,CRS_DEP_TIME,DEP_DELAY,CRS_ARR_TIME,ARR_DELAY,CANCELLED\n"
        "2024-01-05,AA,100,N1,JFK,BOS,0700,12,0815,5,0\n"
        "2024-01-05,AA,101,N1,BOS,JFK,0900,,1015,,1\n"
        "2024-01-05,AA,102,N1,JFK,ORD,2330,30,0130,20,0\n"
    )
    h = History.from_csv(path)
    a, b, c = sorted(h.flights, key=lambda x: x.flight.std)
    assert a.flight.id == "AA100" and a.flight.std == 420 and a.dep_delay == 12
    assert b.cancelled
    assert c.flight.sta == 25 * 60 + 30 and c.arr_delay == 20


def test_reliability_table_bins() -> None:
    rows = reliability_table([0.05, 0.15, 0.9, 0.95], [False, False, True, True], bins=10)
    assert [r["flights"] for r in rows] == [1, 1, 2]
    assert rows[-1]["observed"] == 1.0


def test_turnaround_critical_path() -> None:
    r = turnaround(replications=150)
    assert r.quantile(0.5) < r.quantile(0.95) <= r.min_turn(0.95)
    assert r.criticality["board"] == 1.0 and r.criticality["unload_bags"] == 0.0
    assert "bottleneck" in r.format()
    slow = turnaround(replications=150, crews={"clean": 1, "cater": 1})
    assert slow.quantile(0.5) >= r.quantile(0.5) - 1
    with pytest.raises(ConfigError, match="cycle"):
        turnaround([TurnaroundTask("a", 1, 1, 1, ("b",)), TurnaroundTask("b", 1, 1, 1, ("a",))])


def test_mct_overbooking_and_checkin(schedule: Schedule) -> None:
    mct = recommend_mct(samples=3000)
    made = [r["made"] for r in mct.table]
    assert made == sorted(made) and 30 <= mct.mct <= 120
    ob = overbooking(180, 0.9)
    assert ob.best_limit > 180
    assert overbooking(180, 1.0).best_limit == 180
    plan = checkin_staffing([f for f in schedule.flights if f.origin == "HUB"], replications=2)
    assert plan.staff_hours > 0
    assert max(plan.simulated_wait) < 15


def test_cli_aviation_workflow(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    d = tmp_path / "ex"
    assert main(["aviation", "example", str(d), "--days", "4"]) == 0
    common = [
        str(d / "schedule.csv"),
        "-c",
        str(d / "connections.csv"),
        "--ops",
        str(d / "ops.yaml"),
    ]
    capsys.readouterr()
    assert main(["aviation", "forecast", *common, "-r", "10", "--json"]) == 0
    data = json.loads(capsys.readouterr().out)
    assert data["replications"] == 10
    assert (
        main(
            [
                "aviation",
                "forecast",
                *common,
                "-r",
                "10",
                "--status",
                str(d / "status_1200.csv"),
                "--now",
                "12:00",
                "-o",
                str(d / "f.csv"),
            ]
        )
        == 0
    )
    assert (d / "f.csv").read_text().startswith("id,")
    assert main(["aviation", "whatif", *common, "-r", "5", "--option", "x=cancel F100 F101"]) == 0
    assert (
        main(
            [
                "aviation",
                "calibrate",
                str(d / "history.csv"),
                "--quick",
                "-o",
                str(d / "delays.yaml"),
            ]
        )
        == 0
    )
    assert (
        main(
            [
                "aviation",
                "backtest",
                str(d / "history.csv"),
                "--delays",
                str(d / "delays.yaml"),
                "-r",
                "5",
                "--date",
                "2026-01-01",
            ]
        )
        == 0
    )
    assert main(["aviation", "overbooking", "--seats", "100"]) == 0
    assert main(["aviation", "turnaround", "-r", "50"]) == 0
    capsys.readouterr()
    assert main(["aviation", "forecast", *common, "--now", "12:00"]) == 2
    assert "--now needs --status" in capsys.readouterr().err


def test_dashboard_forecast_endpoint() -> None:
    from simulsi.web.server import aviation_example, aviation_forecast

    ex = aviation_example()
    out = aviation_forecast(
        {**ex, "replications": 10, "now": ex["status_now"], "actions": ["retime F110 10"]}
    )
    assert out["now"] == 720 and len(out["gantt"]) == len(Schedule.synthetic())
    json.dumps(out, allow_nan=False)
    with pytest.raises(ConfigError, match="status"):
        aviation_forecast({**ex, "status_csv": "", "now": "12:00", "replications": 5})


def _prezip(tmp_path: Path) -> Path:
    # SEA is 3 h behind JFK; one aircraft flies SEA-JFK-SEA, local clock times
    path = tmp_path / "prezip.csv"
    rows = [
        "FlightDate,Reporting_Airline,Flight_Number_Reporting_Airline,Tail_Number,Origin,Dest,CRSDepTime,DepDelay,CRSArrTime,ArrDelay,Cancelled,CRSElapsedTime",
    ]
    for day in ("2026-06-01", "2026-06-02"):
        rows += [
            f"{day},AS,10,N1,SEA,JFK,0700,5.00,1520,-3.00,0.00,320.00",
            f"{day},AS,11,N1,JFK,SEA,1630,40.00,1950,30.00,0.00,380.00",
            f"{day},AS,12,N1,SEA,PDX,2100,,2150,,1.00,50.00",
            f"{day},DL,99,N9,JFK,ATL,0800,0.00,1030,0.00,0.00,150.00",
        ]
    path.write_text("\n".join(rows) + "\n")
    return path


def test_history_reads_prezip_and_aligns_time_zones(tmp_path: Path) -> None:
    h = History.from_csv(_prezip(tmp_path), carrier="AS")
    assert h.clock == "JFK" and h.tz_offsets == {"JFK": 0.0, "SEA": -180.0, "PDX": -180.0}
    s = h.schedule("2026-06-01")
    assert s.validate() == [] and {f.id for f in s} == {"AS10", "AS11", "AS12"}
    out = s.by_id["AS10"]
    assert out.std == 10 * 60 and out.sta == 10 * 60 + 320  # 07:00 Seattle = 10:00 New York
    back = s.by_id["AS11"]
    assert back.std == 16 * 60 + 30 and back.sta == back.std + 380
    flown = {a.flight.id: a for a in h.day("2026-06-01")}
    assert flown["AS11"].dep_delay == 40 and flown["AS12"].cancelled
    state = h.state("2026-06-01", 17 * 60 + 30)
    assert state.flights["AS10"].ata is not None and state.flights["AS11"].atd == 17 * 60 + 10
    assert "AS12" not in state.flights
    assert History.from_csv(_prezip(tmp_path), carrier="AS", clock="SEA").tz_offsets["JFK"] == 180


def test_schedule_repair_splits_broken_rotations() -> None:
    from simulsi.aviation import Flight
    from simulsi.aviation.calibration import ActualFlight

    legs = [
        ActualFlight("d", Flight("A", "T", "SEA", "PDX", 600, 650), 600, 650, False),
        ActualFlight("d", Flight("B", "T", "LAX", "SEA", 800, 950), 800, 950, False),  # missing leg
    ]
    h = History(legs)
    assert h.schedule("d", repair=False).validate()
    fixed = h.schedule("d")
    assert fixed.validate() == [] and fixed.by_id["B"].tail == "T#2"


def test_fitted_turns_shape_and_day_effect(schedule: Schedule) -> None:
    from simulsi.aviation import fit_turn_times

    truth = HUB.replace(delays=DelayModel(prob=0.3, mean=35, day_sigma=0.6))
    history = History.simulated(schedule, truth, days=15, seed=4)
    turns = fit_turn_times(history, min_samples=5)
    assert 25 < turns["HUB"] < 60
    quick = fit_delay_model(history, min_samples=10_000)
    sh = quick.shape
    assert len(sh) == 41 and (sum(sh) - (sh[0] + sh[-1]) / 2) / 40 == pytest.approx(1)
    model = calibrate(history, HUB, iterations=3, replications=4, max_days=3)
    assert model.day_sigma > 0.1
    back = DelayModel.from_dict(model.to_dict())
    assert back.shape == pytest.approx(model.shape, abs=1e-4) and back.day_sigma == model.day_sigma


def test_backtest_scores_live_reforecasts(schedule: Schedule) -> None:
    history = History.simulated(schedule, HUB, days=3, seed=2)
    res = backtest(history, HUB, replications=10, live_at=[600, 960])
    assert [r["at"] for r in res.live] == ["10:00", "16:00"]
    assert res.live[0]["flights_scored"] > res.live[1]["flights_scored"]


def test_cli_history_and_bts_options(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    path = str(_prezip(tmp_path))
    assert main(["aviation", "history", path, "--carrier", "AS"]) == 0
    assert "JFK clock" in capsys.readouterr().out
    sched, status = tmp_path / "s.csv", tmp_path / "st.csv"
    assert (
        main(
            [
                "aviation",
                "history",
                path,
                "--carrier",
                "AS",
                "--date",
                "2026-06-01",
                "-o",
                str(sched),
                "--status-at",
                "12:00",
                "--status-out",
                str(status),
            ]
        )
        == 0
    )
    assert Schedule.from_csv(sched).validate() == []
    assert (
        main(
            [
                "aviation",
                "calibrate",
                path,
                "--carrier",
                "AS",
                "--quick",
                "--fit-turns",
                "--write-ops",
                str(tmp_path / "ops.yaml"),
            ]
        )
        == 0
    )
    assert "delays:" in (tmp_path / "ops.yaml").read_text()
    assert (
        main(
            [
                "aviation",
                "backtest",
                path,
                "--carrier",
                "AS",
                "--ops",
                str(tmp_path / "ops.yaml"),
                "-r",
                "5",
                "--live-at",
                "12:00",
            ]
        )
        == 0
    )
