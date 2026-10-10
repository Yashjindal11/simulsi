"""``simulsi aviation ...``: airline operations forecasts, recovery and planning from the command line."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from simulsi.errors import ConfigError

EXIT_OK = 0


def _out(text: str = "") -> None:
    sys.stdout.write(text + "\n")


def _load(args: argparse.Namespace) -> tuple[Any, Any, list[Any]]:
    """Schedule, ops config and weather from the common options."""
    from simulsi.aviation import Schedule, WeatherEvent

    schedule = Schedule.from_csv(args.schedule, getattr(args, "connections", None)).check()
    _, cfg, weather = _load_config_only(args)
    weather += [WeatherEvent.parse(w) for w in getattr(args, "weather", None) or []]
    if getattr(args, "no_weather", False):
        weather = []
    return schedule, cfg, weather


def _state(args: argparse.Namespace) -> Any:
    from simulsi.aviation import OpsState

    if getattr(args, "status", None):
        if not args.now:
            raise ConfigError("--status needs --now (e.g. --now 13:30)")
        return OpsState.from_csv(args.status, args.now)
    if getattr(args, "now", None):
        raise ConfigError(
            "--now needs --status: a CSV of what has departed, landed or been cancelled"
        )
    return None


def _emit(args: argparse.Namespace, obj: Any, text: str) -> None:
    if getattr(args, "json", False):
        from simulsi.aviation.forecast import jsonable

        _out(json.dumps(jsonable(obj), indent=2, default=str))
    else:
        _out(text)


def cmd_example(args: argparse.Namespace) -> int:
    from simulsi.aviation import History, Schedule

    out = Path(args.directory)
    out.mkdir(parents=True, exist_ok=True)
    schedule = Schedule.synthetic(seed=args.seed)
    schedule.to_csv(out / "schedule.csv")
    schedule.connections_csv(out / "connections.csv")
    (out / "ops.yaml").write_text(EXAMPLE_OPS)
    truth_cfg, status = example_status(schedule, args.seed)
    (out / "status_1200.csv").write_text(status)
    History.simulated(schedule, truth_cfg, days=args.days, seed=args.seed).to_csv(
        out / "history.csv"
    )
    _out(f"wrote {out}/schedule.csv, connections.csv, ops.yaml, status_1200.csv, history.csv")
    _out("try:")
    _out(
        f"  simulsi aviation forecast {out}/schedule.csv --connections {out}/connections.csv --ops {out}/ops.yaml"
    )
    _out(
        f"  simulsi aviation forecast {out}/schedule.csv --connections {out}/connections.csv --ops {out}/ops.yaml "
        f"--status {out}/status_1200.csv --now 12:00"
    )
    _out(
        f"  simulsi aviation calibrate {out}/history.csv --ops {out}/ops.yaml -o {out}/delays.yaml"
    )
    return EXIT_OK


def example_status(schedule: Any, seed: int = 0, now: float = 720.0) -> tuple[Any, str]:
    """A "true" day for the example network and its live status CSV at ``now`` (12:00)."""
    from simulsi.aviation import DelayModel, OpsConfig, simulate_day, state_at

    truth_cfg = OpsConfig(
        runway_rate={"HUB": 30},
        gates={"HUB": 10},
        curfew={"HUB": 1410},
        delays=DelayModel(prob=0.25, mean=35),
    )
    truth = simulate_day(schedule, truth_cfg, seed=seed + 1000)
    return truth_cfg, state_at(schedule, truth, now).to_csv()


EXAMPLE_OPS = """\
# Operating rules for the example network (minutes; times as HH:MM)
on_time: 15
min_turn: 35
runway_rate: {HUB: 30}        # departures (and arrivals) per hour; others unconstrained
gates: {HUB: 10}
curfew: {HUB: "23:30"}        # no departures or arrivals after this
duty_limit: 780               # 13 h crew duty
spares: {HUB: 1}
standby_crews: {HUB: 2}
swap_threshold: 90            # spare aircraft / standby crew when this late
cancel_threshold: 240         # cancel a round trip when this late
mct: 35                       # passenger minimum connection time
eu261: true
delays: {prob: 0.25, mean: 35, block_cv: 0.06}
weather:
  - "HUB 15:00-18:00 0.4 p=0.5"   # 50 % chance of storms cutting HUB capacity to 40 %
"""


def cmd_forecast(args: argparse.Namespace) -> int:
    from simulsi.aviation import forecast, parse_action

    schedule, cfg, weather = _load(args)
    actions = [parse_action(a) for a in args.action or []]
    fc = forecast(
        schedule,
        cfg,
        replications=args.replications,
        seed=args.seed,
        weather=weather,
        state=_state(args),
        actions=actions,
    )
    if args.out:
        if args.out.endswith(".json"):
            fc.to_json(args.out)
        else:
            fc.to_csv(args.out)
    _emit(args, fc.to_dict(), fc.format(args.top))
    return EXIT_OK


def cmd_watch(args: argparse.Namespace) -> int:
    import time

    from simulsi.aviation import OpsState, forecast, format_time

    schedule, cfg, weather = _load(args)
    previous: set[str] = set()
    n = 0
    while True:
        state = OpsState.load(args.feed, args.now)
        fc = forecast(
            schedule,
            cfg,
            replications=args.replications,
            seed=args.seed,
            weather=weather,
            state=state,
        )
        s = fc.summary()
        alerts = fc.alerts()
        keys = {f"{a.kind}:{a.flight}:{a.message}" for a in alerts}
        new = [a for a in alerts if f"{a.kind}:{a.flight}:{a.message}" not in previous]
        previous = keys
        _out(
            f"[{format_time(state.now)}] OTP {s['otp']['mean']:.1%} "
            f"({s['otp']['p10']:.1%}-{s['otp']['p90']:.1%}), "
            f"cancellations {s['cancelled']['mean']:.1f}, {len(alerts)} alerts, {len(new)} new"
        )
        for a in new[:10]:
            _out(f"  {a}")
        if args.out:
            fc.to_json(args.out)
        n += 1
        if args.times and n >= args.times:
            return EXIT_OK
        time.sleep(args.every)


def cmd_whatif(args: argparse.Namespace) -> int:
    from simulsi.aviation import compare_plans, parse_action

    schedule, cfg, weather = _load(args)
    options: dict[str, list[Any]] = {}
    for text in args.option or []:
        name, sep, rest = text.partition("=")
        if not sep:
            raise ConfigError(f"--option expects NAME=ACTION[;ACTION], got {text!r}")
        options[name.strip()] = [parse_action(a) for a in rest.split(";") if a.strip()]
    if not options:
        raise ConfigError("give at least one --option NAME=ACTION")
    cmp = compare_plans(
        schedule,
        options,
        cfg,
        replications=args.replications,
        seed=args.seed,
        weather=weather,
        state=_state(args),
    )
    _emit(args, cmp.table(), cmp.format())
    return EXIT_OK


def cmd_recover(args: argparse.Namespace) -> int:
    from simulsi.aviation import recover

    schedule, cfg, weather = _load(args)
    res = recover(
        schedule,
        cfg,
        state=_state(args),
        weather=weather,
        max_actions=args.max_actions,
        replications=args.replications,
        seed=args.seed,
    )
    _emit(args, res.to_dict(), res.format())
    return EXIT_OK


def cmd_reserves(args: argparse.Namespace) -> int:
    from simulsi.aviation import plan_reserves

    schedule, cfg, weather = _load(args)
    plan = plan_reserves(
        schedule,
        cfg,
        airport=args.airport,
        spares=_ints(args.spares),
        standby_crews=_ints(args.standby),
        spare_cost=args.spare_cost,
        standby_cost=args.standby_cost,
        weather=weather,
        replications=args.replications,
        seed=args.seed,
    )
    _emit(args, plan.to_dict(), plan.format())
    return EXIT_OK


def cmd_buffers(args: argparse.Namespace) -> int:
    from simulsi.aviation import optimize_buffers

    schedule, cfg, weather = _load(args)
    plan = optimize_buffers(
        schedule,
        cfg,
        budget=args.budget,
        step=args.step,
        replications=args.replications,
        seed=args.seed,
        weather=weather,
    )
    if args.out:
        plan.schedule.to_csv(args.out)
    _emit(args, {"added": plan.added, "before": plan.before, "after": plan.after}, plan.format())
    return EXIT_OK


def cmd_impact(args: argparse.Namespace) -> int:
    from simulsi.aviation import Schedule, schedule_impact

    base, cfg, weather = _load(args)
    changed = Schedule.from_csv(args.changed, args.changed_connections).check()
    res = schedule_impact(
        base, changed, cfg, replications=args.replications, seed=args.seed, weather=weather
    )
    _emit(args, {"metrics": res.rows, "airports": res.airports}, res.format())
    return EXIT_OK


def _history(args: argparse.Namespace) -> Any:
    from simulsi.aviation import History

    return History.from_csv(args.history, carrier=args.carrier, clock=args.clock)


def _subset(history: Any, dates: list[str]) -> Any:
    from simulsi.aviation import History

    keep = set(dates)
    return History(
        [a for a in history.flights if a.date in keep], history.tz_offsets, history.clock
    )


def cmd_calibrate(args: argparse.Namespace) -> int:
    import yaml

    from simulsi.aviation import calibrate, fit_delay_model, fit_recalibration, fit_turn_times

    history = _history(args)
    if args.days:
        history = _subset(history, history.dates[: args.days])
    _, cfg, _ = _load_config_only(args)
    if args.fit_turns:
        cfg = cfg.replace(
            min_turn_by_airport={**fit_turn_times(history), **cfg.min_turn_by_airport}
        )
    if args.quick:
        model = fit_delay_model(history, min_turn=cfg.min_turn, by=args.by)
    else:
        model = calibrate(history, cfg, iterations=args.iterations, seed=args.seed)
    text = yaml.safe_dump(model.to_dict(), sort_keys=False)
    if args.output:
        Path(args.output).write_text(text)
    if args.write_ops:
        ops = cfg.replace(delays=model)
        if args.recalibrate:
            ops = ops.replace(
                otp_recalibration=fit_recalibration(
                    history, ops, dates=history.dates[-8:], seed=args.seed
                )
            )
        ops.to_yaml(args.write_ops)
    summary = history.summary()
    _out(
        f"history: {summary['days']:.0f} days, {summary['flights']:.0f} flights, "
        f"OTP {summary['otp']:.1%}, mean departure delay {summary['dep_delay.mean']:.1f} min"
        + (f"; times on the {history.clock} clock" if history.clock else "")
    )
    _out(text.rstrip())
    if args.output:
        _out(
            f"wrote {args.output} (use it with --delays, or 'delays: {Path(args.output).name}' in ops.yaml)"
        )
    if args.write_ops:
        _out(f"wrote {args.write_ops} (operating rules with the fitted delays and turn times)")
    return EXIT_OK


def cmd_backtest(args: argparse.Namespace) -> int:
    from simulsi.aviation import backtest, parse_time

    history = _history(args)
    _, cfg, _ = _load_config_only(args)
    dates = args.date or history.dates[args.skip_days :]
    res = backtest(
        history,
        cfg,
        replications=args.replications,
        seed=args.seed,
        dates=dates,
        live_at=[parse_time(t) for t in args.live_at or []],
    )
    _emit(args, res.to_dict(), res.format())
    return EXIT_OK


def cmd_history(args: argparse.Namespace) -> int:
    from simulsi.aviation import parse_time

    history = _history(args)
    summary = history.summary()
    if not args.date:
        _out(
            f"{summary['days']:.0f} days ({history.dates[0]} to {history.dates[-1]}), "
            f"{summary['flights']:,.0f} flights, OTP {summary['otp']:.1%}"
            + (f"; times on the {history.clock} clock" if history.clock else "")
        )
        if history.tz_offsets:
            offsets = sorted(history.tz_offsets.items(), key=lambda kv: (kv[1], kv[0]))
            _out("clock offsets (min): " + ", ".join(f"{a} {v:+.0f}" for a, v in offsets[:40]))
        return EXIT_OK
    schedule = history.schedule(args.date)
    if args.output:
        schedule.to_csv(args.output)
        _out(f"wrote {args.output}: {schedule}")
    if args.status_at:
        if not args.status_out:
            raise ConfigError("--status-at needs --status-out FILE")
        history.state(args.date, parse_time(args.status_at)).to_csv(args.status_out)
        _out(f"wrote {args.status_out}: what was known at {args.status_at}")
    if not args.output and not args.status_at:
        _out(str(schedule))
    return EXIT_OK


def _load_config_only(args: argparse.Namespace) -> tuple[None, Any, list[Any]]:
    """Ops config (with an optional separate delay model) and the weather listed in the ops file."""
    import yaml

    from simulsi.aviation import DelayModel, OpsConfig
    from simulsi.aviation.config import weather_from_config

    cfg, weather = OpsConfig(), []
    if getattr(args, "ops", None):
        try:
            data = yaml.safe_load(Path(args.ops).read_text()) or {}
            if isinstance(data.get("delays"), str):
                data["delays"] = yaml.safe_load(
                    (Path(args.ops).parent / data["delays"]).read_text()
                )
        except (OSError, yaml.YAMLError) as exc:
            raise ConfigError(f"cannot read {args.ops}: {exc}") from exc
        cfg = OpsConfig.from_dict(data)
        weather = weather_from_config(data)
    if getattr(args, "delays", None):
        try:
            delays = yaml.safe_load(Path(args.delays).read_text()) or {}
        except (OSError, yaml.YAMLError) as exc:
            raise ConfigError(f"cannot read {args.delays}: {exc}") from exc
        cfg = cfg.replace(delays=DelayModel.from_dict(delays))
    return None, cfg, weather


def cmd_turnaround(args: argparse.Namespace) -> int:
    from simulsi.aviation import turnaround

    crews = {}
    for item in args.team or []:
        k, _, v = item.partition("=")
        crews[k] = int(v)
    res = turnaround(replications=args.replications, seed=args.seed, crews=crews or None)
    text = (
        res.format()
        + f"\nscheduled turn for {args.reliability:.0%} reliability: {res.min_turn(args.reliability):g} min"
    )
    _emit(args, {"table": res.table(), "min_turn": res.min_turn(args.reliability)}, text)
    return EXIT_OK


def cmd_mct(args: argparse.Namespace) -> int:
    from simulsi.aviation import recommend_mct

    res = recommend_mct(
        reliability=args.reliability, walk=tuple(args.walk), gate_close=args.gate_close
    )
    _emit(args, {"mct": res.mct, "table": res.table}, res.format())
    return EXIT_OK


def cmd_overbooking(args: argparse.Namespace) -> int:
    from simulsi.aviation import overbooking

    res = overbooking(
        args.seats, args.show_rate, fare=args.fare, denied_boarding_cost=args.denied_cost
    )
    _emit(
        args,
        res.to_dict(),
        res.format() + f"\nbest booking limit: {res.best_limit} for {res.capacity} seats",
    )
    return EXIT_OK


def cmd_checkin(args: argparse.Namespace) -> int:
    from simulsi.aviation import checkin_staffing

    schedule, _, _ = _load(args)
    deps = [f for f in schedule.flights if f.origin == args.airport.upper()]
    if not deps:
        raise ConfigError(f"no departures from {args.airport}")
    plan = checkin_staffing(deps, service_minutes=args.service, target_wait=args.target)
    _emit(
        args,
        {"table": plan.table(), "staff_hours": plan.staff_hours},
        plan.format() + f"\ntotal: {plan.staff_hours:.1f} counter-hours",
    )
    return EXIT_OK


def _ints(text: str) -> list[int]:
    try:
        return [int(x) for x in text.split(",") if x.strip()]
    except ValueError as exc:
        raise ConfigError(f"expected comma-separated integers, got {text!r}") from exc


def add_parser(sub: Any) -> None:
    av = sub.add_parser(
        "aviation", help="airline operations: forecasts, live twin, recovery, planning, calibration"
    )
    asub = av.add_subparsers(dest="aviation_command", required=True)

    def common(s: argparse.ArgumentParser, *, state: bool = True) -> None:
        s.add_argument(
            "schedule", help="schedule CSV: flight, tail, origin, dest, std, sta [pax, crew]"
        )
        s.add_argument("--connections", "-c", help="connections CSV: inbound, outbound, pax")
        s.add_argument("--ops", help="operating rules YAML (see 'simulsi aviation example')")
        s.add_argument("--delays", help="delay model YAML from 'simulsi aviation calibrate'")
        s.add_argument("--weather", action="append", metavar="'APT HH:MM-HH:MM [cap] [p=prob]'")
        s.add_argument("--no-weather", action="store_true", help="ignore weather in the ops file")
        s.add_argument("--replications", "-r", type=int, default=200)
        s.add_argument("--seed", type=int, default=0)
        s.add_argument("--json", action="store_true")
        if state:
            s.add_argument("--status", help="live status CSV: flight, atd, ata, etd, status")
            s.add_argument("--now", help="current time (HH:MM); forecast the rest of the day")

    s = asub.add_parser("example", help="write an example schedule, rules, live status and history")
    s.add_argument("directory", nargs="?", default="aviation_example")
    s.add_argument("--days", type=int, default=30, help="days of synthetic history")
    s.add_argument("--seed", type=int, default=0)
    s.set_defaults(func=cmd_example)

    s = asub.add_parser("forecast", help="per-flight OTP, delay ranges, fragile rotations, alerts")
    common(s)
    s.add_argument(
        "--action", action="append", metavar="'cancel F1 F2' | 'retime F1 30' | 'swap T1 T2 12:00'"
    )
    s.add_argument("--top", type=int, default=15)
    s.add_argument("--out", "-o", help="write per-flight forecasts (.csv or .json)")
    s.set_defaults(func=cmd_forecast)

    s = asub.add_parser("watch", help="live twin: re-forecast from a status feed every few minutes")
    common(s, state=False)
    s.add_argument(
        "--feed", required=True, help="status CSV/JSON file or http(s) URL, re-read each time"
    )
    s.add_argument("--now", help="fixed time (default: the feed's 'now', else the local clock)")
    s.add_argument("--every", type=float, default=300.0, help="seconds between forecasts")
    s.add_argument(
        "--times", type=int, default=0, help="stop after N forecasts (0 = run until stopped)"
    )
    s.add_argument("--out", "-o", help="write the latest forecast JSON here each time")
    s.set_defaults(func=cmd_watch, replications=100)

    s = asub.add_parser("whatif", help="compare controller options on the same disturbances")
    common(s)
    s.add_argument("--option", action="append", metavar="NAME=ACTION[;ACTION]")
    s.set_defaults(func=cmd_whatif, replications=100)

    s = asub.add_parser(
        "recover", help="search for actions that lower the expected disruption cost"
    )
    common(s)
    s.add_argument("--max-actions", type=int, default=3)
    s.set_defaults(func=cmd_recover, replications=60)

    s = asub.add_parser("reserves", help="how many spare aircraft and standby crews to hold")
    common(s, state=False)
    s.add_argument("--airport", help="where (default: the hub)")
    s.add_argument("--spares", default="0,1,2,3")
    s.add_argument("--standby", default="0,1,2,3")
    s.add_argument("--spare-cost", type=float, default=15000.0)
    s.add_argument("--standby-cost", type=float, default=2500.0)
    s.set_defaults(func=cmd_reserves, replications=60)

    s = asub.add_parser("buffers", help="spend extra ground time where it stops the most delay")
    common(s, state=False)
    s.add_argument("--budget", type=float, default=60.0, help="total extra minutes")
    s.add_argument("--step", type=float, default=5.0)
    s.add_argument("--out", "-o", help="write the padded schedule CSV")
    s.set_defaults(func=cmd_buffers, replications=100)

    s = asub.add_parser("impact", help="compare two schedules (new rotation, bank structure, ...)")
    common(s, state=False)
    s.add_argument("changed", help="the changed schedule CSV")
    s.add_argument("--changed-connections")
    s.set_defaults(func=cmd_impact, replications=100)

    def bts(s: argparse.ArgumentParser) -> None:
        s.add_argument("history", help="actual flights CSV (generic, or a US BTS on-time file)")
        s.add_argument("--carrier", help="keep one airline, e.g. AS (BTS files hold all of them)")
        s.add_argument("--clock", help="airport whose local time is used for all times")

    s = asub.add_parser("calibrate", help="fit the delay model to history (generic or BTS CSV)")
    bts(s)
    s.add_argument("--ops", help="operating rules used when re-simulating history")
    s.add_argument("--output", "-o", help="write the delay model YAML")
    s.add_argument("--write-ops", metavar="FILE", help="write full operating rules with the fit")
    s.add_argument(
        "--fit-turns", action="store_true", help="also fit minimum turn times per airport"
    )
    s.add_argument(
        "--recalibrate",
        action="store_true",
        help="also fit on-time probability recalibration (Platt) into --write-ops",
    )
    s.add_argument(
        "--days", type=int, help="use only the first N days (keep the rest for a backtest)"
    )
    s.add_argument("--quick", action="store_true", help="direct estimate only, no simulation loop")
    s.add_argument("--by", choices=["origin", "origin_hour"], default="origin")
    s.add_argument("--iterations", type=int, default=6)
    s.add_argument("--seed", type=int, default=0)
    s.set_defaults(func=cmd_calibrate)

    s = asub.add_parser("backtest", help="forecast past days and score the forecasts")
    bts(s)
    s.add_argument("--ops")
    s.add_argument("--delays")
    s.add_argument("--date", action="append", help="only these dates (YYYY-MM-DD)")
    s.add_argument(
        "--skip-days", type=int, default=0, help="skip the first N days (the training days)"
    )
    s.add_argument(
        "--live-at",
        action="append",
        metavar="HH:MM",
        help="also score live re-forecasts at this time",
    )
    s.add_argument("--replications", "-r", type=int, default=100)
    s.add_argument("--seed", type=int, default=0)
    s.add_argument("--json", action="store_true")
    s.set_defaults(func=cmd_backtest)

    s = asub.add_parser(
        "history", help="summarise history, or extract one day's schedule and live status"
    )
    bts(s)
    s.add_argument("--date", help="the day to extract (YYYY-MM-DD)")
    s.add_argument("--output", "-o", help="write that day's schedule CSV")
    s.add_argument("--status-at", metavar="HH:MM", help="also write what was known at this time")
    s.add_argument("--status-out", metavar="FILE")
    s.set_defaults(func=cmd_history)

    s = asub.add_parser("turnaround", help="turnaround critical path and minimum turn time")
    s.add_argument("--reliability", type=float, default=0.95)
    s.add_argument(
        "--team", action="append", metavar="TASK=N", help="limit a shared team, e.g. clean=1"
    )
    s.add_argument("--replications", "-r", type=int, default=2000)
    s.add_argument("--seed", type=int, default=0)
    s.add_argument("--json", action="store_true")
    s.set_defaults(func=cmd_turnaround)

    s = asub.add_parser("mct", help="minimum connection time for a reliability target")
    s.add_argument("--reliability", type=float, default=0.95)
    s.add_argument(
        "--walk", type=float, nargs=3, default=[8.0, 12.0, 25.0], metavar=("MIN", "MODE", "MAX")
    )
    s.add_argument("--gate-close", type=float, default=15.0)
    s.add_argument("--json", action="store_true")
    s.set_defaults(func=cmd_mct)

    s = asub.add_parser("overbooking", help="best booking limit for a flight")
    s.add_argument("--seats", type=int, default=180)
    s.add_argument("--show-rate", type=float, default=0.92)
    s.add_argument("--fare", type=float, default=150.0)
    s.add_argument("--denied-cost", type=float, default=600.0)
    s.add_argument("--json", action="store_true")
    s.set_defaults(func=cmd_overbooking)

    s = asub.add_parser(
        "checkin", help="check-in counters per 15 minutes for an airport's departures"
    )
    common(s, state=False)
    s.add_argument("--airport", required=True)
    s.add_argument("--service", type=float, default=2.5, help="minutes per passenger")
    s.add_argument("--target", type=float, default=10.0, help="target mean wait (minutes)")
    s.set_defaults(func=cmd_checkin)
