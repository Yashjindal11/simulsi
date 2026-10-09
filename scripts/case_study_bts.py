"""Case study: forecast Alaska Airlines' June 2026 operation from public BTS data.

Download the monthly "Reporting Carrier On-Time Performance" file from
https://www.transtats.bts.gov (PREZIP, June 2026), unzip it, then:

    python scripts/case_study_bts.py path/to/On_Time_..._2026_6.csv [--carrier AS] [--reps 100]

Calibrates the delay model on the first 20 days, then backtests the last
10 days: day-ahead forecasts from the schedule alone, and live re-forecasts
during the day from what had happened so far. Prints a Markdown report.
"""

from __future__ import annotations

import argparse
import time
from collections import Counter

from simulsi.aviation import (
    History,
    OpsConfig,
    backtest,
    calibrate,
    fit_turn_times,
    forecast,
    plan_reserves,
)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("csv")
    ap.add_argument("--carrier", default="AS")
    ap.add_argument("--reps", type=int, default=100)
    ap.add_argument("--train-days", type=int, default=20)
    args = ap.parse_args()

    t0 = time.perf_counter()
    history = History.from_csv(args.csv, carrier=args.carrier)
    dates = history.dates
    train = History(
        [a for a in history.flights if a.date in dates[: args.train_days]],
        history.tz_offsets,
        history.clock,
    )
    test_dates = dates[args.train_days :]
    s = history.summary()
    hub, _ = Counter(a.flight.origin for a in history.flights).most_common(1)[0]
    hub_offset = history.tz_offsets.get(hub, 0.0)
    print(f"# {args.carrier}: {dates[0]} to {dates[-1]}\n")
    print(
        f"{s['flights']:,.0f} flights over {s['days']:.0f} days, {s['cancelled']:,.0f} cancelled, "
        f"OTP (A15) {s['otp']:.1%}, mean departure delay {s['dep_delay.mean']:.1f} min. "
        f"Busiest airport {hub}; times on the {history.clock} clock ({hub} is {hub_offset:+.0f} min).\n"
    )

    plain = OpsConfig()
    turns = fit_turn_times(train)
    model = calibrate(
        train, plain.replace(min_turn_by_airport=turns), iterations=5, replications=5, max_days=5
    )
    tuned = plain.replace(min_turn_by_airport=turns, delays=model)
    print(
        f"Calibrated on {args.train_days} days: primary-delay chance {model.prob:.2f}, mean {model.mean:.0f} min, "
        f"block-time CV {model.block_cv:.3f}, actual/scheduled block {model.block_scale:.3f}, "
        f"{len(model.table)} airport-specific entries; minimum turns fitted at {len(turns)} airports "
        f"({hub} {turns.get(hub, plain.min_turn):.0f} min).\n"
    )

    # live re-forecasts at 10:00, 14:00 and 18:00 hub local time
    live_at = [h * 60 - hub_offset for h in (10, 14, 18)]
    for label, cfg, live in (("default delay model", plain, ()), ("calibrated", tuned, live_at)):
        res = backtest(history, cfg, replications=args.reps, dates=test_dates, live_at=live)
        print(f"## Backtest on {len(test_dates)} held-out days - {label}\n")
        print(f"- OTP error per day (mean absolute): {res.otp_mae:.1%}")
        print(
            f"- Brier score {res.brier:.4f} vs climatology {res.brier_climatology:.4f} (skill {res.skill:+.3f})"
        )
        print(
            "- departures at or below predicted quantile: "
            + ", ".join(f"{k} {v:.0%}" for k, v in res.coverage.items())
        )
        print("\n```text\n" + res.format().split("\n\nOTP error")[0] + "\n```\n")
        print("Reliability:\n\n```text")
        from simulsi.analysis.report import format_table

        print(format_table(res.reliability) + "\n```\n")
        if res.live:
            for row in res.live:
                row["at"] = (
                    f"{hub} {int((float(row['at'][:2]) * 60 + float(row['at'][3:]) + hub_offset) // 60):02d}:00"
                )
            print(
                "Live re-forecasts (flights not yet departed):\n\n```text\n"
                + format_table(res.live)
                + "\n```\n"
            )

    day = test_dates[0]
    fc = forecast(history.schedule(day), tuned, replications=args.reps)
    print(f"## Day-ahead view for {day}\n\n```text\n{fc.format(top=10)}\n```\n")
    plan = plan_reserves(
        history.schedule(day),
        tuned,
        airport=hub,
        spares=(0, 1, 2),
        standby_crews=(0, 2),
        replications=max(20, args.reps // 3),
    )
    print(f"## Reserves at {hub} ({day})\n\n```text\n{plan.format()}\n```\n")
    print(f"_Runtime {time.perf_counter() - t0:.0f} s._")


if __name__ == "__main__":
    main()
