"""Case study: forecast US airline operations from public BTS on-time data.

Download monthly "Reporting Carrier On-Time Performance" ZIPs (PREZIP) from
https://www.transtats.bts.gov, then for example:

    python scripts/case_study_bts.py bts_2026_1.zip bts_2026_6.zip --carrier AS --carrier B6

For every month and carrier the first ``--train-days`` days calibrate the
model (turn times, primary delays with their shape, good and bad days per
airport, extra cancellations, probability recalibration); the remaining
days are forecast day ahead and re-forecast live, and scored. Prints a
Markdown report, then a summary across all runs.
"""

from __future__ import annotations

import argparse
import time
from collections import Counter
from pathlib import Path
from typing import Any

from simulsi.analysis.report import format_table
from simulsi.aviation import (
    History,
    OpsConfig,
    backtest,
    calibrate,
    fit_recalibration,
    fit_turn_times,
    forecast,
    plan_reserves,
)


def run(path: str, carrier: str, args: argparse.Namespace) -> dict[str, Any]:
    t0 = time.perf_counter()
    history = History.from_csv(path, carrier=carrier)
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
    causes = history.causes()
    print(f"# {carrier}, {dates[0]} to {dates[-1]} ({Path(path).name})\n")
    print(
        f"{s['flights']:,.0f} flights over {s['days']:.0f} days, {s['cancelled']:,.0f} cancelled, "
        f"OTP (A15) {s['otp']:.1%}, mean departure delay {s['dep_delay.mean']:.1f} min. "
        f"Busiest airport {hub}; times on the {history.clock} clock.\n"
    )
    share = causes["delay_share"]
    print(
        "Delay minutes by cause (BTS): "
        + ", ".join(f"{k.replace('_', ' ')} {v:.0%}" for k, v in share.items())
        + "; cancellations by cause: "
        + ", ".join(f"{k} {v}" for k, v in sorted(causes["cancellations"].items()))
        + "\n"
    )

    plain = OpsConfig()
    turns = fit_turn_times(train)
    base = plain.replace(min_turn_by_airport=turns)
    model = calibrate(train, base, iterations=5, replications=5, max_days=5)
    tuned = base.replace(delays=model)
    platt = fit_recalibration(train, tuned, dates=train.dates[-8:], replications=30)
    final = tuned.replace(otp_recalibration=platt)
    print(
        f"Calibrated on {args.train_days} days: primary-delay chance {model.prob:.2f}, "
        f"mean {model.mean:.0f} min, day-to-day sigma {model.day_sigma:.2f}, good/bad-day "
        f"profiles at {len(model.airport_days)} airports, extra cancellation rate "
        f"{model.cancel_rate:.4f}, minimum turns at {len(turns)} airports "
        f"({hub} {turns.get(hub, plain.min_turn):.0f} min), probability recalibration "
        f"a={platt[0]:+.2f} b={platt[1]:.2f}.\n"
    )
    live_at = [h * 60 - hub_offset for h in (10, 14, 18)]
    rows = []
    results = {}
    for label, cfg, live in (
        ("default", plain, ()),
        ("calibrated", tuned, ()),
        ("calibrated + recalibrated", final, live_at),
    ):
        res = backtest(history, cfg, replications=args.reps, dates=test_dates, live_at=live)
        results[label] = res
        rows.append(
            {
                "model": label,
                "OTP error/day": f"{res.otp_mae:.1%}",
                "Brier": round(res.brier, 4),
                "skill": f"{res.skill:+.3f}",
                "p50/p80/p95 coverage": "/".join(
                    f"{res.coverage[k]:.0%}" for k in ("p50", "p80", "p95")
                ),
                "cancelled pred/actual": f"{sum(d['predicted_cancelled'] for d in res.days):.0f}/"
                f"{sum(d['actual_cancelled'] for d in res.days)}",
            }
        )
    print(f"## Backtest on {len(test_dates)} held-out days\n\n```text\n{format_table(rows)}\n```\n")
    best = results["calibrated + recalibrated"]
    print(
        "Reliability (calibrated + recalibrated):\n\n```text\n"
        + format_table(best.reliability)
        + "\n```\n"
    )
    for row in best.live:
        hh, mm = row["at"].split(":")
        row["at"] = f"{hub} {int((int(hh) * 60 + int(mm) + hub_offset) // 60) % 24:02d}:00"
    print(
        "Live re-forecasts (flights not yet departed):\n\n```text\n"
        + format_table(best.live)
        + "\n```\n"
    )
    if args.details:
        day = test_dates[0]
        fc = forecast(history.schedule(day), final, replications=args.reps)
        print(f"## Day-ahead view for {day}\n\n```text\n{fc.format(top=10)}\n```\n")
        plan = plan_reserves(
            history.schedule(day),
            final,
            airport=hub,
            spares=(0, 1, 2),
            standby_crews=(0,),
            replications=max(20, args.reps // 3),
        )
        print(f"## Reserves at {hub} ({day})\n\n```text\n{plan.format()}\n```\n")
    print(f"_Runtime {time.perf_counter() - t0:.0f} s._\n", flush=True)
    live18 = best.live[-1] if best.live else {}
    return {
        "carrier": carrier,
        "month": dates[0][:7],
        "flights": int(s["flights"]),
        "OTP": f"{s['otp']:.1%}",
        "OTP err default": f"{results['default'].otp_mae:.1%}",
        "OTP err final": f"{best.otp_mae:.1%}",
        "skill default": f"{results['default'].skill:+.3f}",
        "skill final": f"{best.skill:+.3f}",
        "18:00 live vs day-ahead Brier": f"{live18.get('brier_live', float('nan')):.3f} vs "
        f"{live18.get('brier_day_ahead', float('nan')):.3f}",
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+", help="BTS monthly CSV or ZIP files")
    ap.add_argument("--carrier", action="append", help="repeatable (default AS)")
    ap.add_argument("--reps", type=int, default=60)
    ap.add_argument("--train-days", type=int, default=20)
    ap.add_argument(
        "--details", action="store_true", help="also print a day-ahead view and reserves"
    )
    args = ap.parse_args()
    summary = [run(f, c, args) for f in args.files for c in (args.carrier or ["AS"])]
    print("# Summary\n\n```text\n" + format_table(summary) + "\n```")


if __name__ == "__main__":
    main()
