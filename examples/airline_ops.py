"""Example 7 - An airline operations day: forecast, live re-forecast and recovery (synthetic data).

1. The evening before: forecast tomorrow's hub-and-spoke day with a 50 %
   chance of afternoon storms at the hub - per-flight on-time chances,
   fragile rotations and alerts.
2. At noon: a "real" day has been unfolding; re-forecast the rest of it
   from the live state.
3. Operations control: search for actions that lower the expected cost.

    python examples/airline_ops.py
"""

from __future__ import annotations

from simulsi.aviation import (
    DelayModel,
    OpsConfig,
    Schedule,
    forecast,
    recover,
    simulate_day,
    state_at,
    thunderstorm,
)


def main(replications: int = 200) -> None:
    schedule = Schedule.synthetic()
    rules = OpsConfig(
        runway_rate={"HUB": 30},
        gates={"HUB": 10},
        curfew={"HUB": 23 * 60 + 30},
        spares={"HUB": 1},
        standby_crews={"HUB": 2},
        eu261=True,
        delays=DelayModel(prob=0.25, mean=35),
    )
    storms = [thunderstorm("HUB", "15:00", hours=3, probability=0.5)]

    print("== Day ahead ==")
    print(forecast(schedule, rules, replications=replications, weather=storms).format(top=8))

    print("\n== Noon: re-forecast from the live state ==")
    truth = simulate_day(schedule, rules, seed=2026, weather=storms)
    live = state_at(schedule, truth, now=12 * 60)
    print(
        forecast(schedule, rules, replications=replications, weather=storms, state=live).format(
            top=5
        )
    )

    print("\n== Recovery search ==")
    res = recover(
        schedule, rules, state=live, weather=storms, replications=max(10, replications // 4)
    )
    print(res.format())


if __name__ == "__main__":
    main()
