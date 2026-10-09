# Airline operations

`simulsi.aviation` turns a flight schedule into a stochastic model of the
operation. You can use it to:

* forecast tomorrow's flights the day before;
* re-forecast during the day from the live state (a simple *digital twin*);
* test controller actions and search for good ones;
* size reserves, schedule buffers and gates;
* calibrate on history, and check that the forecasts mean what they say.

Everything below works from Python, from the command line (`simulsi
aviation ...`) and, for forecasts, in the dashboard ("Airline ops twin"
page). All examples use a synthetic hub-and-spoke network.

```python
from simulsi.aviation import Schedule, OpsConfig, forecast, thunderstorm

schedule = Schedule.synthetic()                 # 12 aircraft, 72 flights, 9 airports
rules = OpsConfig(runway_rate={"HUB": 30}, gates={"HUB": 10}, curfew={"HUB": 1410})
fc = forecast(schedule, rules, replications=100,
              weather=[thunderstorm("HUB", "15:00", hours=3, probability=0.5)])
print(fc.format(top=5))
```

## How the day is simulated

Each aircraft (*tail*) flies its rotation as a simulsi process. A
departure waits for three things:

* **the aircraft**: the inbound arrival plus the minimum turn time;
* **the crew**: its previous flight plus the crew connection time, which
  may be on another aircraft;
* **any known estimate**, such as an ETD from the live status.

On top of that, each departure gets a random *primary* delay (technical,
ATC, late passengers). At airports with a runway rate or weather, it then
queues for a departure slot. Arrivals at those airports also queue for a
landing slot and wait out closures. At airports with a gate limit they may
wait for a gate.

About an hour before each departure, operations control looks at the
projected delay and may act:

| Situation | Action |
| --- | --- |
| Aircraft projected more than `swap_threshold` late, spare at the airport | Spare takes over the rotation; the late aircraft becomes the spare |
| Inbound crew late, or crew would break `duty_limit` | Call a standby crew, else cancel |
| Projected delay above `cancel_threshold` | Cancel the round trip (the aircraft stays put) |
| Departure or arrival after the airport's `curfew` | Cancel the rest of the rotation |

After the day, a passenger connection counts as missed when either flight
is cancelled, or when the outbound leaves less than `mct` minutes after the
inbound arrives. The cost of the day is:

* delay minutes x `delay_cost_per_minute`;
* plus cancellations x `cancel_cost`;
* plus misconnected passengers x `misconnect_cost_per_pax`;
* plus, with `eu261=True`, EU261 compensation. Weather exemptions are not
  modelled, so this compensation figure is an upper bound.

Every flight draws its random delays from a fixed position. As a result,
two plans that differ only by an action see the same disturbances
(*common random numbers*). This makes comparisons between options sharp.

## Schedules

A schedule is a CSV with one row per flight:

```text
flight,tail,origin,dest,std,sta,pax,crew
F100,T01,HUB,AAA,06:00,06:50,170,C01a
F101,T01,AAA,HUB,07:35,08:25,152,C01a
```

How the columns are read:

* **Times** may be `HH:MM`, with hours past 23 for the next day, minutes,
  or ISO date-times.
* **Arrivals before departure** are taken as next-day arrivals.
* **`pax` and `crew`** are optional. Without crews, crew connections are
  not modelled.
* **Column names are flexible**: `flight_no`, `registration`,
  `from`/`to`, `departure`/`arrival` and similar also work.

Passenger connections go in a second CSV with the columns
`inbound,outbound,pax`.

`Schedule.validate()` lists problems such as:

* an aircraft that "teleports" between airports;
* overlapping legs;
* connections that leave before the inbound arrives.

```python
from simulsi.aviation import Schedule

s = Schedule.synthetic(tails=6)
print(s, s.validate())
print(s.to_csv().splitlines()[:3])
```

## Operating rules

`OpsConfig` holds the rules of the day. Per-airport settings are
dictionaries; an airport that is not listed is unconstrained. In YAML
(`simulsi aviation example` writes one):

```yaml
on_time: 15
min_turn: 35
runway_rate: {HUB: 30}        # departures (and arrivals) per hour
gates: {HUB: 10}
curfew: {HUB: "23:30"}
duty_limit: 780               # minutes
spares: {HUB: 1}
standby_crews: {HUB: 2}
swap_threshold: 90
cancel_threshold: 240
mct: 35
eu261: true
delays: {prob: 0.25, mean: 35, block_cv: 0.06}
weather:
  - "HUB 15:00-18:00 0.4 p=0.5"
```

`delays` is the `DelayModel`. It sets:

* the chance and mean size of a primary delay, overridable per airport
  (`"LHR"`) or per airport and hour (`"LHR@07"`);
* block-time variability, with an optional per-route bias.

Weather events cut an airport's departure and arrival rate to `capacity`
times normal during a window, with a `probability`. Each simulated day
draws whether the event happens. There are presets: `thunderstorm`,
`morning_fog`, `snow` and `closure`.

## The day before: forecasts

`forecast` simulates the day many times and summarises it:

* **per flight**: the chance of an on-time arrival (A15), the chance of
  cancellation, departure-delay quantiles (p50/p80/p95), the main cause of
  delay, crew-duty risk and expected misconnecting passengers;
* **per rotation**: `fc.rotations()` ranks aircraft by expected delay and
  cancellation risk, so you can see which rotations need more buffer;
* **per connection**: the chance each connection is missed
  (`fc.connection_risk`);
* **network**: OTP, cancellations, misconnections and cost, each with an
  80% range;
* **alerts**: flights and connections that cross risk thresholds.

```python
from simulsi.aviation import Schedule, OpsConfig, forecast, closure

fc = forecast(Schedule.synthetic(), OpsConfig(runway_rate={"HUB": 30}),
              replications=60, weather=[closure("HUB", "09:00", 120)])
for alert in fc.alerts()[:3]:
    print(alert)
print(fc.rotations()[0])
```

`fc.to_csv()`, `fc.to_json()` and `fc.gantt()` (rows for a tail-by-time
chart) export the results.

## During the day: the live twin

`OpsState` holds what has happened so far: actual departure and arrival
times, known ETDs, cancellations and aircraft out of service. A status CSV
has the columns `flight,atd,ata,etd,status`; rows with `status=aog` and a
`tail` take an aircraft out of service until its `etd`. Pass the state to
`forecast` to simulate only the rest of the day. Flights that already flew
keep their real times, and everything else is projected from them.
Re-running this every 15 minutes gives a rolling forecast.

`rolling_forecast` lets you test the idea on a known "truth" day. It
re-forecasts at several times and scores each forecast against what
happened. The uncertainty narrows as the day unfolds:

```python
from simulsi.aviation import Schedule, OpsConfig, simulate_day, rolling_forecast

s, rules = Schedule.synthetic(), OpsConfig(runway_rate={"HUB": 30})
truth = simulate_day(s, rules, seed=123)
for row in rolling_forecast(s, rules, truth=truth, times=[300, 720, 1080], replications=40):
    print(row)
```

## What-ifs and recovery

Actions change the plan:

* `Cancel("F1", "F2")` cancels flights;
* `Retime("F1", 20)` holds a departure;
* `Swap("T01", "T02", after=720)` swaps two aircraft's remaining
  rotations at the same airport.

`compare_plans` simulates options on the same disturbances and reports the
paired cost difference with a 95% interval. `recover` searches greedily.
It starts from candidate actions for the riskiest flights: round-trip
cancellations, aircraft swaps, and holds for big connecting groups. It
keeps an action only if its saving is clearly larger than the simulation
noise.

```python
from simulsi.aviation import Schedule, OpsConfig, Cancel, compare_plans, recover, closure

s, rules = Schedule.synthetic(), OpsConfig(runway_rate={"HUB": 30}, curfew={"HUB": 1410})
storm = [closure("HUB", "09:00", 150)]
print(compare_plans(s, {"cancel F100/F101": [Cancel("F100", "F101")]}, rules,
                    replications=30, weather=storm).format())
print(recover(s, rules, weather=storm, replications=20, max_actions=2).format())
```

## Planning

| Function | Question |
| --- | --- |
| `plan_reserves` | How many spare aircraft and standby crews to hold at the hub: reserve cost against expected disruption cost, with the Pareto front |
| `optimize_buffers` | Where to add a budget of extra ground time. It goes to the turns where knock-on delay is largest, weighted by how many later legs each turn protects |
| `schedule_impact` | What a schedule change does (new rotation, retimed bank, banked vs rolling hub), simulated on matched disturbances |
| `network_model` | The day as a regular simulsi `Model` with parameters (`spares`, `standby_crews`, thresholds, `min_turn`, `delay_scale`, `weather`), for experiments, `optimize` and `pareto_search` |

```python
from simulsi.aviation import Schedule, OpsConfig, DelayModel, plan_reserves, schedule_impact

rules = OpsConfig(runway_rate={"HUB": 30}, delays=DelayModel(prob=0.25, mean=60))
print(plan_reserves(Schedule.synthetic(), rules, spares=(0, 1), standby_crews=(0, 2),
                    replications=20).format())
print(schedule_impact(Schedule.synthetic(), Schedule.synthetic(banks=False), rules,
                      names=("banked", "rolling"), replications=20).format())
```

## Passengers and ground

* `checkin_staffing` builds a counter plan per 15 minutes. It starts from
  Erlang C per slot, then a simulation checks the time-varying plan and
  adds counters where queues carry over from busy slots.
* `overbooking` finds the booking limit that maximises expected revenue
  minus denied-boarding cost.
* `eu261_compensation` gives the compensation per flight.
* `turnaround` simulates ground tasks (deboarding, cleaning, catering,
  fuelling, bags, boarding) as processes that wait for their predecessors.
  It reports the turn-time distribution and each task's *criticality*: how
  often it lies on the critical path. `min_turn(0.95)` is the scheduled
  turn that is long enough on 95% of days. Shared teams (`crews={"clean":
  1}`) show bank congestion.
* `recommend_mct` gives the minimum connection time for a reliability
  target, from inbound and outbound delays and walking time.

```python
from simulsi.aviation import turnaround, overbooking, recommend_mct

r = turnaround(replications=300)
print(r.format())
print("booking limit for 180 seats:", overbooking(180, 0.92).best_limit)
print("MCT for 95%:", recommend_mct(samples=5000).mct, "min")
```

## Calibration and backtesting

Calibration starts from `History`, which reads a CSV of actual flights.
It accepts the generic columns `date, flight, tail, origin, dest, std,
sta, atd, ata, cancelled`. It also reads the US BTS on-time files, in both
the download format (`FL_DATE, TAIL_NUM, CRS_DEP_TIME, DEP_DELAY, ...`)
and the monthly ZIP format (`FlightDate, Tail_Number, CRSDepTime, ...`):

* `carrier="AS"` keeps one airline.
* BTS times are local. When scheduled block times are present, each
  airport's offset is inferred from the data and all times are put on one
  clock (`history.clock`, `history.tz_offsets`).
* `history.schedule(date)` splits rotations broken by diversions or
  missing legs, so every day validates.
* `history.state(date, now)` rebuilds what was known at a given time.

There are two ways to fit the delay model:

* **`fit_delay_model`** subtracts each departure's knock-on delay from a
  late inbound aircraft. The rest counts as primary delay, and the function
  estimates its chance and size per airport, the empirical *shape* of
  delays (heavier-tailed than an exponential), and actual against
  scheduled block time.
* **`calibrate`** refines that estimate by simulation. The direct estimate
  also counts runway queues and crew waits, which the simulation adds by
  itself. `calibrate` re-simulates historical days and adjusts each
  airport's parameters until the simulated share of late departures and
  the mean delay match history. It also sets `day_sigma` so that whole
  days vary together as much as real ones do.

`fit_turn_times` estimates each airport's minimum turn from the fastest
turns actually flown.

`backtest` forecasts each past day from its schedule alone and scores the
forecasts:

* **Reliability table**: do flights given a 70% on-time chance arrive on
  time about 70% of the time?
* **Brier score**, against always predicting the average on-time rate.
* **Delay-quantile coverage**: about 80% of departures should be at or
  below the predicted p80.
* **OTP error per day.**
* **Live scoring** with `live_at=[...]`: it also re-forecasts from the
  live state at those times and scores the flights still to depart.

See the [Alaska Airlines case study](case-study-bts.md) for a full run on
a month of real data.

```python
from simulsi.aviation import Schedule, OpsConfig, DelayModel, History, calibrate, backtest

s, rules = Schedule.synthetic(), OpsConfig(runway_rate={"HUB": 30})
history = History.simulated(s, rules.replace(delays=DelayModel(prob=0.3, mean=35)), days=10)
model = calibrate(history, rules, iterations=3, replications=4, max_days=3)
print(model.prob, model.mean)
result = backtest(history, rules.replace(delays=model), replications=20, dates=history.dates[:2])
print(result.coverage, round(result.brier, 3))
```

A day-ahead forecast made from the schedule alone carries little
information about *which* flights will be late. Its Brier skill is
therefore modest, even when it is well calibrated. The value grows during
the day, once the live state is known.

**Machine-learning predictions.** If you have a model that predicts a
flight's departure delay (weather, load factor, history), pass it as
`DelayModel(predictor=fn)`. Here `fn(flight)` returns the expected primary
delay in minutes. The simulation samples around that prediction and
propagates it through aircraft, crews, runways and connections, which a
per-flight ML model cannot do on its own.

## Command line

```text
simulsi aviation example DIR          # schedule, connections, ops.yaml, live status, history
simulsi aviation forecast schedule.csv -c connections.csv --ops ops.yaml [--weather "HUB 15:00-18:00 0.4 p=0.6"]
simulsi aviation forecast ... --status status.csv --now 12:00 [--action "cancel F1 F2"] [-o forecast.csv]
simulsi aviation whatif   ... --option "cancel=cancel F100 F101" --option "swap=swap T01 T02 12:00"
simulsi aviation recover  ... [--status status.csv --now 12:00]
simulsi aviation reserves ... --spares 0,1,2 --standby 0,1,2
simulsi aviation buffers  ... --budget 60 -o padded.csv
simulsi aviation impact   schedule.csv changed.csv ...
simulsi aviation calibrate history.csv --ops ops.yaml -o delays.yaml
simulsi aviation backtest history.csv --ops ops.yaml --delays delays.yaml
simulsi aviation history bts.csv --carrier AS [--date 2026-06-21 -o day.csv --status-at 17:00 --status-out st.csv]
simulsi aviation calibrate bts.csv --carrier AS --days 20 --fit-turns --write-ops as_ops.yaml
simulsi aviation backtest  bts.csv --carrier AS --skip-days 20 --ops as_ops.yaml --live-at 17:00
simulsi aviation turnaround [--team clean=1]
simulsi aviation mct | overbooking --seats 180 --show-rate 0.92 | checkin schedule.csv --airport HUB
```

Add `--json` for machine-readable output.

## Limits

These are deliberate simplifications. Each could be refined if the data
supports it:

* **Distances.** Block time stands in for distance, for example in the
  EU261 bands.
* **Crew rules.** Duty is checked against a single limit; there is no
  rest-rule engine.
* **Spare aircraft.** Spares do not need positioning flights.
* **Swaps.** Swaps of rotations between tails are only proposed at the
  same airport.
* **Airspace and holding.** Runway capacity is a rate per airport; there
  is no airspace or holding-stack model.
* **Passengers.** Misconnected passengers are counted and costed, but not
  rebooked onto later flights.
