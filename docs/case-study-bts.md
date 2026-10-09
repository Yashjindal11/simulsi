# Case study: Alaska Airlines, June 2026

This case study runs `simulsi.aviation` on a full month of public US data.
It asks two questions:

1. How good are the forecasts on days the model has never seen?
2. How much do they improve during the day, once actual times come in?

All the numbers below come from
[`scripts/case_study_bts.py`](https://github.com/Yashjindal11/simulsi/blob/main/scripts/case_study_bts.py)
and can be reproduced from the public file.

## Data

The data is the Bureau of Transportation Statistics *Reporting Carrier
On-Time Performance* file for June 2026. Download the monthly ZIP from
[transtats.bts.gov](https://www.transtats.bts.gov) and unzip it. The
analysis keeps Alaska Airlines (`AS`), which now includes the Hawaiian
inter-island flying:

| | |
| --- | --- |
| Flights | 29,885 over 30 days (about 1,000 a day, 290 aircraft, 83 airports) |
| Cancelled | 196 |
| On-time arrivals (A15) | 81.4% |
| Mean departure delay | 11.5 min |
| Busiest airport | SEA, then HNL, PDX, ANC |

BTS reports **local** clock times. A Seattle-New York rotation therefore
looks broken until the times share one clock. `History.from_csv` infers
each airport's offset from the data itself: it compares scheduled block
times with the local departure and arrival times. It found, for example:

* SEA −3 h and HNL −6 h against Tampa's Eastern time;
* the Alaskan bush airports correctly an hour apart from Hawaii.

Tail numbers give the aircraft rotations. A diverted or unreported leg
breaks a rotation; `History.schedule` splits it into separate pieces so
every day validates.

```text
simulsi aviation history On_Time_..._2026_6.csv --carrier AS
simulsi aviation history On_Time_..._2026_6.csv --carrier AS --date 2026-06-21 -o day.csv \
    --status-at 17:00 --status-out status_1700.csv
```

## Method

The month was split in time:

* **Calibration, 1-20 June:**
  * minimum turn times per airport (`fit_turn_times`; SEA 59 min, the
    Hawaiian outer islands under 30);
  * primary-delay chance and size per airport (`calibrate`, method of
    simulated moments), with the empirical shape of delays (heavier-tailed
    than an exponential);
  * how much whole days vary together (`day_sigma`);
  * the ratio of actual to scheduled block time: 0.972, so flights arrive
    a little ahead of schedule in the air.
* **Test, 21-30 June:** each day was forecast from its schedule alone
  (*day ahead*). It was then re-forecast at 10:00, 14:00 and 18:00 Seattle
  time from the actual departures, arrivals and cancellations up to that
  moment (*live*). Only flights that had not yet departed were scored.

BTS has no crew pairings or passenger connections, so crew and
connection effects are not part of this test.

## Results

| Test days (10) | Default delay model | Calibrated |
| --- | --- | --- |
| OTP error per day (mean absolute) | 8.9 pts | **6.0 pts** |
| Brier score of on-time probabilities | 0.170 | **0.153** |
| Brier skill vs the base rate | −0.11 | +0.00 |
| Departures at or below forecast p50 / p80 / p95 | 59% / 72% / 93% | 61% / 73% / 90% |

**Day ahead**, the calibrated model gets the level of the operation
right:

* It predicts 76-78% on-time against an actual 71-88%.
* Its probabilities are about as sharp as the base rate, so no worse than
  always predicting the average on-time rate.
* It cannot know *which* days will go wrong from a schedule alone; on bad
  days it says so only through its wide 80% range.
* The default model, by contrast, was 9 points too pessimistic: it
  propagated delays through turns that were too short and block times
  that were too long.

**Live**, the gain is clear:

| Re-forecast at (Seattle time) | Flights scored | Brier live | Brier day ahead | Arrival-delay error live | Day ahead |
| --- | --- | --- | --- | --- | --- |
| 10:00 | 6,468 | 0.155 | 0.166 | 15.0 min | 16.2 min |
| 14:00 | 4,131 | 0.153 | 0.173 | 14.2 min | 16.6 min |
| 18:00 | 1,702 | **0.136** | 0.162 | **12.4 min** | 15.5 min |

By the evening the live twin's probabilities are 16% better (Brier) and
its delay estimates 3 minutes closer than the day-ahead forecast. The
reason is that it knows which aircraft are already running late, and the
simulation propagates that through their remaining rotations.

**What the operation looks like.** For 21 June, the forecast:

* lists evening SEA departures on long-delayed rotations as most at risk
  (for example SEA-PHX with a p80 delay above two hours);
* ranks tails by expected knock-on minutes;
* in a reserve study at SEA, finds that one or two spare aircraft pay for
  themselves on a day like this, cutting the expected total cost by about
  2.5%. Standby crews do not, but with no crew data that part of the model
  is idle.

## Limits found on real data

* **Tails.** The top 5% of delays are still under-forecast: 90% of
  departures fall at or below the p95 instead of 95%. Rare long delays
  (maintenance, crew timeouts) need richer causes than one delay
  distribution per airport.
* **Extreme probabilities.** Flights given very low on-time chances
  (under 20%) turned out better more often than predicted. The model
  never lets operations control speed up turns or swap aircraft beyond
  its simple rules, and real controllers do more.
* **Cancellations.** Cancellations are under-predicted on bad days; the
  real ones are often weather or crew-driven.
* **Not tested here.** Without crew pairings, passenger connections and
  ATC data, those parts of the twin were not exercised. An airline's own
  data would add them.

## Reproduce

```text
python scripts/case_study_bts.py On_Time_..._2026_6.csv --carrier AS --reps 100   # about 3 minutes
```

or step by step with the CLI:

```text
simulsi aviation calibrate On_Time_..._2026_6.csv --carrier AS --days 20 --fit-turns --write-ops as_ops.yaml
simulsi aviation backtest  On_Time_..._2026_6.csv --carrier AS --skip-days 20 --ops as_ops.yaml \
    --live-at 13:00 --live-at 17:00 --live-at 21:00      # 10:00/14:00/18:00 Seattle on the Eastern clock
```
