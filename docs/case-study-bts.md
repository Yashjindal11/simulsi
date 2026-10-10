# Case study: US airlines on public BTS data

This case study runs `simulsi.aviation` on public US data: first a full
month of Alaska Airlines in detail, then three carriers over four months.
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
  * how much whole days vary together (`day_sigma`), and each airport's
    own good and bad days, for delays and cancellations;
  * the ratio of actual to scheduled block time: 0.972, so flights arrive
    a little ahead of schedule in the air;
  * a probability correction (`fit_recalibration`, Platt scaling) on the
    last 8 training days.
* **Test, 21-30 June:** each day was forecast from its schedule alone
  (*day ahead*). It was then re-forecast at 10:00, 14:00 and 18:00 Seattle
  time from the actual departures, arrivals and cancellations up to that
  moment (*live*). Only flights that had not yet departed were scored.

BTS has no crew pairings or passenger connections, so crew and
connection effects are not part of this test.

## Results

| Test days (10) | Default delay model | Calibrated | Calibrated + recalibrated |
| --- | --- | --- | --- |
| OTP error per day (mean absolute) | 8.7 pts | 5.9 pts | **4.3 pts** |
| Brier score of on-time probabilities | 0.171 | 0.155 | **0.148** |
| Brier skill vs the base rate | −0.12 | −0.01 | **+0.04** |
| Departures at or below forecast p50 / p80 / p95 | 59% / 72% / 92% | 61% / 73% / 90% | 61% / 73% / 90% |
| Cancellations predicted / actual | 0 / 70 | 94 / 70 | 94 / 70 |

**Day ahead**, the calibrated model gets the level of the operation
right:

* Its daily OTP is within about 4 points of the actual one.
* Recalibrated, its probabilities beat always predicting the average
  on-time rate, though only slightly (Brier skill +0.04).
* It cannot know *which* days will go wrong from a schedule alone; on bad
  days it says so only through its wide 80% range.
* The default model, by contrast, was 9 points too pessimistic: it
  propagated delays through turns that were too short and block times
  that were too long.

**Live**, the gain is clear:

| Re-forecast at (Seattle time) | Flights scored | Brier live | Brier day ahead | Arrival-delay error live | Day ahead |
| --- | --- | --- | --- | --- | --- |
| 10:00 | 6,468 | 0.151 | 0.160 | 15.4 min | 16.7 min |
| 14:00 | 4,131 | 0.151 | 0.166 | 14.4 min | 16.9 min |
| 18:00 | 1,702 | **0.133** | 0.150 | **12.4 min** | 15.7 min |

By the evening the live twin's probabilities are 11% better (Brier) and
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

## Three carriers, four months

The same method was run on Alaska (`AS`), JetBlue (`B6`) and Delta (`DL`)
for January, February, June and July 2026: about 530,000 flights. Each
month was calibrated on its first 20 days and tested on the rest (8-11
days). *Final* is calibrated + recalibrated; *skill* is Brier skill
against always predicting the average on-time rate.

| Carrier | Month | Flights | OTP | OTP error default | OTP error final | Skill default | Skill final | 18:00 Brier live vs day ahead | Cancelled pred / actual |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| AS | Jan | 26,373 | 79.6% | 12.8 | **3.4** | −0.24 | +0.00 | 0.109 vs 0.115 | 173 / 246 |
| B6 | Jan | 18,106 | 69.1% | 12.7 | 10.6 | −0.04 | +0.00 | 0.187 vs 0.206 | 78 / 971 |
| DL | Jan | 79,184 | 80.8% | 9.4 | 9.0 | −0.03 | −0.01 | 0.159 vs 0.175 | 279 / 2,801 |
| AS | Feb | 23,986 | 83.7% | 8.9 | **1.4** | −0.07 | +0.02 | 0.131 vs 0.141 | 57 / 155 |
| B6 | Feb | 17,866 | 68.3% | 21.0 | 14.1 | −0.20 | −0.05 | 0.244 vs 0.285 | 37 / 1,294 |
| DL | Feb | 74,199 | 83.8% | 6.4 | 6.6 | −0.02 | +0.01 | 0.164 vs 0.184 | 167 / 1,180 |
| AS | Jun | 29,885 | 81.4% | 8.7 | **4.3** | −0.12 | +0.04 | 0.133 vs 0.150 | 94 / 70 |
| B6 | Jun | 20,796 | 75.3% | 8.5 | 8.2 | −0.02 | +0.03 | 0.206 vs 0.252 | 107 / 107 |
| DL | Jun | 90,231 | 82.6% | 6.5 | 8.7 | −0.05 | −0.05 | 0.140 vs 0.158 | 474 / 89 |
| AS | Jul | 31,662 | 77.5% | 5.0 | **3.9** | −0.04 | +0.05 | 0.163 vs 0.187 | 106 / 124 |
| B6 | Jul | 23,289 | 62.6% | 17.1 | 13.0 | −0.11 | +0.01 | 0.250 vs 0.280 | 714 / 842 |
| DL | Jul | 94,116 | 74.3% | 4.4 | 6.7 | −0.00 | −0.00 | 0.187 vs 0.211 | 917 / 756 |

OTP errors are mean absolute points per test day.

What holds across all twelve:

* **Live beats day ahead every time.** By 18:00 the Brier score of the
  flights still to depart is 5-18% better than the day-ahead forecast in
  all 12 carrier-months. This is the most robust result.
* **Day-ahead skill is small.** Recalibrated per-flight skill ranges from
  −0.05 to +0.05, positive in 8 of 12. From schedules and history alone
  there is little to say about *which* flights will be late.
* **Calibration fixes the level** in 9 of 12: Alaska's daily OTP error
  falls from 5-13 points to 1-4, JetBlue's from 9-21 to 8-14.

Where it does not:

* **Delta.** Calibration made the level worse in February, June and July.
  Delta is the largest network here (110+ airports) and its days vary a
  lot together. With only 20 training days the fitted day effect hits its
  cap, and when the test days are calmer than the training days (June:
  85% on time against 81%) the forecast stays too pessimistic. A longer
  training window would help.
* **Storms.** Winter storms in January and February cancelled about
  1,000 JetBlue flights and 2,800 Delta flights in a few days. The model
  predicted a small fraction, and those months have the largest OTP
  errors (10-14 points for JetBlue). A model built on schedules and history cannot see a storm
  coming. Give it the storm as a weather scenario instead (for example
  `"JFK 06:00-20:00 0.3 p=1 c=0.4"`, with capacity cut to 30% and 40% of
  departures cancelled) and use it to compare plans under that storm.
* **JetBlue** is the hardest carrier: low OTP (63-75%), a congested
  Northeast base, and larger errors in every month.

## Limits found on real data

* **Tails.** The top 5% of delays are still under-forecast: 90% of
  departures fall at or below the p95 instead of 95%. Rare long delays
  (maintenance, crew timeouts) need richer causes than one delay
  distribution per airport.
* **Extreme probabilities.** Raw simulated probabilities are too sure:
  the fitted Platt slope is 0.16-0.54 across carriers and months. Real controllers
  recover late aircraft in ways the model's simple rules do not, so
  `fit_recalibration` is part of the recommended setup.
* **Cancellations.** Cancellations are predicted well on ordinary days
  (Alaska and JetBlue in June and July) but not in storms (see above).
* **Not tested here.** Without crew pairings, passenger connections and
  ATC data, those parts of the twin were not exercised. An airline's own
  data would add them.

## Reproduce

```text
python scripts/case_study_bts.py On_Time_..._2026_6.csv --carrier AS --reps 100   # about 3 minutes
python scripts/case_study_bts.py bts_2026_1.zip bts_2026_2.zip bts_2026_6.zip bts_2026_7.zip \
    --carrier AS --carrier B6 --carrier DL --reps 50                                # about an hour
```

or step by step with the CLI:

```text
simulsi aviation calibrate On_Time_..._2026_6.csv --carrier AS --days 20 --fit-turns --write-ops as_ops.yaml --recalibrate
simulsi aviation backtest  On_Time_..._2026_6.csv --carrier AS --skip-days 20 --ops as_ops.yaml \
    --live-at 13:00 --live-at 17:00 --live-at 21:00      # 10:00/14:00/18:00 Seattle on the Eastern clock
```
