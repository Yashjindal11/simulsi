# Built-in model gallery

SimulSI ships seven ready-to-run models. Each is a normal `Model` with
documented parameters, key outputs and **presets**: named what-if
scenarios that show how the system reacts when you change something. Use
them to learn the library, to teach, or as starting points for your own
models.

```bash
simulsi models                                   # list them
simulsi models airline                           # parameters, presets, outputs
simulsi whatif builtin:airline --presets         # run every preset, compare
simulsi whatif builtin:traffic_signal --vary cycle=30,60,90,120   # sweep a parameter
simulsi whatif builtin:epidemic --vary r0=1.5,2.5 --vary vaccination=0,0.4   # full grid
```

In the dashboard (`simulsi ui`), pick a model on *Run experiment* and click
**Load what-if presets**, or use *Explore* for grid sweeps, sensitivity and
Monte Carlo. From Python:

```python
from simulsi import Experiment
from simulsi.models import supply_chain

result = Experiment(supply_chain, supply_chain.preset_scenarios(), replications=3, seed=1).run()
print(result.format_summary(["bullwhip.factory", "fill_rate"]))
```

The tables below come from `simulsi whatif ... --presets` (seed 0, common
random numbers; ± is the 95% CI half-width). All data is synthetic, so
read the tables for *direction and size of effects*, not as forecasts.

| Model | What it shows | Time unit |
|---|---|---|
| [`mmc`](#mmc) | the M/M/c queue, checked against Erlang C | any |
| [`airline`](#airline) | delay propagation through aircraft rotations, crews and gates | minutes |
| [`epidemic`](#epidemic) | stochastic SEIR with hospital overflow and lockdown policy | days |
| [`supply_chain`](#supply-chain) | the bullwhip effect in a four-stage supply chain | days |
| [`ride_hailing`](#ride-hailing) | pickup-time feedback, cancellations and surge pricing | minutes |
| [`cloud_autoscaling`](#cloud-autoscaling) | bursty traffic, cold starts and autoscaling policies | seconds |
| [`traffic_signal`](#traffic-signal) | signal timing, Webster's formula, actuated control | seconds |

## mmc

Poisson arrivals, exponential service, `servers` identical servers. The
simulated mean wait matches the Erlang C formula (`simulsi.models.erlang_c`);
the test suite checks this. Presets: `busy` (utilization 0.97: waits
explode), `two_servers` and `pooled_three` (the same utilization with more,
pooled servers: much shorter waits).

## airline

A fleet of aircraft flies rotations through a hub. Late inbound aircraft
and crews that connect from other aircraft pass delay on to the next
departure (*reactionary* delay). Gates are held from arrival to pushback,
spare aircraft rescue badly delayed legs, and a weather window cuts the
runway rate.

`simulsi whatif builtin:airline --presets -r 8`:

| scenario | on-time performance | mean arrival delay (min) | reactionary share | mean gate wait (min) |
|---|---:|---:|---:|---:|
| baseline (15 min buffer) | 0.65 ± 0.06 | 15.9 ± 3.3 | 0.63 | 0.7 |
| tight_schedule (0 buffer) | **0.32** ± 0.02 | 36.4 ± 3.5 | 0.85 | 0.4 |
| padded_schedule (30 min) | **0.79** ± 0.04 | 11.1 ± 3.4 | 0.35 | 2.4 |
| crew_chaos | 0.63 ± 0.06 | 17.3 ± 3.5 | 0.68 | 0.6 |
| no_spares | 0.63 ± 0.06 | 20.8 ± 5.1 | 0.69 | 0.6 |
| thunderstorm | 0.61 ± 0.05 | 18.4 ± 3.6 | 0.66 | 0.6 |
| storm_with_padding | 0.79 ± 0.03 | 9.7 ± 1.5 | 0.30 | 2.5 |
| gate_crunch (4 gates, padded) | 0.50 ± 0.09 | 24.2 ± 7.3 | 0.77 | **17.1** |

Without a buffer most delay is reactionary: one disruption ripples
through the rest of the day. Padding fixes that, but padded aircraft sit
at gates longer, so with too few gates the gain turns into gate queues.

## epidemic

Individuals move Susceptible → Exposed → Infectious → Removed one event at
a time (Gillespie's algorithm), so outbreaks differ run to run and small
ones can die out. Some cases need a bed; patients who cannot get one within
`bed_patience` days go untreated with a higher fatality rate. A lockdown
starts at `lockdown_trigger` bed occupancy.

`simulsi whatif builtin:epidemic --presets -r 5`:

| scenario | attack rate | peak infectious | deaths | untreated | lockdown days |
|---|---:|---:|---:|---:|---:|
| baseline (R0 2.5) | 0.76 ± 0.06 | 425 | 9.2 | 0 | 37 |
| mild_strain (R0 1.3) | 0.39 ± 0.12 | 103 | 3.8 | 0 | 0 |
| aggressive_strain (R0 4) | 0.86 ± 0.02 | 573 | 13.2 | 10.8 | 51 |
| vaccinated_60pct | **0.02** ± 0.03 | 12 | 0.4 | 0 | 0 |
| no_lockdown | 0.90 ± 0.00 | 719 | **24.8** | 43.8 | 0 |
| early_lockdown | 0.69 ± 0.03 | 193 | 8.8 | 0 | 103 |
| surge_beds (80 beds) | 0.90 ± 0.00 | 719 | 12.6 | 0 | 0 |

Vaccinating 60% pushes the effective R0 below 1, so outbreaks die out.
Without a lockdown the hospital overflows and deaths roughly double;
doubling beds removes the overflow instead, at the cost of the full epidemic.

## supply chain

The "beer game": customers → retailer → wholesaler → distributor →
factory. Each stage forecasts from the orders it receives and orders up to
a target, so demand noise is amplified upstream (the *bullwhip effect*).
`bullwhip.<stage>` is the variance of that stage's orders divided by the
variance of customer demand.

`simulsi whatif builtin:supply_chain --presets -r 5`:

| scenario | bullwhip at retailer | bullwhip at factory | fill rate | cost per day |
|---|---:|---:|---:|---:|
| baseline | 4.1 ± 0.3 | 34 ± 2 | 1.00 | 1,155 |
| long_lead_times (5 days) | 9.0 ± 0.6 | 38 ± 2 | 0.995 | 3,030 |
| jumpy_forecasts (alpha 0.8) | **16.7** ± 2.5 | 38 ± 2 | 0.999 | 6,719 |
| shared_pos_data | 4.1 ± 0.3 | **17.0** ± 1.6 | 1.00 | **620** |
| demand_shock (+30%) | 3.4 ± 0.2 | 18.7 ± 0.9 | 1.00 | 1,406 |
| lean_no_safety | 3.7 ± 0.1 | 29 ± 3 | 0.999 | 715 |
| tight_factory | 3.7 ± 0.2 | 0.3 ± 0.2 | **0.50** | 10,750 |

Sharing point-of-sale data halves the bullwhip at the factory and halves
costs. A capacity-limited factory cannot amplify anything (its orders are
capped) but leaves half of customer demand unmet.

## ride hailing

Riders arrive with morning and evening peaks. Pickup time grows like
`1 / sqrt(idle drivers)`, so a busy fleet spends more time driving to
pickups, which keeps it busy: the "wild goose chase". Riders give up when
no driver is found quickly or the pickup estimate is too long. Surge
pricing raises the price as the busy share of drivers climbs: some riders
decline, more drivers log on.

`simulsi whatif builtin:ride_hailing --presets -r 4`:

| scenario | service level | mean pickup (min) | surge (mean) | driver earnings / h |
|---|---:|---:|---:|---:|
| baseline (60 drivers) | 0.66 ± 0.03 | 8.7 | 1.00 | 15.9 |
| small_fleet (40) | 0.45 ± 0.01 | 9.5 | 1.00 | 16.3 |
| big_fleet (90) | **0.86** ± 0.02 | 5.6 | 1.00 | 14.0 |
| surge_pricing | 0.69 ± 0.02 | 6.3 | 1.27 | **20.3** |
| concert_night | 0.52 ± 0.02 | 8.7 | 1.00 | 15.9 |
| concert_night_with_surge | 0.58 ± 0.01 | 6.7 | 1.34 | **23.0** |
| impatient_riders | 0.65 ± 0.02 | 6.7 | 1.00 | 15.9 |

Surge pricing serves *more* riders even though some decline the price,
because the extra drivers and the shed demand break the long-pickup spiral.

## cloud autoscaling

Requests arrive at a base rate with random traffic bursts; each server
handles one at a time and requests time out after `timeout` seconds. Every
`check_interval` the autoscaler adds `scale_step` servers (ready only after
`spin_up` seconds) or removes one.

`simulsi whatif builtin:cloud_autoscaling --presets -r 4`:

| scenario | SLO attainment | error rate | p95 latency (s) | cost |
|---|---:|---:|---:|---:|
| baseline | 0.86 ± 0.09 | 0.012 | 1.39 | 0.81 |
| slow_cold_start (120 s) | 0.85 ± 0.14 | 0.008 | 1.34 | 0.94 |
| instant_start (5 s) | 0.87 ± 0.06 | 0.007 | 1.33 | 0.80 |
| overprovisioned (10 min) | 0.87 ± 0.08 | 0.005 | 1.40 | 1.01 |
| lazy_autoscaler | 0.72 ± 0.21 | 0.051 | 1.79 | 0.66 |
| flash_crowd (6x bursts) | 0.67 ± 0.28 | **0.107** | 1.84 | 0.94 |
| flash_crowd_ready | 0.87 ± 0.08 | 0.021 | 1.31 | 1.10 |

Bursts make the results noisy (look at the intervals): one long burst can
dominate an hour. Against a 6x flash crowd, adding one server per check is
far too slow; checking every 5 s and adding 4 servers at a time keeps the
error rate near the baseline for about a third more cost.

## traffic signal

Two approaches share a signal. In fixed-time mode the simulated delay can
be checked against Webster's (1958) formula, reported as
`webster.delay_<approach>`; with `control = "actuated"` the green extends
while vehicles keep arriving and ends when the queue clears.

`simulsi whatif builtin:traffic_signal --vary cycle=30,45,60,90,120,180 -r 4`:

| cycle (s) | simulated mean delay (s) | Webster, north-south (s) |
|---:|---:|---:|
| 30 | 14.8 ± 2.2 | 16.7 |
| 45 | 15.2 ± 0.8 | 14.3 |
| 60 | 16.9 ± 0.8 | 15.5 |
| 90 | 22.0 ± 1.5 | 19.2 |
| 120 | 26.3 ± 1.5 | 23.5 |
| 180 | 36.6 ± 3.4 | 32.2 |

Presets (one run, seed 1): actuated control cuts mean delay from 16.9 s to
12.9 s; at rush hour (`flow_ns` 850 veh/h) fixed timing is close to
saturation and delay jumps to 40.7 s, while actuated control holds it at
15.5 s. With `wrong_split` (35% green to the busier approach) that approach
becomes oversaturated: Webster's formula has no answer (NaN) and the queue
grows for the whole hour.

## Writing your own

Each model is a single file in
[`src/simulsi/models/`](https://github.com/Yashjindal11/simulsi/tree/main/src/simulsi/models):
a build function plus a `Model(...)` with parameters, outputs and
`presets={name: overrides}`. Copy one as a template; the
[core concepts](concepts.md) explain the building blocks.
