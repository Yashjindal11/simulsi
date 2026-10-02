# Benchmark results

Measured 2026-10-02T13:29:52+00:00 with `python benchmarks/run_benchmarks.py`.

- Python 3.12.15 (CPython), NumPy 2.5.3
- macOS-26.6.2-arm64-arm-64bit, 10 logical CPUs
- SimulSI 0.1.0.dev0

```text
name                      size   events  seconds  events_per_second  peak_memory_mb
---------------------  -------  -------  -------  -----------------  --------------
events                   10000    10000  0.02287          437,189.5  0.02138
events                  100000   100000    0.228          438,655.7  0.02455
events                 1000000  1000000    2.281          438,466.2  None
processes                10000    38501    0.154          250,000.3  0.3073
processes               100000   385581    1.595          241,692.2  0.5414
processes              1000000  3850935    16.34          235,736.8  None
experiment(workers=1)       16  1119489    4.587          244,072.2  None
experiment(workers=2)       16  1119489    3.816          293,356.6  None
experiment(workers=4)       16  1119489      3.4          329,285.1  None
```

* `events`: pure engine overhead - callback events in 100 interleaved chains (size = events).
* `processes`: M/M/2 queue with generator processes and a resource (size = customers).
* `experiment(workers=N)`: 16 replications of the built-in M/M/1 model, 20,000 time units each (process start-up costs ~1 s on macOS, so parallelism pays off only for longer runs).
* `peak_memory_mb`: tracemalloc peak during a separate run (only for sizes <= 100k).
