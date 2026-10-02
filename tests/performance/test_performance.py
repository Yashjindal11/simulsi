"""Loose performance regression guards. Thresholds are deliberately generous
(roughly 10x below what a laptop achieves) so they only catch real regressions
such as accidental O(n^2) behaviour."""

from __future__ import annotations

import time

import pytest

from simulsi.benchmarks import run_engine_benchmark

pytestmark = pytest.mark.slow


def test_event_throughput_floor() -> None:
    r = run_engine_benchmark("events", 50_000)
    assert r.events == 50_000
    assert r.events_per_second > 30_000, r.to_dict()


def test_process_throughput_floor() -> None:
    r = run_engine_benchmark("processes", 10_000)
    assert r.events_per_second > 15_000, r.to_dict()


def test_scaling_is_roughly_linear() -> None:
    small = run_engine_benchmark("events", 20_000)
    t0 = time.perf_counter()
    big = run_engine_benchmark("events", 200_000)
    elapsed = time.perf_counter() - t0
    # 10x the events should take far less than 40x the time (heap is O(log n))
    assert big.seconds < 40 * small.seconds, (small.seconds, big.seconds, elapsed)
