# Contributing to SimulSI

Thanks for helping! SimulSI aims to be a small, correct, well-documented
simulation framework. Contributions of all sizes are welcome: bug reports,
examples from your domain, documentation fixes, statistical methods, and
performance work.

## Development setup

```bash
git clone https://github.com/Yashjindal11/simulsi.git
cd simulsi
python3.12 -m venv .venv
.venv/bin/pip install -e ".[dev]"
scripts/check.sh            # format check, lint, mypy --strict, tests
python scripts/run_examples.py   # every example, quick mode
```

The optional dashboard lives in `web/frontend` (Node 20+):

```bash
cd web/frontend && npm ci && npm run build   # outputs to src/simulsi/web/static
```

## Ground rules

- **Correctness first.** New statistical or queueing behaviour needs a test
  against theory or a reference implementation (see
  `tests/statistical/test_queueing_theory.py` and `tests/unit/test_statistics.py`).
- **Reproducibility is a feature.** Never use global random state. Draw from
  `sim.stream("name")` or a `RandomStream` passed in. Same seed must give the
  same result, serial or parallel.
- **No invented numbers.** Benchmarks and accuracy claims in docs must come
  from scripts in the repository that anyone can re-run.
- **Keep the core light.** Core dependencies are NumPy, SciPy, pydantic and
  PyYAML. Plotting, pandas and Parquet stay optional extras.
- **Typed code.** `mypy --strict` must pass for `src/`.
- **Safe by default.** Configuration is data; never `eval`, `exec` or
  unpickle user-supplied files.

## Pull requests

1. Open an issue first for larger changes so we can agree on the design.
2. Keep PRs focused; include tests and docs for public behaviour.
3. Use [Conventional Commits](https://www.conventionalcommits.org/)
   (`feat:`, `fix:`, `docs:`, `test:`, `perf:`, `refactor:`, `ci:`, `chore:`).
4. Update `CHANGELOG.md` under "Unreleased" if users will notice the change.

## Adding an example

Examples live in `examples/<domain>.py`, define a module-level `model`
(a `simulsi.Model`) and a `main(replications=...)` function, and must run in a
few seconds with `replications=3`. `tests/integration/test_examples.py`
checks that each one builds, is reproducible and passes `validate_model`.
Use synthetic data only.

## Releasing (maintainers)

1. Update `src/simulsi/_version.py` and move "Unreleased" notes in
   `CHANGELOG.md` to a new version heading.
2. Run `scripts/check.sh`, `python scripts/run_examples.py` and build the
   dashboard.
3. Tag `vX.Y.Z` and push the tag; the release workflow builds and publishes
   the GitHub release.
