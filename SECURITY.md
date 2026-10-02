# Security Policy

## Reporting a vulnerability

Please **do not** open a public issue. Report privately through
[GitHub security advisories](https://github.com/Yashjindal11/simulsi/security/advisories/new).
Include the affected version (`simulsi --version`), a description and a
minimal reproduction. You can expect an acknowledgement within a week. Fixes
are released as patch versions and credited in the changelog unless you
prefer otherwise.

Supported versions: the latest minor release.

## Threat model

SimulSI is a local library and command-line tool. Its security properties:

| Concern | Behaviour |
|---|---|
| Configuration files | Parsed with `yaml.safe_load` and validated against a strict schema (unknown keys are errors). YAML tags such as `!!python/object` are rejected. Configuration is data; it is never `eval`'d or executed. |
| Model references | The `model:` key names Python code to run (`file.py[:attr]`, `package.module:attr` or `builtin:<name>`). Running a model *is* running code you chose, exactly like pointing `pytest` at a test file. File references in a config must resolve inside the config's directory unless you pass `--allow-outside`, which blocks `../../` path traversal. Only reference model files you trust. |
| Distributions in config | Built only from a fixed registry (`{"distribution": "exponential", ...}`); no dynamic imports. |
| Deserialisation | Results are loaded from JSON only. No `pickle` is ever read from disk. (Pickle is used in memory to send models to local worker processes.) |
| Shell / subprocesses | SimulSI never runs shell commands. Provenance capture calls `git rev-parse HEAD` / `git status` with a fixed argument list, no shell, and a timeout. |
| Network & telemetry | None. Simulations, results and logs stay on your machine; nothing is uploaded, there is no telemetry and no API key is needed. |
| Output paths | `output.directory` in a config must resolve inside the config's directory (choose any other location explicitly with `-o`). The dashboard serves static files only from its own package directory and loads results only from paths given on the command line. |
| Web dashboard | `simulsi ui` binds to `127.0.0.1` by default and runs only built-in or explicitly loaded models. It has no authentication: do not expose it on a network. |
| Resource exhaustion | Experiment sizes, worker counts and file sizes are bounded by validation; event logs can be capped with `max_log_records`. A model that schedules events forever is still your code's responsibility - always give runs a `duration`. |
