#!/usr/bin/env bash
# Run the same checks as CI: format check, lint, type check, tests.
set -euo pipefail
cd "$(dirname "$0")/.."
PY=${PY:-.venv/bin}
"$PY/ruff" format --check .
"$PY/ruff" check .
"$PY/mypy"
"$PY/pytest" "$@"
