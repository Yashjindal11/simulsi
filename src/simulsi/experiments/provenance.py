"""Provenance: what produced a result (code version, environment, time)."""

from __future__ import annotations

import os
import platform
import shutil
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from simulsi._version import __version__


def git_commit(cwd: str | Path | None = None) -> dict[str, Any] | None:
    """Current git commit and dirty flag, or ``None`` outside a repository."""
    git = shutil.which("git")
    if git is None:
        return None
    try:
        sha = subprocess.run(  # noqa: S603 - fixed argv, no shell
            [git, "rev-parse", "HEAD"],
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=5,
            check=True,
        ).stdout.strip()
        dirty = subprocess.run(  # noqa: S603
            [git, "status", "--porcelain", "--untracked-files=no"],
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=5,
            check=True,
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return None
    return {"commit": sha, "dirty": bool(dirty)}


def environment() -> dict[str, Any]:
    import numpy
    import scipy

    return {
        "simulsi": __version__,
        "python": sys.version.split()[0],
        "implementation": platform.python_implementation(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "cpu_count": os.cpu_count(),
        "numpy": numpy.__version__,
        "scipy": scipy.__version__,
    }


def utc_now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")
