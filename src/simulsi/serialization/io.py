"""Safe, dependency-light export and import of results.

* JSON is always available. NaN and infinities are written as ``null`` so
  files stay valid JSON for other tools (browsers reject bare ``NaN``).
* CSV uses the standard library.
* Parquet needs the optional ``pyarrow`` + ``pandas`` extras.

Loading never uses ``pickle`` or ``eval``: only JSON is read back.
"""

from __future__ import annotations

import csv
import json
import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

import numpy as np


def to_jsonable(obj: Any) -> Any:
    """Convert numpy scalars/arrays, dataclasses, tuples and non-finite floats for JSON."""
    if isinstance(obj, float | np.floating):
        f = float(obj)
        return f if math.isfinite(f) else None
    if isinstance(obj, bool | np.bool_):
        return bool(obj)
    if isinstance(obj, int | np.integer):
        return int(obj)
    if isinstance(obj, str) or obj is None:
        return obj
    if isinstance(obj, np.ndarray):
        return [to_jsonable(x) for x in obj.tolist()]
    if is_dataclass(obj) and not isinstance(obj, type):
        return to_jsonable(asdict(obj))
    if hasattr(obj, "to_dict"):
        return to_jsonable(obj.to_dict())
    if isinstance(obj, Mapping):
        return {str(k): to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, Iterable):
        return [to_jsonable(x) for x in obj]
    return repr(obj)


def nan_if_none(value: Any) -> float:
    return math.nan if value is None else float(value)


def write_json(data: Any, path: str | Path, *, indent: int | None = 2) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as fh:
        json.dump(to_jsonable(data), fh, indent=indent, allow_nan=False)
        fh.write("\n")
    return p


def read_json(path: str | Path) -> Any:
    with Path(path).open(encoding="utf-8") as fh:
        return json.load(fh)


def write_csv(rows: Sequence[Mapping[str, Any]], path: str | Path) -> Path:
    """Write dict rows; the header is the union of keys in first-seen order."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    columns: dict[str, None] = {}
    for row in rows:
        for k in row:
            columns.setdefault(k, None)
    with p.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(columns), extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({k: _csv_cell(v) for k, v in row.items()})
    return p


def _csv_cell(v: Any) -> Any:
    if isinstance(v, float) and not math.isfinite(v):
        return ""
    if isinstance(v, Mapping | list | tuple):
        return json.dumps(to_jsonable(v), allow_nan=False)
    return v


def read_csv(path: str | Path) -> list[dict[str, str]]:
    with Path(path).open(encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


def write_parquet(rows: Sequence[Mapping[str, Any]], path: str | Path) -> Path:
    try:
        import pandas as pd
    except ImportError as exc:  # pragma: no cover - depends on optional extras
        raise ImportError("Parquet export needs: pip install 'simulsi[parquet]'") from exc
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    flat = [
        {k: _csv_cell(v) if isinstance(v, Mapping | list | tuple) else v for k, v in r.items()}
        for r in rows
    ]
    try:
        pd.DataFrame(flat).to_parquet(p, index=False)
    except ImportError as exc:  # pragma: no cover
        raise ImportError("Parquet export needs: pip install 'simulsi[parquet]'") from exc
    return p


def safe_child(base: str | Path, name: str) -> Path:
    """Resolve ``base/name`` and refuse paths that escape ``base`` (path traversal)."""
    root = Path(base).resolve()
    target = (root / name).resolve()
    if target != root and root not in target.parents:
        raise ValueError(f"path {name!r} escapes {str(root)!r}")
    return target
