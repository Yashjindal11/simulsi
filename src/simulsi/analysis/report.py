"""Tables for terminal output (plain text) and notebooks (HTML)."""

from __future__ import annotations

import html
import math
from collections.abc import Mapping, Sequence
from typing import Any


def fmt_value(v: Any) -> str:
    if isinstance(v, bool):
        return str(v)
    if isinstance(v, int):
        return str(v)
    if isinstance(v, float):
        if math.isnan(v):
            return "-"
        if math.isinf(v):
            return "inf" if v > 0 else "-inf"
        a = abs(v)
        if a != 0 and (a >= 1e6 or a < 1e-3):
            return f"{v:.3e}"
        return f"{v:.4g}" if a < 1000 else f"{v:,.1f}"
    return str(v)


def format_table(rows: Sequence[Mapping[str, Any]], columns: Sequence[str] | None = None) -> str:
    if not rows:
        return "(no rows)"
    cols = list(columns) if columns is not None else list(rows[0].keys())
    cells = [[fmt_value(r.get(c, "")) for c in cols] for r in rows]
    widths = [max(len(c), *(len(row[i]) for row in cells)) for i, c in enumerate(cols)]
    numeric = [
        all(isinstance(r.get(c), int | float) and not isinstance(r.get(c), bool) for r in rows)
        for c in cols
    ]

    def line(values: Sequence[str]) -> str:
        return "  ".join(
            v.rjust(w) if num else v.ljust(w)
            for v, w, num in zip(values, widths, numeric, strict=True)
        ).rstrip()

    out = [line(cols), line(["-" * w for w in widths])]
    out += [line(row) for row in cells]
    return "\n".join(out)


def html_table(
    rows: Sequence[Mapping[str, Any]],
    columns: Sequence[str] | None = None,
    *,
    caption: str | None = None,
    max_rows: int = 200,
) -> str:
    """An HTML ``<table>`` (all text escaped), e.g. for Jupyter ``_repr_html_``."""
    if not rows:
        return "<p><em>(no rows)</em></p>"
    cols = list(columns) if columns is not None else list(rows[0].keys())
    esc = html.escape
    out = ['<table class="simulsi">']
    if caption:
        out.append(f"<caption>{esc(caption)}</caption>")
    out.append("<thead><tr>" + "".join(f"<th>{esc(str(c))}</th>" for c in cols) + "</tr></thead>")
    out.append("<tbody>")
    for r in rows[:max_rows]:
        cells = []
        for c in cols:
            v = r.get(c, "")
            num = isinstance(v, int | float) and not isinstance(v, bool)
            style = ' style="text-align:right"' if num else ""
            cells.append(f"<td{style}>{esc(fmt_value(v))}</td>")
        out.append("<tr>" + "".join(cells) + "</tr>")
    out.append("</tbody></table>")
    if len(rows) > max_rows:
        out.append(f"<p><em>{len(rows) - max_rows} more rows not shown</em></p>")
    return "".join(out)
