from __future__ import annotations

import os
from collections import defaultdict
from typing import Optional

from ..models import CaseRecord
from .common import aggregate_value, error_bounds, plt, style_axes


def plot(records: list[CaseRecord], out_path: str, agg: str, error_bars: str) -> Optional[str]:
    grouped: dict[str, dict[int, list[float]]] = defaultdict(lambda: defaultdict(list))
    for rec in records:
        if rec.n is None or rec.time_ms is None:
            continue
        grouped[rec.backend][rec.n].append(rec.time_ms)

    if not grouped:
        return None

    fig, ax = plt.subplots(figsize=(10, 6))
    for backend, n_map in sorted(grouped.items()):
        xs = sorted(n_map.keys())
        ys = []
        lowers = []
        uppers = []
        for n in xs:
            center = aggregate_value(n_map[n], agg)
            lo, hi = error_bounds(n_map[n], center, error_bars)
            ys.append(center)
            lowers.append(lo)
            uppers.append(hi)
        ax.plot(xs, ys, marker="o", linewidth=2, label=backend)
        if error_bars != "none":
            ax.fill_between(xs, lowers, uppers, alpha=0.15)

    ax.set_xscale("log", base=2)
    style_axes(ax, "Runtime vs Input Size", "N (log2 scale)", "Time (ms)")
    ax.legend()
    fig.tight_layout()

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
    return out_path
