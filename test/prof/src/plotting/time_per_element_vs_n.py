from __future__ import annotations

import os
from collections import defaultdict
from typing import Optional

from ..models import CaseRecord
from .common import aggregate_value, error_bounds, plt, style_axes


def plot(records: list[CaseRecord], out_path: str, agg: str, error_bars: str) -> Optional[str]:
    grouped: dict[str, dict[int, list[float]]] = defaultdict(lambda: defaultdict(list))
    for rec in records:
        if rec.n is None or rec.n <= 0 or rec.time_ms is None or rec.time_ms < 0:
            continue
        t_per_elem = rec.time_ms / float(rec.n)
        grouped[rec.backend][rec.n].append(t_per_elem)

    if not grouped:
        return None

    fig, ax = plt.subplots(figsize=(10, 6))
    for label, n_map in sorted(grouped.items()):
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
        ax.plot(xs, ys, marker="o", linewidth=2, label=label)
        if error_bars != "none":
            ax.fill_between(xs, lowers, uppers, alpha=0.15)

    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    style_axes(ax, "Time per Element vs Input Size", "N (log2 scale)", "Time / element (ms)")
    ax.legend()
    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
    return out_path
