from __future__ import annotations

import os
from collections import defaultdict
from typing import Optional

from ..models import MeasurementRecord
from .common import aggregate_value, plt, style_axes


def plot(records: list[MeasurementRecord], out_path: str, agg: str, compare_n: Optional[int]) -> Optional[str]:
    energy_by_source: dict[str, list[float]] = defaultdict(list)
    for rec in records:
        if compare_n is not None and rec.n != compare_n:
            continue
        if rec.energy_joules is not None and rec.energy_joules >= 0:
            energy_by_source[rec.source].append(rec.energy_joules)

    if not energy_by_source:
        return None

    sources = sorted(energy_by_source.keys())
    ys = [aggregate_value(energy_by_source[s], agg) for s in sources]

    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.bar(sources, ys)
    for bar, y in zip(bars, ys):
        ax.text(bar.get_x() + bar.get_width() / 2, y, f"{y:.3f}", ha="center", va="bottom")

    style_axes(ax, "Average Energy by Source", "Source", "Energy (J)")
    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
    return out_path
