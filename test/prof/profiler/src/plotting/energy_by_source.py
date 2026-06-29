from __future__ import annotations

import os
from collections import defaultdict
from typing import Optional

from ..models import MeasurementRecord
from .common import aggregate_value, metric_title_suffix, plt, select_energy_joules, style_axes


def plot(
    records: list[MeasurementRecord], out_path: str, agg: str, compare_n: Optional[int], metric: str = "total"
) -> Optional[str]:
    energy_by_source: dict[str, list[float]] = defaultdict(list)
    for rec in records:
        if compare_n is not None and rec.n != compare_n:
            continue
        energy = select_energy_joules(rec, metric)
        if energy is not None and energy >= 0:
            energy_by_source[rec.source].append(energy)

    if not energy_by_source:
        return None

    sources = sorted(energy_by_source.keys())
    ys = [aggregate_value(energy_by_source[s], agg) for s in sources]

    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.bar(sources, ys)
    for bar, y in zip(bars, ys):
        ax.text(bar.get_x() + bar.get_width() / 2, y, f"{y:.3f}", ha="center", va="bottom")

    style_axes(ax, "End-to-end Energy per op by Source" + metric_title_suffix(metric), "Source", "Energy per op (J)")
    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
    return out_path
