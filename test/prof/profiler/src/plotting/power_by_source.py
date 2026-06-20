from __future__ import annotations

import os
from collections import defaultdict
from typing import Optional

from ..models import MeasurementRecord
from .common import aggregate_value, metric_title_suffix, plt, select_power_watts, style_axes


def plot(
    records: list[MeasurementRecord], out_path: str, agg: str, compare_n: Optional[int], metric: str = "total"
) -> Optional[str]:
    power_by_source: dict[str, list[float]] = defaultdict(list)
    for rec in records:
        if compare_n is not None and rec.n != compare_n:
            continue
        watts = select_power_watts(rec, metric)
        if watts is not None and watts >= 0:
            power_by_source[rec.source].append(watts)

    if not power_by_source:
        return None

    sources = sorted(power_by_source.keys())
    ys = [aggregate_value(power_by_source[s], agg) for s in sources]

    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.bar(sources, ys)
    for bar, y in zip(bars, ys):
        ax.text(bar.get_x() + bar.get_width() / 2, y, f"{y:.3f}", ha="center", va="bottom")

    style_axes(ax, "Average Power by Source" + metric_title_suffix(metric), "Source", "Power (W)")
    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
    return out_path
