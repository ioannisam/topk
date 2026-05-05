from __future__ import annotations

import os
from collections import defaultdict
from typing import Optional

from ..models import MeasurementRecord
from .common import aggregate_value, error_bounds, label_with_algorithm, plt, style_axes


def plot(
    records: list[MeasurementRecord], out_path: str, agg: str, compare_n: Optional[int], error_bars: str
) -> Optional[str]:
    grouped: dict[str, list[float]] = defaultdict(list)
    algorithms = {rec.algorithm for rec in records if rec.algorithm}
    include_algorithm = len(algorithms) > 1
    for rec in records:
        if rec.energy_joules is None or rec.energy_joules < 0:
            continue
        if compare_n is not None and rec.n != compare_n:
            continue
        base_label = rec.backend if rec.backend else rec.source
        label = label_with_algorithm(base_label, rec.algorithm, include_algorithm)
        grouped[label].append(rec.energy_joules)

    if not grouped:
        return None

    labels = sorted(grouped.keys())
    ys = []
    yerr_low = []
    yerr_high = []
    for label in labels:
        center = aggregate_value(grouped[label], agg)
        lo, hi = error_bounds(grouped[label], center, error_bars)
        ys.append(center)
        yerr_low.append(max(0.0, center - lo))
        yerr_high.append(max(0.0, hi - center))

    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(labels, ys, yerr=[yerr_low, yerr_high] if error_bars != "none" else None, capsize=4)
    for bar, y in zip(bars, ys):
        ax.text(bar.get_x() + bar.get_width() / 2, y, f"{y:.3f}", ha="center", va="bottom")

    title = "Energy by Backend"
    if compare_n is not None:
        title += f" (N={compare_n})"
    style_axes(ax, title, "Backend", "Energy (J)")
    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
    return out_path
