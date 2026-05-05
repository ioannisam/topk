from __future__ import annotations

import os
from collections import defaultdict
from typing import Optional

from ..models import MeasurementRecord
from .common import aggregate_value, error_bounds, label_with_algorithm, plt, style_axes


def plot(records: list[MeasurementRecord], out_path: str, agg: str, error_bars: str) -> Optional[str]:
    grouped: dict[str, dict[int, list[float]]] = defaultdict(lambda: defaultdict(list))
    algorithms = {rec.algorithm for rec in records if rec.algorithm}
    include_algorithm = len(algorithms) > 1
    for rec in records:
        if rec.average_watts is None or rec.average_watts <= 0 or rec.n is None:
            continue
        base_label = rec.backend if rec.backend else rec.source
        label = label_with_algorithm(base_label, rec.algorithm, include_algorithm)
        grouped[label][rec.n].append(rec.average_watts)

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
    style_axes(ax, "Power vs Input Size", "N (log2 scale)", "Power (W, log scale)")
    ax.legend()
    fig.tight_layout()

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
    return out_path
