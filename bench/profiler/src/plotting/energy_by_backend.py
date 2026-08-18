from __future__ import annotations

import os
from collections import defaultdict
from typing import Optional

from ..models import MeasurementRecord
from .common import (
    save_k_figure,
    aggregate_value,
    error_bounds,
    label_with_algorithm,
    metric_title_suffix,
    plt,
    select_energy_joules,
    style_axes,
)


def plot(
    records: list[MeasurementRecord],
    out_path: str,
    agg: str,
    compare_n: Optional[int],
    error_bars: str,
    metric: str = "total",
) -> Optional[list[str]]:
    grouped: dict[int, dict[str, dict[int, list[float]]]] = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    algorithms = {rec.algorithm for rec in records if rec.algorithm}
    include_algorithm = len(algorithms) > 1
    for rec in records:
        if rec.k is None or rec.n is None:
            continue
        energy = select_energy_joules(rec, metric)
        if energy is None or energy <= 0:
            continue
        base_label = rec.backend if rec.backend else rec.source
        label = label_with_algorithm(base_label, rec.algorithm, include_algorithm)
        grouped[rec.k][label][rec.n].append(energy)

    if not grouped:
        return None

    base_dir = os.path.dirname(out_path) or "."
    base_name, ext = os.path.splitext(os.path.basename(out_path))
    outputs: list[str] = []

    for k, label_map in sorted(grouped.items()):
        all_ns = {n for n_map in label_map.values() for n in n_map}
        target_n = compare_n if (compare_n is not None and compare_n in all_ns) else max(all_ns)

        labels, ys, yerr_low, yerr_high = [], [], [], []
        for label in sorted(label_map.keys()):
            vals = label_map[label].get(target_n)
            if not vals:
                continue
            center = aggregate_value(vals, agg)
            lo, hi = error_bounds(vals, center, error_bars)
            labels.append(label)
            ys.append(center)
            yerr_low.append(max(0.0, center - lo))
            yerr_high.append(max(0.0, hi - center))

        if not labels:
            continue

        fig, ax = plt.subplots(figsize=(10, 5))
        bars = ax.bar(labels, ys, yerr=[yerr_low, yerr_high] if error_bars != "none" else None, capsize=4)
        for bar, y in zip(bars, ys):
            ax.text(bar.get_x() + bar.get_width() / 2, y, f"{y:.3f}", ha="center", va="bottom")

        style_axes(
            ax,
            f"End-to-end Energy per op by Backend (K={k}, N={target_n})" + metric_title_suffix(metric),
            "Backend",
            "Energy per op (J)",
        )
        fig.tight_layout()
        outputs.append(save_k_figure(fig, base_dir, base_name, k, ext))

    return outputs
