from __future__ import annotations

import os
from collections import defaultdict
from typing import Optional

from ..models import MeasurementRecord
from .common import (
    aggregate_value,
    error_bounds,
    label_with_algorithm,
    metric_title_suffix,
    plt,
    select_energy_joules,
    style_axes,
)


def plot(
    records: list[MeasurementRecord], out_path: str, agg: str, error_bars: str, metric: str = "total"
) -> Optional[list[str]]:
    grouped: dict[int, dict[str, dict[int, list[float]]]] = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    algorithms = {rec.algorithm for rec in records if rec.algorithm}
    include_algorithm = len(algorithms) > 1
    for rec in records:
        if rec.k is None or rec.n is None or rec.elapsed_seconds is None:
            continue
        energy = select_energy_joules(rec, metric)
        if energy is None or energy <= 0 or rec.elapsed_seconds <= 0:
            continue
        base_label = rec.backend if rec.backend else rec.source
        label = label_with_algorithm(base_label, rec.algorithm, include_algorithm)
        grouped[rec.k][label][rec.n].append(energy * rec.elapsed_seconds)

    if not grouped:
        return None

    base_dir = os.path.dirname(out_path) or "."
    base_name, ext = os.path.splitext(os.path.basename(out_path))
    outputs: list[str] = []

    for k, label_map in sorted(grouped.items()):
        fig, ax = plt.subplots(figsize=(10, 6))
        for label, n_map in sorted(label_map.items()):
            xs = sorted(n_map.keys())
            ys, lowers, uppers = [], [], []
            for n in xs:
                center = aggregate_value(n_map[n], agg)
                lo, hi = error_bounds(n_map[n], center, error_bars)
                ys.append(center)
                lowers.append(lo)
                uppers.append(hi)
            (line,) = ax.plot(xs, ys, marker="o", linewidth=2, label=label)
            if error_bars != "none":
                ax.fill_between(xs, lowers, uppers, alpha=0.15, color=line.get_color())

        ax.set_xscale("log", base=2)
        ax.set_yscale("log")
        style_axes(
            ax,
            f"EDP vs Input Size (K = {k})" + metric_title_suffix(metric),
            "N (log2 scale)",
            "Energy-Delay Product (J*s, log scale)",
        )
        ax.legend()
        fig.tight_layout()

        os.makedirs(base_dir, exist_ok=True)
        out_file = os.path.join(base_dir, f"{base_name}_k{k}{ext}")
        fig.savefig(out_file, dpi=160)
        plt.close(fig)
        outputs.append(out_file)

    return outputs
