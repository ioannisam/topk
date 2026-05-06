from __future__ import annotations

import math
import os
from collections import defaultdict
from typing import Optional

from ..models import CaseRecord
from .common import aggregate_value, label_with_algorithm, plt, select_time_ms, style_axes


def plot(records: list[CaseRecord], out_path: str, agg: str) -> Optional[list[str]]:
    gt_algo_lists: dict[int, dict[tuple[str, str, Optional[int], str], list[float]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for rec in records:
        if rec.backend != "gt" or rec.k is None:
            continue
        t_algo = select_time_ms(rec, "algorithmic")
        if t_algo is not None and t_algo > 0:
            gt_algo_lists[rec.k][(rec.dtype, rec.mode, rec.n, rec.case_name)].append(t_algo)

    gt_algo: dict[int, dict[tuple[str, str, Optional[int], str], float]] = defaultdict(dict)
    for k, config_map in gt_algo_lists.items():
        for config_tuple, times in config_map.items():
            gt_algo[k][config_tuple] = aggregate_value(times, agg)

    series: dict[int, dict[str, dict[int, list[float]]]] = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))

    algorithms = {rec.algorithm for rec in records if rec.algorithm}
    include_algorithm = len(algorithms) > 1

    for rec in records:
        if rec.backend == "gt" or rec.n is None or rec.k is None:
            continue

        gt_t_algo = gt_algo.get(rec.k, {}).get((rec.dtype, rec.mode, rec.n, rec.case_name))
        t_algo = select_time_ms(rec, "algorithmic")

        if gt_t_algo is not None and t_algo is not None and t_algo > 0:
            speedup = gt_t_algo / t_algo
            if math.isfinite(speedup) and speedup > 0:
                label = label_with_algorithm(rec.backend, rec.algorithm, include_algorithm)
                series[rec.k][label][rec.n].append(speedup)

    if not series:
        return None

    base_dir = os.path.dirname(out_path) or "."
    base_name, ext = os.path.splitext(os.path.basename(out_path))
    outputs: list[str] = []

    for k, label_map in sorted(series.items()):
        fig, ax = plt.subplots(figsize=(10, 6))
        has_data = False

        for label, n_map in sorted(label_map.items()):
            xs = sorted(n_map.keys())
            ys = [aggregate_value(n_map[n], agg) for n in xs]
            if xs:
                has_data = True
                ax.plot(xs, ys, marker="o", linewidth=2, label=label)

        if not has_data:
            plt.close(fig)
            continue

        ax.axhline(1.0, color="gray", linestyle="--", linewidth=1)
        ax.set_xscale("log", base=2)
        style_axes(ax, f"Algorithmic Speedup vs GT (K = {k})", "N (log2 scale)", "Speedup")
        ax.legend(title="Configuration", loc="upper left")
        fig.tight_layout()

        os.makedirs(base_dir, exist_ok=True)
        out_file = os.path.join(base_dir, f"{base_name}_k{k}{ext}")
        fig.savefig(out_file, dpi=160, bbox_inches="tight")
        plt.close(fig)
        outputs.append(out_file)

    return outputs
