from __future__ import annotations

import math
import os
from collections import defaultdict
from typing import Optional

from ..models import CaseRecord
from .common import aggregate_value, plt, select_time_ms, style_axes


def plot(records: list[CaseRecord], out_path: str, agg: str) -> Optional[list[str]]:
    # gt_algo_lists: gt_backend -> k -> (dtype, mode, dist, n) -> list[time]
    gt_algo_lists: dict[str, dict[int, dict[tuple[str, str, str, int], list[float]]]] = defaultdict(
        lambda: defaultdict(lambda: defaultdict(list))
    )
    for rec in records:
        if rec.algorithm != "gt" or rec.k is None or rec.n is None:
            continue
        t_algo = select_time_ms(rec, "algorithmic")
        if t_algo is not None and t_algo > 0:
            gt_algo_lists[rec.backend][rec.k][(rec.dtype, rec.mode, rec.dist, rec.n)].append(t_algo)

    gt_algo: dict[str, dict[int, dict[tuple[str, str, str, int], float]]] = defaultdict(lambda: defaultdict(dict))
    for b, k_map in gt_algo_lists.items():
        for k, config_map in k_map.items():
            for config_tuple, times in config_map.items():
                gt_algo[b][k][config_tuple] = aggregate_value(times, agg)

    # series: gt_backend -> k -> "backend-algorithm" label -> n -> list[speedup]
    series: dict[str, dict[int, dict[str, dict[int, list[float]]]]] = defaultdict(
        lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    )

    for rec in records:
        if rec.algorithm == "gt" or rec.n is None or rec.k is None:
            continue
        t_algo = select_time_ms(rec, "algorithmic")
        if t_algo is None or t_algo <= 0:
            continue

        config = (rec.dtype, rec.mode, rec.dist, rec.n)
        label = f"{rec.backend}-{rec.algorithm}"
        for gt_backend, k_map in gt_algo.items():
            gt_t_algo = k_map.get(rec.k, {}).get(config)
            if gt_t_algo is None:
                continue
            speedup = gt_t_algo / t_algo
            if math.isfinite(speedup) and speedup > 0:
                series[gt_backend][rec.k][label][rec.n].append(speedup)

    if not series:
        return None

    base_dir = os.path.dirname(out_path) or "."
    base_name, ext = os.path.splitext(os.path.basename(out_path))
    outputs: list[str] = []

    for gt_backend, k_map in sorted(series.items()):
        for k, label_map in sorted(k_map.items()):
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
            style_axes(
                ax,
                f"Cross-Backend Speedup vs {gt_backend.upper()} GT (K = {k})",
                "N (log2 scale)",
                "Speedup",
            )
            ax.legend(title="Backend-Algorithm", loc="upper left")
            fig.tight_layout()

            os.makedirs(base_dir, exist_ok=True)
            out_file = os.path.join(base_dir, f"{base_name}_{gt_backend}_k{k}{ext}")
            fig.savefig(out_file, dpi=160, bbox_inches="tight")
            plt.close(fig)
            outputs.append(out_file)

    return outputs
