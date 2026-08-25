from __future__ import annotations
import os
from collections import defaultdict
from typing import Optional
from ..models import CaseRecord, MeasurementRecord
from .common import (
    aggregate_value,
    label_with_algorithm,
    metric_title_suffix,
    plt,
    select_energy_joules,
    select_time_ms,
    style_axes,
)


def plot(
    case_records: list[CaseRecord],
    measurement_records: list[MeasurementRecord],
    out_path: str,
    agg: str,
    compare_n: Optional[int],
    metric: str = "total",
) -> Optional[str]:
    time_map: dict[tuple[str, str, str, Optional[int], Optional[int], str], list[float]] = defaultdict(list)
    for rec in case_records:
        if rec.n is None or (compare_n is not None and rec.n != compare_n):
            continue
        key = (rec.backend, rec.dtype, rec.mode, rec.k, rec.n, rec.algorithm)
        t = select_time_ms(rec, "e2e")
        if t is not None:
            time_map[key].append(t)

    energy_map: dict[tuple[str, str, str, Optional[int], Optional[int], str], list[float]] = defaultdict(list)
    for rec in measurement_records:
        energy = select_energy_joules(rec, metric)
        if energy is None or energy <= 0 or rec.n is None or (compare_n is not None and rec.n != compare_n):
            continue
        backend = rec.backend if rec.backend else rec.source
        key = (backend, rec.dtype, rec.mode, rec.k, rec.n, rec.algorithm)
        energy_map[key].append(energy)

    points: dict[str, list[tuple[float, float]]] = defaultdict(list)
    algorithms = {rec.algorithm for rec in case_records if rec.algorithm} | {
        rec.algorithm for rec in measurement_records if rec.algorithm
    }
    include_algorithm = len(algorithms) > 1

    for key, t_list in time_map.items():
        e_list = energy_map.get(key)
        if not e_list:
            continue
        t_ms, e_j = aggregate_value(t_list, agg), aggregate_value(e_list, agg)
        if t_ms > 0 and e_j > 0:
            label = label_with_algorithm(key[0], key[5], include_algorithm)
            points[label].append((t_ms, e_j))

    if not points:
        return None

    fig, ax = plt.subplots(figsize=(10, 6))
    for label, pts in sorted(points.items()):
        xs, ys = [p[0] for p in pts], [p[1] for p in pts]
        ax.scatter(xs, ys, s=55, alpha=0.85, marker="o", label=label)

    ax.set_xscale("log")
    ax.set_yscale("log")
    title = "End-to-end Time vs Energy (Pareto View)" + metric_title_suffix(metric)
    if compare_n is not None:
        title += f" (N={compare_n})"
    style_axes(ax, title, "End-to-end time per op (ms)", f"End-to-end energy per op (J){metric_title_suffix(metric)}")
    ax.legend(title="Configuration", loc="best")
    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
    return out_path
