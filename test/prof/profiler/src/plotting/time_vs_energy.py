from __future__ import annotations

import os
from collections import defaultdict
from typing import Optional

from ..models import CaseRecord, MeasurementRecord
from .common import add_time_metric_legend, aggregate_value, label_with_algorithm, plt, select_time_ms, style_axes


def plot(
    case_records: list[CaseRecord],
    measurement_records: list[MeasurementRecord],
    out_path: str,
    agg: str,
    compare_n: Optional[int],
) -> Optional[str]:
    time_map: dict[tuple[str, str, str, Optional[int], Optional[int], str], dict[str, list[float]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for rec in case_records:
        if rec.n is None:
            continue
        if compare_n is not None and rec.n != compare_n:
            continue
        key = (rec.backend, rec.dtype, rec.mode, rec.k, rec.n, rec.algorithm)
        for metric in ("algorithmic", "end-to-end"):
            time_ms = select_time_ms(rec, metric)
            if time_ms is None:
                continue
            time_map[key][metric].append(time_ms)

    energy_map: dict[tuple[str, str, str, Optional[int], Optional[int], str], list[float]] = defaultdict(list)
    for rec in measurement_records:
        if rec.energy_joules is None or rec.energy_joules < 0 or rec.n is None:
            continue
        if compare_n is not None and rec.n != compare_n:
            continue
        backend = rec.backend if rec.backend else rec.source
        key = (backend, rec.dtype, rec.mode, rec.k, rec.n, rec.algorithm)
        energy_map[key].append(rec.energy_joules)

    points_algo: dict[str, list[tuple[float, float]]] = defaultdict(list)
    points_e2e: dict[str, list[tuple[float, float]]] = defaultdict(list)
    algorithms = {rec.algorithm for rec in case_records if rec.algorithm}
    algorithms.update(rec.algorithm for rec in measurement_records if rec.algorithm)
    include_algorithm = len(algorithms) > 1
    for key, metric_times in time_map.items():
        energies = energy_map.get(key)
        if not energies:
            continue
        e_j = aggregate_value(energies, agg)
        if e_j <= 0:
            continue
        label = label_with_algorithm(key[0], key[5], include_algorithm)

        algo_times = metric_times.get("algorithmic", [])
        e2e_times = metric_times.get("end-to-end", [])
        if algo_times:
            t_ms = aggregate_value(algo_times, agg)
            if t_ms > 0:
                points_algo[label].append((t_ms, e_j))
        if e2e_times:
            t_ms = aggregate_value(e2e_times, agg)
            if t_ms > 0:
                points_e2e[label].append((t_ms, e_j))

    if not points_algo and not points_e2e:
        return None

    fig, ax = plt.subplots(figsize=(10, 6))
    # Plot algorithmic first (config legend uses these labels/colors)
    for label in sorted(set(points_algo.keys()) | set(points_e2e.keys())):
        color = None
        if label in points_algo:
            xs = [p[0] for p in points_algo[label]]
            ys = [p[1] for p in points_algo[label]]
            coll = ax.scatter(xs, ys, s=55, alpha=0.85, marker="o", label=label)
            try:
                color = coll.get_facecolor()[0]
            except Exception:
                color = None

        if label in points_e2e:
            xs = [p[0] for p in points_e2e[label]]
            ys = [p[1] for p in points_e2e[label]]
            ax.scatter(xs, ys, s=55, alpha=0.85, marker="x", color=color, label="_nolegend_")

    ax.set_xscale("log")
    ax.set_yscale("log")
    title = "Time vs Energy (Pareto View)"
    if compare_n is not None:
        title += f" (N={compare_n})"
    style_axes(ax, title, "Time (ms)", "Energy (J)")
    config_leg = ax.legend(title="Configuration", loc="best")
    ax.add_artist(config_leg)
    add_time_metric_legend(ax, loc="lower right", kind="marker")
    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
    return out_path
