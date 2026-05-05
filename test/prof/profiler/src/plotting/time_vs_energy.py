from __future__ import annotations

import os
from collections import defaultdict
from typing import Optional

from ..models import CaseRecord, MeasurementRecord
from .common import aggregate_value, plt, style_axes


def plot(
    case_records: list[CaseRecord],
    measurement_records: list[MeasurementRecord],
    out_path: str,
    agg: str,
    compare_n: Optional[int],
) -> Optional[str]:
    time_map: dict[tuple[str, str, str, Optional[int], Optional[int]], list[float]] = defaultdict(list)
    for rec in case_records:
        if rec.time_ms is None or rec.n is None:
            continue
        if compare_n is not None and rec.n != compare_n:
            continue
        key = (rec.backend, rec.dtype, rec.mode, rec.k, rec.n)
        time_map[key].append(rec.time_ms)

    energy_map: dict[tuple[str, str, str, Optional[int], Optional[int]], list[float]] = defaultdict(list)
    for rec in measurement_records:
        if rec.energy_joules is None or rec.energy_joules < 0 or rec.n is None:
            continue
        if compare_n is not None and rec.n != compare_n:
            continue
        backend = rec.backend if rec.backend else rec.source
        key = (backend, rec.dtype, rec.mode, rec.k, rec.n)
        energy_map[key].append(rec.energy_joules)

    points_by_backend: dict[str, list[tuple[float, float]]] = defaultdict(list)
    for key, times in time_map.items():
        energies = energy_map.get(key)
        if not energies:
            continue
        t_ms = aggregate_value(times, agg)
        e_j = aggregate_value(energies, agg)
        if t_ms > 0 and e_j > 0:
            points_by_backend[key[0]].append((t_ms, e_j))

    if not points_by_backend:
        return None

    fig, ax = plt.subplots(figsize=(10, 6))
    for backend, points in sorted(points_by_backend.items()):
        xs = [p[0] for p in points]
        ys = [p[1] for p in points]
        ax.scatter(xs, ys, s=55, alpha=0.8, label=backend)

    ax.set_xscale("log")
    ax.set_yscale("log")
    title = "Time vs Energy (Pareto View)"
    if compare_n is not None:
        title += f" (N={compare_n})"
    style_axes(ax, title, "Time (ms)", "Energy (J)")
    ax.legend()
    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
    return out_path
