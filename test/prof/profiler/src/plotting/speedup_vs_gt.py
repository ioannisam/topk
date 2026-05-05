from __future__ import annotations

import math
import os
from collections import defaultdict
from typing import Optional

from ..models import CaseRecord
from .common import plt, style_axes


def plot(records: list[CaseRecord], out_path: str) -> Optional[str]:
    gt_map: dict[tuple[str, str, Optional[int], Optional[int], str], float] = {}
    others: dict[str, dict[int, list[float]]] = defaultdict(lambda: defaultdict(list))

    for rec in records:
        if rec.time_ms is None:
            continue
        key = (rec.dtype, rec.mode, rec.k, rec.n, rec.case_name)
        if rec.backend == "gt":
            gt_map[key] = rec.time_ms

    for rec in records:
        if rec.backend == "gt" or rec.time_ms is None or rec.n is None:
            continue
        key = (rec.dtype, rec.mode, rec.k, rec.n, rec.case_name)
        gt_time = gt_map.get(key)
        if gt_time is None or rec.time_ms <= 0:
            continue
        speedup = gt_time / rec.time_ms
        if math.isfinite(speedup) and speedup > 0:
            others[rec.backend][rec.n].append(speedup)

    if not others:
        return None

    fig, ax = plt.subplots(figsize=(10, 6))
    for backend, n_map in sorted(others.items()):
        xs = sorted(n_map.keys())
        ys = [sum(n_map[n]) / len(n_map[n]) for n in xs]
        ax.plot(xs, ys, marker="o", linewidth=2, label=backend)

    ax.axhline(1.0, color="gray", linestyle="--", linewidth=1)
    ax.set_xscale("log", base=2)
    style_axes(ax, "Speedup vs GT (gt_time / backend_time)", "N (log2 scale)", "Speedup")
    ax.legend()
    fig.tight_layout()

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
    return out_path
