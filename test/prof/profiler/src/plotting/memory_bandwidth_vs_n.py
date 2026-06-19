from __future__ import annotations

import os
from collections import defaultdict
from typing import Iterable, Optional

from ..models import CaseRecord
from .common import aggregate_value, error_bounds, plt, style_axes


def get_bytes_per_element(dtype: str) -> int:
    d = dtype.lower()
    if "double" in d:
        return 8
    if "fp16" in d:
        return 2
    return 4


def plot(
    records: Iterable[CaseRecord],
    out_path: str,
    agg: str = "mean",
    error_bars: str = "p10-p90",
    title: str = "Effective Memory Bandwidth vs. Input Size (N)",
) -> Optional[str]:
    grouped: dict[tuple[str, str], dict[int, list[float]]] = defaultdict(lambda: defaultdict(list))
    for r in records:
        if r.n is None or r.time_ms is None or r.time_ms <= 0:
            continue
        bpe = get_bytes_per_element(r.dtype)
        bw_gbps = (r.n * bpe) / (r.time_ms * 1e6)
        grouped[(r.backend, r.algorithm)][r.n].append(bw_gbps)

    if not grouped:
        return None

    fig, ax = plt.subplots(figsize=(10, 6))
    for (backend, algo), n_map in sorted(grouped.items()):
        xs = sorted(n_map.keys())
        ys, lowers, uppers = [], [], []
        for n in xs:
            center = aggregate_value(n_map[n], agg)
            lo, hi = error_bounds(n_map[n], center, error_bars)
            ys.append(center)
            lowers.append(lo)
            uppers.append(hi)
        (line,) = ax.plot(xs, ys, marker="o", linewidth=2, label=f"{backend} - {algo}")
        if error_bars != "none":
            ax.fill_between(xs, lowers, uppers, alpha=0.15, color=line.get_color())

    ax.set_xscale("log", base=2)
    style_axes(ax, title, "Input Size N (elements)", "Effective Bandwidth (GB/s)")
    ax.legend()
    fig.tight_layout()

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
    return out_path
