from __future__ import annotations

import os
from collections import defaultdict
from typing import Iterable, Optional

from ..models import CaseRecord
from .common import aggregate_value, error_bounds, plt, select_time_ms, style_axes
from .roofline import backend_color


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
    ceilings: Optional[dict[str, float]] = None,
    metric: str = "algorithmic",
) -> Optional[str]:
    # Bandwidth is compared against device-local ceilings, so the algorithmic window is the
    # correct basis: end-to-end would charge GPU/NPU runs for host transfers the roofline
    # microbenchmark never performs.
    grouped: dict[tuple[str, str], dict[int, list[float]]] = defaultdict(lambda: defaultdict(list))
    for r in records:
        time_ms = select_time_ms(r, metric)
        if r.n is None or time_ms is None or time_ms <= 0:
            continue
        bpe = get_bytes_per_element(r.dtype)
        bw_gbps = (r.n * bpe) / (time_ms * 1e6)
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

    plotted_backends = {backend for backend, _ in grouped.keys()}
    for backend, peak in sorted((ceilings or {}).items()):
        if backend not in plotted_backends or peak <= 0:
            continue
        ax.axhline(
            peak,
            linestyle=(0, (6, 4)),
            linewidth=1.8,
            color=backend_color(backend),
            alpha=0.9,
            label=f"{backend.upper()} measured peak ({peak:.0f} GB/s)",
        )

    ax.set_xscale("log", base=2)
    basis = "algorithmic" if metric == "algorithmic" else "end-to-end"
    style_axes(ax, title, "Input Size N (elements)", f"Effective Bandwidth (GB/s, {basis})")
    ax.legend()
    fig.tight_layout()

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
    return out_path
