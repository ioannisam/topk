from __future__ import annotations

import os
from collections import defaultdict
from typing import Iterable, Optional

from ..models import CaseRecord, RooflinePoint
from .common import aggregate_value, error_bounds, plt, select_time_ms, style_axes
from .roofline import backend_color, bandwidth_ladders, roof_at


def get_bytes_per_element(dtype: str) -> int:
    d = dtype.lower()
    if "double" in d:
        return 8
    if "half" in d:
        return 2
    return 4


def plot(
    records: Iterable[CaseRecord],
    out_path: str,
    agg: str = "mean",
    error_bars: str = "p10-p90",
    title: str = "Effective Memory Bandwidth vs. Input Size (N)",
    roofline_points: Optional[Iterable[RooflinePoint]] = None,
    metric: str = "algorithmic",
) -> Optional[str]:
    # Bandwidth is compared against device-local ceilings, so the algorithmic window is the
    # correct basis: end-to-end would charge GPU/NPU runs for host transfers the roofline
    # microbenchmark never performs.
    #
    # The numerator is the traffic the backend reports actually moving, not n*sizeof(T).
    # A truncated bitonic network rereads its working set about seven times, so charging it a
    # single pass understated its bandwidth by the same factor and made every implementation
    # look far from the wall.
    grouped: dict[tuple[str, str], dict[int, list[float]]] = defaultdict(lambda: defaultdict(list))
    working_set: dict[tuple[str, str], dict[int, float]] = defaultdict(dict)
    missing_traffic: set[tuple[str, str, str, int, int]] = set()
    for r in records:
        time_ms = select_time_ms(r, metric)
        if r.n is None or time_ms is None or time_ms <= 0:
            continue
        if not r.bytes_moved or r.bytes_moved <= 0:
            missing_traffic.add((r.backend, r.dtype, r.algorithm, r.n, r.k))
            continue
        bw_gbps = r.bytes_moved / (time_ms * 1e6)
        grouped[(r.backend, r.algorithm)][r.n].append(bw_gbps)
        working_set[(r.backend, r.algorithm)][r.n] = r.n * get_bytes_per_element(r.dtype)

    if missing_traffic:
        cases = ", ".join(f"{b}/{d}/{a} n={n} k={k}" for b, d, a, n, k in sorted(missing_traffic))
        print(f"Warning: skipped {len(missing_traffic)} case(s) reporting zero bytes moved: {cases}")
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

    # The roof is a function of working-set size, not a constant: a run that fits in L2 is
    # nowhere near the DRAM wall, so a flat line would be the wrong comparison for most of
    # this sweep.
    ladders = bandwidth_ladders(roofline_points or [])
    plotted_backends = {backend for backend, _ in grouped.keys()}
    for backend in sorted(plotted_backends):
        ladder = ladders.get(backend)
        if not ladder:
            continue
        sizes = sorted({n for key, n_map in working_set.items() if key[0] == backend for n in n_map})
        if not sizes:
            continue
        roof_xs, roof_ys = [], []
        for n in sizes:
            bytes_for_n = next(
                (n_map[n] for key, n_map in working_set.items() if key[0] == backend and n in n_map), None
            )
            roof = roof_at(ladder, bytes_for_n) if bytes_for_n else None
            if roof is not None:
                roof_xs.append(n)
                roof_ys.append(roof)
        if roof_xs:
            ax.plot(
                roof_xs,
                roof_ys,
                linestyle=(0, (6, 4)),
                linewidth=1.8,
                color=backend_color(backend),
                alpha=0.9,
                label=f"{backend.upper()} measured roof at this working set",
            )

    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    basis = "algorithmic" if metric == "algorithmic" else "end-to-end"
    style_axes(ax, title, "Input Size N (elements)", f"Effective Bandwidth (GB/s, {basis})")
    ax.legend(fontsize=8)
    fig.tight_layout()

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
    return out_path
