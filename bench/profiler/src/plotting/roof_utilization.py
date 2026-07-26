from __future__ import annotations

import os
from collections import defaultdict
from typing import Iterable

from ..models import CaseRecord, RooflinePoint
from .common import aggregate_value, plt, select_time_ms, style_axes
from .memory_bandwidth_vs_n import get_bytes_per_element
from .roofline import bandwidth_ladders, compute_ceilings, roof_at

# A kernel this close to a roof is limited by it; the exact cut is a reporting convention, not
# a measurement, so it is drawn as a band rather than asserted per point.
BOUND_THRESHOLD_PCT = 80.0

ALGO_COLORS = {"bitonic": "#4269D0", "map_reduce": "#C2410C", "gt": "#059669"}


def plot(
    records: Iterable[CaseRecord],
    roofline_points: Iterable[RooflinePoint],
    out_dir: str,
    agg: str = "mean",
    metric: str = "algorithmic",
) -> list[str]:
    """Per backend, how much of each measured roof the kernels actually reach.

    Distance to the bandwidth roof and distance to the compare-exchange roof together classify
    the bound: near the memory roof is memory bound, near the compute roof is compute bound,
    and far from both means the limiter is latency, dispatch, or occupancy.
    """
    point_list = list(roofline_points)
    ladders = bandwidth_ladders(point_list)
    cmp_ceilings = compute_ceilings(point_list)
    if not ladders and not cmp_ceilings:
        return []

    mem: dict[tuple[str, str], dict[int, list[float]]] = defaultdict(lambda: defaultdict(list))
    compute: dict[tuple[str, str], dict[int, list[float]]] = defaultdict(lambda: defaultdict(list))

    for r in records:
        time_ms = select_time_ms(r, metric)
        if r.n is None or time_ms is None or time_ms <= 0 or not r.bytes_moved:
            continue

        ladder = ladders.get(r.backend)
        if ladder:
            roof = roof_at(ladder, r.n * get_bytes_per_element(r.dtype))
            if roof and roof > 0:
                achieved = r.bytes_moved / (time_ms * 1e6)
                mem[(r.backend, r.algorithm)][r.n].append(100.0 * achieved / roof)

        cmp_roof = cmp_ceilings.get(r.backend)
        if cmp_roof and cmp_roof > 0 and r.compare_ops:
            achieved = r.compare_ops / (time_ms * 1e6)
            compute[(r.backend, r.algorithm)][r.n].append(100.0 * achieved / cmp_roof)

    backends = sorted({b for b, _ in mem} | {b for b, _ in compute})
    written: list[str] = []

    for backend in backends:
        fig, ax = plt.subplots(figsize=(10, 6))

        for (b, algo), n_map in sorted(mem.items()):
            if b != backend:
                continue
            xs = sorted(n_map)
            ax.plot(
                xs,
                [aggregate_value(n_map[n], agg) for n in xs],
                marker="o",
                linewidth=2.4,
                color=ALGO_COLORS.get(algo, "#6B7280"),
                label=f"{algo} - % of memory roof",
            )

        for (b, algo), n_map in sorted(compute.items()):
            if b != backend:
                continue
            xs = sorted(n_map)
            ax.plot(
                xs,
                [aggregate_value(n_map[n], agg) for n in xs],
                marker="x",
                markersize=7,
                linestyle=(0, (6, 3)),
                linewidth=1.8,
                color=ALGO_COLORS.get(algo, "#6B7280"),
                label=f"{algo} - % of compute roof",
            )

        ax.axhspan(BOUND_THRESHOLD_PCT, 100.0, color="#DC2626", alpha=0.08)
        ax.axhline(BOUND_THRESHOLD_PCT, color="#DC2626", linestyle=":", linewidth=1.2, alpha=0.7)
        ax.annotate(
            f"at the roof (>{BOUND_THRESHOLD_PCT:.0f}%)",
            xy=(0.01, BOUND_THRESHOLD_PCT),
            xycoords=("axes fraction", "data"),
            xytext=(0, 4),
            textcoords="offset points",
            fontsize=8,
            color="#DC2626",
        )

        ax.set_xscale("log", base=2)
        ax.set_yscale("log")
        style_axes(
            ax,
            f"{backend.upper()}: Roof Utilization vs. Input Size (N)",
            "Input Size N (elements)",
            "Percent of measured roof (%)",
        )
        ax.legend(fontsize=8)
        fig.tight_layout()

        os.makedirs(out_dir, exist_ok=True)
        out_file = os.path.join(out_dir, f"{backend}_roof_utilization.png")
        fig.savefig(out_file, dpi=160)
        plt.close(fig)
        written.append(out_file)

    return written
