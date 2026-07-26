from __future__ import annotations

import math
import os
from collections import defaultdict
from typing import Iterable, Optional

from ..models import CaseRecord, RooflinePoint
from .common import plt, select_time_ms, style_axes

BACKEND_COLORS = {
    "cpu": "#4269D0",
    "gpu": "#C2410C",
    "npu": "#7C3AED",
}

STREAM_KERNELS = ("read", "fma", "cmp")

# Below this the working set is too small to keep every thread busy, so the point measures
# available parallelism rather than a bandwidth ceiling. Matches walls.CACHE_LADDER_MIN_BYTES.
LADDER_MIN_BYTES = 1024 * 1024


def backend_color(backend: str) -> str:
    return BACKEND_COLORS.get(backend.lower(), "#6B7280")


def backend_ceilings(points: Iterable[RooflinePoint]) -> dict[str, float]:
    """Peak measured read bandwidth (GB/s) per backend, for overlaying on bandwidth plots."""
    ceilings: dict[str, float] = {}
    for p in points:
        if p.kernel not in STREAM_KERNELS or p.gbytes_per_s <= 0:
            continue
        current = ceilings.get(p.backend)
        if current is None or p.gbytes_per_s > current:
            ceilings[p.backend] = p.gbytes_per_s
    return ceilings


def compute_ceilings(points: Iterable[RooflinePoint]) -> dict[str, float]:
    """Peak measured compare-exchange rate (Gcmp/s) per backend.

    Top-k does compare-exchanges, not FLOPs, so the compute roof has to be measured in the
    same op currency as the kernels being placed against it.
    """
    ceilings: dict[str, float] = {}
    for p in points:
        if p.kernel != "cmp" or p.gflops_per_s <= 0:
            continue
        current = ceilings.get(p.backend)
        if current is None or p.gflops_per_s > current:
            ceilings[p.backend] = p.gflops_per_s
    return ceilings


def bandwidth_ladders(points: Iterable[RooflinePoint]) -> dict[str, list[tuple[float, float]]]:
    """Per backend, the measured (working-set bytes, GB/s) ladder across the cache hierarchy.

    A single flat DRAM ceiling is the wrong roof for a size sweep that crosses L2 and L3: a
    cache-resident run is nowhere near the DRAM wall and should not be drawn as if it were.
    """
    ladders: dict[str, list[tuple[float, float]]] = defaultdict(list)
    for p in points:
        if p.kernel != "cache_read" or p.gbytes_per_s <= 0 or p.elements <= 0:
            continue
        working_set = p.elements * 4.0
        if working_set < LADDER_MIN_BYTES:
            continue
        ladders[p.backend].append((working_set, p.gbytes_per_s))

    stream = backend_ceilings(points)
    out: dict[str, list[tuple[float, float]]] = {}
    for backend, series in ladders.items():
        ordered = sorted(series)
        # Anchor the far end at the DRAM-resident stream measurement.
        peak = stream.get(backend)
        if peak is not None and ordered and peak > 0:
            largest = ordered[-1][0]
            ordered.append((largest * 2.0, min(ordered[-1][1], peak)))
        out[backend] = ordered
    return out


def roof_at(ladder: list[tuple[float, float]], working_set_bytes: float) -> Optional[float]:
    """Attainable read bandwidth for a working set of this size, log-interpolated on the ladder."""
    if not ladder or working_set_bytes <= 0:
        return None
    if working_set_bytes <= ladder[0][0]:
        return ladder[0][1]
    if working_set_bytes >= ladder[-1][0]:
        return ladder[-1][1]

    for (x0, y0), (x1, y1) in zip(ladder, ladder[1:]):
        if x0 <= working_set_bytes <= x1:
            if x1 == x0:
                return y1
            t = (math.log2(working_set_bytes) - math.log2(x0)) / (math.log2(x1) - math.log2(x0))
            return y0 + t * (y1 - y0)
    return ladder[-1][1]


ALGO_MARKERS = {"bitonic": "o", "map_reduce": "s", "gt": "^"}


def plot_kernels(
    records: Iterable[CaseRecord],
    points: Iterable[RooflinePoint],
    out_path: str,
    metric: str = "algorithmic",
    title: str = "Top-k Kernels on the Measured Roofline",
) -> Optional[str]:
    """Place each top-k run at its measured (compare-ops/byte, compare-ops/s).

    This is the figure that answers compute-vs-memory bound: a point riding the sloped roof is
    bandwidth limited, one riding the flat roof is compute limited, one well below either is
    limited by something else (latency, dispatch, or occupancy).
    """
    point_list = list(points)
    bw_ceilings = backend_ceilings(point_list)
    cmp_ceilings = compute_ceilings(point_list)
    if not bw_ceilings or not cmp_ceilings:
        return None

    grouped: dict[tuple[str, str], list[tuple[float, float]]] = defaultdict(list)
    for r in records:
        time_ms = select_time_ms(r, metric)
        if not r.bytes_moved or not r.compare_ops or time_ms is None or time_ms <= 0:
            continue
        intensity = r.compare_ops / r.bytes_moved
        rate = r.compare_ops / (time_ms * 1e6)
        grouped[(r.backend, r.algorithm)].append((intensity, rate))

    if not grouped:
        return None

    fig, ax = plt.subplots(figsize=(10, 6))

    all_ai = [ai for series in grouped.values() for ai, _ in series]
    ai_lo = min(all_ai) / 4.0
    ai_hi = max(all_ai) * 4.0

    for backend in sorted(bw_ceilings):
        peak_bw = bw_ceilings[backend]
        peak_cmp = cmp_ceilings.get(backend)
        if peak_cmp is None or peak_bw <= 0:
            continue
        color = backend_color(backend)
        ridge = peak_cmp / peak_bw
        roof_x = [ai_lo, ridge, ai_hi]
        ax.plot(
            roof_x,
            [min(peak_bw * x, peak_cmp) for x in roof_x],
            linewidth=2,
            color=color,
            alpha=0.85,
            label=f"{backend.upper()} roof ({peak_bw:.0f} GB/s, {peak_cmp:.0f} Gcmp/s)",
        )

    for (backend, algo), series in sorted(grouped.items()):
        ordered = sorted(series)
        ax.plot(
            [x for x, _ in ordered],
            [y for _, y in ordered],
            linestyle="none",
            marker=ALGO_MARKERS.get(algo, "D"),
            markersize=7,
            markerfacecolor=backend_color(backend),
            markeredgecolor="white",
            markeredgewidth=1.2,
            alpha=0.9,
            zorder=3,
            label=f"{backend} - {algo}",
        )

    ax.set_xscale("log", base=2)
    ax.set_yscale("log", base=10)
    style_axes(
        ax,
        title,
        "Operational Intensity (compare-ops / byte moved)",
        "Attained Rate (Gcmp/s)",
    )
    ax.legend(fontsize=8, loc="lower right")
    fig.tight_layout()

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
    return out_path


def plot(
    points: Iterable[RooflinePoint],
    out_path: str,
    title: str = "Measured Roofline: Attainable Performance vs. Arithmetic Intensity",
) -> Optional[str]:
    grouped: dict[str, list[RooflinePoint]] = defaultdict(list)
    for p in points:
        if p.kernel == "fma" and p.arithmetic_intensity > 0 and p.gflops_per_s > 0:
            grouped[p.backend].append(p)

    if not grouped:
        return None

    fig, ax = plt.subplots(figsize=(10, 6))

    all_ai = [p.arithmetic_intensity for pts in grouped.values() for p in pts]
    ai_lo = min(all_ai) / 4.0
    ai_hi = max(all_ai) * 4.0

    for backend in sorted(grouped.keys()):
        pts = sorted(grouped[backend], key=lambda p: p.arithmetic_intensity)
        color = backend_color(backend)

        peak_bw = max(p.gbytes_per_s for p in pts)
        peak_gflops = max(p.gflops_per_s for p in pts)
        ridge_ai = peak_gflops / peak_bw if peak_bw > 0 else 0.0

        roof_x = [ai_lo, ridge_ai, ai_hi]
        roof_y = [min(peak_bw * x, peak_gflops) for x in roof_x]
        ax.plot(roof_x, roof_y, linewidth=2, color=color, label=f"{backend.upper()} roofline")

        ax.plot(
            [p.arithmetic_intensity for p in pts],
            [p.gflops_per_s for p in pts],
            linestyle="none",
            marker="o",
            markersize=8,
            markerfacecolor=color,
            markeredgecolor="white",
            markeredgewidth=2.0,
            zorder=3,
        )

        if ridge_ai > 0:
            ax.annotate(
                f"ridge {ridge_ai:.1f} FLOP/byte\n{peak_bw:.0f} GB/s · {peak_gflops / 1000.0:.2f} TFLOP/s",
                xy=(ridge_ai, peak_gflops),
                xytext=(6, -28),
                textcoords="offset points",
                fontsize=9,
                color="#374151",
            )

    ax.set_xscale("log", base=2)
    ax.set_yscale("log", base=10)
    style_axes(ax, title, "Arithmetic Intensity (FLOP/byte)", "Attainable Performance (GFLOP/s)")
    ax.legend(loc="lower right")
    fig.tight_layout()

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
    return out_path
