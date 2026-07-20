from __future__ import annotations

import os
from collections import defaultdict
from typing import Iterable, Optional

from ..models import RooflinePoint
from .common import plt, style_axes

BACKEND_COLORS = {
    "cpu": "#4269D0",
    "gpu": "#C2410C",
    "npu": "#7C3AED",
}

STREAM_KERNELS = ("read", "fma")


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
