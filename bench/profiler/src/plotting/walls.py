from __future__ import annotations

import os
from collections import defaultdict
from typing import Iterable, Optional

from ..models import RooflinePoint
from .common import plt, style_axes
from .roofline import LADDER_MIN_BYTES, backend_color

TRANSFER_KERNELS = {
    "h2d_pageable": ("GPU host to device (pageable)", "-"),
    "h2d_pinned": ("GPU host to device (pinned)", "--"),
    "d2h_pageable": ("GPU device to host (pageable)", "-."),
    "d2h_pinned": ("GPU device to host (pinned)", ":"),
    "stage_write": ("NPU host write into buffer", "-"),
    "stage_write_sync": ("NPU write + sync (real path)", "--"),
    "sync_only": ("NPU sync only", ":"),
}

CACHE_MARKERS = {
    "cpu": [(8 * 1024 * 1024, "L2 8 MiB"), (16 * 1024 * 1024, "L3 16 MiB")],
    "gpu": [(32 * 1024 * 1024, "L2 32 MiB")],
}


def _bytes_axis(ax) -> None:
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.xaxis.set_major_formatter(
        plt.FuncFormatter(lambda v, _: f"{v / 1048576:g}M" if v >= 1048576 else f"{v / 1024:g}K")
    )


def plot_cache_ladder(points: Iterable[RooflinePoint], out_path: str) -> Optional[str]:
    grouped: dict[str, list[tuple[float, float]]] = defaultdict(list)
    for p in points:
        if p.kernel != "cache_read" or p.gbytes_per_s <= 0 or p.elements <= 0:
            continue
        working_set = p.elements * 4.0
        if working_set < LADDER_MIN_BYTES:
            continue
        grouped[p.backend].append((working_set, p.gbytes_per_s))

    if not grouped:
        return None

    fig, ax = plt.subplots(figsize=(10, 6))
    for backend in sorted(grouped):
        series = sorted(grouped[backend])
        ax.plot(
            [x for x, _ in series],
            [y for _, y in series],
            marker="o",
            linewidth=2,
            color=backend_color(backend),
            label=backend.upper(),
        )
        for size, label in CACHE_MARKERS.get(backend, []):
            ax.axvline(size, color=backend_color(backend), linestyle=":", linewidth=1, alpha=0.6)
            ax.annotate(
                label,
                xy=(size, 0.03),
                xycoords=("data", "axes fraction"),
                xytext=(4, 0),
                textcoords="offset points",
                fontsize=8,
                color=backend_color(backend),
            )

    _bytes_axis(ax)
    style_axes(
        ax,
        "Memory Wall by Working-Set Size (where each cache level ends)",
        "Working set (bytes, log scale)",
        "Read bandwidth (GB/s, log scale)",
    )
    ax.legend()
    fig.tight_layout()

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
    return out_path


def plot_transfer_walls(points: Iterable[RooflinePoint], out_path: str) -> Optional[str]:
    grouped: dict[tuple[str, str], list[tuple[float, float]]] = defaultdict(list)
    for p in points:
        if p.kernel not in TRANSFER_KERNELS or p.gbytes_per_s <= 0 or p.elements <= 0:
            continue
        grouped[(p.backend, p.kernel)].append((p.elements * 4.0, p.gbytes_per_s))

    if not grouped:
        return None

    fig, ax = plt.subplots(figsize=(10, 6))
    for (backend, kernel), series in sorted(grouped.items()):
        label, style = TRANSFER_KERNELS[kernel]
        ordered = sorted(series)
        ax.plot(
            [x for x, _ in ordered],
            [y for _, y in ordered],
            marker="o",
            linestyle=style,
            linewidth=2,
            color=backend_color(backend),
            label=label,
        )

    # Latency floors are fixed per-call costs, not rates, so they cannot be drawn as a
    # roof. They are reported as microseconds in a corner annotation instead.
    floors = [
        f"{p.backend.upper()} {p.kernel.removeprefix('latency_')}: {p.ms_min * 1000:.1f} us"
        for p in points
        if p.kernel.startswith("latency_") and p.ms_min > 0
    ]
    if floors:
        ax.annotate(
            "Latency floors (fixed per call)\n" + "\n".join(sorted(floors)),
            xy=(0.02, 0.02),
            xycoords="axes fraction",
            fontsize=8,
            color="#374151",
            va="bottom",
            bbox={"boxstyle": "round", "facecolor": "white", "edgecolor": "#D1D5DB", "alpha": 0.9},
        )

    _bytes_axis(ax)
    style_axes(
        ax,
        "Transfer Walls: Bandwidth vs Transfer Size",
        "Transfer size (bytes, log scale)",
        "Bandwidth (GB/s, log scale)",
    )
    ax.legend(fontsize=8)
    fig.tight_layout()

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
    return out_path
