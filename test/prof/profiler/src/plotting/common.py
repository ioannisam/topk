from __future__ import annotations

import math
import sys
from statistics import mean, median, pstdev
from typing import Any, Literal

try:
    import matplotlib

    if "--show" not in sys.argv:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    MATPLOTLIB_AVAILABLE = True
except Exception:
    plt = None
    MATPLOTLIB_AVAILABLE = False


E2E_DASH_PATTERN = (0, (8, 3))


def time_metric_style(metric: str, *, color: Any = None, label: str | None = None) -> dict[str, Any]:
    if metric == "algorithmic":
        return {
            "linestyle": "-",
            "linewidth": 3.0,
            "marker": "o",
            "markersize": 5,
            "color": color,
            "alpha": 0.4,
            "label": label,
            "zorder": 2,
        }
    return {
        "linestyle": E2E_DASH_PATTERN,
        "linewidth": 2.0,
        "marker": "x",
        "markersize": 8,
        "markeredgewidth": 2.0,
        "color": color,
        "label": label,
        "zorder": 3,
    }


def add_time_metric_legend(
    ax,
    *,
    loc: str = "lower right",
    title: str = "Time metric",
    kind: Literal["line", "marker"] = "line",
) -> None:
    """Add a legend that explains algorithmic vs end-to-end series styling."""
    try:
        from matplotlib.lines import Line2D

        if kind == "marker":
            handles = [
                Line2D([0], [0], color="black", linestyle="none", marker="o", markersize=6, label="Algorithmic"),
                Line2D([0], [0], color="black", linestyle="none", marker="x", markersize=6, label="End-to-end"),
            ]
        else:
            handles = [
                Line2D([0], [0], color="black", linestyle="-", linewidth=2.2, label="Algorithmic"),
                Line2D([0], [0], color="black", linestyle="--", linewidth=2.2, label="End-to-end"),
            ]
        ax.legend(handles=handles, title=title, loc=loc)
    except Exception:
        return


def add_energy_metric_legend(ax, *, loc: str = "lower right") -> None:
    """Legend explaining estimated-algorithmic vs measured-end-to-end energy series."""
    try:
        from matplotlib.lines import Line2D

        handles = [
            Line2D([0], [0], color="black", linestyle="-", linewidth=2.2, label="End-to-end (host + backend)"),
            Line2D([0], [0], color="black", linestyle="--", linewidth=2.2, label="Algorithmic (backend only)"),
        ]
        ax.legend(handles=handles, title="Energy metric", loc=loc)
    except Exception:
        return


def style_axes(ax, title: str, xlabel: str, ylabel: str) -> None:
    ax.set_title(title, fontsize=14)
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)

    ax.set_axisbelow(True)
    ax.grid(True, which="major", alpha=0.28, linestyle=":")
    ax.grid(True, which="minor", alpha=0.14, linestyle=":")
    ax.tick_params(axis="both", which="major", labelsize=11)

    try:
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    except Exception:
        pass


def label_with_algorithm(label: str, algorithm: str, include_algorithm: bool) -> str:
    if include_algorithm and algorithm:
        return f"{label}/{algorithm}"
    return label


def select_time_ms(rec, metric: str) -> float | None:
    if metric == "algorithmic":
        return rec.time_algorithmic_ms if rec.time_algorithmic_ms is not None else rec.time_end_to_end_ms
    return rec.time_end_to_end_ms if rec.time_end_to_end_ms is not None else rec.time_algorithmic_ms


def select_energy_joules(rec, metric: str, scope: str = "e2e") -> float | None:
    net = metric == "net"
    ops = getattr(rec, "bench_ops", 1) or 1

    if getattr(rec, "inproc_available", False):
        if scope == "algorithmic":
            energy = rec.inproc_net_algo_joules if net else rec.inproc_algo_joules
        else:
            energy = rec.inproc_net_e2e_joules if net else rec.inproc_e2e_joules
        if energy is not None:
            return energy / ops

    energy = rec.net_energy_joules if (net and rec.net_energy_joules is not None) else rec.energy_joules
    return energy / ops if energy is not None else None


def select_elapsed_seconds(rec) -> float | None:
    # Per-operation end-to-end wall time, matching select_energy_joules' window.
    if rec.elapsed_seconds is None:
        return None
    ops = getattr(rec, "bench_ops", 1) or 1
    return rec.elapsed_seconds / ops


def select_power_watts(rec, metric: str) -> float | None:
    # Average power is energy/time, invariant under per-op normalization.
    if metric == "net" and rec.net_average_watts is not None:
        return rec.net_average_watts
    return rec.average_watts


def metric_title_suffix(metric: str) -> str:
    return " (net of idle)" if metric == "net" else ""


def percentile(values: list[float], pct: float) -> float:
    if not values:
        raise ValueError("percentile on empty list")
    if len(values) == 1:
        return values[0]
    xs = sorted(values)
    rank = (len(xs) - 1) * pct
    lo = int(math.floor(rank))
    hi = int(math.ceil(rank))
    if lo == hi:
        return xs[lo]
    frac = rank - lo
    return xs[lo] * (1.0 - frac) + xs[hi] * frac


def aggregate_value(values: list[float], agg: str) -> float:
    if agg == "median":
        return median(values)
    return mean(values)


def error_bounds(values: list[float], center: float, error_bars: str) -> tuple[float, float]:
    if error_bars == "none" or len(values) < 2:
        return center, center
    if error_bars == "std":
        s = pstdev(values)
        return center - s, center + s
    return percentile(values, 0.10), percentile(values, 0.90)
