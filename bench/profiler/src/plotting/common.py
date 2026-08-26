from __future__ import annotations

import math
import os
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


_DTYPE_CONTEXT: str | None = None


def set_dtype_context(dtype: str | None) -> None:
    """Set the dtype every subsequent figure is subtitled with, or None to clear it.

    Plots are written into per-dtype directories, so the figures themselves carried no
    record of which type produced them and stopped being self-describing the moment one
    was pulled out of its folder. The profiler emits one dtype at a time, so a single
    setter is enough; machine-level figures clear it because they characterise the
    hardware rather than any one dtype.
    """
    global _DTYPE_CONTEXT
    _DTYPE_CONTEXT = dtype or None


def add_subtitle(ax, subtitle: str | None = None) -> None:
    text = subtitle if subtitle is not None else _DTYPE_CONTEXT
    if not text:
        return
    ax.text(
        0.5,
        1.008,
        f"dtype: {text}",
        transform=ax.transAxes,
        ha="center",
        va="bottom",
        fontsize=10,
        color="#6B7280",
    )


def style_axes(ax, title: str, xlabel: str, ylabel: str, subtitle: str | None = None) -> None:
    ax.set_title(title, fontsize=14, pad=22 if (subtitle or _DTYPE_CONTEXT) else 12)
    add_subtitle(ax, subtitle)
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
        return rec.time_algorithmic_ms
    return rec.time_end_to_end_ms


def measured_loop_fraction(rec) -> float | None:
    """Share of process wall time spent inside the benchmark loop.

    The external wrapper meters the whole process: startup, input generation, the CPU reference
    sort when verify=true, and teardown all sit outside the timed region. Scaling by this fraction
    keeps whole-process energy comparable with the in-process counters, which see only the loop.
    """
    loop_s = getattr(rec, "inproc_loop_seconds", None)
    total_s = rec.elapsed_seconds
    if not loop_s or not total_s or total_s <= 0:
        return None
    return min(loop_s / total_s, 1.0)


def select_energy_joules(rec, metric: str, scope: str = "e2e") -> float | None:
    net = metric == "net"
    ops = getattr(rec, "bench_ops", 1) or 1

    if getattr(rec, "inproc_available", False):
        if scope == "algorithmic":
            energy = rec.inproc_net_algo_joules if net else rec.inproc_algo_joules
        else:
            energy = rec.inproc_net_e2e_joules if net else rec.inproc_e2e_joules
        if energy is not None:
            iterations = getattr(rec, "inproc_iterations", None) or ops
            return energy / iterations

    if scope == "algorithmic":
        return None

    if net:
        if rec.net_energy_joules is None:
            return None
        energy = rec.net_energy_joules
    else:
        energy = rec.energy_joules
    if energy is None:
        return None
    fraction = measured_loop_fraction(rec)
    if fraction is not None:
        energy *= fraction
    return energy / ops


def select_elapsed_seconds(rec) -> float | None:
    ops = getattr(rec, "bench_ops", 1) or 1
    e2e_s = getattr(rec, "inproc_e2e_seconds", None)
    if e2e_s:
        iterations = getattr(rec, "inproc_iterations", None) or ops
        return e2e_s / iterations
    if rec.elapsed_seconds is None:
        return None
    elapsed = rec.elapsed_seconds
    fraction = measured_loop_fraction(rec)
    if fraction is not None:
        elapsed *= fraction
    return elapsed / ops


def select_power_watts(rec, metric: str) -> float | None:
    if metric == "net":
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
        return max(center - s, min(values)), center + s
    return percentile(values, 0.10), percentile(values, 0.90)


def save_k_figure(fig, base_dir: str, base_name: str, k, ext: str, **savefig_kwargs) -> str:
    """Write <base_dir>/<base_name>_k<k><ext> and close the figure."""
    os.makedirs(base_dir, exist_ok=True)
    out_file = os.path.join(base_dir, f"{base_name}_k{k}{ext}")
    fig.savefig(out_file, dpi=160, **savefig_kwargs)
    plt.close(fig)
    return out_file
