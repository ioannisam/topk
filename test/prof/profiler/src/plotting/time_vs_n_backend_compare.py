from __future__ import annotations

import os
from collections import defaultdict
from typing import Optional

from ..models import CaseRecord
from .common import add_time_metric_legend, aggregate_value, error_bounds, plt, select_time_ms, style_axes, time_metric_style


def plot(records: list[CaseRecord], out_path: str, agg: str, error_bars: str) -> Optional[list[str]]:
    """Per-algorithm backend comparison (time vs N).

    Produces two figures:
    - bitonic: all backends on one plot
    - map_reduce: all backends on one plot

    Solid lines are algorithmic time, dashed lines are end-to-end time.
    """

    grouped: dict[str, dict[str, dict[str, dict[int, list[float]]]]] = defaultdict(
        lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    )
    for rec in records:
        if rec.backend in {"", "gt"}:
            continue
        if rec.algorithm not in {"bitonic", "map_reduce"}:
            continue
        if rec.n is None:
            continue
        for metric in ("algorithmic", "end-to-end"):
            time_ms = select_time_ms(rec, metric)
            if time_ms is None or time_ms <= 0:
                continue
            grouped[rec.algorithm][rec.backend][metric][rec.n].append(time_ms)

    if not grouped:
        return None

    base_dir = os.path.dirname(out_path) or "."
    base_name, ext = os.path.splitext(os.path.basename(out_path))
    if not ext:
        ext = ".png"

    outputs: list[str] = []
    for algorithm in ("bitonic", "map_reduce"):
        backend_map = grouped.get(algorithm)
        if not backend_map:
            continue

        fig, ax = plt.subplots(figsize=(10, 6))
        for backend in sorted(backend_map.keys()):
            metric_map = backend_map[backend]
            algo_series = metric_map.get("algorithmic", {})
            e2e_series = metric_map.get("end-to-end", {})
            color = None

            if algo_series:
                xs = sorted(algo_series.keys())
                ys = []
                lowers = []
                uppers = []
                for n in xs:
                    center = aggregate_value(algo_series[n], agg)
                    lo, hi = error_bounds(algo_series[n], center, error_bars)
                    ys.append(center)
                    lowers.append(lo)
                    uppers.append(hi)
                (line,) = ax.plot(xs, ys, **time_metric_style("algorithmic", label=backend))
                color = line.get_color()
                if error_bars != "none":
                    ax.fill_between(xs, lowers, uppers, alpha=0.15, color=color)

            if e2e_series:
                xs = sorted(e2e_series.keys())
                ys = [aggregate_value(e2e_series[n], agg) for n in xs]
                ax.plot(xs, ys, **time_metric_style("end-to-end", color=color, label="_nolegend_"))

        ax.set_xscale("log", base=2)
        ax.set_yscale("log")
        style_axes(ax, f"Runtime vs Input Size ({algorithm}): backend comparison", "N (log2 scale)", "Time (ms, log scale)")
        config_leg = ax.legend(title="Backend", loc="best")
        ax.add_artist(config_leg)
        add_time_metric_legend(ax, loc="lower right")
        fig.tight_layout()

        os.makedirs(base_dir, exist_ok=True)
        out_file = os.path.join(base_dir, f"{base_name}_{algorithm}{ext}")
        fig.savefig(out_file, dpi=160)
        plt.close(fig)
        outputs.append(out_file)

    return outputs
