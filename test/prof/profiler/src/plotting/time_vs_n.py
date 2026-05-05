from __future__ import annotations

import os
from collections import defaultdict
from typing import Optional

from ..models import CaseRecord
from .common import add_time_metric_legend, aggregate_value, error_bounds, label_with_algorithm, plt, select_time_ms, style_axes, time_metric_style


def plot(records: list[CaseRecord], out_path: str, agg: str, error_bars: str) -> Optional[str]:
    grouped: dict[str, dict[str, dict[int, list[float]]]] = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    algorithms = {rec.algorithm for rec in records if rec.algorithm}
    include_algorithm = len(algorithms) > 1
    for rec in records:
        if rec.n is None:
            continue
        label = label_with_algorithm(rec.backend, rec.algorithm, include_algorithm)
        for metric in ("algorithmic", "end-to-end"):
            time_ms = select_time_ms(rec, metric)
            if time_ms is None or time_ms <= 0:
                continue
            grouped[label][metric][rec.n].append(time_ms)

    if not grouped:
        return None

    fig, ax = plt.subplots(figsize=(10, 6))
    for label in sorted(grouped.keys()):
        algo_map = grouped[label].get("algorithmic", {})
        e2e_map = grouped[label].get("end-to-end", {})
        color = None

        if algo_map:
            xs = sorted(algo_map.keys())
            ys = []
            lowers = []
            uppers = []
            for n in xs:
                center = aggregate_value(algo_map[n], agg)
                lo, hi = error_bounds(algo_map[n], center, error_bars)
                ys.append(center)
                lowers.append(lo)
                uppers.append(hi)
            (line,) = ax.plot(xs, ys, **time_metric_style("algorithmic", label=label))
            color = line.get_color()
            if error_bars != "none":
                ax.fill_between(xs, lowers, uppers, alpha=0.15, color=color)

        if e2e_map:
            xs = sorted(e2e_map.keys())
            ys = []
            for n in xs:
                center = aggregate_value(e2e_map[n], agg)
                ys.append(center)
            ax.plot(xs, ys, **time_metric_style("end-to-end", color=color, label="_nolegend_"))

    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    style_axes(ax, "Runtime vs Input Size", "N (log2 scale)", "Time (ms, log scale)")

    config_leg = ax.legend(title="Configuration", loc="best")
    ax.add_artist(config_leg)
    add_time_metric_legend(ax, loc="lower right")
    fig.tight_layout()

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
    return out_path
