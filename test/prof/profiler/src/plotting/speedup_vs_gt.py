from __future__ import annotations

import math
import os
from collections import defaultdict
from typing import Optional

from ..models import CaseRecord
from .common import add_time_metric_legend, label_with_algorithm, plt, select_time_ms, style_axes, time_metric_style


def _make_speedup_series(records: list[CaseRecord]) -> dict[str, dict[str, dict[int, list[float]]]]:
    """Build speedup series grouped by configuration label.

    Returns: series[label][metric][n] = list[speedup]
    metric keys: "algorithmic" and "end-to-end".
    """
    gt_algo: dict[tuple[str, str, Optional[int], Optional[int], str], float] = {}
    gt_e2e: dict[tuple[str, str, Optional[int], Optional[int], str], float] = {}

    for rec in records:
        key = (rec.dtype, rec.mode, rec.k, rec.n, rec.case_name)
        if rec.backend != "gt":
            continue
        t_algo = select_time_ms(rec, "algorithmic")
        t_e2e = select_time_ms(rec, "end-to-end")
        if t_algo is not None and t_algo > 0:
            gt_algo[key] = t_algo
        if t_e2e is not None and t_e2e > 0:
            gt_e2e[key] = t_e2e

    series: dict[str, dict[str, dict[int, list[float]]]] = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    algorithms = {rec.algorithm for rec in records if rec.algorithm}
    include_algorithm = len(algorithms) > 1

    for rec in records:
        if rec.backend == "gt" or rec.n is None:
            continue

        key = (rec.dtype, rec.mode, rec.k, rec.n, rec.case_name)
        gt_t_algo = gt_algo.get(key)
        gt_t_e2e = gt_e2e.get(key)

        label = label_with_algorithm(rec.backend, rec.algorithm, include_algorithm)

        # Algorithmic speedup
        t_algo = select_time_ms(rec, "algorithmic")
        if gt_t_algo is not None and t_algo is not None and t_algo > 0:
            speedup = gt_t_algo / t_algo
            if math.isfinite(speedup) and speedup > 0:
                series[label]["algorithmic"][rec.n].append(speedup)

        # End-to-end speedup
        t_e2e = select_time_ms(rec, "end-to-end")
        if gt_t_e2e is not None and t_e2e is not None and t_e2e > 0:
            speedup = gt_t_e2e / t_e2e
            if math.isfinite(speedup) and speedup > 0:
                series[label]["end-to-end"][rec.n].append(speedup)

    return series


def _plot_series(
    series: dict[str, dict[str, dict[int, list[float]]]],
    *,
    title: str,
    out_path: str,
) -> Optional[str]:
    if not series:
        return None

    fig, ax = plt.subplots(figsize=(10, 6))
    for label in sorted(series.keys()):
        algo_map = series[label].get("algorithmic", {})
        e2e_map = series[label].get("end-to-end", {})
        if not algo_map and not e2e_map:
            continue

        # Use the algorithmic line as the "color source" so the dashed e2e
        # line matches the same configuration color.
        color = None
        if algo_map:
            xs = sorted(algo_map.keys())
            ys = [sum(algo_map[n]) / len(algo_map[n]) for n in xs]
            (line,) = ax.plot(xs, ys, **time_metric_style("algorithmic", label=label))
            color = line.get_color()

        if e2e_map:
            xs = sorted(e2e_map.keys())
            ys = [sum(e2e_map[n]) / len(e2e_map[n]) for n in xs]
            ax.plot(xs, ys, **time_metric_style("end-to-end", color=color, label="_nolegend_"))

    ax.axhline(1.0, color="gray", linestyle="--", linewidth=1)
    ax.set_xscale("log", base=2)
    style_axes(ax, title, "N (log2 scale)", "Speedup")

    # Two legends: one for configurations (colors) and one for metric (line style).
    config_leg = ax.legend(title="Configuration", loc="best")
    ax.add_artist(config_leg)
    add_time_metric_legend(ax, loc="lower right")

    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
    return out_path


def plot(records: list[CaseRecord], out_path: str) -> Optional[str]:
    """Single speedup-vs-GT plot (all backends/algorithms).

    Solid lines are algorithmic speedup, dashed lines are end-to-end speedup.
    """
    series = _make_speedup_series(records)
    return _plot_series(series, title="Speedup vs GT (gt_time / backend_time)", out_path=out_path)
