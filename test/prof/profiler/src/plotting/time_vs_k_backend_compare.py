from __future__ import annotations

import os
from collections import defaultdict
from typing import Optional

from ..models import CaseRecord
from .common import (
    add_time_metric_legend,
    aggregate_value,
    error_bounds,
    plt,
    select_time_ms,
    style_axes,
    time_metric_style,
)


def plot(records: list[CaseRecord], out_path: str, agg: str, error_bars: str) -> Optional[list[str]]:
    # grouped[n][algorithm][backend][metric][k] = list[time]
    grouped: dict[int, dict[str, dict[str, dict[str, dict[int, list[float]]]]]] = defaultdict(
        lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(list))))
    )

    for rec in records:
        if rec.backend in {"", "gt"} or rec.algorithm not in {"bitonic", "map_reduce"}:
            continue
        if rec.n is None or rec.k is None:
            continue

        for metric in ("algorithmic", "end-to-end"):
            time_ms = select_time_ms(rec, metric)
            if time_ms is None or time_ms <= 0:
                continue
            grouped[rec.n][rec.algorithm][rec.backend][metric][rec.k].append(time_ms)

    if not grouped:
        return None

    base_dir, base_name = os.path.dirname(out_path) or ".", os.path.splitext(os.path.basename(out_path))[0]
    outputs: list[str] = []

    # Generate one plot per N, per Algorithm
    for n, algo_map in sorted(grouped.items()):
        for algorithm, backend_map in sorted(algo_map.items()):
            # Skip if we only have one K value tested for this N
            all_ks = {k for b in backend_map.values() for m in b.values() for k in m.keys()}
            if len(all_ks) < 2:
                continue

            fig, ax = plt.subplots(figsize=(10, 6))
            for backend, metric_map in sorted(backend_map.items()):
                algo_series = metric_map.get("algorithmic", {})
                e2e_series = metric_map.get("end-to-end", {})
                color = None

                if algo_series:
                    xs = sorted(algo_series.keys())
                    ys, lowers, uppers = [], [], []
                    for k in xs:
                        center = aggregate_value(algo_series[k], agg)
                        lo, hi = error_bounds(algo_series[k], center, error_bars)
                        ys.append(center)
                        lowers.append(lo)
                        uppers.append(hi)
                    (line,) = ax.plot(xs, ys, **time_metric_style("algorithmic", label=backend))
                    color = line.get_color()
                    if error_bars != "none":
                        ax.fill_between(xs, lowers, uppers, alpha=0.15, color=color)

                if e2e_series:
                    xs = sorted(e2e_series.keys())
                    ys = [aggregate_value(e2e_series[k], agg) for k in xs]
                    ax.plot(xs, ys, **time_metric_style("end-to-end", color=color, label="_nolegend_"))

            ax.set_xscale("log", base=2)
            ax.set_yscale("log")
            style_axes(ax, f"Runtime vs K Scaling (N={n}, {algorithm})", "K (log2 scale)", "Time (ms, log scale)")
            config_leg = ax.legend(title="Backend", loc="best")
            ax.add_artist(config_leg)
            add_time_metric_legend(ax, loc="lower right")
            fig.tight_layout()

            os.makedirs(base_dir, exist_ok=True)
            out_file = os.path.join(base_dir, f"{base_name}_{algorithm}_N{n}.png")
            fig.savefig(out_file, dpi=160)
            plt.close(fig)
            outputs.append(out_file)

    return outputs
