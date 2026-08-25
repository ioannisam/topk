from __future__ import annotations

import os
from collections import defaultdict
from typing import Optional

from ..models import MeasurementRecord
from .common import (
    add_energy_metric_legend,
    aggregate_value,
    error_bounds,
    plt,
    select_energy_joules,
    style_axes,
    time_metric_style,
)


def plot(
    records: list[MeasurementRecord], out_path: str, agg: str, error_bars: str, metric: str = "total"
) -> Optional[list[str]]:

    grouped: dict[str, dict[str, dict[str, dict[int, list[float]]]]] = defaultdict(
        lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    )
    for rec in records:
        if rec.n is None or rec.k is None:
            continue
        backend = rec.backend if rec.backend else rec.source
        if not backend:
            continue
        for scope in ("end-to-end", "algorithmic"):
            energy = select_energy_joules(rec, metric, scope)
            if energy is None or energy <= 0:
                continue
            grouped[backend][rec.algorithm][scope][rec.n].append(energy)

    if not grouped:
        return None

    base_dir, base_name = os.path.dirname(out_path) or ".", os.path.splitext(os.path.basename(out_path))[0]
    outputs: list[str] = []

    for backend, algo_map in sorted(grouped.items()):
        fig, ax = plt.subplots(figsize=(10, 6))
        for algorithm, scope_map in sorted(algo_map.items()):
            e2e_series = scope_map.get("end-to-end", {})
            algo_series = scope_map.get("algorithmic", {})
            color = None
            if e2e_series:
                xs = sorted(e2e_series.keys())
                ys, lowers, uppers = [], [], []
                for n in xs:
                    center = aggregate_value(e2e_series[n], agg)
                    lo, hi = error_bounds(e2e_series[n], center, error_bars)
                    ys.append(center)
                    lowers.append(lo)
                    uppers.append(hi)
                (line,) = ax.plot(xs, ys, **time_metric_style("algorithmic", label=algorithm if algorithm else backend))
                color = line.get_color()
                if error_bars != "none":
                    ax.fill_between(xs, lowers, uppers, alpha=0.15, color=color)
            if algo_series:
                xs = sorted(algo_series.keys())
                ys = [aggregate_value(algo_series[n], agg) for n in xs]
                ax.plot(xs, ys, **time_metric_style("end-to-end", color=color, label="_nolegend_"))

        ax.set_xscale("log", base=2)
        ax.set_yscale("log")
        style_axes(
            ax,
            f"Energy Metrics ({backend}): End-to-End (host + backend) vs Algorithmic (backend only)",
            "N (log2 scale)",
            "Energy per op (J, log scale)",
        )
        config_leg = ax.legend(title="Algorithm", loc="best")
        ax.add_artist(config_leg)
        add_energy_metric_legend(ax, loc="lower right")
        fig.tight_layout()

        os.makedirs(base_dir, exist_ok=True)
        out_file = os.path.join(base_dir, f"{base_name}_{backend}.png")
        fig.savefig(out_file, dpi=160)
        plt.close(fig)
        outputs.append(out_file)

    return outputs
