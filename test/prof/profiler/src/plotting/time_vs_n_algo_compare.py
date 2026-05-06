from __future__ import annotations
import os
from collections import defaultdict
from typing import Optional
from ..models import CaseRecord
from .common import aggregate_value, error_bounds, plt, select_time_ms, style_axes


def plot(records: list[CaseRecord], out_path: str, agg: str, error_bars: str) -> Optional[list[str]]:
    # Group by: K -> backend -> algorithm -> n -> times
    grouped: dict[int, dict[str, dict[str, dict[int, list[float]]]]] = defaultdict(
        lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    )
    for rec in records:
        if (
            rec.backend in {"", "gt"}
            or rec.algorithm not in {"bitonic", "map_reduce"}
            or rec.n is None
            or rec.k is None
        ):
            continue
        time_ms = select_time_ms(rec, "algorithmic")
        if time_ms is None or time_ms <= 0:
            continue
        grouped[rec.k][rec.backend][rec.algorithm][rec.n].append(time_ms)

    if not grouped:
        return None
    base_dir, base_name = os.path.dirname(out_path) or ".", os.path.splitext(os.path.basename(out_path))[0]
    outputs: list[str] = []

    for k, backend_map in sorted(grouped.items()):
        for backend, algo_map in sorted(backend_map.items()):
            fig, ax = plt.subplots(figsize=(10, 6))
            has_data = False
            for algorithm in ("bitonic", "map_reduce"):
                if algorithm not in algo_map:
                    continue
                xs = sorted(algo_map[algorithm].keys())
                ys, lowers, uppers = [], [], []
                for n in xs:
                    center = aggregate_value(algo_map[algorithm][n], agg)
                    lo, hi = error_bounds(algo_map[algorithm][n], center, error_bars)
                    ys.append(center)
                    lowers.append(lo)
                    uppers.append(hi)
                if xs:
                    has_data = True
                    (line,) = ax.plot(xs, ys, marker="o", linewidth=2, label=algorithm)
                    if error_bars != "none":
                        ax.fill_between(xs, lowers, uppers, alpha=0.15, color=line.get_color())

            if not has_data:
                plt.close(fig)
                continue

            ax.set_xscale("log", base=2)
            ax.set_yscale("log")
            style_axes(
                ax,
                f"Algorithmic Runtime ({backend}): bitonic vs map_reduce (K = {k})",
                "N (log2 scale)",
                "Time (ms, log scale)",
            )
            ax.legend(title="Algorithm", loc="upper left")
            fig.tight_layout()

            os.makedirs(base_dir, exist_ok=True)
            out_file = os.path.join(base_dir, f"{base_name}_{backend}_k{k}.png")
            fig.savefig(out_file, dpi=160, bbox_inches="tight")
            plt.close(fig)
            outputs.append(out_file)

    return outputs
