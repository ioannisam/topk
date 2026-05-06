from __future__ import annotations

import os
from collections import defaultdict
from typing import Optional

import matplotlib.cm as cm
import matplotlib.colors as mcolors

from ..models import CaseRecord
from .common import aggregate_value, plt, select_time_ms, style_axes


def plot(records: list[CaseRecord], out_path: str, agg: str) -> Optional[list[str]]:
    # Group by: backend -> algorithm -> k -> n -> list of times
    grouped: dict[str, dict[str, dict[int, dict[int, list[float]]]]] = defaultdict(
        lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    )

    for rec in records:
        if rec.backend in {"", "gt"} or rec.n is None or rec.k is None:
            continue
        t = select_time_ms(rec, "algorithmic")
        if t is not None and t > 0:
            grouped[rec.backend][rec.algorithm][rec.k][rec.n].append(t)

    if not grouped:
        return None

    base_dir = os.path.dirname(out_path) or "."
    base_name = os.path.splitext(os.path.basename(out_path))[0]
    outputs: list[str] = []

    for backend, algo_map in grouped.items():
        for algo, k_map in algo_map.items():
            fig, ax = plt.subplots(figsize=(10, 6))
            sorted_ks = sorted(k_map.keys())

            cmap = cm.get_cmap("viridis")
            if len(sorted_ks) > 1 and min(sorted_ks) > 0:
                norm = mcolors.LogNorm(vmin=min(sorted_ks), vmax=max(sorted_ks))
            else:
                norm = mcolors.Normalize(vmin=0, vmax=max(1, len(sorted_ks)))

            for i, k in enumerate(sorted_ks):
                n_map = k_map[k]
                xs = sorted(n_map.keys())
                ys = [aggregate_value(n_map[n], agg) for n in xs]

                # Apply color based on K's magnitude
                color = cmap(norm(k)) if min(sorted_ks) > 0 else cmap(i / max(1, len(sorted_ks) - 1))
                ax.plot(xs, ys, marker="o", linewidth=2, color=color, label=f"K={k}")

            ax.set_xscale("log", base=2)
            ax.set_yscale("log")
            style_axes(ax, f"Runtime vs N for varying K ({backend} - {algo})", "N (log2 scale)", "Time (ms, log scale)")

            ax.legend(title="Top-K", loc="upper left", fontsize="small", ncol=1)
            fig.tight_layout()

            os.makedirs(base_dir, exist_ok=True)
            out_file = os.path.join(base_dir, f"{base_name}_{backend}_{algo}.png")
            fig.savefig(out_file, dpi=160)
            plt.close(fig)
            outputs.append(out_file)

    return outputs
