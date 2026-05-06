from __future__ import annotations

import os
from collections import defaultdict
from typing import Optional

from ..models import CaseRecord
from .common import aggregate_value, error_bounds, label_with_algorithm, plt, select_time_ms, style_axes

def plot(records: list[CaseRecord], out_path: str, agg: str, error_bars: str) -> Optional[list[str]]:
    # Grouped by: K -> label -> N -> list of times
    grouped: dict[int, dict[str, dict[int, list[float]]]] = defaultdict(
        lambda: defaultdict(lambda: defaultdict(list))
    )
    
    algorithms = {rec.algorithm for rec in records if rec.algorithm}
    include_algorithm = len(algorithms) > 1
    
    for rec in records:
        if rec.n is None or rec.k is None: continue
        label = label_with_algorithm(rec.backend, rec.algorithm, include_algorithm)
        time_ms = select_time_ms(rec, "algorithmic")
        if time_ms is None or time_ms <= 0: continue
        
        grouped[rec.k][label][rec.n].append(time_ms)

    if not grouped: return None

    base_dir = os.path.dirname(out_path) or "."
    base_name, ext = os.path.splitext(os.path.basename(out_path))
    outputs: list[str] = []

    # Generate one plot per K
    for k, label_map in sorted(grouped.items()):
        fig, ax = plt.subplots(figsize=(10, 6))
        has_data = False
        
        for label, n_map in sorted(label_map.items()):
            xs = sorted(n_map.keys())
            ys, lowers, uppers = [], [], []
            for n in xs:
                center = aggregate_value(n_map[n], agg)
                lo, hi = error_bounds(n_map[n], center, error_bars)
                ys.append(center)
                lowers.append(lo)
                uppers.append(hi)
            
            if xs:
                has_data = True
                (line,) = ax.plot(xs, ys, marker="o", linewidth=2, label=label)
                if error_bars != "none": 
                    ax.fill_between(xs, lowers, uppers, alpha=0.15, color=line.get_color())

        if not has_data:
            plt.close(fig)
            continue

        ax.set_xscale("log", base=2)
        ax.set_yscale("log")
        style_axes(ax, f"Algorithmic Runtime vs Input Size (K = {k})", "N (log2 scale)", "Time (ms, log scale)")
        ax.legend(title="Configuration", loc="upper left")
        fig.tight_layout()
        
        os.makedirs(base_dir, exist_ok=True)
        # Suffix the filename with _k<value>
        out_file = os.path.join(base_dir, f"{base_name}_k{k}{ext}")
        fig.savefig(out_file, dpi=160, bbox_inches="tight")
        plt.close(fig)
        outputs.append(out_file)

    return outputs
