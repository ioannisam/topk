from __future__ import annotations

import os
import numpy as np
from collections import defaultdict
from typing import Optional
import matplotlib.colors as mcolors

from ..models import CaseRecord
from .common import aggregate_value, plt, select_time_ms


def plot(records: list[CaseRecord], out_path: str, agg: str) -> Optional[list[str]]:
    # Group By: backend -> algorithm -> (n, k) -> list of times
    grouped: dict[str, dict[str, dict[tuple[int, int], list[float]]]] = defaultdict(
        lambda: defaultdict(lambda: defaultdict(list))
    )

    for rec in records:
        if not rec.backend or rec.n is None or rec.k is None:
            continue
        t = select_time_ms(rec, "algorithmic")
        if t is not None and t > 0:
            grouped[rec.backend][rec.algorithm][(rec.n, rec.k)].append(t)

    if not grouped:
        return None

    base_dir = os.path.dirname(out_path) or "."
    base_name = os.path.splitext(os.path.basename(out_path))[0]
    outputs: list[str] = []

    for backend, algo_map in grouped.items():
        for algo, points in algo_map.items():
            unique_ns = sorted(list(set(n for (n, k) in points.keys())))
            unique_ks = sorted(list(set(k for (n, k) in points.keys())))

            if not unique_ns or not unique_ks:
                continue

            # data into a 2D numpy matrix
            matrix = np.full((len(unique_ks), len(unique_ns)), np.nan)
            for (n, k), times in points.items():
                n_idx = unique_ns.index(n)
                k_idx = unique_ks.index(k)
                matrix[k_idx, n_idx] = aggregate_value(times, agg)

            fig, ax = plt.subplots(figsize=(10, 8))

            # mask NaNs and apply logarithmic color normalization
            masked_matrix = np.ma.masked_invalid(matrix)
            im = ax.imshow(
                masked_matrix,
                origin="lower",
                cmap="turbo",
                aspect="auto",
                norm=mcolors.LogNorm(vmin=np.nanmin(matrix), vmax=np.nanmax(matrix)),
            )

            ax.set_xticks(np.arange(len(unique_ns)))
            ax.set_yticks(np.arange(len(unique_ks)))
            ax.set_xticklabels([f"$2^{{{int(np.log2(n))}}}$" if n > 0 else "0" for n in unique_ns])
            ax.set_yticklabels([str(k) for k in unique_ks])

            ax.set_xlabel("Input Size N")
            ax.set_ylabel("Requested Top-K")
            ax.set_title(f"Heatmap: Algorithmic Time ({backend} - {algo})")

            # pad to match the colorbar
            cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            cbar.set_label("Time (ms, log scale)")

            fig.tight_layout()
            os.makedirs(base_dir, exist_ok=True)
            out_file = os.path.join(base_dir, f"{base_name}_{backend}_{algo}.png")
            fig.savefig(out_file, dpi=160)
            plt.close(fig)
            outputs.append(out_file)

    return outputs
