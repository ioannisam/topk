from __future__ import annotations

import os
from typing import Iterable, Optional

try:
    import matplotlib.pyplot as plt
    import numpy as np

    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False

from ..models import CaseRecord


def get_bytes_per_element(dtype: str) -> int:
    d = dtype.lower()
    if "double" in d:
        return 8
    if "fp16" in d:
        return 2
    return 4


def plot(
    records: Iterable[CaseRecord],
    out_path: str,
    agg: str = "mean",
    error_bars: str = "p10-p90",
    title: str = "Effective Memory Bandwidth vs. Input Size (N)",
) -> Optional[str]:
    if not MATPLOTLIB_AVAILABLE or not records:
        return None

    # Group data by (backend, algorithm)
    data: dict[tuple[str, str], dict[int, list[float]]] = {}

    for r in records:
        if r.n is None or r.time_ms is None or r.time_ms <= 0:
            continue

        bpe = get_bytes_per_element(r.dtype)
        bw_gbps = (r.n * bpe) / (r.time_ms * 1e6)

        key = (r.backend, r.algorithm)
        if key not in data:
            data[key] = {}
        if r.n not in data[key]:
            data[key][r.n] = []

        data[key][r.n].append(bw_gbps)

    if not data:
        return None

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.figure(figsize=(10, 6))

    for (backend, algo), n_dict in data.items():
        ns = sorted(n_dict.keys())

        if agg == "median":
            y = [float(np.median(n_dict[n])) for n in ns]
        else:
            y = [float(np.mean(n_dict[n])) for n in ns]

        label = f"{backend} - {algo}"
        p = plt.plot(ns, y, marker="o", label=label)
        color = p[0].get_color()

        if error_bars == "p10-p90":
            y10 = [float(np.percentile(n_dict[n], 10)) for n in ns]
            y90 = [float(np.percentile(n_dict[n], 90)) for n in ns]
            plt.fill_between(ns, y10, y90, color=color, alpha=0.2)
        elif error_bars == "std":
            ystd = [float(np.std(n_dict[n])) for n in ns]
            y_upper = [yy + s for yy, s in zip(y, ystd)]
            y_lower = [max(0, yy - s) for yy, s in zip(y, ystd)]
            plt.fill_between(ns, y_lower, y_upper, color=color, alpha=0.2)

    plt.xscale("log", base=2)
    plt.xlabel("Input Size N (elements)")
    plt.ylabel("Effective Bandwidth (GB/s)")
    plt.title(title)
    plt.legend()
    plt.grid(True, which="both", ls="--", alpha=0.7)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()

    return out_path
