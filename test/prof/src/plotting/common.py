from __future__ import annotations

import math
from statistics import mean, median, pstdev

try:
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except Exception:
    plt = None
    MATPLOTLIB_AVAILABLE = False


def style_axes(ax, title: str, xlabel: str, ylabel: str) -> None:
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.25)


def percentile(values: list[float], pct: float) -> float:
    if not values:
        raise ValueError("percentile on empty list")
    if len(values) == 1:
        return values[0]
    xs = sorted(values)
    rank = (len(xs) - 1) * pct
    lo = int(math.floor(rank))
    hi = int(math.ceil(rank))
    if lo == hi:
        return xs[lo]
    frac = rank - lo
    return xs[lo] * (1.0 - frac) + xs[hi] * frac


def aggregate_value(values: list[float], agg: str) -> float:
    if agg == "median":
        return median(values)
    return mean(values)


def error_bounds(values: list[float], center: float, error_bars: str) -> tuple[float, float]:
    if error_bars == "none" or len(values) < 2:
        return center, center
    if error_bars == "std":
        s = pstdev(values)
        return center - s, center + s
    return percentile(values, 0.10), percentile(values, 0.90)
