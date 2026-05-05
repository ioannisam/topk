from __future__ import annotations

import os
from collections import defaultdict
from typing import Optional

from ..models import CaseRecord
from .common import plt, style_axes


def plot(records: list[CaseRecord], out_path: str) -> Optional[str]:
    stats: dict[str, list[bool]] = defaultdict(list)
    for rec in records:
        stats[rec.backend].append(rec.pass_bool)

    if not stats:
        return None

    backends = sorted(stats.keys())
    rates = [100.0 * (sum(stats[b]) / len(stats[b])) for b in backends]

    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.bar(backends, rates)
    for bar, rate in zip(bars, rates):
        ax.text(bar.get_x() + bar.get_width() / 2, rate + 1.0, f"{rate:.1f}%", ha="center")

    style_axes(ax, "Pass Rate by Backend", "Backend", "Pass rate (%)")
    ax.set_ylim(0, 105)
    fig.tight_layout()

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
    return out_path
