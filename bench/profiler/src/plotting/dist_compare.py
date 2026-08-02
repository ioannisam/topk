from __future__ import annotations

import os
from collections import defaultdict
from typing import Optional

from ..models import CaseRecord
from .common import aggregate_value, error_bounds, plt, save_k_figure, select_time_ms, style_axes

DIST_ORDER = ["uniform", "normal", "trimodal", "sorted", "reverse", "adversarial"]

ALGO_LINESTYLE = {"bitonic": "-", "map_reduce": (0, (8, 3)), "gt": ":"}


def dist_rank(dist: str) -> int:
    return DIST_ORDER.index(dist) if dist in DIST_ORDER else len(DIST_ORDER)


def dist_color(dist: str):
    return plt.get_cmap("tab10")(dist_rank(dist) % 10)


def sorted_dists(dists) -> list[str]:
    return sorted(dists, key=lambda d: (dist_rank(d), d))


def add_dist_legend(ax, dists: list[str], algorithms: list[str]) -> None:
    from matplotlib.lines import Line2D

    handles = [Line2D([0], [0], color=dist_color(d), linewidth=2.4, label=d) for d in dists]
    handles += [
        Line2D([0], [0], color="black", linestyle=ALGO_LINESTYLE.get(a, "-"), linewidth=2.0, label=a)
        for a in algorithms
    ]
    ax.legend(
        handles=handles,
        title="Distribution / algorithm",
        loc="upper left",
        bbox_to_anchor=(1.02, 1.0),
        fontsize=9,
    )


def plot_time_vs_n(records: list[CaseRecord], out_path: str, agg: str, error_bars: str) -> Optional[list[str]]:
    grouped: dict[int, dict[str, dict[tuple[str, str], dict[int, list[float]]]]] = defaultdict(
        lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    )

    for rec in records:
        if not rec.backend or not rec.dist or rec.n is None or rec.k is None:
            continue
        time_ms = select_time_ms(rec, "algorithmic")
        if time_ms is None or time_ms <= 0:
            continue
        grouped[rec.k][rec.backend][(rec.dist, rec.algorithm)][rec.n].append(time_ms)

    if not grouped:
        return None

    all_dists = {d for k_map in grouped.values() for b_map in k_map.values() for (d, _) in b_map}
    if len(all_dists) < 2:
        return None

    base_dir = os.path.dirname(out_path) or "."
    base_name, ext = os.path.splitext(os.path.basename(out_path))
    outputs: list[str] = []

    for k, backend_map in sorted(grouped.items()):
        for backend in sorted(backend_map):
            series = backend_map[backend]
            fig, ax = plt.subplots(figsize=(10, 6))
            has_data = False

            for dist, algorithm in sorted(series, key=lambda p: (dist_rank(p[0]), p[0], p[1])):
                n_map = series[(dist, algorithm)]
                xs = sorted(n_map)
                if not xs:
                    continue
                ys, lowers, uppers = [], [], []
                for n in xs:
                    center = aggregate_value(n_map[n], agg)
                    lo, hi = error_bounds(n_map[n], center, error_bars)
                    ys.append(center)
                    lowers.append(lo)
                    uppers.append(hi)

                has_data = True
                ax.plot(
                    xs,
                    ys,
                    marker="o",
                    markersize=4,
                    linewidth=2,
                    color=dist_color(dist),
                    linestyle=ALGO_LINESTYLE.get(algorithm, "-"),
                )
                if error_bars != "none":
                    ax.fill_between(xs, lowers, uppers, alpha=0.12, color=dist_color(dist))

            if not has_data:
                plt.close(fig)
                continue

            ax.set_xscale("log", base=2)
            ax.set_yscale("log")
            style_axes(
                ax,
                f"{backend.upper()} Algorithmic Runtime by Input Distribution (K = {k})",
                "N (log2 scale)",
                "Time (ms, log scale)",
            )
            add_dist_legend(
                ax,
                sorted_dists({d for (d, _) in series}),
                sorted({a for (_, a) in series}),
            )
            fig.tight_layout()

            os.makedirs(base_dir, exist_ok=True)
            out_file = os.path.join(base_dir, f"{base_name}_{backend}_k{k}{ext}")
            fig.savefig(out_file, dpi=160, bbox_inches="tight")
            plt.close(fig)
            outputs.append(out_file)

    return outputs


def plot_sensitivity(
    records: list[CaseRecord], out_path: str, agg: str, compare_n: Optional[int]
) -> Optional[list[str]]:
    grouped: dict[int, dict[str, dict[str, dict[int, list[float]]]]] = defaultdict(
        lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    )

    for rec in records:
        if not rec.backend or not rec.dist or rec.n is None or rec.k is None:
            continue
        time_ms = select_time_ms(rec, "algorithmic")
        if time_ms is None or time_ms <= 0:
            continue
        grouped[rec.k][f"{rec.backend}/{rec.algorithm}"][rec.dist][rec.n].append(time_ms)

    if not grouped:
        return None

    all_dists = {d for k_map in grouped.values() for c_map in k_map.values() for d in c_map}
    if "uniform" not in all_dists or len(all_dists) < 2:
        return None

    base_dir = os.path.dirname(out_path) or "."
    base_name, ext = os.path.splitext(os.path.basename(out_path))
    outputs: list[str] = []

    for k, combo_map in sorted(grouped.items()):
        all_ns = {n for d_map in combo_map.values() for n_map in d_map.values() for n in n_map}
        if not all_ns:
            continue
        target_n = compare_n if (compare_n is not None and compare_n in all_ns) else max(all_ns)

        combos = sorted(combo_map)
        dists = sorted_dists({d for d_map in combo_map.values() for d in d_map})
        ratios: dict[str, list[Optional[float]]] = {d: [] for d in dists}

        for combo in combos:
            baseline_vals = combo_map[combo].get("uniform", {}).get(target_n)
            baseline = aggregate_value(baseline_vals, agg) if baseline_vals else None
            for dist in dists:
                vals = combo_map[combo].get(dist, {}).get(target_n)
                if not vals or not baseline or baseline <= 0:
                    ratios[dist].append(None)
                    continue
                ratios[dist].append(aggregate_value(vals, agg) / baseline)

        if not any(v is not None for series in ratios.values() for v in series):
            continue

        observed = [v for series in ratios.values() for v in series if v is not None]
        floor = min(0.5, min(observed) * 0.8)
        ceiling = max(2.0, max(observed) * 2.2)

        fig, ax = plt.subplots(figsize=(max(10, 1.6 * len(combos)), 6))
        width = 0.8 / max(1, len(dists))
        for idx, dist in enumerate(dists):
            xs = [i + (idx - (len(dists) - 1) / 2) * width for i in range(len(combos))]
            heights = [(v if v is not None else floor) - floor for v in ratios[dist]]
            bars = ax.bar(xs, heights, width=width, bottom=floor, color=dist_color(dist), label=dist)
            for bar, value in zip(bars, ratios[dist]):
                if value is None:
                    continue
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    value,
                    f"{value:.2f}",
                    ha="center",
                    va="bottom",
                    fontsize=7,
                    rotation=90,
                )

        ax.set_yscale("log")
        ax.set_ylim(floor, ceiling)
        ax.axhline(1.0, color="black", linewidth=1.0, linestyle="--", alpha=0.6)
        ax.set_xticks(range(len(combos)))
        ax.set_xticklabels(combos, rotation=30, ha="right")
        style_axes(
            ax,
            f"Runtime Sensitivity to Input Distribution (K = {k}, N = {target_n})",
            "Backend / algorithm",
            "Time relative to uniform (log scale)",
        )
        ax.legend(title="Distribution", loc="upper left", fontsize=9)
        fig.tight_layout()
        outputs.append(save_k_figure(fig, base_dir, base_name, k, ext, bbox_inches="tight"))

    return outputs
