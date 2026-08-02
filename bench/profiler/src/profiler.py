#!/usr/bin/env python3
from __future__ import annotations

import argparse
import glob
import os

from .csv_io import write_case_csv, write_measurement_csv, write_roofline_csv
from .filtering import filter_measurements, filter_records
from .parsing import attach_inprocess_energy, parse_measurements, parse_roofline, parse_test_output
from .plotting.common import MATPLOTLIB_AVAILABLE, set_dtype_context
from .plotting import dist_compare
from .plotting import edp_vs_n
from .plotting import energy_by_backend
from .plotting import energy_by_source
from .plotting import energy_per_element_vs_n
from .plotting import energy_vs_n
from .plotting import energy_vs_n_metric_compare
from .plotting import pass_rate
from .plotting import power_by_backend
from .plotting import power_by_source
from .plotting import power_vs_n
from .plotting import speedup_vs_gt
from .plotting import time_per_element_vs_n
from .plotting import time_vs_n_algo_compare
from .plotting import time_vs_energy
from .plotting import time_vs_n
from .plotting import time_vs_n_backend_compare
from .plotting import time_vs_n_metric_compare
from .plotting import time_vs_n_k_colored
from .plotting import heatmap_time
from .plotting import memory_bandwidth_vs_n
from .plotting import roof_utilization
from .plotting import roofline, walls


def metric_path(out_dir: str, name: str, metric: str) -> str:
    base, ext = os.path.splitext(name)
    suffix = "" if metric == "total" else f"_{metric}"
    return os.path.join(out_dir, f"{base}{suffix}{ext}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Parse testcase output and generate profiler plots.")
    parser.add_argument(
        "--input",
        nargs="+",
        default=["bench/results/raw/cases/output.json"],
        help="Path(s) to testcase output file(s) (.json or .txt).",
    )
    parser.add_argument(
        "--output-dir",
        default="bench/results/plots",
        help=(
            "Root directory where plots are written "
            "(plots are placed under <output-dir>/<dtype>/{time,correctness,energy,memory}/)."
        ),
    )
    parser.add_argument(
        "--plot",
        nargs="+",
        default=["all"],
        choices=[
            "all",
            "none",
            "time-vs-n",
            "time-vs-n-algo-compare",
            "time-vs-n-backend-compare",
            "time-vs-n-metric-compare",
            "speedup-vs-gt",
            "pass-rate",
            "energy-by-source",
            "power-by-source",
            "energy-vs-n",
            "energy-vs-n-metric-compare",
            "power-vs-n",
            "energy-by-backend",
            "power-by-backend",
            "time-vs-energy",
            "edp-vs-n",
            "energy-per-element-vs-n",
            "time-per-element-vs-n",
            "memory-bandwidth-vs-n",
            "roof-utilization",
            "roofline",
            "roofline-kernels",
            "time-vs-n-k-colored",
            "heatmap-time",
            "dist-compare",
        ],
        help="Plot(s) to generate.",
    )
    parser.add_argument(
        "--dtype",
        nargs="*",
        default=[],
        help="Optional dtype filter(s), e.g. int float.",
    )
    parser.add_argument(
        "--algorithm",
        nargs="*",
        default=[],
        help="Optional algorithm filter(s), e.g. bitonic map_reduce.",
    )
    parser.add_argument(
        "--mode",
        choices=["max", "min"],
        default=None,
        help="Optional mode filter.",
    )
    parser.add_argument("--k", type=int, default=None, help="Optional top-k filter.")
    parser.add_argument(
        "--fanout-k",
        type=int,
        nargs="*",
        default=[],
        help=(
            "Restrict which k values get their own figure in the per-k plot families. "
            "Plots that carry k as an axis (heatmap-time, time-vs-n-k-colored) keep every "
            "measured k regardless. Default: every k present."
        ),
    )
    parser.add_argument(
        "--dist",
        nargs="*",
        default=[],
        help="Optional input-distribution filter(s), e.g. uniform trimodal adversarial.",
    )
    parser.add_argument(
        "--backend",
        nargs="*",
        default=[],
        help="Optional backend filter(s), e.g. cpu gt gpu npu.",
    )
    parser.add_argument(
        "--timing-csv-out",
        default="bench/results/derived/cases.csv",
        help="CSV export path for parsed case records. Pass '' to skip.",
    )
    parser.add_argument(
        "--measurement-input",
        nargs="*",
        default=[],
        help="Optional measurement text files from measure_rapl.sh / measure_smi.sh.",
    )
    parser.add_argument(
        "--measurement-glob",
        default="bench/results/raw/energy/measurements/*.txt",
        help="Glob for auto-discovered measurement files.",
    )
    parser.add_argument(
        "--energy-csv-out",
        default="bench/results/derived/energy.csv",
        help="CSV export path for parsed measurement records. Pass '' to skip.",
    )
    parser.add_argument(
        "--agg",
        choices=["mean", "median"],
        default="mean",
        help="Aggregation method across repeated runs (default: mean).",
    )
    parser.add_argument(
        "--error-bars",
        choices=["none", "p10-p90", "std"],
        default="std",
        help="Error band style for line/bar plots (default: std).",
    )
    parser.add_argument(
        "--energy-metric",
        choices=["total", "net", "both"],
        default="both",
        help="Energy/power metric: raw total, idle-baseline-subtracted net, or both (default: both).",
    )
    parser.add_argument(
        "--compare-n",
        type=int,
        default=None,
        help="Optional fixed N for backend comparison bars and Pareto plot.",
    )
    parser.add_argument(
        "--roofline-input",
        nargs="*",
        default=["bench/results/raw/roofline/roofline.json"],
        help="Roofline measurement JSON from bench/lib/roofline.py.",
    )
    parser.add_argument(
        "--roofline-csv-out",
        default="bench/results/derived/roofline.csv",
        help="CSV export path for roofline sweep points (machine-level, not per dtype). Pass '' to skip.",
    )
    parser.add_argument("--show", action="store_true", help="Show figures interactively.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    all_records: list = []
    for input_path in args.input:
        if os.path.isfile(input_path):
            all_records.extend(parse_test_output(input_path))
        else:
            print(f"Warning: testcase input not found: {input_path}")

    dtypes_filter = {d.lower() for d in args.dtype}
    algorithms = {a.lower() for a in args.algorithm}
    backends = {b.lower() for b in args.backend}
    dists = {d.lower() for d in args.dist}
    fanout_ks = set(args.fanout_k)
    energy_metrics = ["total", "net"] if args.energy_metric == "both" else [args.energy_metric]

    measurement_paths = list(args.measurement_input)
    measurement_paths.extend(sorted(glob.glob(args.measurement_glob)))
    measurement_paths = list(dict.fromkeys(measurement_paths))

    measurement_records = parse_measurements(measurement_paths)
    roofline_points = parse_roofline([p for p in args.roofline_input if os.path.isfile(p)])

    bench_ops_by_key: dict = {}
    global_bench_ops = None
    for rec in all_records:
        if rec.bench_ops:
            global_bench_ops = rec.bench_ops
            bench_ops_by_key[(rec.backend, rec.dtype, rec.algorithm, rec.n, rec.k)] = rec.bench_ops
    for rec in measurement_records:
        ops = bench_ops_by_key.get((rec.backend, rec.dtype, rec.algorithm, rec.n, rec.k), global_bench_ops)
        if ops:
            rec.bench_ops = ops

    inproc_by_key: dict = {}
    for rec in all_records:
        if getattr(rec, "energy_status", "") == "ok":
            inproc_by_key[(rec.backend, rec.dtype, rec.algorithm, rec.n, rec.k, rec.dist, rec.seed, rec.rep)] = rec
    attach_inprocess_energy(measurement_records, inproc_by_key)

    # Dtype selection
    discovered_dtypes = sorted(
        {
            *(rec.dtype for rec in all_records if rec.dtype),
            *(rec.dtype for rec in measurement_records if rec.dtype),
        }
    )
    dtypes_to_plot = sorted(dtypes_filter) if dtypes_filter else discovered_dtypes
    if not dtypes_to_plot:
        dtypes_to_plot = ["unknown"]

    requested = set(args.plot)
    if "all" in requested:
        requested = {
            "time-vs-n",
            "time-vs-n-algo-compare",
            "time-vs-n-backend-compare",
            "time-vs-n-metric-compare",
            "speedup-vs-gt",
            "pass-rate",
            "energy-by-source",
            "power-by-source",
            "energy-vs-n",
            "energy-vs-n-metric-compare",
            "power-vs-n",
            "energy-by-backend",
            "power-by-backend",
            "time-vs-energy",
            "edp-vs-n",
            "energy-per-element-vs-n",
            "time-per-element-vs-n",
            "memory-bandwidth-vs-n",
            "roof-utilization",
            "roofline",
            "roofline-kernels",
            "time-vs-n-k-colored",
            "heatmap-time",
            "dist-compare",
        }
    if "none" in requested:
        requested.remove("none")

    if not requested:
        if not all_records and not measurement_records:
            print("No data available.")
            return 2
        if not args.timing_csv_out and not args.energy_csv_out and not args.roofline_csv_out:
            return 0

    if requested and not MATPLOTLIB_AVAILABLE:
        print("matplotlib is not available. Run with --plot none for CSV-only mode.")
        return 1

    generated: list[str] = []

    def collect_output(out: object) -> None:
        if not out:
            return
        if isinstance(out, list):
            generated.extend(out)
        else:
            generated.append(out)

    # Generate plots per dtype into separate directories.
    for dtype in dtypes_to_plot:
        dtype_dir = dtype if dtype else "unknown"
        out_root = os.path.join(args.output_dir, dtype_dir)
        out_time = os.path.join(out_root, "time")
        out_correctness = os.path.join(out_root, "correctness")
        out_energy = os.path.join(out_root, "energy")
        out_memory = os.path.join(out_root, "memory")
        out_dist = os.path.join(out_root, "dist")

        set_dtype_context(dtype_dir)

        dtype_set = {dtype} if dtype and dtype != "unknown" else set()

        records = filter_records(
            all_records,
            dtypes=dtype_set,
            algorithms=algorithms,
            mode=args.mode,
            k_value=args.k,
            backends=backends,
            dists=dists,
        )
        dtype_measurements = filter_measurements(
            measurement_records,
            dtypes=dtype_set,
            algorithms=algorithms,
            mode=args.mode,
            k_value=args.k,
            backends=backends,
            dists=dists,
        )

        # Create GT-stripped subsets for plots that shouldn't show the Ground Truth
        records_no_gt = [r for r in records if r.algorithm != "gt"]
        dtype_measurements_no_gt = [m for m in dtype_measurements if m.algorithm != "gt"]

        records_fan = [r for r in records if not fanout_ks or r.k in fanout_ks]
        records_no_gt_fan = [r for r in records_no_gt if not fanout_ks or r.k in fanout_ks]
        measurements_no_gt_fan = [
            m for m in dtype_measurements_no_gt if not fanout_ks or m.k is None or m.k in fanout_ks
        ]

        if not records and ("time-vs-n" in requested or "speedup-vs-gt" in requested or "pass-rate" in requested):
            print(f"No testcase records for dtype '{dtype_dir}' matched the selected filters.")
        if not dtype_measurements and any(
            p in requested
            for p in {
                "energy-by-source",
                "power-by-source",
                "energy-vs-n",
                "power-vs-n",
                "energy-by-backend",
                "power-by-backend",
                "time-vs-energy",
                "edp-vs-n",
                "energy-per-element-vs-n",
            }
        ):
            print(f"No measurement records for dtype '{dtype_dir}' matched the selected filters.")

        # CSVs keep all records (including GT) for raw data completeness
        if args.timing_csv_out and records:
            base, ext = os.path.splitext(args.timing_csv_out)
            ext = ext or ".csv"
            csv_path = f"{base}_{dtype_dir}{ext}" if len(dtypes_to_plot) > 1 else args.timing_csv_out
            write_case_csv(records, csv_path)
            print(f"Wrote CSV: {csv_path}")

        if args.energy_csv_out and dtype_measurements:
            base, ext = os.path.splitext(args.energy_csv_out)
            ext = ext or ".csv"
            csv_path = f"{base}_{dtype_dir}{ext}" if len(dtypes_to_plot) > 1 else args.energy_csv_out
            write_measurement_csv(dtype_measurements, csv_path)
            print(f"Wrote measurement CSV: {csv_path}")

        if "time-vs-n" in requested:
            out = time_vs_n.plot(
                records_no_gt_fan,
                os.path.join(out_time, "time_vs_n.png"),
                args.agg,
                args.error_bars,
            )
            if out:
                collect_output(out)
            else:
                print(f"Skipped time-vs-n ({dtype_dir}): no timing points found.")

        # HAS GT - Gets full 'records'
        if "time-vs-n-algo-compare" in requested:
            out = time_vs_n_algo_compare.plot(
                records_fan,
                os.path.join(out_time, "time_vs_n_algo_compare.png"),
                args.agg,
                args.error_bars,
            )
            if out:
                collect_output(out)
            else:
                print(f"Skipped time-vs-n-algo-compare ({dtype_dir}): no timing points found.")

        if "time-vs-n-backend-compare" in requested:
            out = time_vs_n_backend_compare.plot(
                records_no_gt_fan,
                os.path.join(out_time, "time_vs_n_backend_compare.png"),
                args.agg,
                args.error_bars,
            )
            if out:
                collect_output(out)
            else:
                print(f"Skipped time-vs-n-backend-compare ({dtype_dir}): no timing points found.")

        if "time-vs-n-metric-compare" in requested:
            out = time_vs_n_metric_compare.plot(
                records_no_gt,
                os.path.join(out_time, "time_vs_n_metric_compare.png"),
                args.agg,
                args.error_bars,
            )
            if out:
                collect_output(out)
            else:
                print(f"Skipped time-vs-n-metric-compare ({dtype_dir}): no timing points found.")

        if "time-vs-n-k-colored" in requested:
            out = time_vs_n_k_colored.plot(
                records_no_gt,
                os.path.join(out_time, "time_vs_n_k_colored.png"),
                args.agg,
            )
            if out:
                collect_output(out)
            else:
                print(f"Skipped time-vs-n-k-colored ({dtype_dir}): missing data.")

        if "heatmap-time" in requested:
            out = heatmap_time.plot(
                records_no_gt,
                os.path.join(out_time, "heatmap_time.png"),
                args.agg,
            )
            if out:
                collect_output(out)
            else:
                print(f"Skipped heatmap-time ({dtype_dir}): missing data.")

        if "dist-compare" in requested:
            produced = False
            for out in (
                dist_compare.plot_time_vs_n(
                    records_no_gt_fan,
                    os.path.join(out_dist, "time_vs_n_dist_compare.png"),
                    args.agg,
                    args.error_bars,
                ),
                dist_compare.plot_sensitivity(
                    records_fan,
                    os.path.join(out_dist, "dist_sensitivity.png"),
                    args.agg,
                    args.compare_n,
                ),
            ):
                if out:
                    collect_output(out)
                    produced = True
            if not produced:
                print(f"Skipped dist-compare ({dtype_dir}): fewer than two input distributions in the data.")

        if "speedup-vs-gt" in requested:
            out = speedup_vs_gt.plot(records_fan, os.path.join(out_time, "speedup_vs_gt.png"), args.agg)
            if out:
                collect_output(out)
            else:
                print(f"Skipped speedup-vs-gt ({dtype_dir}): no gt/backend matches found.")

        if "time-per-element-vs-n" in requested:
            out = time_per_element_vs_n.plot(
                records_no_gt_fan,
                os.path.join(out_time, "time_per_element_vs_n.png"),
                args.agg,
                args.error_bars,
            )
            if out:
                collect_output(out)
            else:
                print(f"Skipped time-per-element-vs-n ({dtype_dir}): no testcase records with N and time were found.")

        if "memory-bandwidth-vs-n" in requested:
            # Extract all unique K values tested for this dtype
            unique_ks = sorted(list({r.k for r in records_fan if r.k is not None}))

            for current_k in unique_ks:
                # all backends, no gt
                records_no_gt_k = [r for r in records_no_gt if r.k == current_k]

                if records_no_gt_k:
                    out_main = memory_bandwidth_vs_n.plot(
                        records_no_gt_k,
                        os.path.join(out_memory, f"memory_bandwidth_vs_n_k{current_k}.png"),
                        args.agg,
                        args.error_bars,
                        title=f"Effective Memory Bandwidth vs. N (All Backends, K={current_k})",
                        roofline_points=roofline_points,
                    )
                    if out_main:
                        collect_output(out_main)

                # per backend, with gt
                unique_backends = sorted(list({r.backend for r in records if r.backend}))
                for b in unique_backends:
                    backend_records_k = [r for r in records if r.backend == b and r.k == current_k]
                    if not backend_records_k:
                        continue

                    out_b = memory_bandwidth_vs_n.plot(
                        backend_records_k,
                        os.path.join(out_memory, f"{b}_memory_bandwidth_vs_n_k{current_k}.png"),
                        args.agg,
                        args.error_bars,
                        title=f"{b.upper()} Effective Memory Bandwidth vs. N (with GT, K={current_k})",
                        roofline_points=roofline_points,
                    )
                    if out_b:
                        collect_output(out_b)

        if "roof-utilization" in requested:
            outs = roof_utilization.plot(records, roofline_points, out_memory, args.agg)
            for out in outs:
                collect_output(out)
            if not outs:
                reason = (
                    "no roofline sweep points; run 'make measure-roofline' first"
                    if not roofline_points
                    else "no testcase records carry traffic counters"
                )
                print(f"Skipped roof-utilization ({dtype_dir}): {reason}.")

        if "roofline-kernels" in requested:
            out = roofline.plot_kernels(
                records,
                roofline_points,
                os.path.join(out_memory, "roofline_kernels.png"),
                title=f"Top-k Kernels on the Measured Roofline ({dtype_dir})",
            )
            if out:
                collect_output(out)
            else:
                reason = (
                    "no roofline sweep points; run 'make measure-roofline' first"
                    if not roofline_points
                    else "no testcase records carry traffic counters"
                )
                print(f"Skipped roofline-kernels ({dtype_dir}): {reason}.")

        if "pass-rate" in requested:
            out = pass_rate.plot(records_no_gt, os.path.join(out_correctness, "pass_rate.png"))
            if out:
                collect_output(out)
            else:
                print(f"Skipped pass-rate ({dtype_dir}): no records found.")

        # Energy tools use stripped data
        if "energy-by-source" in requested:
            produced = False
            for metric in energy_metrics:
                out = energy_by_source.plot(
                    dtype_measurements_no_gt,
                    metric_path(out_energy, "energy_by_source.png", metric),
                    args.agg,
                    args.compare_n,
                    metric,
                )
                if out:
                    collect_output(out)
                    produced = True
            if not produced:
                print(f"Skipped energy-by-source ({dtype_dir}): no measurement energy points found.")

        if "power-by-source" in requested:
            produced = False
            for metric in energy_metrics:
                out = power_by_source.plot(
                    dtype_measurements_no_gt,
                    metric_path(out_energy, "power_by_source.png", metric),
                    args.agg,
                    args.compare_n,
                    metric,
                )
                if out:
                    collect_output(out)
                    produced = True
            if not produced:
                print(f"Skipped power-by-source ({dtype_dir}): no measurement power points found.")

        if "energy-vs-n" in requested:
            produced = False
            for metric in energy_metrics:
                out = energy_vs_n.plot(
                    measurements_no_gt_fan,
                    metric_path(out_energy, "energy_vs_n.png", metric),
                    args.agg,
                    args.error_bars,
                    metric,
                )
                if out:
                    collect_output(out)
                    produced = True
            if not produced:
                print(f"Skipped energy-vs-n ({dtype_dir}): no measurement records with inferred N were found.")

        if "energy-vs-n-metric-compare" in requested:
            produced = False
            for metric in energy_metrics:
                out = energy_vs_n_metric_compare.plot(
                    dtype_measurements_no_gt,
                    metric_path(out_energy, "energy_vs_n_metric_compare.png", metric),
                    args.agg,
                    args.error_bars,
                    metric,
                )
                if out:
                    collect_output(out)
                    produced = True
            if not produced:
                print(f"Skipped energy-vs-n-metric-compare ({dtype_dir}): no measurement records with times found.")

        if "power-vs-n" in requested:
            produced = False
            for metric in energy_metrics:
                out = power_vs_n.plot(
                    measurements_no_gt_fan,
                    metric_path(out_energy, "power_vs_n.png", metric),
                    args.agg,
                    args.error_bars,
                    metric,
                )
                if out:
                    collect_output(out)
                    produced = True
            if not produced:
                print(f"Skipped power-vs-n ({dtype_dir}): no measurement records with inferred N were found.")

        if "energy-by-backend" in requested:
            produced = False
            for metric in energy_metrics:
                out = energy_by_backend.plot(
                    measurements_no_gt_fan,
                    metric_path(out_energy, "energy_by_backend.png", metric),
                    args.agg,
                    args.compare_n,
                    args.error_bars,
                    metric,
                )
                if out:
                    collect_output(out)
                    produced = True
            if not produced:
                print(f"Skipped energy-by-backend ({dtype_dir}): no matching measurement energy points found.")

        if "power-by-backend" in requested:
            produced = False
            for metric in energy_metrics:
                out = power_by_backend.plot(
                    measurements_no_gt_fan,
                    metric_path(out_energy, "power_by_backend.png", metric),
                    args.agg,
                    args.compare_n,
                    args.error_bars,
                    metric,
                )
                if out:
                    collect_output(out)
                    produced = True
            if not produced:
                print(f"Skipped power-by-backend ({dtype_dir}): no matching measurement power points found.")

        if "time-vs-energy" in requested:
            out = time_vs_energy.plot(
                records_no_gt,
                dtype_measurements_no_gt,
                os.path.join(out_energy, "time_vs_energy.png"),
                args.agg,
                args.compare_n,
            )
            if out:
                collect_output(out)
            else:
                print(f"Skipped time-vs-energy ({dtype_dir}): no matched runtime and energy records were found.")

        if "edp-vs-n" in requested:
            produced = False
            for metric in energy_metrics:
                out = edp_vs_n.plot(
                    measurements_no_gt_fan,
                    metric_path(out_energy, "edp_vs_n.png", metric),
                    args.agg,
                    args.error_bars,
                    metric,
                )
                if out:
                    collect_output(out)
                    produced = True
            if not produced:
                print(
                    f"Skipped edp-vs-n ({dtype_dir}): no measurement records with elapsed time and energy were found."
                )

        if "energy-per-element-vs-n" in requested:
            produced = False
            for metric in energy_metrics:
                out = energy_per_element_vs_n.plot(
                    measurements_no_gt_fan,
                    metric_path(out_energy, "energy_per_element_vs_n.png", metric),
                    args.agg,
                    args.error_bars,
                    metric,
                )
                if out:
                    collect_output(out)
                    produced = True
            if not produced:
                print(
                    f"Skipped energy-per-element-vs-n ({dtype_dir}): "
                    "no measurement records with N and energy were found."
                )

    if args.roofline_csv_out:
        if roofline_points:
            write_roofline_csv(roofline_points, args.roofline_csv_out)
            print(f"Wrote roofline CSV: {args.roofline_csv_out}")
        else:
            print("Skipped roofline CSV: no sweep points found. Run 'make measure-roofline' first.")

    # The roofline characterizes the machine, not a dtype, so it is emitted once and
    # without a dtype subtitle.
    set_dtype_context(None)

    if "roofline" in requested:
        out = roofline.plot(roofline_points, os.path.join(args.output_dir, "machine", "roofline.png"))
        if out:
            collect_output(out)
        else:
            print("Skipped roofline: no sweep points found. Run 'make measure-roofline' first.")

        for producer, name in (
            (walls.plot_cache_ladder, "cache_ladder.png"),
            (walls.plot_transfer_walls, "transfer_walls.png"),
        ):
            out = producer(roofline_points, os.path.join(args.output_dir, "machine", name))
            if out:
                collect_output(out)
            else:
                print(f"Skipped {name}: run 'make measure-roofline' with exp=all first.")

    if generated:
        print("Generated plot files:")
        for path in generated:
            print(f"- {path}")

    if args.show:
        from .plotting.common import plt

        plt.show()

    if not generated and not args.timing_csv_out and not args.energy_csv_out:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
