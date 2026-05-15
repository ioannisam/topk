#!/usr/bin/env python3
from __future__ import annotations

import argparse
import glob
import os

from .csv_io import write_case_csv, write_measurement_csv
from .filtering import filter_measurements, filter_records
from .parsing import parse_measurements, parse_test_output
from .plotting.common import MATPLOTLIB_AVAILABLE
from .plotting import edp_vs_n
from .plotting import energy_by_backend
from .plotting import energy_by_source
from .plotting import energy_per_element_vs_n
from .plotting import energy_vs_n
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
from .plotting import time_vs_k_backend_compare
from .plotting import time_vs_n_k_colored
from .plotting import heatmap_time


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Parse testcase output and generate profiler plots.")
    parser.add_argument(
        "--input",
        nargs="+",
        default=["test/prof/results/test_output.json"],
        help="Path(s) to testcase output file(s) (.json or .txt).",
    )
    parser.add_argument(
        "--output-dir",
        default="test/prof/results/plots",
        help=(
            "Root directory where plots are written "
            "(plots are placed under <output-dir>/<dtype>/{time,correctness,energy}/)."
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
            "power-vs-n",
            "energy-by-backend",
            "power-by-backend",
            "time-vs-energy",
            "edp-vs-n",
            "energy-per-element-vs-n",
            "time-per-element-vs-n",
            # "time-vs-k-backend-compare",
            "time-vs-n-k-colored",
            "heatmap-time",
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
        "--backend",
        nargs="*",
        default=[],
        help="Optional backend filter(s), e.g. cpu gt gpu npu.",
    )
    parser.add_argument(
        "--csv-out",
        default="",
        help="Optional CSV export path for parsed case records.",
    )
    parser.add_argument(
        "--measurement-input",
        nargs="*",
        default=[],
        help="Optional measurement text files from measure_rapl.sh / measure_smi.sh.",
    )
    parser.add_argument(
        "--measurement-glob",
        default="test/prof/results/measurements/*.txt",
        help="Glob for auto-discovered measurement files.",
    )
    parser.add_argument(
        "--measurement-csv-out",
        default="",
        help="Optional CSV export path for parsed measurement records.",
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
        default="p10-p90",
        help="Error band style for line/bar plots (default: p10-p90).",
    )
    parser.add_argument(
        "--compare-n",
        type=int,
        default=None,
        help="Optional fixed N for backend comparison bars and Pareto plot.",
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

    measurement_paths = list(args.measurement_input)
    measurement_paths.extend(sorted(glob.glob(args.measurement_glob)))
    measurement_paths = list(dict.fromkeys(measurement_paths))

    measurement_records = parse_measurements(measurement_paths)

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
            "power-vs-n",
            "energy-by-backend",
            "power-by-backend",
            "time-vs-energy",
            "edp-vs-n",
            "energy-per-element-vs-n",
            "time-per-element-vs-n",
            # "time-vs-k-backend-compare",
            "time-vs-n-k-colored",
            "heatmap-time",
        }
    if "none" in requested:
        requested.remove("none")

    if not requested:
        if not all_records and not measurement_records:
            print("No data available.")
            return 2
        return 0

    if not MATPLOTLIB_AVAILABLE:
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

        dtype_set = {dtype} if dtype and dtype != "unknown" else set()

        records = filter_records(
            all_records,
            dtypes=dtype_set,
            algorithms=algorithms,
            mode=args.mode,
            k_value=args.k,
            backends=backends,
        )
        dtype_measurements = filter_measurements(
            measurement_records,
            dtypes=dtype_set,
            algorithms=algorithms,
            mode=args.mode,
            k_value=args.k,
            backends=backends,
        )

        # Create GT-stripped subsets for plots that shouldn't show the Ground Truth
        records_no_gt = [r for r in records if r.algorithm != "gt"]
        dtype_measurements_no_gt = [m for m in dtype_measurements if m.algorithm != "gt"]

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
        if args.csv_out:
            base, ext = os.path.splitext(args.csv_out)
            ext = ext or ".csv"
            csv_path = f"{base}_{dtype_dir}{ext}" if len(dtypes_to_plot) > 1 else args.csv_out
            write_case_csv(records, csv_path)
            print(f"Wrote CSV: {csv_path}")

        if args.measurement_csv_out:
            base, ext = os.path.splitext(args.measurement_csv_out)
            ext = ext or ".csv"
            csv_path = f"{base}_{dtype_dir}{ext}" if len(dtypes_to_plot) > 1 else args.measurement_csv_out
            write_measurement_csv(dtype_measurements, csv_path)
            print(f"Wrote measurement CSV: {csv_path}")

        if "time-vs-n" in requested:
            out = time_vs_n.plot(
                records_no_gt,
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
                records,
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
                records_no_gt,
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

        if "time-vs-k-backend-compare" in requested:
            out = time_vs_k_backend_compare.plot(
                records_no_gt,
                os.path.join(out_time, "time_vs_k_backend_compare.png"),
                args.agg,
                args.error_bars,
            )
            if out:
                collect_output(out)
            else:
                print(f"Skipped time-vs-k-backend-compare ({dtype_dir}): missing multi-k timing points.")

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

        # HAS GT - Gets full 'records'
        if "speedup-vs-gt" in requested:
            out = speedup_vs_gt.plot(records, os.path.join(out_time, "speedup_vs_gt.png"), args.agg)
            if out:
                collect_output(out)
            else:
                print(f"Skipped speedup-vs-gt ({dtype_dir}): no gt/backend matches found.")

        if "time-per-element-vs-n" in requested:
            out = time_per_element_vs_n.plot(
                records_no_gt,
                os.path.join(out_time, "time_per_element_vs_n.png"),
                args.agg,
                args.error_bars,
            )
            if out:
                collect_output(out)
            else:
                print(f"Skipped time-per-element-vs-n ({dtype_dir}): no testcase records with N and time were found.")

        if "pass-rate" in requested:
            out = pass_rate.plot(records_no_gt, os.path.join(out_correctness, "pass_rate.png"))
            if out:
                collect_output(out)
            else:
                print(f"Skipped pass-rate ({dtype_dir}): no records found.")

        # Energy tools use stripped data
        if "energy-by-source" in requested:
            out = energy_by_source.plot(
                dtype_measurements_no_gt,
                os.path.join(out_energy, "energy_by_source.png"),
                args.agg,
                args.compare_n,
            )
            if out:
                collect_output(out)
            else:
                print(f"Skipped energy-by-source ({dtype_dir}): no measurement energy points found.")

        if "power-by-source" in requested:
            out = power_by_source.plot(
                dtype_measurements_no_gt,
                os.path.join(out_energy, "power_by_source.png"),
                args.agg,
                args.compare_n,
            )
            if out:
                collect_output(out)
            else:
                print(f"Skipped power-by-source ({dtype_dir}): no measurement power points found.")

        if "energy-vs-n" in requested:
            out = energy_vs_n.plot(
                dtype_measurements_no_gt,
                os.path.join(out_energy, "energy_vs_n.png"),
                args.agg,
                args.error_bars,
            )
            if out:
                collect_output(out)
            else:
                print(f"Skipped energy-vs-n ({dtype_dir}): no measurement records with inferred N were found.")

        if "power-vs-n" in requested:
            out = power_vs_n.plot(
                dtype_measurements_no_gt,
                os.path.join(out_energy, "power_vs_n.png"),
                args.agg,
                args.error_bars,
            )
            if out:
                collect_output(out)
            else:
                print(f"Skipped power-vs-n ({dtype_dir}): no measurement records with inferred N were found.")

        if "energy-by-backend" in requested:
            out = energy_by_backend.plot(
                dtype_measurements_no_gt,
                os.path.join(out_energy, "energy_by_backend.png"),
                args.agg,
                args.compare_n,
                args.error_bars,
            )
            if out:
                collect_output(out)
            else:
                print(f"Skipped energy-by-backend ({dtype_dir}): no matching measurement energy points found.")

        if "power-by-backend" in requested:
            out = power_by_backend.plot(
                dtype_measurements_no_gt,
                os.path.join(out_energy, "power_by_backend.png"),
                args.agg,
                args.compare_n,
                args.error_bars,
            )
            if out:
                collect_output(out)
            else:
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
            out = edp_vs_n.plot(
                dtype_measurements_no_gt,
                os.path.join(out_energy, "edp_vs_n.png"),
                args.agg,
                args.error_bars,
            )
            if out:
                collect_output(out)
            else:
                print(
                    f"Skipped edp-vs-n ({dtype_dir}): no measurement records with elapsed time and energy were found."
                )

        if "energy-per-element-vs-n" in requested:
            out = energy_per_element_vs_n.plot(
                dtype_measurements_no_gt,
                os.path.join(out_energy, "energy_per_element_vs_n.png"),
                args.agg,
                args.error_bars,
            )
            if out:
                collect_output(out)
            else:
                print(
                    f"Skipped energy-per-element-vs-n ({dtype_dir}): "
                    "no measurement records with N and energy were found."
                )

    if generated:
        print("Generated plot files:")
        for path in generated:
            print(f"- {path}")

    if args.show:
        from .plotting.common import plt

        plt.show()

    if not generated and not args.csv_out and not args.measurement_csv_out:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
