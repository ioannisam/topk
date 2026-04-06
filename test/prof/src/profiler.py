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
from .plotting import time_vs_energy
from .plotting import time_vs_n


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Parse testcase output and generate profiler plots.")
    parser.add_argument(
        "--input",
        default="test/prof/results/test_output.txt",
        help="Path to testcase output text file.",
    )
    parser.add_argument(
        "--output-dir",
        default="test/prof/results/plots",
        help="Directory where plots are written.",
    )
    parser.add_argument(
        "--plot",
        nargs="+",
        default=["all"],
        choices=[
            "all",
            "none",
            "time-vs-n",
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
        default="median",
        help="Aggregation method across repeated runs (default: median).",
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

    all_records = []
    if os.path.isfile(args.input):
        all_records = parse_test_output(args.input)
    else:
        print(f"Warning: testcase input not found: {args.input}")

    dtypes = {d.lower() for d in args.dtype}
    backends = {b.lower() for b in args.backend}

    records = filter_records(
        all_records,
        dtypes=dtypes,
        mode=args.mode,
        k_value=args.k,
        backends=backends,
    )

    if not records:
        print("No testcase records matched the selected filters.")

    if args.csv_out:
        write_case_csv(records, args.csv_out)
        print(f"Wrote CSV: {args.csv_out}")

    measurement_paths = list(args.measurement_input)
    measurement_paths.extend(sorted(glob.glob(args.measurement_glob)))
    measurement_paths = list(dict.fromkeys(measurement_paths))

    measurement_records = parse_measurements(measurement_paths)
    measurement_records = filter_measurements(
        measurement_records,
        dtypes=dtypes,
        mode=args.mode,
        k_value=args.k,
        backends=backends,
    )

    if args.measurement_csv_out:
        write_measurement_csv(measurement_records, args.measurement_csv_out)
        print(f"Wrote measurement CSV: {args.measurement_csv_out}")

    requested = set(args.plot)
    if "all" in requested:
        requested = {
            "time-vs-n",
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
        }
    if "none" in requested:
        requested.remove("none")

    if not requested:
        if not records and not measurement_records:
            print("No data available.")
            return 2
        return 0

    if not MATPLOTLIB_AVAILABLE:
        print("matplotlib is not available. Run with --plot none for CSV-only mode.")
        return 1

    generated: list[str] = []

    if "time-vs-n" in requested:
        out = time_vs_n.plot(records, os.path.join(args.output_dir, "time_vs_n.png"), args.agg, args.error_bars)
        if out:
            generated.append(out)
        else:
            print("Skipped time-vs-n: no timing points found.")

    if "speedup-vs-gt" in requested:
        out = speedup_vs_gt.plot(records, os.path.join(args.output_dir, "speedup_vs_gt.png"))
        if out:
            generated.append(out)
        else:
            print("Skipped speedup-vs-gt: no gt/backend matches found.")

    if "pass-rate" in requested:
        out = pass_rate.plot(records, os.path.join(args.output_dir, "pass_rate.png"))
        if out:
            generated.append(out)
        else:
            print("Skipped pass-rate: no records found.")

    if "energy-by-source" in requested:
        out = energy_by_source.plot(
            measurement_records,
            os.path.join(args.output_dir, "energy_by_source.png"),
            args.agg,
            args.compare_n,
        )
        if out:
            generated.append(out)
        else:
            print("Skipped energy-by-source: no measurement energy points found.")

    if "power-by-source" in requested:
        out = power_by_source.plot(
            measurement_records,
            os.path.join(args.output_dir, "power_by_source.png"),
            args.agg,
            args.compare_n,
        )
        if out:
            generated.append(out)
        else:
            print("Skipped power-by-source: no measurement power points found.")

    if "energy-vs-n" in requested:
        out = energy_vs_n.plot(
            measurement_records,
            os.path.join(args.output_dir, "energy_vs_n.png"),
            args.agg,
            args.error_bars,
        )
        if out:
            generated.append(out)
        else:
            print("Skipped energy-vs-n: no measurement records with inferred N were found.")

    if "power-vs-n" in requested:
        out = power_vs_n.plot(
            measurement_records,
            os.path.join(args.output_dir, "power_vs_n.png"),
            args.agg,
            args.error_bars,
        )
        if out:
            generated.append(out)
        else:
            print("Skipped power-vs-n: no measurement records with inferred N were found.")

    if "energy-by-backend" in requested:
        out = energy_by_backend.plot(
            measurement_records,
            os.path.join(args.output_dir, "energy_by_backend.png"),
            args.agg,
            args.compare_n,
            args.error_bars,
        )
        if out:
            generated.append(out)
        else:
            print("Skipped energy-by-backend: no matching measurement energy points found.")

    if "power-by-backend" in requested:
        out = power_by_backend.plot(
            measurement_records,
            os.path.join(args.output_dir, "power_by_backend.png"),
            args.agg,
            args.compare_n,
            args.error_bars,
        )
        if out:
            generated.append(out)
        else:
            print("Skipped power-by-backend: no matching measurement power points found.")

    if "time-vs-energy" in requested:
        out = time_vs_energy.plot(
            records,
            measurement_records,
            os.path.join(args.output_dir, "time_vs_energy.png"),
            args.agg,
            args.compare_n,
        )
        if out:
            generated.append(out)
        else:
            print("Skipped time-vs-energy: no matched runtime and energy records were found.")

    if "edp-vs-n" in requested:
        out = edp_vs_n.plot(
            measurement_records,
            os.path.join(args.output_dir, "edp_vs_n.png"),
            args.agg,
            args.error_bars,
        )
        if out:
            generated.append(out)
        else:
            print("Skipped edp-vs-n: no measurement records with elapsed time and energy were found.")

    if "energy-per-element-vs-n" in requested:
        out = energy_per_element_vs_n.plot(
            measurement_records,
            os.path.join(args.output_dir, "energy_per_element_vs_n.png"),
            args.agg,
            args.error_bars,
        )
        if out:
            generated.append(out)
        else:
            print("Skipped energy-per-element-vs-n: no measurement records with N and energy were found.")

    if "time-per-element-vs-n" in requested:
        out = time_per_element_vs_n.plot(
            records,
            os.path.join(args.output_dir, "time_per_element_vs_n.png"),
            args.agg,
            args.error_bars,
        )
        if out:
            generated.append(out)
        else:
            print("Skipped time-per-element-vs-n: no testcase records with N and time were found.")

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
