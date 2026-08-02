from __future__ import annotations

import csv
import os
from typing import Iterable

from .models import CaseRecord, MeasurementRecord


def write_case_csv(records: Iterable[CaseRecord], out_path: str) -> None:
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "backend",
                "case_name",
                "status",
                "reason",
                "dtype",
                "algorithm",
                "mode",
                "k",
                "n",
                "dist",
                "seed",
                "rep",
                "timing_label",
                "time_ms",
                "timing_label_e2e",
                "time_end_to_end_ms",
                "timing_label_algorithmic",
                "time_algorithmic_ms",
                "bytes_moved",
                "bytes_model",
                "compare_ops",
                "ops_model",
            ]
        )
        for rec in records:
            writer.writerow(
                [
                    rec.backend,
                    rec.case_name,
                    rec.status,
                    rec.reason,
                    rec.dtype,
                    rec.algorithm,
                    rec.mode,
                    rec.k if rec.k is not None else "",
                    rec.n if rec.n is not None else "",
                    rec.dist,
                    "" if rec.seed is None else rec.seed,
                    "" if rec.rep is None else rec.rep,
                    rec.timing_label,
                    "" if rec.time_ms is None else f"{rec.time_ms:.12g}",
                    rec.timing_label_e2e,
                    "" if rec.time_end_to_end_ms is None else f"{rec.time_end_to_end_ms:.12g}",
                    rec.timing_label_algorithmic,
                    "" if rec.time_algorithmic_ms is None else f"{rec.time_algorithmic_ms:.12g}",
                    "" if rec.bytes_moved is None else f"{rec.bytes_moved:.12g}",
                    rec.bytes_model,
                    "" if rec.compare_ops is None else f"{rec.compare_ops:.12g}",
                    rec.ops_model,
                ]
            )


def write_measurement_csv(records: Iterable[MeasurementRecord], out_path: str) -> None:
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "source",
                "backend",
                "dtype",
                "algorithm",
                "mode",
                "k",
                "n",
                "dist",
                "seed",
                "rep",
                "file_path",
                "elapsed_seconds",
                "energy_joules",
                "average_watts",
                "baseline_watts",
                "net_energy_joules",
                "net_average_watts",
                "command_exit_code",
                "inproc_counters",
                "inproc_e2e_joules",
                "inproc_algo_joules",
                "inproc_e2e_seconds",
                "inproc_algo_seconds",
                "inproc_net_e2e_joules",
                "inproc_net_algo_joules",
                "inproc_loop_joules",
                "inproc_loop_seconds",
            ]
        )
        for rec in records:
            writer.writerow(
                [
                    rec.source,
                    rec.backend,
                    rec.dtype,
                    rec.algorithm,
                    rec.mode,
                    "" if rec.k is None else rec.k,
                    "" if rec.n is None else rec.n,
                    rec.dist,
                    "" if rec.seed is None else rec.seed,
                    "" if rec.rep is None else rec.rep,
                    rec.file_path,
                    "" if rec.elapsed_seconds is None else f"{rec.elapsed_seconds:.12g}",
                    "" if rec.energy_joules is None else f"{rec.energy_joules:.12g}",
                    "" if rec.average_watts is None else f"{rec.average_watts:.12g}",
                    "" if rec.baseline_watts is None else f"{rec.baseline_watts:.12g}",
                    "" if rec.net_energy_joules is None else f"{rec.net_energy_joules:.12g}",
                    "" if rec.net_average_watts is None else f"{rec.net_average_watts:.12g}",
                    "" if rec.command_exit_code is None else rec.command_exit_code,
                    rec.inproc_counters,
                    "" if rec.inproc_e2e_joules is None else f"{rec.inproc_e2e_joules:.12g}",
                    "" if rec.inproc_algo_joules is None else f"{rec.inproc_algo_joules:.12g}",
                    "" if rec.inproc_e2e_seconds is None else f"{rec.inproc_e2e_seconds:.12g}",
                    "" if rec.inproc_algo_seconds is None else f"{rec.inproc_algo_seconds:.12g}",
                    "" if rec.inproc_net_e2e_joules is None else f"{rec.inproc_net_e2e_joules:.12g}",
                    "" if rec.inproc_net_algo_joules is None else f"{rec.inproc_net_algo_joules:.12g}",
                    "" if rec.inproc_loop_joules is None else f"{rec.inproc_loop_joules:.12g}",
                    "" if rec.inproc_loop_seconds is None else f"{rec.inproc_loop_seconds:.12g}",
                ]
            )


def write_roofline_csv(points: Iterable, out_path: str) -> None:
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "backend",
                "kernel",
                "ops_per_elem",
                "elements",
                "bytes_moved",
                "ops",
                "ms_mean",
                "ms_stdev",
                "ms_min",
                "gbytes_per_s",
                "gops_per_s",
                "operational_intensity",
                "joules_per_iter",
            ]
        )
        for p in points:
            writer.writerow(
                [
                    p.backend,
                    p.kernel,
                    p.ops_per_elem,
                    p.elements,
                    f"{p.bytes_moved:.12g}",
                    f"{p.ops:.12g}",
                    f"{p.ms_mean:.12g}",
                    f"{p.ms_stdev:.12g}",
                    f"{p.ms_min:.12g}",
                    f"{p.gbytes_per_s:.12g}",
                    f"{p.gops_per_s:.12g}",
                    f"{p.operational_intensity:.12g}",
                    f"{p.joules_per_iter:.12g}",
                ]
            )
