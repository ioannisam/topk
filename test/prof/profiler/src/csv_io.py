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
                "mode",
                "k",
                "n",
                "timing_label",
                "time_ms",
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
                    rec.mode,
                    rec.k if rec.k is not None else "",
                    rec.n if rec.n is not None else "",
                    rec.timing_label,
                    "" if rec.time_ms is None else f"{rec.time_ms:.12g}",
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
                "mode",
                "k",
                "n",
                "file_path",
                "elapsed_seconds",
                "energy_joules",
                "average_watts",
                "command_exit_code",
            ]
        )
        for rec in records:
            writer.writerow(
                [
                    rec.source,
                    rec.backend,
                    rec.dtype,
                    rec.mode,
                    "" if rec.k is None else rec.k,
                    "" if rec.n is None else rec.n,
                    rec.file_path,
                    "" if rec.elapsed_seconds is None else f"{rec.elapsed_seconds:.12g}",
                    "" if rec.energy_joules is None else f"{rec.energy_joules:.12g}",
                    "" if rec.average_watts is None else f"{rec.average_watts:.12g}",
                    "" if rec.command_exit_code is None else rec.command_exit_code,
                ]
            )
