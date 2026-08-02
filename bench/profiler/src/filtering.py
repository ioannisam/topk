from __future__ import annotations

from typing import Iterable, Optional

from .models import CaseRecord, MeasurementRecord


def filter_records(
    records: Iterable[CaseRecord],
    dtypes: set[str],
    algorithms: set[str],
    mode: Optional[str],
    k_value: Optional[int],
    backends: set[str],
    dists: Optional[set[str]] = None,
) -> list[CaseRecord]:
    out: list[CaseRecord] = []
    for rec in records:
        # drop failed runs
        if not rec.pass_bool:
            continue

        if dtypes and rec.dtype not in dtypes:
            continue
        if algorithms and rec.algorithm not in algorithms:
            continue
        if mode and rec.mode != mode:
            continue
        if k_value is not None and rec.k != k_value:
            continue
        if backends and rec.backend not in backends:
            continue
        if dists and rec.dist and rec.dist not in dists:
            continue
        out.append(rec)
    return out


def filter_measurements(
    records: Iterable[MeasurementRecord],
    dtypes: set[str],
    algorithms: set[str],
    mode: Optional[str],
    k_value: Optional[int],
    backends: set[str],
    dists: Optional[set[str]] = None,
) -> list[MeasurementRecord]:
    out: list[MeasurementRecord] = []
    for rec in records:
        # drop failed energy measurements
        if rec.command_exit_code is not None and rec.command_exit_code != 0:
            continue

        if dtypes and rec.dtype and rec.dtype not in dtypes:
            continue
        if algorithms and rec.algorithm not in algorithms:
            continue
        if mode and rec.mode and rec.mode != mode:
            continue
        if k_value is not None and rec.k is not None and rec.k != k_value:
            continue
        if backends and rec.backend and rec.backend not in backends:
            continue
        if dists and rec.dist and rec.dist not in dists:
            continue
        out.append(rec)
    return out
