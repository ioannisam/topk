from __future__ import annotations

import os
import re
from typing import Iterable, Optional

from .models import CaseRecord, MeasurementRecord


BACKEND_HEADER_RE = re.compile(r"^== Backend:\s*(?P<backend>[^=]+?)\s*==$")
CASE_HEADER_RE = re.compile(r"^### Case:\s*(?P<name>.+)$")
KV_RE = re.compile(r"^\s{2}(?P<key>[^:]+):\s*(?P<value>.+)$")
MEASURE_KV_RE = re.compile(r"^-\s+(?P<key>[a-zA-Z0-9_]+):\s*(?P<value>.+)$")


def parse_int(value: str) -> Optional[int]:
    try:
        return int(value)
    except Exception:
        return None


def parse_float(value: str) -> Optional[float]:
    text = value.strip()
    if text.lower() == "skipped":
        return None
    try:
        return float(text)
    except Exception:
        return None


def infer_measurement_metadata(path: str) -> dict[str, object]:
    base = os.path.basename(path).lower()
    if base.endswith(".txt"):
        base = base[:-4]
    tokens = base.split("_")

    backend = ""
    dtype = ""
    mode = ""
    source_hint = ""
    k: Optional[int] = None
    n: Optional[int] = None

    for token in tokens:
        if token in {"cpu", "gpu", "npu", "gt"}:
            backend = token
        elif token in {"rapl", "smi"}:
            source_hint = token
        elif token in {"int", "float", "uint"}:
            dtype = token
        elif token in {"max", "min"}:
            mode = token
        elif token.startswith("q") and token[1:].isdigit():
            q = int(token[1:])
            n = 2 ** q
        elif token.startswith("k") and token[1:].isdigit():
            k = int(token[1:])

    return {
        "backend": backend,
        "dtype": dtype,
        "mode": mode,
        "k": k,
        "n": n,
        "source_hint": source_hint,
    }


def parse_measurement_file(path: str) -> list[MeasurementRecord]:
    records: list[MeasurementRecord] = []
    current: Optional[MeasurementRecord] = None
    meta = infer_measurement_metadata(path)

    with open(path, "r", encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.strip()
            if line == "RAPL measurement":
                if current is not None:
                    records.append(current)
                current = MeasurementRecord(
                    source="rapl",
                    file_path=path,
                    backend=str(meta["backend"]),
                    dtype=str(meta["dtype"]),
                    mode=str(meta["mode"]),
                    k=meta["k"],
                    n=meta["n"],
                )
                continue
            if line == "GPU measurement":
                if current is not None:
                    records.append(current)
                current = MeasurementRecord(
                    source="smi",
                    file_path=path,
                    backend=str(meta["backend"]),
                    dtype=str(meta["dtype"]),
                    mode=str(meta["mode"]),
                    k=meta["k"],
                    n=meta["n"],
                )
                continue

            if current is None:
                continue

            kv_match = MEASURE_KV_RE.match(line)
            if not kv_match:
                continue
            key = kv_match.group("key")
            value = kv_match.group("value")

            if key == "elapsed_seconds":
                current.elapsed_seconds = parse_float(value)
            elif key == "energy_joules":
                current.energy_joules = parse_float(value)
            elif key == "average_watts":
                current.average_watts = parse_float(value)
            elif key == "command_exit_code":
                current.command_exit_code = parse_int(value)

    if current is not None:
        records.append(current)

    if not records and str(meta["source_hint"]):
        records.append(
            MeasurementRecord(
                source=str(meta["source_hint"]),
                file_path=path,
                backend=str(meta["backend"]),
                dtype=str(meta["dtype"]),
                mode=str(meta["mode"]),
                k=meta["k"],
                n=meta["n"],
            )
        )

    return records


def parse_measurements(paths: Iterable[str]) -> list[MeasurementRecord]:
    records: list[MeasurementRecord] = []
    for path in paths:
        if not os.path.isfile(path):
            continue
        records.extend(parse_measurement_file(path))
    return records


def parse_test_output(path: str) -> list[CaseRecord]:
    records: list[CaseRecord] = []
    current_backend = ""
    current: Optional[CaseRecord] = None

    with open(path, "r", encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.rstrip("\n")

            backend_match = BACKEND_HEADER_RE.match(line)
            if backend_match:
                current_backend = backend_match.group("backend").strip().lower()
                continue

            case_match = CASE_HEADER_RE.match(line)
            if case_match:
                if current is not None:
                    records.append(current)
                current = CaseRecord(
                    backend=current_backend,
                    case_name=case_match.group("name").strip(),
                )
                continue

            if current is None:
                continue

            if line.startswith("Status:"):
                current.status = line.split(":", 1)[1].strip()
                continue
            if line.startswith("Reason:"):
                current.reason = line.split(":", 1)[1].strip()
                continue

            kv_match = KV_RE.match(line)
            if not kv_match:
                continue

            key = kv_match.group("key").strip()
            value = kv_match.group("value").strip()

            if key == "Data type":
                current.dtype = value.lower()
            elif key == "Mode":
                current.mode = value.lower()
            elif key == "Requested top-k":
                current.k = parse_int(value)
            elif key == "Input size N (2^q)":
                current.n = parse_int(value)
            elif key.endswith("time (ms)"):
                t = parse_float(value)
                if t is not None and current.time_ms is None:
                    current.timing_label = key
                    current.time_ms = t

    if current is not None:
        records.append(current)

    return records
