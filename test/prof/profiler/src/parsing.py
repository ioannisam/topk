from __future__ import annotations

import json
import os
import re
from typing import Iterable, Optional

from .models import CaseRecord, MeasurementRecord


BACKEND_HEADER_RE = re.compile(r"^== Backend:\s*(?P<backend>[^=]+?)\s*==$")
CASE_HEADER_RE = re.compile(r"^### Case:\s*(?P<name>.+)$")
KV_RE = re.compile(r"^\s{2}(?P<key>[^:]+):\s*(?P<value>.+)$")
MEASURE_KV_RE = re.compile(r"^-\s+(?P<key>[a-zA-Z0-9_]+):\s*(?P<value>.+)$")
COMMAND_RE = re.compile(r"^Command:\s*(?P<cmd>.+)$")


def infer_backend_from_command(cmd: str) -> str:
    text = cmd.lower()
    if "ground_truth" in text or "/gt/" in text: return "gt"
    if "/cpu/" in text or "cpu/build/topk" in text: return "cpu"
    if "/gpu/" in text or "gpu/build/topk" in text: return "gpu"
    if "/npu/" in text or "npu/build/topk" in text: return "npu"
    return ""


def infer_algorithm_and_dtype_from_command(cmd: str) -> tuple[str, str]:
    lower = cmd.lower()
    m_algo = re.search(r"(?:^|\s)algo=(bitonic|map_reduce|mapreduce)(?:\s|$)", lower)
    m_dtype = re.search(r"(?:^|\s)dtype=(int|uint|float|double|fp16)(?:\s|$)", lower)
    algo = ""
    dtype = ""
    if m_algo:
        algo = m_algo.group(1)
        if algo == "mapreduce": algo = "map_reduce"
    if m_dtype:
        dtype = m_dtype.group(1)
    if algo or dtype:
        return algo, dtype

    m = re.search(r"/(bitonic|map_reduce)/([^/]+)/[^/]+\.case\b", cmd)
    if not m: return "", ""
    return m.group(1), m.group(2)


def parse_int(value: str) -> Optional[int]:
    try: return int(value)
    except Exception: return None


def parse_float(value: str) -> Optional[float]:
    text = value.strip()
    if text.lower() == "skipped": return None
    try: return float(text)
    except Exception: return None


def select_time_fields(timings: dict[str, float]) -> tuple[str, Optional[float], str, Optional[float]]:
    if not timings: return "", None, "", None

    e2e_priority = [
        "Trunc bitonic end-to-end time (ms)", "Trunc bitonic time (ms)",
        "Map-reduce top-k end-to-end time (ms)", "Map-reduce top-k time (ms)",
        "Ground truth average partial-sort time (ms)", "Ground truth average select/sort time (ms)",
        "Ground truth select/sort time (ms)", "Full bitonic end-to-end time (ms)", "Full bitonic time (ms)",
    ]

    algo_priority = [
        "Trunc bitonic algorithmic time (ms)", "Map-reduce top-k algorithmic time (ms)", "Full bitonic algorithmic time (ms)",
    ]

    e2e_label = ""
    e2e_value: Optional[float] = None
    algo_label = ""
    algo_value: Optional[float] = None

    for label in e2e_priority:
        if label in timings:
            e2e_label = label
            e2e_value = timings[label]
            break

    for label in algo_priority:
        if label in timings:
            algo_label = label
            algo_value = timings[label]
            break

    if e2e_value is None:
        label, value = next(iter(timings.items()))
        e2e_label = label
        e2e_value = value

    if algo_value is None:
        algo_label = e2e_label
        algo_value = e2e_value

    return e2e_label, e2e_value, algo_label, algo_value


def infer_measurement_metadata(path: str) -> dict[str, object]:
    base = os.path.basename(path).lower()
    if base.endswith(".txt"): base = base[:-4]
    tokens = base.split("_")

    backend, dtype, algorithm, mode, source_hint = "", "", "", "", ""
    k: Optional[int] = None
    n: Optional[int] = None

    for token in tokens:
        if token in {"cpu", "gpu", "npu", "gt"}: backend = token
        elif token in {"bitonic", "map_reduce", "mapreduce"}: algorithm = "map_reduce" if token == "mapreduce" else token
        elif token in {"rapl", "smi"}: source_hint = token
        elif token in {"int", "uint", "float", "double", "fp16"}: dtype = token
        elif token in {"max", "min"}: mode = token
        elif token.startswith("q") and token[1:].isdigit(): n = 2 ** int(token[1:])
        elif token.startswith("k") and token[1:].isdigit(): k = int(token[1:])

    if backend == "gt": algorithm = ""
    return {"backend": backend, "dtype": dtype, "algorithm": algorithm, "mode": mode, "k": k, "n": n, "source_hint": source_hint}


def parse_measurement_file(path: str) -> list[MeasurementRecord]:
    records: list[MeasurementRecord] = []
    current: Optional[MeasurementRecord] = None
    meta = infer_measurement_metadata(path)

    with open(path, "r", encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.strip()
            if line == "RAPL measurement":
                if current is not None: records.append(current)
                current = MeasurementRecord(source="rapl", file_path=path, backend=str(meta["backend"]), dtype=str(meta["dtype"]), algorithm=str(meta["algorithm"]), mode=str(meta["mode"]), k=meta["k"], n=meta["n"])
                continue
            if line == "GPU measurement":
                if current is not None: records.append(current)
                current = MeasurementRecord(source="smi", file_path=path, backend=str(meta["backend"]), dtype=str(meta["dtype"]), algorithm=str(meta["algorithm"]), mode=str(meta["mode"]), k=meta["k"], n=meta["n"])
                continue

            if current is None: continue

            kv_match = MEASURE_KV_RE.match(line)
            if not kv_match: continue
            key, value = kv_match.group("key"), kv_match.group("value")

            if key == "elapsed_seconds": current.elapsed_seconds = parse_float(value)
            elif key == "energy_joules": current.energy_joules = parse_float(value)
            elif key == "average_watts": current.average_watts = parse_float(value)
            elif key == "command_exit_code": current.command_exit_code = parse_int(value)

    if current is not None: records.append(current)
    if not records and str(meta["source_hint"]):
        records.append(MeasurementRecord(source=str(meta["source_hint"]), file_path=path, backend=str(meta["backend"]), dtype=str(meta["dtype"]), algorithm=str(meta["algorithm"]), mode=str(meta["mode"]), k=meta["k"], n=meta["n"]))

    return records


def parse_measurements(paths: Iterable[str]) -> list[MeasurementRecord]:
    records: list[MeasurementRecord] = []
    for path in paths:
        if not os.path.isfile(path): continue
        records.extend(parse_measurement_file(path))
    return records


def _parse_test_output_json(path: str) -> list[CaseRecord]:
    """Parse structural data directly from runner.py JSON output."""
    records: list[CaseRecord] = []
    with open(path, "r", encoding="utf-8") as f:
        try:
            data = json.load(f)
        except json.JSONDecodeError:
            print(f"Warning: Failed to decode JSON from {path}")
            return []

    for result in data.get("results", []):
        backend = result.get("backend", "").lower()
        algo = result.get("algorithm", "").lower()
        if algo == "mapreduce": algo = "map_reduce"
        if backend == "gt": algo = ""

        case_name = result.get("case_name", "")
        mode = "max" if "max" in case_name.lower() else ("min" if "min" in case_name.lower() else "")

        q = result.get("q")
        n = (1 << q) if q is not None else None

        current = CaseRecord(
            backend=backend,
            case_name=case_name,
            status=result.get("status", "UNKNOWN"),
            reason=result.get("reason", ""),
            dtype=result.get("type", "").lower(),
            algorithm=algo,
            mode=mode,
            k=result.get("k"),
            n=n,
        )

        # Timings are still trapped in the stdout string, we extract just those
        current_timings: dict[str, float] = {}
        stdout = result.get("stdout", "")
        for line in stdout.splitlines():
            line = line.strip()
            if "time (ms)" in line and ":" in line:
                key, val = line.split(":", 1)
                t = parse_float(val)
                if t is not None:
                    current_timings[key.strip()] = t

        e2e_label, e2e_ms, algo_label, algo_ms = select_time_fields(current_timings)
        current.timing_label_e2e = e2e_label
        current.time_end_to_end_ms = e2e_ms
        current.timing_label_algorithmic = algo_label
        current.time_algorithmic_ms = algo_ms
        current.timing_label = e2e_label
        current.time_ms = e2e_ms if e2e_ms is not None else algo_ms

        records.append(current)

    return records


def _parse_test_output_txt(path: str) -> list[CaseRecord]:
    """Fallback legacy parser for raw .txt files."""
    records: list[CaseRecord] = []
    current_backend = ""
    current: Optional[CaseRecord] = None
    current_timings: dict[str, float] = {}

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
                    e2e_label, e2e_ms, algo_label, algo_ms = select_time_fields(current_timings)
                    current.timing_label_e2e = e2e_label
                    current.time_end_to_end_ms = e2e_ms
                    current.timing_label_algorithmic = algo_label
                    current.time_algorithmic_ms = algo_ms
                    current.timing_label = e2e_label
                    current.time_ms = e2e_ms if e2e_ms is not None else algo_ms
                    records.append(current)
                current_timings = {}
                current = CaseRecord(
                    backend=current_backend,
                    case_name=case_match.group("name").strip(),
                )
                continue

            if current is None: continue

            cmd_match = COMMAND_RE.match(line)
            if cmd_match:
                cmd = cmd_match.group("cmd").strip()
                inferred_backend = infer_backend_from_command(cmd)
                inferred_algo, inferred_dtype = infer_algorithm_and_dtype_from_command(cmd)
                if inferred_backend: current.backend = inferred_backend
                if inferred_algo and current.backend != "gt": current.algorithm = inferred_algo
                if inferred_dtype and not current.dtype: current.dtype = inferred_dtype
                continue

            if line.startswith("Status:"):
                current.status = line.split(":", 1)[1].strip()
                continue
            if line.startswith("Reason:"):
                current.reason = line.split(":", 1)[1].strip()
                continue

            kv_match = KV_RE.match(line)
            if not kv_match: continue

            key, value = kv_match.group("key").strip(), kv_match.group("value").strip()
            if key == "Data type": current.dtype = value.lower()
            elif key == "Algorithm":
                alg = value.lower()
                current.algorithm = "" if current_backend == "gt" else ("map_reduce" if alg == "mapreduce" else alg)
            elif key == "Mode": current.mode = value.lower()
            elif key == "Requested top-k": current.k = parse_int(value)
            elif key == "Input size N (2^q)": current.n = parse_int(value)
            elif key.endswith("time (ms)"):
                t = parse_float(value)
                if t is not None: current_timings[key] = t

    if current is not None:
        e2e_label, e2e_ms, algo_label, algo_ms = select_time_fields(current_timings)
        current.timing_label_e2e = e2e_label
        current.time_end_to_end_ms = e2e_ms
        current.timing_label_algorithmic = algo_label
        current.time_algorithmic_ms = algo_ms
        current.timing_label = e2e_label
        current.time_ms = e2e_ms if e2e_ms is not None else algo_ms
        records.append(current)

    return records


def parse_test_output(path: str) -> list[CaseRecord]:
    """Smart dispatcher: uses fast JSON parsing if available, else falls back to Regex text parsing."""
    if path.endswith(".json"):
        return _parse_test_output_json(path)
    return _parse_test_output_txt(path)
