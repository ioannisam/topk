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
    if "/cpu/" in text or "cpu/build/topk" in text: return "cpu"
    if "/gpu/" in text or "gpu/build/topk" in text: return "gpu"
    if "/npu/" in text or "npu/build/topk" in text: return "npu"
    return ""


def infer_algorithm_and_dtype_from_command(cmd: str) -> tuple[str, str]:
    lower = cmd.lower()
    m_algo = re.search(r"(?:^|\s)algo=(bitonic|map_reduce|mapreduce|gt)(?:\s|$)", lower)
    m_dtype = re.search(r"(?:^|\s)dtype=(int|uint|float|double|fp16)(?:\s|$)", lower)
    algo = m_algo.group(1) if m_algo else ""
    if algo == "mapreduce":
        algo = "map_reduce"
    dtype = m_dtype.group(1) if m_dtype else ""
    return algo, dtype


def select_time_fields(timings: dict[str, float]) -> tuple[str, Optional[float], str, Optional[float]]:
    e2e_label = ""
    e2e_ms: Optional[float] = None
    algo_label = ""
    algo_ms: Optional[float] = None

    for label, val in timings.items():
        low = label.lower()
        if "end-to-end" in low:
            e2e_label = label
            e2e_ms = val
        elif "algorithmic" in low:
            algo_label = label
            algo_ms = val

    if e2e_ms is None and algo_ms is None:
        for label, val in timings.items():
            low = label.lower()
            if "time (ms)" in low and "end-to-end" not in low and "algorithmic" not in low:
                algo_label = label
                algo_ms = val
                break

    return e2e_label, e2e_ms, algo_label, algo_ms


def parse_int(val: str) -> Optional[int]:
    try:
        return int(val)
    except ValueError:
        return None


def parse_float(val: str) -> Optional[float]:
    try:
        return float(val)
    except ValueError:
        return None


def _parse_test_output_json(path: str) -> list[CaseRecord]:
    records: list[CaseRecord] = []
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        print(f"Warning: Could not parse JSON from {path}: {e}")
        return []

    results = data.get("results", [])
    for result in results:
        backend = result.get("backend", "").lower()
        algo = result.get("algorithm", "").lower()
        if algo == "mapreduce":
            algo = "map_reduce"

        rec = CaseRecord(
            backend=backend,
            case_name=result.get("case_name", ""),
            status=result.get("status", "UNKNOWN"),
            reason=result.get("reason", ""),
            dtype=result.get("type", ""),
            algorithm=algo,
            mode="max" if "_max" in result.get("case_name", "") else ("min" if "_min" in result.get("case_name", "") else ""),
            k=result.get("k"),
            n=None,
        )
        
        q = result.get("q")
        if q is not None:
            rec.n = 1 << int(q)
            
        stdout = result.get("stdout", "")
        timings: dict[str, float] = {}
        for line in stdout.splitlines():
            line = line.strip()
            if not line:
                continue
            kv_match = KV_RE.match(f"  {line}") 
            if kv_match:
                key, val = kv_match.group("key").strip(), kv_match.group("value").strip()
                if key.endswith("time (ms)"):
                    t = parse_float(val)
                    if t is not None:
                        timings[key] = t
                        
        e2e_label, e2e_ms, algo_label, algo_ms = select_time_fields(timings)
        rec.timing_label_e2e = e2e_label
        rec.time_end_to_end_ms = e2e_ms
        rec.timing_label_algorithmic = algo_label
        rec.time_algorithmic_ms = algo_ms
        rec.timing_label = e2e_label
        rec.time_ms = e2e_ms if e2e_ms is not None else algo_ms

        records.append(rec)
        
    return records


def _parse_test_output_text(path: str) -> list[CaseRecord]:
    records: list[CaseRecord] = []
    current_backend = ""
    current: Optional[CaseRecord] = None
    current_timings: dict[str, float] = {}

    try:
        with open(path, "r", encoding="utf-8") as f:
            lines = f.readlines()
    except FileNotFoundError:
        print(f"Warning: {path} not found.")
        return []

    for line in lines:
        line = line.rstrip("\n")

        b_match = BACKEND_HEADER_RE.match(line)
        if b_match:
            current_backend = b_match.group("backend").strip().lower()
            continue

        c_match = CASE_HEADER_RE.match(line)
        if c_match:
            if current is not None:
                e2e_label, e2e_ms, algo_label, algo_ms = select_time_fields(current_timings)
                current.timing_label_e2e = e2e_label
                current.time_end_to_end_ms = e2e_ms
                current.timing_label_algorithmic = algo_label
                current.time_algorithmic_ms = algo_ms
                current.timing_label = e2e_label
                current.time_ms = e2e_ms if e2e_ms is not None else algo_ms
                records.append(current)

            current = CaseRecord(backend=current_backend, case_name=c_match.group("name").strip())
            current_timings = {}
            continue

        if current is not None:
            if line.startswith("Status:"):
                current.status = line.split(":", 1)[1].strip()
            elif line.startswith("Reason:"):
                current.reason = line.split(":", 1)[1].strip()
            elif line.startswith("Command:"):
                cmd = line.split(":", 1)[1].strip()
                if not current.backend:
                    current.backend = infer_backend_from_command(cmd)
                algo, dtype = infer_algorithm_and_dtype_from_command(cmd)
                if algo and not current.algorithm:
                    current.algorithm = algo
                if dtype and not current.dtype:
                    current.dtype = dtype

            kv_match = KV_RE.match(line)
            if kv_match:
                key, value = kv_match.group("key").strip(), kv_match.group("value").strip()
                if key == "Data type":
                    current.dtype = value.lower()
                elif key == "Algorithm":
                    alg = value.lower()
                    current.algorithm = "map_reduce" if alg == "mapreduce" else alg
                elif key == "Mode":
                    current.mode = value.lower()
                elif key == "Requested top-k":
                    current.k = parse_int(value)
                elif key == "Input size N (2^q)":
                    current.n = parse_int(value)
                elif key.endswith("time (ms)"):
                    t = parse_float(value)
                    if t is not None:
                        current_timings[key] = t

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
    if path.endswith(".json"):
        return _parse_test_output_json(path)
    return _parse_test_output_text(path)


def parse_measurements(paths: Iterable[str]) -> list[MeasurementRecord]:
    records: list[MeasurementRecord] = []
    for path in paths:
        try:
            with open(path, "r", encoding="utf-8") as f:
                lines = f.readlines()
        except FileNotFoundError:
            continue

        base = os.path.basename(path)
        parts = base.replace(".txt", "").split("_")
        
        backend = ""
        dtype = ""
        algo = ""
        
        if len(parts) >= 5:
            backend = parts[2]
            dtype = parts[3]
            algo = parts[4]

        rec = MeasurementRecord(source=parts[-1] if parts else "unknown", file_path=path)
        rec.backend = backend.lower()
        rec.dtype = dtype.lower()
        rec.algorithm = "map_reduce" if algo.lower() == "mapreduce" else algo.lower()

        for line in lines:
            line = line.rstrip("\n")
            if line.startswith("Command:"):
                cmd = line.split(":", 1)[1].strip()
                if not rec.backend:
                    rec.backend = infer_backend_from_command(cmd)
                cmd_algo, cmd_dtype = infer_algorithm_and_dtype_from_command(cmd)
                if not rec.algorithm and cmd_algo:
                    rec.algorithm = cmd_algo
                if not rec.dtype and cmd_dtype:
                    rec.dtype = cmd_dtype

                m_q = re.search(r"(?:^|\s)q=(\d+)(?:\s|$)", cmd)
                if m_q:
                    rec.n = 1 << int(m_q.group(1))
                m_k = re.search(r"(?:^|\s)k=(\d+)(?:\s|$)", cmd)
                if m_k:
                    rec.k = int(m_k.group(1))
                m_mode = re.search(r"(?:^|\s)mode=(min|max)(?:\s|$)", cmd)
                if m_mode:
                    rec.mode = m_mode.group(1)

            m_kv = MEASURE_KV_RE.match(line)
            if m_kv:
                k, v = m_kv.group("key").strip(), m_kv.group("value").strip()
                if k == "elapsed_seconds":
                    rec.elapsed_seconds = parse_float(v)
                elif k == "energy_joules":
                    rec.energy_joules = parse_float(v)
                elif k == "average_watts":
                    rec.average_watts = parse_float(v)
                elif k == "command_exit_code":
                    rec.command_exit_code = parse_int(v)

        records.append(rec)
    return records