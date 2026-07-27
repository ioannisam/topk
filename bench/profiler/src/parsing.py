from __future__ import annotations

import json
import os
import re
from typing import Iterable, Optional

from .models import CaseRecord, MeasurementRecord, RooflinePoint


BACKEND_HEADER_RE = re.compile(r"^== Backend:\s*(?P<backend>[^=]+?)\s*==$")
CASE_HEADER_RE = re.compile(r"^### Case:\s*(?P<name>.+)$")
KV_RE = re.compile(r"^\s{2}(?P<key>[^:]+):\s*(?P<value>.+)$")
MEASURE_KV_RE = re.compile(r"^-\s+(?P<key>[a-zA-Z0-9_]+):\s*(?P<value>.+)$")
COMMAND_RE = re.compile(r"^Command:\s*(?P<cmd>.+)$")
VARIANT_RE = re.compile(r"_(?P<dist>uniform|normal|sorted|reverse)_s(?P<seed>\d+)_rep(?P<rep>\d+)_(?P<src>[a-z]+)$")
MEASUREMENT_NAME_RE = re.compile(
    r"^\d{8}_\d{6}_(?P<backend>cpu|gpu|npu)_(?P<dtype>int|uint|float|double|half)_"
    r"(?P<algorithm>bitonic|map_reduce|gt)_"
)


def infer_backend_from_command(cmd: str) -> str:
    text = cmd.lower()
    if "/cpu/" in text or "cpu/build/topk" in text:
        return "cpu"
    if "/gpu/" in text or "gpu/build/topk" in text:
        return "gpu"
    if "/npu/" in text or "npu/build/topk" in text:
        return "npu"
    return ""


def infer_algorithm_and_dtype_from_command(cmd: str) -> tuple[str, str]:
    lower = cmd.lower()
    m_algo = re.search(r"(?:^|\s)algo=(bitonic|map_reduce|mapreduce|gt)(?:\s|$)", lower)
    m_dtype = re.search(r"(?:^|\s)dtype=(int|uint|float|double|half)(?:\s|$)", lower)
    algo = m_algo.group(1) if m_algo else ""
    if algo == "mapreduce":
        algo = "map_reduce"
    dtype = m_dtype.group(1) if m_dtype else ""
    return algo, dtype


def _pick_timing(timings: dict[str, float], kind: str) -> tuple[str, Optional[float]]:
    matches = [(label, val) for label, val in timings.items() if kind in label.lower()]
    if not matches:
        return "", None
    for label, val in matches:
        if "trunc" in label.lower():
            return label, val
    return matches[0]


def select_time_fields(timings: dict[str, float]) -> tuple[str, Optional[float], str, Optional[float]]:
    e2e_label, e2e_ms = _pick_timing(timings, "end-to-end")
    algo_label, algo_ms = _pick_timing(timings, "algorithmic")

    if e2e_ms is None and algo_ms is None:
        for label, val in timings.items():
            low = label.lower()
            if "time (ms)" in low and "end-to-end" not in low and "algorithmic" not in low:
                algo_label = label
                algo_ms = val
                break

    return e2e_label, e2e_ms, algo_label, algo_ms


def select_stdev_fields(stdevs: dict[str, float]) -> tuple[Optional[float], Optional[float]]:
    return _pick_timing(stdevs, "end-to-end")[1], _pick_timing(stdevs, "algorithmic")[1]


ENERGY_FIELD_MAP = {
    "Energy e2e joules": "energy_e2e_joules",
    "Energy algo joules": "energy_algo_joules",
    "Energy loop joules": "energy_loop_joules",
    "Energy e2e package joules": "energy_e2e_package_joules",
    "Energy algo package joules": "energy_algo_package_joules",
    "Energy loop package joules": "energy_loop_package_joules",
    "Energy e2e core joules": "energy_e2e_core_joules",
    "Energy algo core joules": "energy_algo_core_joules",
    "Energy e2e device joules": "energy_e2e_device_joules",
    "Energy algo device joules": "energy_algo_device_joules",
    "Energy loop device joules": "energy_loop_device_joules",
    "Energy e2e seconds": "energy_e2e_seconds",
    "Energy algo seconds": "energy_algo_seconds",
    "Energy loop seconds": "energy_loop_seconds",
}


def apply_energy_kv(rec, key: str, val: str) -> bool:
    if key == "Energy status":
        rec.energy_status = val
        return True
    if key == "Energy counters":
        rec.energy_counters = val
        return True
    if key == "Energy iterations":
        rec.energy_iterations = parse_int(val)
        return True
    field = ENERGY_FIELD_MAP.get(key)
    if field is not None:
        setattr(rec, field, parse_float(val))
        return True
    return False


TRAFFIC_FIELD_MAP = {
    "Traffic bytes moved": "bytes_moved",
    "Traffic compare ops": "compare_ops",
}

TRAFFIC_MODEL_MAP = {
    "Traffic bytes model": "bytes_model",
    "Traffic ops model": "ops_model",
}


def apply_traffic_kv(rec, key: str, val: str) -> bool:
    field = TRAFFIC_FIELD_MAP.get(key)
    if field is not None:
        setattr(rec, field, parse_float(val))
        return True
    field = TRAFFIC_MODEL_MAP.get(key)
    if field is not None:
        setattr(rec, field, val)
        return True
    return False


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
            mode="max"
            if "_max" in result.get("case_name", "")
            else ("min" if "_min" in result.get("case_name", "") else ""),
            k=result.get("k"),
            n=None,
        )

        q = result.get("q")
        if q is not None:
            rec.n = 1 << int(q)

        rec.dist = result.get("dist", "")
        rec.seed = result.get("seed")
        rec.rep = result.get("rep")

        stdout = result.get("stdout", "")
        timings: dict[str, float] = {}
        stdevs: dict[str, float] = {}
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
                elif key.endswith("stdev (ms)"):
                    s = parse_float(val)
                    if s is not None:
                        stdevs[key] = s
                elif key == "Benchmark iterations":
                    rec.bench_ops = parse_int(val)
                else:
                    if not apply_traffic_kv(rec, key, val):
                        apply_energy_kv(rec, key, val)

        e2e_label, e2e_ms, algo_label, algo_ms = select_time_fields(timings)
        rec.timing_label_e2e = e2e_label
        rec.time_end_to_end_ms = e2e_ms
        rec.timing_label_algorithmic = algo_label
        rec.time_algorithmic_ms = algo_ms
        rec.timing_label = e2e_label
        rec.time_ms = e2e_ms if e2e_ms is not None else algo_ms

        e2e_std, algo_std = select_stdev_fields(stdevs)
        rec.time_end_to_end_stdev_ms = e2e_std
        rec.time_algorithmic_stdev_ms = algo_std

        records.append(rec)

    return records


def _parse_test_output_text(path: str) -> list[CaseRecord]:
    records: list[CaseRecord] = []
    current_backend = ""
    current: Optional[CaseRecord] = None
    current_timings: dict[str, float] = {}
    current_stdevs: dict[str, float] = {}

    def finalize(rec: CaseRecord, timings: dict[str, float], stdevs: dict[str, float]) -> None:
        e2e_label, e2e_ms, algo_label, algo_ms = select_time_fields(timings)
        rec.timing_label_e2e = e2e_label
        rec.time_end_to_end_ms = e2e_ms
        rec.timing_label_algorithmic = algo_label
        rec.time_algorithmic_ms = algo_ms
        rec.timing_label = e2e_label
        rec.time_ms = e2e_ms if e2e_ms is not None else algo_ms
        e2e_std, algo_std = select_stdev_fields(stdevs)
        rec.time_end_to_end_stdev_ms = e2e_std
        rec.time_algorithmic_stdev_ms = algo_std

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
                finalize(current, current_timings, current_stdevs)
                records.append(current)

            current = CaseRecord(backend=current_backend, case_name=c_match.group("name").strip())
            current_timings = {}
            current_stdevs = {}
            continue

        if current is not None:
            if line.startswith("Status:"):
                current.status = line.split(":", 1)[1].strip()
            elif line.startswith("Reason:"):
                current.reason = line.split(":", 1)[1].strip()
            elif line.startswith("Distribution:"):
                current.dist = line.split(":", 1)[1].strip().lower()
            elif line.startswith("Seed:"):
                current.seed = parse_int(line.split(":", 1)[1].strip())
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
                elif key == "Input distribution":
                    current.dist = value.lower()
                elif key == "Benchmark iterations":
                    current.bench_ops = parse_int(value)
                elif key.endswith("time (ms)"):
                    t = parse_float(value)
                    if t is not None:
                        current_timings[key] = t
                elif key.endswith("stdev (ms)"):
                    s = parse_float(value)
                    if s is not None:
                        current_stdevs[key] = s
                else:
                    if not apply_traffic_kv(current, key, value):
                        apply_energy_kv(current, key, value)

    if current is not None:
        finalize(current, current_timings, current_stdevs)
        records.append(current)

    return records


def parse_test_output(path: str) -> list[CaseRecord]:
    if path.endswith(".json"):
        return _parse_test_output_json(path)
    return _parse_test_output_text(path)


def attach_inprocess_energy(measurements, inproc_by_key) -> None:
    for rec in measurements:
        case = inproc_by_key.get((rec.backend, rec.dtype, rec.algorithm, rec.n, rec.k, rec.dist, rec.seed, rec.rep))
        if case is None:
            continue

        rec.inproc_available = True
        rec.inproc_counters = case.energy_counters
        rec.inproc_iterations = case.energy_iterations
        rec.inproc_e2e_joules = case.energy_e2e_joules
        rec.inproc_algo_joules = case.energy_algo_joules
        rec.inproc_e2e_seconds = case.energy_e2e_seconds
        rec.inproc_algo_seconds = case.energy_algo_seconds
        rec.inproc_loop_joules = case.energy_loop_joules
        rec.inproc_loop_seconds = case.energy_loop_seconds

        idle_w = rec.baseline_watts

        if idle_w is not None:
            if rec.inproc_e2e_joules is not None and rec.inproc_e2e_seconds is not None:
                rec.inproc_net_e2e_joules = max(rec.inproc_e2e_joules - idle_w * rec.inproc_e2e_seconds, 0.0)
            if rec.inproc_algo_joules is not None and rec.inproc_algo_seconds is not None:
                rec.inproc_net_algo_joules = max(rec.inproc_algo_joules - idle_w * rec.inproc_algo_seconds, 0.0)


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

        name_match = MEASUREMENT_NAME_RE.match(base)
        if name_match is not None:
            backend = name_match.group("backend")
            dtype = name_match.group("dtype")
            algo = name_match.group("algorithm")

        rec = MeasurementRecord(source=parts[-1] if parts else "unknown", file_path=path)
        v = VARIANT_RE.search(base.replace(".txt", ""))
        if v is not None:
            rec.dist = v.group("dist")
            rec.seed = parse_int(v.group("seed"))
            rec.rep = parse_int(v.group("rep"))
        rec.backend = backend.lower()
        rec.dtype = dtype.lower()
        rec.algorithm = "map_reduce" if algo.lower() == "mapreduce" else algo.lower()

        for line in lines:
            line = line.rstrip("\n")
            if line.startswith("Command:"):
                cmd = line.split(":", 1)[1].strip()
                if not rec.backend:
                    rec.backend = infer_backend_from_command(cmd)
                # The command carries the exact algo=/dtype= tokens, so prefer it over
                # the filename split (which breaks on the underscore in "map_reduce").
                cmd_algo, cmd_dtype = infer_algorithm_and_dtype_from_command(cmd)
                if cmd_algo:
                    rec.algorithm = cmd_algo
                if cmd_dtype:
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
                elif k == "baseline_watts":
                    rec.baseline_watts = parse_float(v)
                elif k == "board_baseline_watts":
                    rec.board_baseline_watts = parse_float(v)
                elif k == "net_energy_joules":
                    rec.net_energy_joules = parse_float(v)
                elif k == "net_average_watts":
                    rec.net_average_watts = parse_float(v)
                elif k == "core_energy_joules":
                    rec.core_energy_joules = parse_float(v)
                elif k == "net_core_energy_joules":
                    rec.net_core_energy_joules = parse_float(v)
                elif k == "board_energy_joules":
                    rec.board_energy_joules = parse_float(v)
                elif k == "net_board_energy_joules":
                    rec.net_board_energy_joules = parse_float(v)
                elif k == "command_exit_code":
                    rec.command_exit_code = parse_int(v)

        records.append(rec)
    return records


def parse_roofline(paths: Iterable[str]) -> list[RooflinePoint]:
    points: list[RooflinePoint] = []
    for path in paths:
        try:
            with open(path, "r", encoding="utf-8") as f:
                payload = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            continue

        for entry in payload.get("points", []):
            try:
                points.append(
                    RooflinePoint(
                        backend=str(entry["backend"]).lower(),
                        kernel=str(entry["kernel"]).lower(),
                        ops_per_elem=int(entry["ops_per_elem"]),
                        elements=int(entry["elements"]),
                        bytes_moved=float(entry["bytes_moved"]),
                        ops=float(entry["ops"]),
                        ms_mean=float(entry["ms_mean"]),
                        ms_stdev=float(entry["ms_stdev"]),
                        ms_min=float(entry["ms_min"]),
                        gbytes_per_s=float(entry["gbytes_per_s"]),
                        gops_per_s=float(entry["gops_per_s"]),
                        operational_intensity=float(entry["operational_intensity"]),
                        joules_per_iter=float(entry["joules_per_iter"]),
                    )
                )
            except (KeyError, TypeError, ValueError):
                continue
    return points
