from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass
class CaseRecord:
    backend: str
    case_name: str
    status: str = "UNKNOWN"
    reason: str = ""
    dtype: str = ""
    algorithm: str = ""
    mode: str = ""
    k: Optional[int] = None
    n: Optional[int] = None
    dist: str = ""
    seed: Optional[int] = None
    rep: Optional[int] = None
    bench_ops: Optional[int] = None
    timing_label: str = ""
    time_ms: Optional[float] = None
    timing_label_e2e: str = ""
    time_end_to_end_ms: Optional[float] = None
    timing_label_algorithmic: str = ""
    time_algorithmic_ms: Optional[float] = None
    time_end_to_end_stdev_ms: Optional[float] = None
    time_algorithmic_stdev_ms: Optional[float] = None
    energy_status: str = ""
    energy_counters: str = ""
    energy_iterations: Optional[int] = None
    energy_e2e_joules: Optional[float] = None
    energy_algo_joules: Optional[float] = None
    energy_loop_joules: Optional[float] = None
    energy_e2e_package_joules: Optional[float] = None
    energy_algo_package_joules: Optional[float] = None
    energy_loop_package_joules: Optional[float] = None
    energy_e2e_core_joules: Optional[float] = None
    energy_algo_core_joules: Optional[float] = None
    energy_loop_core_joules: Optional[float] = None
    energy_e2e_device_joules: Optional[float] = None
    energy_algo_device_joules: Optional[float] = None
    energy_loop_device_joules: Optional[float] = None
    energy_e2e_seconds: Optional[float] = None
    energy_algo_seconds: Optional[float] = None
    energy_loop_seconds: Optional[float] = None
    bytes_moved: Optional[float] = None
    bytes_model: str = ""
    compare_ops: Optional[float] = None
    ops_model: str = ""

    @property
    def bytes_exact(self) -> bool:
        return self.bytes_model == "exact"

    @property
    def pass_bool(self) -> bool:
        return self.status.upper() == "PASS"


@dataclass
class RooflinePoint:
    backend: str
    kernel: str
    ops_per_elem: int
    elements: int
    bytes_moved: float
    ops: float
    ms_mean: float
    ms_stdev: float
    ms_min: float
    gbytes_per_s: float
    gops_per_s: float
    operational_intensity: float
    joules_per_iter: float


@dataclass
class MeasurementRecord:
    source: str
    file_path: str
    backend: str = ""
    dtype: str = ""
    algorithm: str = ""
    mode: str = ""
    k: Optional[int] = None
    n: Optional[int] = None
    elapsed_seconds: Optional[float] = None
    energy_joules: Optional[float] = None
    average_watts: Optional[float] = None
    baseline_watts: Optional[float] = None
    net_energy_joules: Optional[float] = None
    net_average_watts: Optional[float] = None
    command_exit_code: Optional[int] = None
    bench_ops: int = 1
    dist: str = ""
    seed: Optional[int] = None
    rep: Optional[int] = None
    board_baseline_watts: Optional[float] = None
    core_baseline_watts: Optional[float] = None
    inproc_available: bool = False
    inproc_counters: str = ""
    inproc_iterations: Optional[int] = None
    inproc_e2e_joules: Optional[float] = None
    inproc_algo_joules: Optional[float] = None
    inproc_e2e_seconds: Optional[float] = None
    inproc_algo_seconds: Optional[float] = None
    inproc_net_e2e_joules: Optional[float] = None
    inproc_net_algo_joules: Optional[float] = None
    inproc_loop_joules: Optional[float] = None
    inproc_loop_seconds: Optional[float] = None
    core_energy_joules: Optional[float] = None
    net_core_energy_joules: Optional[float] = None
    board_energy_joules: Optional[float] = None
    net_board_energy_joules: Optional[float] = None
