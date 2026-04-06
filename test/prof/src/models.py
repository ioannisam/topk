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
    mode: str = ""
    k: Optional[int] = None
    n: Optional[int] = None
    timing_label: str = ""
    time_ms: Optional[float] = None

    @property
    def pass_bool(self) -> bool:
        return self.status.upper() == "PASS"


@dataclass
class MeasurementRecord:
    source: str
    file_path: str
    backend: str = ""
    dtype: str = ""
    mode: str = ""
    k: Optional[int] = None
    n: Optional[int] = None
    elapsed_seconds: Optional[float] = None
    energy_joules: Optional[float] = None
    average_watts: Optional[float] = None
    command_exit_code: Optional[int] = None
