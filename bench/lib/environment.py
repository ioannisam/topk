#!/usr/bin/env python3
import platform
import re
import subprocess


def run_tool(cmd):
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=15)
    except Exception:
        return ""
    return result.stdout if result.returncode == 0 else ""


def read_file(path):
    try:
        with open(path, encoding="utf-8") as f:
            return f.read()
    except OSError:
        return ""


def first_match(text, pattern):
    match = re.search(pattern, text, re.MULTILINE)
    return match.group(1).strip() if match else None


def git_revision(root_dir):
    revision = run_tool(["git", "-C", root_dir, "rev-parse", "--short=12", "HEAD"]).strip()
    if not revision:
        return None
    if run_tool(["git", "-C", root_dir, "status", "--porcelain"]).strip():
        return f"{revision}-dirty"
    return revision


def xrt_versions():
    report = run_tool(["xrt-smi", "examine", "--batch"])
    return (
        first_match(report, r"^\s*Version\s*:\s*(.+)$"),
        first_match(report, r"^\s*NPU Firmware Version\s*:\s*(.+)$"),
    )


def capture_environment(root_dir):
    xrt, npu_firmware = xrt_versions()
    return {
        "git_revision": git_revision(root_dir),
        "hostname": platform.node(),
        "os": first_match(read_file("/etc/os-release"), r'^PRETTY_NAME="?([^"\n]+)"?'),
        "kernel": platform.release(),
        "cpu": first_match(read_file("/proc/cpuinfo"), r"^model name\s*:\s*(.+)$"),
        "gcc": first_match(run_tool(["gcc", "--version"]), r"^gcc .*?([0-9]+\.[0-9]+\.[0-9]+)"),
        "nvidia_driver": first_match(
            run_tool(["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"]), r"^([0-9.]+)"
        ),
        "cuda": first_match(run_tool(["nvcc", "--version"]), r"release [0-9.]+, V([0-9.]+)"),
        "xrt": xrt,
        "npu_firmware": npu_firmware,
    }
