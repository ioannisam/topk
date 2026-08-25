#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime

from environment import capture_environment


def find_root_dir():
    current_dir = os.path.abspath(os.path.dirname(__file__))
    while current_dir != "/":
        if os.path.exists(os.path.join(current_dir, "CMakeLists.txt")):
            return current_dir
        current_dir = os.path.dirname(current_dir)
    print("error: could not locate repo root (CMakeLists.txt)", file=sys.stderr)
    sys.exit(1)


ROOT_DIR = find_root_dir()

SCHEMA = [
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

INT_FIELDS = {"ops_per_elem", "elements"}


def resolve_binary_path(backend):
    paths = {
        "cpu": "build/CPU/roofline",
        "gpu": "build/GPU/roofline",
        "npu": "build/NPU/roofline",
    }
    return os.path.join(ROOT_DIR, paths.get(backend, ""))


def parse_roofline_stdout(text):
    points = []
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped.startswith("ROOFLINE,"):
            continue
        fields = stripped.split(",")[1:]
        if len(fields) != len(SCHEMA):
            continue
        point = {}
        for name, raw in zip(SCHEMA, fields):
            if name in ("backend", "kernel"):
                point[name] = raw
            elif name in INT_FIELDS:
                point[name] = int(raw)
            else:
                point[name] = float(raw)
        points.append(point)
    return points


def parse_args():
    parser = argparse.ArgumentParser(
        description="Measure per-backend bandwidth ceilings and arithmetic-intensity sweeps."
    )
    parser.add_argument("backends", nargs="*", default=["cpu", "gpu"], help="Backends to measure (cpu gpu npu).")
    parser.add_argument(
        "--exp",
        default="all",
        choices=["stream", "sweep", "cache", "transfer", "both", "all"],
        help="Experiment to run.",
    )
    parser.add_argument("--bytes", default="256M", help="Working set per backend, e.g. 256M or 1G.")
    parser.add_argument("--ops", default="0,1,2,4,8,16,32,64,128,256", help="Comma-separated ops/element sweep points.")
    parser.add_argument("--sizes", default="", help="Comma-separated working-set/transfer sizes, e.g. 32K,1M,64M.")
    parser.add_argument("--threads", type=int, default=0, help="CPU worker threads (0 = hardware concurrency).")
    parser.add_argument("--seed", type=int, default=42, help="Input generation seed.")
    parser.add_argument("--cooldown", type=int, default=10, help="Seconds to idle between backends.")
    parser.add_argument("--timeout-seconds", type=float, default=600.0, help="Per-backend subprocess timeout")
    parser.add_argument("--output-json", default=os.path.join(ROOT_DIR, "bench/results/raw/roofline/roofline.json"))
    parser.add_argument("--output-raw", default=os.path.join(ROOT_DIR, "bench/results/raw/roofline/roofline.txt"))
    return parser.parse_args()


def main():
    args = parse_args()

    json_data = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "environment": capture_environment(ROOT_DIR),
        "experiment": args.exp,
        "bytes": args.bytes,
        "ops": args.ops,
        "sizes": args.sizes,
        "seed": args.seed,
        "points": [],
        "failures": [],
    }
    raw_chunks = []

    for index, backend in enumerate(args.backends):
        if index > 0 and args.cooldown > 0:
            print(f"  cooldown {args.cooldown}s")
            time.sleep(args.cooldown)

        binary = resolve_binary_path(backend)
        if not os.path.isfile(binary):
            print(f"skip {backend}: binary not found at {binary}")
            json_data["failures"].append({"backend": backend, "reason": f"binary not found: {binary}"})
            continue

        cmd = [
            binary,
            f"exp={args.exp}",
            f"bytes={args.bytes}",
            f"ops={args.ops}",
            f"seed={args.seed}",
            "debug=true",
        ]
        if args.threads > 0:
            cmd.append(f"threads={args.threads}")
        if args.sizes:
            cmd.append(f"sizes={args.sizes}")

        env = dict(os.environ)
        if backend == "gpu":
            env.setdefault("TOPK_ENERGY_DEVICE", "1")
        if backend == "npu":
            env.setdefault("NPU_OFFLOAD_XCLBIN", os.path.join(ROOT_DIR, "build/NPU/map_reduce.xclbin"))

        print(f"=== roofline: {backend} ===")
        print("  " + " ".join(cmd))
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, env=env, timeout=args.timeout_seconds)
        except subprocess.TimeoutExpired as exc:
            stdout = exc.stdout or ""
            stderr = (exc.stderr or "") + f"\ntimed out after {args.timeout_seconds}s"
            raw_chunks.append(f"===== {backend} =====\n{stdout}{stderr}")
            print(f"  failed (timeout after {args.timeout_seconds}s)")
            json_data["failures"].append({"backend": backend, "reason": stderr.strip()})
            continue

        raw_chunks.append(f"===== {backend} =====\n{result.stdout}{result.stderr}")

        if result.returncode != 0:
            print(f"  failed (exit {result.returncode}): {result.stderr.strip()}")
            json_data["failures"].append({"backend": backend, "reason": result.stderr.strip()})
            continue

        points = parse_roofline_stdout(result.stdout)
        if not points:
            print("  failed: no ROOFLINE rows in output")
            json_data["failures"].append({"backend": backend, "reason": "no ROOFLINE rows in output"})
            continue

        json_data["points"].extend(points)
        for point in points:
            print(
                f"  {point['kernel']:<18} ops={point['ops_per_elem']:<4} "
                f"{point['gbytes_per_s']:>8.2f} GB/s  {point['gops_per_s']:>10.2f} Gop/s"
            )

    os.makedirs(os.path.dirname(args.output_json), exist_ok=True)
    with open(args.output_json, "w", encoding="utf-8") as f_json:
        json.dump(json_data, f_json, indent=2)
    with open(args.output_raw, "w", encoding="utf-8") as f_raw:
        f_raw.write("\n".join(raw_chunks))

    print()
    print(f"Wrote {len(json_data['points'])} roofline points to: {args.output_json}")
    print(f"Wrote raw output to: {args.output_raw}")

    sys.exit(1 if json_data["failures"] else 0)


if __name__ == "__main__":
    main()
