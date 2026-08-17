#!/usr/bin/env python3
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


def resolve_binary_path(backend):
    paths = {
        "cpu": "build/CPU/topk",
        "gpu": "build/GPU/topk",
        "npu": "build/NPU/topk",
    }
    return os.path.join(ROOT_DIR, paths.get(backend, ""))


def resolve_energy_mode(backend, requested_mode):
    if requested_mode == "auto":
        return "gpu" if backend == "gpu" else "rapl"
    return requested_mode


def resolve_npu_xclbin(backend, algo):
    if backend != "npu":
        return ""
    path = os.path.join(ROOT_DIR, "build/NPU", f"{algo}.xclbin")
    return path if os.path.isfile(path) else ""


def extract_field_watts(report_text, key):
    prefix = f"- {key}:"
    for line in report_text.splitlines():
        s = line.strip()
        if s.startswith(prefix):
            try:
                return float(s.split(":", 1)[1].strip())
            except ValueError:
                return None
    return None


def capture_baseline_report(measure_cmd):
    try:
        result = subprocess.run(measure_cmd, capture_output=True, text=True, timeout=30)
    except Exception:
        return ""
    return result.stdout


def parse_args():
    parser = argparse.ArgumentParser(description="Dynamic Top-K Test Runner")
    parser.add_argument("backends", nargs="+", help="Backends to test (e.g., cpu gpu npu)")
    parser.add_argument(
        "--types", nargs="*", default=["double", "float", "half", "int", "uint"], help="Datatypes to test"
    )
    parser.add_argument(
        "--energy", choices=["none", "auto", "rapl", "gpu"], default="none", help="Energy measurement wrapper"
    )
    parser.add_argument("--energy-out-dir", default=os.path.join(ROOT_DIR, "bench/results/raw/energy/measurements"))
    parser.add_argument("--rapl-path", default="")
    parser.add_argument("--repeats", type=int, default=1, help="Repeats per (case, dist, seed) (for averaging)")
    parser.add_argument(
        "--dists", nargs="*", default=["uniform"], help="Input distributions: uniform normal sorted reverse"
    )
    parser.add_argument("--seeds", type=int, default=1, help="Number of distinct seeds swept per case")
    parser.add_argument("--baseline-seconds", type=float, default=3.0, help="Idle baseline sampling duration (seconds)")
    parser.add_argument(
        "--cooldown",
        type=float,
        default=0.0,
        help="Thermal cooldown (seconds) between backends (shared thermal budget)",
    )
    parser.add_argument("--gpu-index", type=int, default=0)
    parser.add_argument("--gpu-interval-ms", type=int, default=100)
    parser.add_argument("--q-min", type=int, default=1)
    parser.add_argument("--q-max", type=int, default=21)
    parser.add_argument("--k", type=int, nargs="+", default=[8, 256, 4096, 131072], help="List of K values to test")
    parser.add_argument("--mode", choices=["max", "min"], default="max")
    parser.add_argument("--run", choices=["trunc", "full", "both"], default="trunc")
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--seed-base", type=int, default=100)
    parser.add_argument("--min", type=int, default=0)
    parser.add_argument("--max", type=int, default=1000)
    parser.add_argument("--verify", choices=["true", "false"], default="true")
    parser.add_argument(
        "--case-timeout-seconds",
        type=float,
        default=900.0,
        help="Per-case subprocess timeout (gpu map_reduce at k=131072, q=24 already takes ~185s)",
    )
    parser.add_argument("--output-raw", default=os.path.join(ROOT_DIR, "bench/results/raw/cases/output.txt"))
    parser.add_argument("--output-json", default=os.path.join(ROOT_DIR, "bench/results/raw/cases/output.json"))
    return parser.parse_args()


def main():
    args = parse_args()

    backends = [b.lower().strip(", ") for b in args.backends]
    if len(backends) == 1 and "," in backends[0]:
        backends = backends[0].split(",")

    types = [t.lower().strip(", ") for t in args.types]
    if len(types) == 1 and "," in types[0]:
        types = types[0].split(",")

    dists = [d.lower().strip(", ") for d in args.dists]
    if len(dists) == 1 and "," in dists[0]:
        dists = dists[0].split(",")

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs(os.path.dirname(args.output_raw), exist_ok=True)
    os.makedirs(os.path.dirname(args.output_json), exist_ok=True)
    if args.energy != "none":
        os.makedirs(args.energy_out_dir, exist_ok=True)

    json_data = {
        "metadata": {
            "timestamp": run_id,
            "backends": backends,
            "types": types,
            "config": vars(args),
            "environment": capture_environment(ROOT_DIR),
        },
        "results": [],
        "summary": {"total_pass": 0, "total_fail": 0, "skipped_backends": []},
    }

    print("Running testcase suite")
    print(f"Backends: {','.join(backends)}")
    print(f"Types   : {','.join(types)}")
    print(f"Q range : {args.q_min}..{args.q_max}")
    print(f"Defaults: k=min({args.k},N) mode={args.mode} run={args.run} threads={args.threads} verify={args.verify}")
    print(f"Dists   : {','.join(dists)} (seeds/case={max(1, args.seeds)}, repeats={args.repeats})")
    print(f"Random range: {args.min}..{args.max}")
    print(f"Energy  : {args.energy}")

    with open(args.output_raw, "w", encoding="utf-8") as f_raw:
        f_raw.write("Testcase Run Output\n")
        f_raw.write(f"Backends: {','.join(backends)}\n")
        f_raw.write(f"Types   : {','.join(types)}\n")
        f_raw.write(f"Q range : {args.q_min}..{args.q_max}\n")
        f_raw.write(
            f"Defaults: k=min({args.k},N) mode={args.mode} run={args.run} threads={args.threads} verify={args.verify}\n"
        )
        f_raw.write(f"Random range: {args.min}..{args.max}\n")
        f_raw.write(f"Energy  : {args.energy}\n\n")

    expected_marker = "Top-k correctness vs CPU sorted reference: OK"

    rapl_baseline_w = None
    rapl_core_baseline_w = None
    gpu_baseline_w = None
    gpu_board_baseline_w = None
    if args.energy != "none":
        sleep_cmd = ["sleep", str(args.baseline_seconds)]
        modes = {resolve_energy_mode(b, args.energy) for b in backends}
        if "rapl" in modes:
            rapl_cmd = [os.path.join(ROOT_DIR, "bench/lib/measure_rapl.sh")]
            if args.rapl_path:
                rapl_cmd += ["--path", args.rapl_path]
            rapl_cmd += ["--"] + sleep_cmd
            rapl_report = capture_baseline_report(rapl_cmd)
            rapl_baseline_w = extract_field_watts(rapl_report, "average_watts")
            rapl_core_baseline_w = extract_field_watts(rapl_report, "core_average_watts")
            print(f"Idle RAPL baseline: package={rapl_baseline_w} W core={rapl_core_baseline_w} W")
        if "gpu" in modes:
            gpu_cmd = [
                os.path.join(ROOT_DIR, "bench/lib/measure_smi.sh"),
                "--gpu-index",
                str(args.gpu_index),
                "--interval-ms",
                str(args.gpu_interval_ms),
                "--",
            ] + sleep_cmd
            gpu_report = capture_baseline_report(gpu_cmd)
            gpu_baseline_w = extract_field_watts(gpu_report, "average_watts")
            gpu_board_baseline_w = extract_field_watts(gpu_report, "board_average_watts")
            print(f"Idle GPU baseline: total={gpu_baseline_w} W board={gpu_board_baseline_w} W")

    for backend_idx, backend in enumerate(backends):
        if backend_idx > 0 and args.cooldown > 0:
            print(f"\nCooldown {args.cooldown:.0f}s before {backend} (shared thermal budget)...")
            time.sleep(args.cooldown)

        binary_path = resolve_binary_path(backend)
        if not os.path.isfile(binary_path) or not os.access(binary_path, os.X_OK):
            # Not a case failure: keep it out of the pass/fail tally so the rate stays a
            # correctness measure, and surface it separately.
            print(f"\nBackend binary not found or not executable: {binary_path}")
            json_data["summary"]["skipped_backends"].append(backend)
            continue

        print(f"\nBackend: {backend}")
        print(f"Binary : {binary_path}")

        with open(args.output_raw, "a", encoding="utf-8") as f_raw:
            f_raw.write(f"== Backend: {backend} ==\n")
            f_raw.write(f"Binary: {binary_path}\n\n")

        backend_pass = 0
        backend_fail = 0

        for dtype in types:
            print(f"  Type {dtype}: dynamic cases")
            with open(args.output_raw, "a", encoding="utf-8") as f_raw:
                f_raw.write(f"-- Type: {dtype} --\n")
                f_raw.write("Cases: dynamic\n\n")

            type_pass = 0
            type_fail = 0

            algorithms = ["bitonic", "map_reduce", "gt"]
            if backend == "npu":
                # gt runs on the host; every other algo needs its own xclbin to run at all.
                algorithms = [a for a in algorithms if a == "gt" or resolve_npu_xclbin(backend, a)]

            for algo in algorithms:
                for q in range(args.q_min, args.q_max + 1):
                    n = 1 << q

                    effective_ks = sorted(list(set(k_val for k_val in args.k if k_val <= n)))
                    for k_eff in effective_ks:
                        # Spread q and k apart so distinct cases cannot land on the same seed
                        # (q + k collided for e.g. (q=10,k=8) and (q=8,k=10)).
                        base_seed = args.seed_base + q * 1000003 + k_eff
                        seed_list = [base_seed + s for s in range(max(1, args.seeds))]
                        base_case_name = f"q{q:02d}_k{k_eff}_{args.mode}"

                        # Sweep (distribution, seed); each combination is an independent
                        # sample that feeds the error bands, repeated for energy averaging.
                        for dist in dists:
                            for seed in seed_list:
                                case_args = [
                                    f"q={q}",
                                    f"k={k_eff}",
                                    f"mode={args.mode}",
                                    f"dtype={dtype}",
                                    f"algo={algo}",
                                    f"run={args.run}",
                                    "debug=false",
                                    f"threads={args.threads}",
                                    f"seed={seed}",
                                    f"verify={args.verify}",
                                    f"min={args.min}",
                                    f"max={args.max}",
                                    f"dist={dist}",
                                ]

                                energy_mode = resolve_energy_mode(backend, args.energy)
                                baseline_w = (
                                    rapl_baseline_w
                                    if energy_mode == "rapl"
                                    else (gpu_baseline_w if energy_mode == "gpu" else None)
                                )

                                case_env = os.environ.copy()
                                if backend == "gpu":
                                    case_env["TOPK_ENERGY_DEVICE"] = "1"
                                    case_env["TOPK_GPU_INDEX"] = str(args.gpu_index)
                                npu_xclbin = resolve_npu_xclbin(backend, algo)
                                if npu_xclbin:
                                    case_env["NPU_OFFLOAD_XCLBIN"] = npu_xclbin
                                else:
                                    # An inherited value points at the wrong algo's xclbin.
                                    case_env.pop("NPU_OFFLOAD_XCLBIN", None)

                                for rep in range(1, args.repeats + 1):
                                    # Always tag the variant: the profiler joins measurement
                                    # files to case records on (dist, seed, rep), and it can
                                    # only recover them from the filename.
                                    suffix = f"_{dist}_s{seed}_rep{rep}"
                                    if energy_mode == "none":
                                        run_cmd = [binary_path] + case_args
                                        energy_case_file = ""
                                    else:
                                        energy_case_file = os.path.join(
                                            args.energy_out_dir,
                                            f"{run_id}_{backend}_{dtype}_{algo}_{base_case_name}{suffix}_{energy_mode}.txt",
                                        )
                                        if energy_mode == "rapl":
                                            run_cmd = [
                                                os.path.join(ROOT_DIR, "bench/lib/measure_rapl.sh"),
                                                "--out",
                                                energy_case_file,
                                            ]
                                            if args.rapl_path:
                                                run_cmd += ["--path", args.rapl_path]
                                        else:
                                            run_cmd = [
                                                os.path.join(ROOT_DIR, "bench/lib/measure_smi.sh"),
                                                "--out",
                                                energy_case_file,
                                                "--gpu-index",
                                                str(args.gpu_index),
                                                "--interval-ms",
                                                str(args.gpu_interval_ms),
                                            ]
                                        if baseline_w is not None:
                                            run_cmd += ["--baseline-watts", str(baseline_w)]
                                        if energy_mode == "rapl" and rapl_core_baseline_w is not None:
                                            run_cmd += ["--core-baseline-watts", str(rapl_core_baseline_w)]
                                        if energy_mode == "gpu" and gpu_board_baseline_w is not None:
                                            run_cmd += ["--board-baseline-watts", str(gpu_board_baseline_w)]
                                        run_cmd += ["--", binary_path] + case_args
                                    try:
                                        result = subprocess.run(
                                            run_cmd,
                                            capture_output=True,
                                            text=True,
                                            env=case_env,
                                            timeout=args.case_timeout_seconds,
                                        )
                                        stdout = result.stdout
                                        stderr = result.stderr
                                        returncode = result.returncode
                                    except subprocess.TimeoutExpired as exc:
                                        stdout = exc.stdout or ""
                                        stderr = (exc.stderr or "") + f"\ntimed out after {args.case_timeout_seconds}s"
                                        returncode = -1

                                    label = f"{base_case_name} {dist} seed={seed}" + (
                                        f" rep={rep}" if args.repeats > 1 else ""
                                    )

                                    case_status = "FAIL"
                                    case_reason = "non-zero exit"

                                    if returncode == 0:
                                        if args.verify == "true" and expected_marker not in stdout:
                                            case_reason = "PASS marker missing"
                                            print(f"    [FAIL] {label} (algo={algo}) ({case_reason})")
                                            type_fail += 1
                                        else:
                                            case_status = "PASS"
                                            # Do not claim a correctness check that never ran.
                                            case_reason = "ok" if args.verify == "true" else "unverified"
                                            print(f"    [PASS] {label} (algo={algo})")
                                            type_pass += 1
                                    else:
                                        reason_detail = stderr.strip().splitlines()
                                        if reason_detail:
                                            case_reason = f"non-zero exit: {reason_detail[-1]}"
                                        print(f"    [FAIL] {label} (algo={algo}) ({case_reason})")
                                        type_fail += 1

                                    # Write Raw File
                                    with open(args.output_raw, "a", encoding="utf-8") as f_raw:
                                        f_raw.write(f"### Case: {base_case_name}\n")
                                        f_raw.write(f"Distribution: {dist}\n")
                                        f_raw.write(f"Seed: {seed}\n")
                                        f_raw.write(f"Status: {case_status}\n")
                                        f_raw.write(f"Reason: {case_reason}\n")
                                        f_raw.write(f"Command: {' '.join(run_cmd)}\n")
                                        if energy_case_file:
                                            f_raw.write(f"Measurement file: {energy_case_file}\n")
                                        f_raw.write(f"Output:\n{stdout}\n")
                                        if stderr.strip():
                                            f_raw.write(f"Stderr:\n{stderr}\n")
                                        f_raw.write("\n")

                                    # Append to JSON structure
                                    json_data["results"].append(
                                        {
                                            "backend": backend,
                                            "type": dtype,
                                            "algorithm": algo,
                                            "case_name": base_case_name,
                                            "q": q,
                                            "k": k_eff,
                                            "dist": dist,
                                            "seed": seed,
                                            "rep": rep,
                                            "status": case_status,
                                            "reason": case_reason,
                                            "command": " ".join(run_cmd),
                                            "energy_file": energy_case_file,
                                            "exit_code": returncode,
                                            "stdout": stdout.strip(),
                                            "stderr": stderr.strip(),
                                        }
                                    )

            print(f"    Type {dtype} summary: pass={type_pass} fail={type_fail}")
            backend_pass += type_pass
            backend_fail += type_fail

        print(f"  Backend {backend} summary: pass={backend_pass} fail={backend_fail}")
        json_data["summary"]["total_pass"] += backend_pass
        json_data["summary"]["total_fail"] += backend_fail

    # Dump JSON File
    with open(args.output_json, "w", encoding="utf-8") as f_json:
        json.dump(json_data, f_json, indent=2)

    print("\nTotal Summary")
    print(f"  Passed: {json_data['summary']['total_pass']}")
    print(f"  Failed: {json_data['summary']['total_fail']}")
    if json_data["summary"]["skipped_backends"]:
        print(f"  Skipped backends (no binary): {','.join(json_data['summary']['skipped_backends'])}")
    if args.verify != "true":
        print("  NOTE: --verify false, so PASS means 'ran without error', not 'output correct'")
    print(f"Wrote detailed text output to: {args.output_raw}")
    print(f"Wrote structured JSON output to: {args.output_json}")

    sys.exit(1 if json_data["summary"]["total_fail"] > 0 else 0)


if __name__ == "__main__":
    main()
