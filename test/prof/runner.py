#!/usr/bin/env python3
import argparse
import json
import os
import subprocess
import sys
from datetime import datetime


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


def extract_avg_watts(report_text):
    for line in report_text.splitlines():
        s = line.strip()
        if s.startswith("- average_watts:"):
            try:
                return float(s.split(":", 1)[1].strip())
            except ValueError:
                return None
    return None


def capture_baseline_watts(measure_cmd):
    try:
        result = subprocess.run(measure_cmd, capture_output=True, text=True)
    except Exception:
        return None
    return extract_avg_watts(result.stdout)


def parse_args():
    parser = argparse.ArgumentParser(description="Dynamic Top-K Test Runner")
    parser.add_argument("backends", nargs="+", help="Backends to test (e.g., cpu gpu npu)")
    parser.add_argument(
        "--types", nargs="*", default=["double", "float", "fp16", "int", "uint"], help="Datatypes to test"
    )
    parser.add_argument(
        "--energy", choices=["none", "auto", "rapl", "gpu"], default="none", help="Energy measurement wrapper"
    )
    parser.add_argument("--energy-out-dir", default=os.path.join(ROOT_DIR, "test/prof/results/energy"))
    parser.add_argument("--rapl-path", default="")
    parser.add_argument("--repeats", type=int, default=1, help="Energy measurement repeats per case (for averaging)")
    parser.add_argument("--baseline-seconds", type=float, default=3.0, help="Idle baseline sampling duration (seconds)")
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
    parser.add_argument("--output-raw", default=os.path.join(ROOT_DIR, "test/prof/results/cases_output.txt"))
    parser.add_argument("--output-json", default=os.path.join(ROOT_DIR, "test/prof/results/cases_output.json"))
    return parser.parse_args()


def main():
    args = parse_args()

    backends = [b.lower().strip(", ") for b in args.backends]
    if len(backends) == 1 and "," in backends[0]:
        backends = backends[0].split(",")

    types = [t.lower().strip(", ") for t in args.types]
    if len(types) == 1 and "," in types[0]:
        types = types[0].split(",")

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs(os.path.dirname(args.output_raw), exist_ok=True)
    if args.energy != "none":
        os.makedirs(args.energy_out_dir, exist_ok=True)

    json_data = {
        "metadata": {"timestamp": run_id, "backends": backends, "types": types, "config": vars(args)},
        "results": [],
        "summary": {"total_pass": 0, "total_fail": 0},
    }

    print("Running testcase suite")
    print(f"Backends: {','.join(backends)}")
    print(f"Types   : {','.join(types)}")
    print(f"Q range : {args.q_min}..{args.q_max}")
    print(f"Defaults: k=min({args.k},N) mode={args.mode} run={args.run} threads={args.threads} verify={args.verify}")
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
    gpu_baseline_w = None
    if args.energy != "none":
        sleep_cmd = ["sleep", str(args.baseline_seconds)]
        modes = {resolve_energy_mode(b, args.energy) for b in backends}
        if "rapl" in modes:
            rapl_cmd = [os.path.join(ROOT_DIR, "test/prof/energy/measure_rapl.sh")]
            if args.rapl_path:
                rapl_cmd += ["--path", args.rapl_path]
            rapl_cmd += ["--"] + sleep_cmd
            rapl_baseline_w = capture_baseline_watts(rapl_cmd)
            print(f"Idle RAPL baseline (package): {rapl_baseline_w} W")
        if "gpu" in modes:
            gpu_cmd = [
                os.path.join(ROOT_DIR, "test/prof/energy/measure_smi.sh"),
                "--gpu-index",
                str(args.gpu_index),
                "--interval-ms",
                str(args.gpu_interval_ms),
                "--",
            ] + sleep_cmd
            gpu_baseline_w = capture_baseline_watts(gpu_cmd)
            print(f"Idle GPU baseline: {gpu_baseline_w} W")

    for backend in backends:
        binary_path = resolve_binary_path(backend)
        if not os.path.isfile(binary_path) or not os.access(binary_path, os.X_OK):
            print(f"\nBackend binary not found or not executable: {binary_path}")
            json_data["summary"]["total_fail"] += 1
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
            if backend == "npu" and not os.path.isfile(os.path.join(ROOT_DIR, "build/NPU/map_reduce.xclbin")):
                algorithms = ["bitonic", "gt"]

            for algo in algorithms:
                for q in range(args.q_min, args.q_max + 1):
                    n = 1 << q

                    effective_ks = sorted(list(set(k_val for k_val in args.k if k_val <= n)))
                    for k_eff in effective_ks:
                        seed = args.seed_base + q + k_eff
                        case_name = f"q{q:02d}_k{k_eff}_{args.mode}"

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
                        ]

                        energy_mode = resolve_energy_mode(backend, args.energy)
                        baseline_w = (
                            rapl_baseline_w
                            if energy_mode == "rapl"
                            else (gpu_baseline_w if energy_mode == "gpu" else None)
                        )

                        case_env = os.environ.copy()
                        npu_xclbin = resolve_npu_xclbin(backend, algo)
                        if npu_xclbin:
                            case_env["NPU_OFFLOAD_XCLBIN"] = npu_xclbin

                        # Execute Subprocess (repeated for energy averaging)
                        result = None
                        run_cmd = []
                        energy_case_file = ""
                        for rep in range(1, args.repeats + 1):
                            suffix = f"_rep{rep}" if args.repeats > 1 else ""
                            if energy_mode == "none":
                                run_cmd = [binary_path] + case_args
                                energy_case_file = ""
                            else:
                                energy_case_file = os.path.join(
                                    args.energy_out_dir,
                                    f"{run_id}_{backend}_{dtype}_{algo}_{case_name}{suffix}_{energy_mode}.txt",
                                )
                                if energy_mode == "rapl":
                                    run_cmd = [
                                        os.path.join(ROOT_DIR, "test/prof/energy/measure_rapl.sh"),
                                        "--out",
                                        energy_case_file,
                                    ]
                                    if args.rapl_path:
                                        run_cmd += ["--path", args.rapl_path]
                                else:
                                    run_cmd = [
                                        os.path.join(ROOT_DIR, "test/prof/energy/measure_smi.sh"),
                                        "--out",
                                        energy_case_file,
                                        "--gpu-index",
                                        str(args.gpu_index),
                                        "--interval-ms",
                                        str(args.gpu_interval_ms),
                                    ]
                                if baseline_w is not None:
                                    run_cmd += ["--baseline-watts", str(baseline_w)]
                                run_cmd += ["--", binary_path] + case_args
                            result = subprocess.run(run_cmd, capture_output=True, text=True, env=case_env)

                        stdout = result.stdout

                        case_status = "FAIL"
                        case_reason = "non-zero exit"

                        if result.returncode == 0:
                            if args.verify == "true" and expected_marker not in stdout:
                                case_reason = "PASS marker missing"
                                print(f"    [FAIL] {case_name} (algo={algo}) ({case_reason})")
                                type_fail += 1
                            else:
                                case_status = "PASS"
                                case_reason = "ok"
                                print(f"    [PASS] {case_name} (algo={algo})")
                                type_pass += 1
                        else:
                            print(f"    [FAIL] {case_name} (algo={algo}) (non-zero exit)")
                            type_fail += 1

                        # Write Raw File
                        with open(args.output_raw, "a", encoding="utf-8") as f_raw:
                            f_raw.write(f"### Case: {case_name}\n")
                            f_raw.write(f"Status: {case_status}\n")
                            f_raw.write(f"Reason: {case_reason}\n")
                            f_raw.write(f"Command: {' '.join(run_cmd)}\n")
                            if energy_case_file:
                                f_raw.write(f"Measurement file: {energy_case_file}\n")
                            f_raw.write(f"Output:\n{stdout}\n\n")

                        # Append to JSON structure
                        json_data["results"].append(
                            {
                                "backend": backend,
                                "type": dtype,
                                "algorithm": algo,
                                "case_name": case_name,
                                "q": q,
                                "k": k_eff,
                                "status": case_status,
                                "reason": case_reason,
                                "command": " ".join(run_cmd),
                                "energy_file": energy_case_file,
                                "exit_code": result.returncode,
                                "stdout": stdout.strip(),
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
    print(f"Wrote detailed text output to: {args.output_raw}")
    print(f"Wrote structured JSON output to: {args.output_json}")

    sys.exit(1 if json_data["summary"]["total_fail"] > 0 else 0)


if __name__ == "__main__":
    main()
