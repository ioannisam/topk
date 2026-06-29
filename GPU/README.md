# GPU TOP-K

This directory contains a CUDA backend implementation of the bitonic top-k pipeline, aligned with the CPU and NPU backend structure and shared API contracts.

It supports:
- Full bitonic sorting network (reference path)
- Trunc bitonic execution for top-k output
- Optional fp16 execution path through CUDA half conversion

## Layout

- include/algorithm.hpp + src/algorithm.cu: CUDA network execution kernels and device query utilities
- include/reporting.hpp + src/reporting.cpp: GPU-specific configuration and debug reporting hooks
- include/runner.hpp + src/runner.cpp: GPU backend hook wiring into the shared top-k pipeline
- src/main.cpp: CLI entrypoint and error handling (uses shared common parser)

Shared backend-agnostic foundation lives in ../common.

## Build

```bash
cd GPU
cmake -S . -B build
cmake --build build -j
```

Optional CMake flags:

```bash
cmake -S . -B build \
  -DDEBUG=ON
```

- DEBUG: default runtime debug mode (debug or nodebug)

## Run

```bash
./build/topk q=<q> [k=<k>] [mode=min|max] [dtype=<type>] [algo=bitonic|map_reduce|gt] [run=full|trunc|both] [debug=true|false] [threads=<num>] [seed=<seed>] [verify=true|false] [min=<int>] [max=<int>] [dist=uniform|normal|sorted|reverse]
```

Only key=value arguments are accepted. Required key: q.

Notes:
- N = 2^q total elements
- k defaults to N
- mode defaults to max
- run mode defaults to trunc
- seed defaults to 42
- debug defaults from DEBUG
- verify defaults to false
- random range defaults to min=0 and max=1000
- threads is accepted by shared CLI for parity across backends, but GPU launch geometry is selected internally by the CUDA backend
- dtype supports int, uint, float, double, fp16 (fp16 uses float input with CUDA half compute path)
- algo chooses bitonic (default), map_reduce, or gt (Thrust full-sort reference baseline)
- dist selects the input distribution: uniform (default), normal, sorted, reverse

## Example

```bash
./build/topk q=13 k=128 mode=min dtype=float run=both debug=true seed=42 verify=true min=0 max=1000
```

This backend reports:
- CUDA device in configuration/debug sections
- Full and trunc execution timing
- Comparator skip metrics
- Optional CPU sorted-reference correctness check when verify=true

## Profiling Workflow (Dynamic Cases)

Use the dynamic profiler runner (no static `.case` files):

```bash
cd /home/ioannis/Development/Thesis
./test/prof/cases/run_testcases.sh gpu float --q-min 1 --q-max 16 --verify true --min 0 --max 1000
```
