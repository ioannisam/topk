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
./build/topk q=<q> [k=<k>] [mode=min|max] [dtype=<type>] [run=full|trunc|both] [debug=true|false] [threads=<num>] [seed=<seed>]
```

Only key=value arguments are accepted. Required key: q.

Or run from a testcase file:

```bash
./build/topk <testcase-file>
# or
./build/topk --case <testcase-file>
# or
./build/topk case=<testcase-file>
```

Notes:
- N = 2^q total elements
- k defaults to N
- mode defaults to max
- run mode defaults to trunc
- seed defaults to 42
- debug defaults from DEBUG
- threads is accepted by shared CLI for parity across backends, but GPU launch geometry is selected internally by the CUDA backend
- dtype supports int, uint, float, double, fp16 (fp16 uses float input with CUDA half compute path)

## Example

```bash
./build/topk q=13 k=128 mode=min dtype=float run=both debug=true seed=42
```

This backend reports:
- CUDA device in configuration/debug sections
- Full and trunc execution timing
- Comparator skip metrics
- Optional full vs trunc correctness check when run=both
