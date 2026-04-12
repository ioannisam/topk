# NPU TOP-K

This directory contains an NPU backend scaffold for the bitonic top-k pipeline, aligned with the CPU/GPU backend structure and shared API contracts.

Current design is offload-only:
- Runtime requires a valid XRT device and a kernel xclbin.
- Host side handles kernel loading, buffer management and per-layer dispatch.

## Layout

- `include/algorithm.hpp` + `src/algorithm.cpp`: NPU offload orchestration and runtime queries
- `include/reporting.hpp` + `src/reporting.cpp`: NPU-specific configuration/debug reporting hooks
- `include/runner.hpp` + `src/runner.cpp`: NPU backend hook wiring into shared top-k pipeline
- `src/main.cpp`: CLI entrypoint and error handling (uses shared common parser)

Shared backend-agnostic foundation lives in `../common`.

## Build

```bash
cd NPU
cmake -S . -B build
cmake --build build -j
```

Optional CMake flags:

```bash
cmake -S . -B build -DDEBUG=ON
```

## Run

```bash
./build/topk q=<q> [k=<k>] [mode=min|max] [dtype=<type>] [run=full|trunc|both] [debug=true|false] [seed=<seed>]
```

Only `key=value` arguments are accepted. Required key: `q`.

Or run from a testcase file:

```bash
./build/topk <testcase-file>
./build/topk --case <testcase-file>
./build/topk case=<testcase-file>
```

## Offload Configuration

Required environment variables:

- `NPU_OFFLOAD_XCLBIN`: absolute/relative path to compiled xclbin

Optional environment variables:

- `NPU_OFFLOAD_KERNEL`: kernel symbol name inside xclbin (default: `bitonic_layer`)
- `NPU_OFFLOAD_WAIT_MS`: dispatch wait timeout in milliseconds (default: `5000`)
- `NPU_OFFLOAD_OPCODE`: DPU opcode override when DPU ABI is detected (default: `3`)
- `NPU_OFFLOAD_NINSTR_BYTES`: if set to `1|true|on`, pass instruction byte-count instead of word-count for DPU ABI
- `NPU_OFFLOAD_INSTR`: optional override path for DPU instruction binary

Behavior:

- If `NPU_OFFLOAD_XCLBIN` is missing, execution fails with an error.
- If kernel launch fails, execution fails with an error.
- No host-thread/CPU fallback is performed by this backend.

Example:

```bash
cd /home/ioannis/Development/Thesis
export NPU_OFFLOAD_XCLBIN=/path/to/bitonic_layer.xclbin
export NPU_OFFLOAD_KERNEL=bitonic_layer

./NPU/build/topk q=13 k=128 mode=max dtype=int run=both debug=true seed=42
```

## Kernel ABI Contract (Host <-> NPU)

The kernel must accept arguments in this order for the standard ABI path:

1. `data_bo` (buffer): input/output array (`n * sizeof(T)` bytes)
2. `pairs_bo` (buffer): truncated comparator indices (`uint32_t` base indices), can be dummy for full mode
3. `n` (`uint32_t`): element count
4. `stage` (`uint32_t`): bitonic stage (`k`)
5. `step` (`uint32_t`): bitonic stride (`j`)
6. `begin` (`uint32_t`): start offset inside `pairs_bo` for trunc mode
7. `count` (`uint32_t`): active comparator count for this layer
8. `trunc` (`uint32_t`): `0` for full mode, `1` for trunc mode

Required per-dispatch semantics:

- Full mode (`trunc=0`): perform `count = n/2` compare-swap operations for layer `(stage, step)`.
- Trunc mode (`trunc=1`): perform `count` compare-swaps using `i = pairs_bo[begin + t]`, `ixj = i ^ step`.
- Ascending decision must match: `ascending = ((i & stage) == 0)`.

Current host offload dtype support:

- `int`, `uint`, `float` (4-byte element types)

`double` and `fp16` are not offloaded by current host code.

## Runtime Requirements

Before expecting NPU dispatches to complete, verify:

- `xrt-smi --version` reports expected XRT/runtime stack for your platform.
- `xrt-smi examine` lists your NPU device.
- `modinfo amdxdna` reports the loaded kernel module and firmware path.

If basic XRT sample workloads fail on your machine, treat that as a platform/runtime issue before debugging bitonic kernel logic.
