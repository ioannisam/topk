# NPU TOP-K

This directory contains an NPU backend scaffold for the bitonic top-k pipeline, aligned with the CPU/GPU backend structure and shared API contracts.

Current design is offload-only:
- Runtime requires a valid XRT device and a kernel `.xclbin` along with its `.bin` instructions sequence.
- Host side handles kernel loading, buffer management, stride packing, and per-chunk dispatch.

## Layout

- `include/algorithm.hpp` + `src/algorithm.cpp`: NPU offload orchestration, stride-packing fallback, and runtime queries
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
NPU_OFFLOAD_XCLBIN=./build/NPU/bitonic.xclbin \
	make run-npu ARGS="q=<q> [k=<k>] [algo=bitonic] [mode=min|max] [dtype=<type>] [run=full|trunc|both] [debug=true|false] [threads=<num>] [seed=<seed>] [verify=true|false] [min=<int>] [max=<int>]"
```

Only `key=value` arguments are accepted. Required key: `q`.

Defaults:
- `k=N`, `mode=max`, `run=trunc`, `seed=42`
- `verify=false`
- random range `min=0`, `max=1000`

Notes:

- The `run-npu` target lives at the repository root and forwards `ARGS` to the NPU CLI.
- `verify=true` enables an additional CPU sorted-reference correctness check after NPU execution.

## Offload Configuration

Required environment variables:

- `NPU_OFFLOAD_XCLBIN`: absolute/relative path to compiled `.xclbin`. *(Note: the host expects the accompanying `.bin` instruction file to reside in the same directory).*

Optional environment variables:

- `NPU_OFFLOAD_KERNEL`: kernel symbol name inside xclbin (default: `MLIR_AIE`)
- `NPU_OFFLOAD_WAIT_MS`: dispatch wait timeout in milliseconds (default: `5000`)

Behavior:

- If `NPU_OFFLOAD_XCLBIN` is missing, execution fails with an error.
- If kernel launch fails, execution fails with an error.
- The bitonic offload follows the Truncated Bitonic Sort streaming model. Each of the 4 NPU columns runs one core that sorts its 1024-element tiles into ascending runs of length 16 (truncation: only short sorted runs are built, not a full sort), using hardware lane-shuffle compare-exchanges. The host then merges the sorted runs into a running top-k via a small max-heap with per-run early-out ("merge-and-purge"). Tiles are dispatched in double-buffered batches so host packing/merging overlaps NPU compute. There is no host-side global bitonic merge.
- Each value is mapped to a monotonic `int32` key before the device sort (the AIE core compares with signed `int32` min/max), so `int`, `uint`, and `float` — including negative floats — all sort correctly; the reduction runs in key space and only the surviving keys are mapped back.

Example:

```bash
cd /home/ioannis/Development/Thesis
NPU_OFFLOAD_XCLBIN=./build/NPU/bitonic.xclbin \
	make run-npu ARGS="q=17 k=15 algo=bitonic run=trunc verify=true"
```

## Kernel ABI Contract (Host <-> NPU)

The AIE kernel logic is built via MLIR (`aiecc`). The bitonic sort kernel takes no
per-tile configuration (the sort network is fixed), so its `runtime_sequence(out, inp)`
maps the XRT run arguments as follows:

1. **Arg 0**: Opcode (`uint32_t`: `3` for execute)
2. **Arg 1**: Instruction Buffer (`xrt::bo` cacheable memory containing `.bin` payload)
3. **Arg 2**: Instruction Word Count (`uint32_t`)
4. **Arg 3**: Output Data Buffer (`xrt::bo` `dst_bo`, sorted `int32` key tiles)
5. **Arg 4**: Input Data Buffer (`xrt::bo` `src_bo`, `int32` key tiles)

The `map_reduce` kernel keeps a separate ABI with an extra config buffer (threshold,
mode, sentinel) at **Arg 3**, shifting `dst_bo`/`src_bo` to **Arg 4**/**Arg 5**.

Current host offload dtype support:

- `int`, `uint`, `float` (4-byte element types)

`double` and `fp16` are not offloaded by current host code.

## Runtime Requirements

Before expecting NPU dispatches to complete, verify:

- `xrt-smi --version` reports expected XRT/runtime stack for your platform.
- `xrt-smi examine` lists your NPU device.
- `modinfo amdxdna` reports the loaded kernel module and firmware path.
