# CPU TOP-K

This directory contains a parallel CPU implementation of bitonic top-k using C++ and `std::thread`.

It supports:
- Full bitonic sorting network (reference path)
- Trunc bitonic execution for top-k output

## Layout

- `include/layers.hpp` + `src/layers.cpp`: `Layer` object and bitonic layer schedule generation
- `include/algorithm.hpp` + `src/algorithm.cpp`: keep-mask generation and parallel network execution
- `include/utils.hpp` + `src/utils.cpp`: comparator counting utilities (CPU-specific)
- `include/reporting.hpp` + `src/reporting.cpp`: run/debug/timing/output print helpers
- `include/runner.hpp` + `src/runner.cpp`: top-k run pipeline and flow orchestration
- `src/main.cpp`: CLI entrypoint and error handling (uses shared common parser)

Shared backend-agnostic foundation now lives in `../common`:

- `../common/include/common/config.hpp` + `../common/src/config.cpp`: config types and CLI/testcase parsing
- `../common/include/common/random.hpp` + `../common/src/random.cpp`: deterministic random input generation
- `../common/include/common/validation.hpp`: output formatting and testcase-answer validation helpers
- `../common/include/common/reporting.hpp` + `../common/src/reporting.cpp`: reusable reporting primitives (configuration, timing, output)

## Build

```bash
cd CPU
cmake -S . -B build
cmake --build build -j
```

Optional CMake flags:

```bash
cmake -S . -B build \
  -DDEBUG=ON
```

- `DEBUG`: default runtime debug mode (`debug`/`nodebug`)

## Run

```bash
./build/topk q=<q> [k=<k>] [mode=min|max] [dtype=<type>] [run=full|trunc|both] [debug=true|false] [threads=<num>] [seed=<seed>]
```

Only `key=value` arguments are accepted. Required key: `q`.
Arguments are accepted in any order, but examples below use the canonical order above for consistency.

Or run from a testcase file:

```bash
./build/topk <testcase-file>
# or
./build/topk --case <testcase-file>
# or
./build/topk case=<testcase-file>
```

- `N = 2^q` total elements
- execution threads default to hardware concurrency (capped by `N`)
- `k` defaults to `N`
- `seed` defaults to `42`
- `mode` defaults to `max`
- `debug` defaults from `DEBUG`
- run mode defaults to `trunc`
- `threads=<num>` optionally overrides execution threads
- `dtype=<type>` selects value type: `int`, `uint`, `float`, `double`, `fp16` (fp16 requires compiler support)

Example:

```bash
./build/topk q=13 k=128 mode=min dtype=float run=both debug=true threads=16 seed=42
```

Runtime tokens:
- `mode=min|max`: smallest or largest k values
- `debug=true|false`: enable/disable debug metrics
- `run=trunc|full|both`: select network execution mode

## Testcase File Format

Testcase files can hold everything a full CLI command includes.

Rules:
- First non-comment line: command arguments (same tokens you would pass to `./build/topk`, without the program name).
- Last non-comment line: expected top-k output values in sorted order.
- Lines starting with `#` are ignored.
- `check=true|false` is allowed only in testcase mode.

Minimal example:

```text
q=4 k=5 mode=max dtype=int run=both debug=false threads=4 seed=7 check=true
979 978 780 724 539
```

Optional tagged format is also supported:

```text
command: q=4 k=5 mode=max dtype=int run=both debug=false threads=4 seed=7 check=true
answer: 979 978 780 724 539
```

Validation behavior:
- `both` compares trunc top-k vs full-network top-k.
- If testcase command includes `check=true`, output is also validated against testcase expected-answer line.
- If testcase command includes `check=false`, testcase expected-answer validation is skipped.
