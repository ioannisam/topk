# CPU TOP-K

This directory contains a parallel CPU implementation of bitonic top-k using C++ and `std::thread`.

It supports:
- Full bitonic sorting network (reference path)
- Truncated bitonic execution for top-k output

## Layout

- `include/config.hpp` + `src/config.cpp`: `Config` object, argument parsing and config construction
- `include/layers.hpp` + `src/layers.cpp`: `Layer` object and bitonic layer schedule generation
- `include/algorithm.hpp` + `src/algorithm.cpp`: keep-mask generation and parallel network execution
- `include/utils.hpp` + `src/utils.cpp`: comparator counting utilities
- `include/reporting.hpp` + `src/reporting.cpp`: run/debug/timing/output print helpers
- `include/runner.hpp` + `src/runner.cpp`: top-k run pipeline and flow orchestration
- `src/main.cpp`: CLI entrypoint and error handling

## Build

```bash
cd CPU
cmake -S . -B build
cmake --build build -j
```

Optional CMake flags:

```bash
cmake -S . -B build \
  -DDEBUG=ON \
  -DCHECK=OFF
```

- `DEBUG`: default runtime debug mode (`debug`/`nodebug`)
- `CHECK`: default runtime full-reference run + correctness check (`check`/`nocheck`)

## Run

```bash
./build/topk <q> [k] [seed] [min|max] [sort|nosort] [debug|nodebug] [check|nocheck] [threads=<num>] [dtype=<type>]
```

- `N = 2^q` total elements
- execution threads default to hardware concurrency (capped by `N`)
- `k` defaults to `N`
- `seed` defaults to `42`
- `mode` defaults to `max`
- `sort` defaults to `nosort`
- `debug` defaults from `DEBUG`
- `check` defaults from `CHECK`
- `threads=<num>` optionally overrides execution threads
- `dtype=<type>` selects value type: `int`, `uint`, `float`, `double`, `fp16` (fp16 requires compiler support)

Example:

```bash
./build/topk 13 128 42 min sort debug check threads=16 dtype=float
```

Runtime tokens:
- `min`: smallest k values
- `max`: largest k values
- `sort`: sort final reported top-k values
- `nosort`: keep network order in final report
- `debug`: print diagnostics (threads, layers, comparator counts, skip ratio, speedup)
- `nodebug`: keep diagnostics minimal
- `check`: run full bitonic reference and validate top-k correctness
- `nocheck`: skip full-reference path for faster runs
