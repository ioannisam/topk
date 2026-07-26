# CPU TOP-K

This directory contains a parallel CPU implementation of bitonic top-k using C++ and `std::thread`.

It supports:
- Full bitonic sorting network (reference path)
- Trunc bitonic execution for top-k output
- Map-reduce top-k execution (`algo=map_reduce`): per-tile heap top-k + final nth_element reduction

## Layout

- `include/algorithm.hpp` + `src/algorithm.cpp`: CPU parallel network execution kernels
- `include/reporting.hpp` + `src/reporting.cpp`: CPU-specific configuration/debug reporting hooks
- `include/runner.hpp` + `src/runner.cpp`: CPU backend hook wiring into shared top-k pipeline
- `src/main.cpp`: CLI entrypoint and error handling (uses shared common parser)

Shared backend-agnostic foundation now lives in `../common`:

- `../common/include/common/config.hpp` + `../common/src/config.cpp`: config types and key=value CLI parsing
- `../common/include/common/bitonic.hpp` + `../common/src/bitonic.cpp`: bitonic layers, masks, and comparator counting
- `../common/include/common/random.hpp` + `../common/src/random.cpp`: deterministic random input generation
- `../common/include/common/validation.hpp`: output formatting and numeric comparison helpers
- `../common/include/common/reporting.hpp` + `../common/src/reporting.cpp`: reusable reporting primitives (configuration, timing, output)
- `../common/include/common/runner.hpp`: shared top-k runner flow and hook interface reused by CPU/GPU/NPU.

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
./build/topk q=<q> [k=<k>] [mode=min|max] [dtype=<type>] [algo=bitonic|map_reduce|gt] [run=full|trunc|both] [debug=true|false] [threads=<num>] [seed=<seed>] [verify=true|false] [min=<int>] [max=<int>] [dist=uniform|normal|sorted|reverse]
```

Only `key=value` arguments are accepted. Required key: `q`.
Arguments are accepted in any order, but examples below use the canonical order above for consistency.

- `N = 2^q` total elements
- execution threads default to hardware concurrency (capped by `N`)
- `k` defaults to `N`
- `seed` defaults to `42`
- `mode` defaults to `max`
- `debug` defaults from `DEBUG`
- run mode defaults to `trunc`
- `verify` defaults to `false`
- random range defaults to `min=0`, `max=1000`
- `threads=<num>` optionally overrides execution threads
- `dtype=<type>` selects value type: `int`, `uint`, `float`, `double`, `half` (half requires compiler support)
- `algo=<name>` chooses backend algorithm: `bitonic` (default), `map_reduce`, or `gt` (std::partial_sort top-k baseline)
- `dist=<name>` selects input distribution: `uniform` (default), `normal`, `sorted`, `reverse`

Example:

```bash
./build/topk q=13 k=128 mode=min dtype=float algo=map_reduce run=both debug=true threads=16 seed=42 verify=true min=0 max=1000
```

Runtime tokens:
- `mode=min|max`: smallest or largest k values
- `algo=bitonic|map_reduce`: sorting-network path or map-reduce path
- `debug=true|false`: enable/disable debug metrics
- `run=trunc|full|both`: select network execution mode
- `verify=true|false`: validate final top-k against a CPU full-sort reference
- `min=<int>`, `max=<int>`: inclusive random input range (`max` must be `>= min`)

For `algo=map_reduce`, runtime is a single map-reduce pass. Use `verify=true` for explicit correctness validation against the CPU sorted reference.

## Profiling Workflow (Dynamic Cases)

The old static `.case` workflow has been removed. Use dynamic generation instead:

```bash
cd /home/ioannis/Development/Thesis
python3 ./bench/lib/runner.py cpu --types int --q-min 1 --q-max 16 --verify true --min 0 --max 1000
```
