# CPU TOP-K

This directory contains a parallel CPU implementation of bitonic top-k using C++ and `std::thread`.

It supports:
- Full bitonic sorting network (reference path)
- Truncated bitonic execution for top-k output

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
./build/topk <q> <p> [k] [seed] [min|max] [sort|nosort] [debug|nodebug] [check|nocheck] [threads=<num>]
```

- `N = 2^(q+p)` total elements
- `2^p` virtual bitonic ranks (algorithm layout)
- `2^q` elements per virtual rank
- execution threads default to hardware concurrency (capped by `N`)
- `k` defaults to `N`
- `seed` defaults to `42`
- `mode` defaults to `max`
- `sort` defaults to `nosort`
- `debug` defaults from `DEBUG`
- `check` defaults from `CHECK`
- `threads=<num>` optionally overrides execution threads

Example:

```bash
./build/topk 10 3 128 42 min sort debug check threads=16
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
