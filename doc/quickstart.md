# Quickstart: Build and Profile

This file gives the shortest path to build all backends and generate profiler plots.

Prerequisites:
- Build toolchains installed (see setup guide)

## Build Everything

Build all backends in one shot:

```bash
make build-all
```

Build only one backend:

```bash
make build-cpu
make build-gpu
make build-npu
```

Build outputs are placed under [build](build).

## Use the Profiler

1) Run testcase sweeps to generate profiler inputs:

```bash
make run-cases ARGS="cpu gpu npu --types int float --q-max 16 --run trunc --verify true"
```

2) Generate plots from the latest run:

```bash
make run-profiler ARGS="--plot all"
```

Notes:
- The profiler auto-bootstraps its Python venv the first time it runs.
- Testcase outputs and plots are stored under [test/prof/results](test/prof/results).
