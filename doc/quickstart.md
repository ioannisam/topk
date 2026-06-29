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

To sweep input distributions and seeds (each `(dist, seed)` is an independent
sample that feeds the error bands):

```bash
make run-cases ARGS="cpu gpu npu --types int float --q-max 16 \
  --dists uniform normal sorted reverse --seeds 5 --verify true"
```

2) Generate plots from the latest run:

```bash
make run-profiler ARGS="--plot all"
```

Notes:
- The profiler auto-bootstraps its Python venv the first time it runs.
- Testcase outputs and plots are stored under [test/prof/results](test/prof/results).

## Benchmark Methodology

- **Timing:** each case runs 5 warmup + 50 measured iterations; the binary reports
  the **mean ± standard deviation** (and the min as a best-case line) for both the
  end-to-end and algorithmic channels. Plots aggregate across the `(dist, seed)`
  samples with `--agg mean --error-bars std` by default.
- **Input distributions** (`dist=`): `uniform` (default), `normal` (Gaussian about
  the mid-range), `sorted` (ascending), `reverse` (descending). Bitonic is
  data-oblivious, so its timing is a control; `map_reduce` and the library baselines
  are data-dependent.
- **Energy:** reported as **per-operation, end-to-end** Joules. The wrapper integrates
  power over the whole process (the binary reports `Benchmark iterations`, warmup +
  measured), and the profiler divides by that count so energy shares the same window as
  the e2e time. Energy is paired with **e2e** time (EDP, Pareto), never with algorithmic
  time. The meters can't resolve a sub-millisecond kernel, so a separate "algorithmic
  energy" is not measurable. Trust the larger `q` points, where the measured loop
  dominates and fixed process overhead is amortized away. Scope note: energy is RAPL
  package for CPU/NPU and GPU board for GPU, so the NPU figure is host energy to drive
  the NPU (no XDNA power telemetry exists) and the GPU figure excludes host transfer
  energy. The numbers are consistent per device, but not a single shared scope.
- **Baselines** (`algo=gt`): CPU/NPU use `std::partial_sort` (the standard-library
  heap-based top-k, which for `k << n` is far faster than `nth_element` thanks to its
  cache-resident size-k heap and high rejection rate); GPU uses Thrust's `thrust::sort`
  + truncate. Both are industry-standard library calls. Thrust sorts all N, so treat the
  GPU `gt` as a full-sort *reference* baseline rather than a tuned top-k.
