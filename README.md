# TOP-K

This is an effort to re-implement the truncated bitonic sort algorithm originally presented in
[N. Sismanis, N. Pitsianis and X. Sun, "Parallel search of k-nearest neighbors with synchronous operations"](https://ieeexplore.ieee.org/document/6408667)
in NPU and GPU hardware and compare the performance against modern high-performance GPU and CPU libraries and implementations, including `k-select`.

## Repository structure

- **`CPU/`, `GPU/`, `NPU/`** — the three backend implementations (AVX-512, CUDA, and MLIR-AIE respectively). Each has its own README with build instructions and file layout.
- **`common/`** — backend-agnostic code shared by all three: CLI parsing, bitonic network construction, the roofline model, energy measurement, reporting.
- **`bench/`** — the benchmarking harness: `measure/` runs sweeps, `profiler/` turns raw results into plots, `lib/` holds shared measurement tooling, `results/` stores the raw, derived, and plotted data.
- **`scripts/`** — maintenance scripts: linting, pinning system conditions for measurement, A/B and A-testing, spec validation.
- **`doc/`** — setup, quickstart, and validation guides, plus the full thesis write-up (see below).
- **`Makefile`** — the single entry point for building, running, measuring, plotting and benchmarking; run `make help` for the full target list.

## Where to find more

| Looking for...                               | See                                                          |
|----------------------------------------------|---------------------------------------------------------------|
| How to build and run one backend             | `CPU/README.md`, `GPU/README.md`, `NPU/README.md`             |
| Environment/toolchain setup                  | `doc/setup.md`                                                |
| Fastest path to build and benchmark          | `doc/quickstart.md`                                           |
| This machine's hardware specs                | `doc/specs.md`                                                |
| How to validate the toolchain                | `doc/validation.md`                                           |
| Full methodology, results and discussion     | `doc/thesis/` (Greek and English editions — `make thesis`)    |
