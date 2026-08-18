# Quickstart: Build, Run and Benchmark

This file gives the shortest path to build all backends, run a single case, 
and reproduce the benchmark data and plots.

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

## Run a Single Case

Each backend is a standalone binary, invoked via `make run-<backend>
ARGS="..."`:

```bash
make run-cpu ARGS="q=20 k=256 dtype=float algo=bitonic dist=uniform verify=true"
make run-gpu ARGS="q=20 k=256 dtype=float algo=bitonic verify=true"
make run-npu ARGS="q=20 k=256 dtype=int algo=bitonic verify=true"
```

Full argument grammar (`q=` is the only required key; everything else falls
back to a sensible default: `algo=bitonic`, `dtype=int`, `dist=uniform`,
`mode=max`, `run=trunc`, `seed=42`):

```
q=<q> [k=<k>] [mode=min|max] [dtype=int|uint|float|double|half]
[algo=bitonic|map_reduce|gt] [run=full|trunc|both] [debug=true|false]
[threads=<num>] [seed=<seed>] [verify=true|false] [min=<int>] [max=<int>]
[dist=uniform|normal|trimodal|sorted|reverse|adversarial]
```

`q` sets the input size as `n = 2^q`. `verify=true` cross-checks the result
against the `gt` baseline before returning. For NPU, the offload `.xclbin`
is picked automatically to match `algo=`; override with `NPU_OFFLOAD_XCLBIN`
if needed.

## Benchmark

To reproduce the full measurement set and plots unattended (cleans and
rebuilds everything, pins the machine, runs all four measurement sweeps,
plots all four, then restores conditions):

```bash
make benchmark            # GPU power cap defaults to 50 W
make benchmark GPU_W=60   # or pick your own cap
```

This asks for `sudo` once up front (needed for RAPL and for pinning), then
runs unattended. Results land under [bench/results](bench/results).

To run one measurement axis at a time instead of the full sweep, use the
scoped pipelines — each is measure + plot for one concern:

```bash
make benchmark-cases      # timing
make benchmark-energy     # energy (needs sudo for RAPL)
make benchmark-dists      # input-distribution sensitivity
make benchmark-roofline   # bandwidth + compare-exchange roofs
```

For a custom sweep (e.g. a narrower type/size grid while iterating), drive
`measure-*` and `plot` directly with your own `ARGS`:

```bash
./scripts/pin_conditions.sh 60          # pin governor/boost/GPU cap first
make measure-cases ARGS="cpu gpu npu --types int float --q-max 16 --run trunc --verify true"
make plot ARGS="--plot all"
ACTION=restore ./scripts/pin_conditions.sh
```

Notes:
- The profiler auto-bootstraps its Python venv the first time it runs.
- Unpinned runs (default governor is `powersave`, boost on) carry
  frequency/thermal variance that mean±stdev doesn't capture — pin before
  any sweep you intend to cite.
- `make benchmark` handles pinning and restoring conditions for you; the
  manual path above is for anyone driving `measure-*` outside of it.

For the full rationale behind these measurements (what the `algo` vs.
`end-to-end` fields mean, how energy and traffic are counted per backend,
the roofline model and the correctness checks) see the methodology
chapter of the thesis in [doc/thesis](doc/thesis) rather than this file;
