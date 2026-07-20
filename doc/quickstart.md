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
make measure-cases ARGS="cpu gpu npu --types int float --q-max 16 --run trunc --verify true"
```

To sweep input distributions and seeds (each `(dist, seed)` is an independent
sample that feeds the error bands):

```bash
make measure-cases ARGS="cpu gpu npu --types int float --q-max 16 \
  --dists uniform normal sorted reverse --seeds 5 --verify true"
```

2) Generate plots from the latest run:

```bash
make plot ARGS="--plot all"
```

Notes:
- The profiler auto-bootstraps its Python venv the first time it runs.
- Testcase outputs and plots are stored under [bench/results](bench/results).

## Benchmark Methodology

- **Pinned conditions (run first):** before a sweep, pin the machine into a fixed,
  documented state so timing/energy don't drift with frequency scaling or thermals:

  ```bash
  ./scripts/pin_conditions.sh 60      # governor=performance, boost off, GPU persistence + 60 W cap
  ACTION=show ./scripts/pin_conditions.sh   # record the exact state in the thesis
  ```

  Pick **one** GPU power cap and keep it for every run. The defaults are *not* pinned
  (the machine ships on the `powersave` governor with boost on), so unpinned runs carry
  uncontrolled frequency/thermal variance that mean±stdev does not capture. The laptop
  shares one thermal budget across CPU/GPU/NPU, so leave a cooldown between backends and
  don't run other heavy work during a sweep.
- **Timing:** each case runs 5 warmup + 50 measured iterations; the binary reports
  the **mean ± standard deviation** (and the min as a best-case line) for both the
  end-to-end and algorithmic channels. Plots aggregate across the `(dist, seed)`
  samples with `--agg mean --error-bars std` by default.
- **Input distributions** (`dist=`): `uniform` (default), `normal` (Gaussian about
  the mid-range), `sorted` (ascending), `reverse` (descending). Bitonic is
  data-oblivious, so its timing is a control; `map_reduce` and the library baselines
  are data-dependent.
- **Energy: in-process counters are the primary source.** The binary reads the energy
  counters itself, bracketing the *same* regions its timers bracket
  (`common/src/energy.cpp`, wired into `run_benchmark`). Reported under `== Energy ==`
  as a **sum over the 50 measured iterations**, not a per-iteration mean: RAPL
  quantises each individual delta, but the counters free-run, so the sum stays accurate.
  The profiler divides by `Benchmark iterations` for per-operation figures.
  - Reading `energy_uj` needs root, so energy sweeps run under `sudo` (as they already
    did for `measure_rapl.sh`). Without it the binary prints `rapl:unavailable` and
    reports **no** energy rather than reporting zeros as data.
  - **Why in-process:** the external wrapper measures the *whole process*, but the
    benchmark loop is only 1.6%–92% of that window and the fraction varies
    systematically **by backend** — so wrapper energy *biases* cross-backend comparison
    rather than merely adding noise. The wrapper is retained as a cross-check
    (process total must be ≥ the in-process total), not as the primary number.
  - Three windows are reported per run: `e2e` (sum of per-iteration end-to-end
    brackets), `algo` (sum of per-iteration algorithmic brackets) and `loop` (one
    bracket around the entire measured loop). **`e2e` and `algo` are the reported
    numbers** — they match the two time channels exactly. `loop` is strictly larger
    because it also contains the harness scaffolding between iterations (each iteration
    restores its input from a backup copy, outside the timed region); for CPU bitonic at
    q=20 it spans ~1.9x the e2e time. Use `loop` only as an upper-bound cross-check and
    as the measure of how much of the process is scaffolding.
- **Symmetric e2e scope (host + backend):** every backend counts the host RAPL
  **package** and the GPU adds its **board**. So e2e is the whole cost-to-solution for
  all three and the GPU does not hide its host transfer/launch cost.
- **Channel semantics per backend:**
  - **CPU** → algorithmic ≡ e2e (the CPU *is* the accelerator; no host to strip). Both
    channels share a single counter read, so they are exactly equal by construction.
  - **NPU** → e2e brackets the host-side call; algorithmic brackets the offload region
    inside `npu::*::run_topk`. The difference is XRT dispatch and host marshalling.
  - **GPU** → e2e brackets the host-side call (including H2D/D2H); algorithmic brackets
    the CUDA-event region. The difference is transfer and allocation cost.
- **GPU board power is integrated in-process, not read as an energy counter.**
  `nvmlDeviceGetTotalEnergyConsumption` is unusable on this part: it advances in ~20 J
  steps every ~96 ms, so bracketing a millisecond kernel returns a random multiple of
  the quantum. Three identical q=24 runs gave 40.5 / 18.6 / 8.4 J of "algorithmic"
  energy while the *time* channel varied by 0.2%; at idle it reports 13.1 W against
  10.8 W of integrated power. Instead a background thread samples
  `nvmlDeviceGetPowerUsage` every 2 ms and integrates it, which reproduces the measured
  idle board baseline (10.64 W) and yields 20.5–26.1 W under load across repeat runs.
  Enabled for the GPU backend via `TOPK_ENERGY_DEVICE=1` (set by `runner.py`).
  - The underlying power reading only changes every ~231 ms, so a single 7 ms
    algorithmic window is a stale-power estimate. Summed over 50 iterations the
    staleness averages out, but **GPU algorithmic energy is lower-confidence than
    CPU/NPU**, where RAPL updates at ~1 ms. Treat the GPU algo/e2e *split* as
    approximate; the `loop` total is solid.
- **The RAPL `core` domain is unusable on this part — do not build claims on it.**
  Measured directly: under a pure 8-thread spin loop, package rises 6.66 -> 59.55 W
  (+52.9 W) while `core` rises only 0.51 -> 6.53 W (+6.0 W). The `core` domain captures
  ~11% of real core energy. It is still recorded, but `package - core` is **not** a valid
  way to isolate the NPU subsystem from the host cores on this SoC and any earlier
  result derived that way should be discarded.
- **The `wait` channel measures time reliably, energy *not at all*.** A scope inside
  `wait_for_runlist_or_throw` brackets every NPU dispatch wait. The **wait seconds** and
  **wait count** are trustworthy (steady_clock, ns resolution) and are the useful output:
  at q=22/k=256 the host waits ~6.5 ms of a ~151 ms offload region across 200 dispatches,
  i.e. **only ~4-5% of the offload is spent waiting for the NPU**. The **wait joules**
  are *not* usable: RAPL updates every ~1208 us in ~21.7 mJ steps, while each wait is
  ~34 us — 39x shorter. Five identical runs gave 17.9 / 21.4 / 14.7 / 16.8 / 46.0 W,
  resting on 4.6-14.7 counter ticks. Do not derive an NPU power figure from it.
  Resolving NPU-only draw needs a **sustained probe**: prepare one batch, then loop
  execute+wait for >= 1 s with the host otherwise idle, so the wait window exceeds the
  counter period.
- **What the NPU numbers can and cannot show.** RAPL package covers the whole SoC: host
  cores, NPU tiles, uncore and memory controller share one counter. So an NPU run is
  measured as "total SoC energy while the NPU is working," including host-side XRT
  dispatch and marshalling. This is a fair **cost-to-solution** comparison and it is
  symmetric across backends — but it cannot on its own demonstrate that the NPU performs
  the computation with less energy, because the NPU's own draw is never separable from
  the host's. Isolating the NPU tile would need external instrumentation.
- **Net (idle-subtracted) energy** is computed by the profiler as
  `gross - baseline_watts x window_seconds`, using the per-window durations the binary
  reports. `baseline_watts` is already the **combined** idle figure for that backend's
  counter set (for the GPU, host package + board), so it is not added to the board
  baseline again.
- Use the **net** (idle-subtracted) metric, especially for the NPU (removes the static
  uncore) and GPU (cancels the power sampler's own host overhead). True per-component
  cross-device parity (isolating the NPU tile) would still need a wall-socket meter.
- **Roofline (`make measure-roofline`, plotted by `roofline` / `memory-bandwidth-vs-n`).**
  Measures each backend's achievable bandwidth and compute ceilings by sweeping
  arithmetic intensity, writing `bench/results/raw/roofline/roofline.json`. Measured here:
  **CPU 57.5 GB/s / 905 GFLOP/s** (ridge AI 15.7, observed turnover at AI 16.25) and
  **GPU 346.0 GB/s / 10207 GFLOP/s** (ridge AI 29.5). The predicted ridge matching the
  observed turnover is the internal check that the sweep is well formed.
  - **Compare like with like: the roofline uses a 512 MiB DRAM-resident working set.**
    Top-k at n=2^20 with 4-byte elements is only 4 MiB and fits in the 16 MiB L3, so it
    reads from cache and can legitimately *exceed* the DRAM ceiling — CPU `map_reduce`
    at n=2^20 measures 97.5 GB/s, or 170% of roof, which is an artefact of the
    comparison, not a result. Only n=2^24 (64 MiB > L3) is genuinely DRAM-bound, so
    **quote roofline efficiency at n=2^24 only**.
  - At n=2^24 the efficiencies are: CPU `map_reduce` **89.9%** of roof, GPU
    `map_reduce` 45.2%, GPU `bitonic` 15.9%, CPU `bitonic` 11.8%. The CPU top-k is
    therefore already near the memory-bandwidth limit, which bounds how much any
    accelerator can win on this problem.
  - **The NPU roofline measures the offload *data path*, not AIE compute.** There is no
    variable-arithmetic-intensity AIE kernel available (writing one needs a new xclbin
    via mlir-aie), so `build/NPU/roofline` reports no `fma` sweep and therefore no
    compute roof. What it does report is the host-side staging path that feeds the NPU:
    `read` and `copy` over the mapped `host_only` buffer, plus `stage_h2d`/`stage_d2h`
    for the `xrt::bo::sync` cost. Measured: **read 55.5 GB/s, copy 44.3 GB/s**, which
    tracks the CPU roof (57.5 GB/s) because a `host_only` BO is host DRAM - that
    agreement is the sanity check, not an NPU result.
  - Read the NPU ceiling as "how fast the host can stream the buffer the NPU is fed
    from," not "AIE fabric bandwidth." The useful inference is that NPU top-k reaches
    only 5-6 GB/s against a 55 GB/s staging path, so the limit is per-batch dispatch and
    encode overhead rather than raw streaming bandwidth. `stage_h2d`/`stage_d2h` report
    ~122 GB/s, i.e. faster than DRAM, which means `sync` on a `host_only` BO is close to
    a no-op and no bulk transfer is taking place.
- **Baselines** (`algo=gt`): CPU/NPU use `std::partial_sort` (the standard-library
  heap-based top-k, which for `k << n` is far faster than `nth_element` thanks to its
  cache-resident size-k heap and high rejection rate); GPU uses Thrust's `thrust::sort`
  + truncate. Both are industry-standard library calls. Thrust sorts all N, so treat the
  GPU `gt` as a full-sort *reference* baseline rather than a tuned top-k.
