# Specifications of This Machine

## System Information
* **Host Model:** Lenovo Legion 5 15AHP10
* **Operating System:** Arch Linux x86_64
* **Kernel:** Linux 7.1.5-arch1-2 (re-check with `make specs`; this drifts on every kernel update)
* **Desktop Environment:** KDE Plasma 6.6.3 on Wayland
* **System RAM:** ~32 GB DDR5 installed (`MemTotal` 32138656 kB ≈ 30.6 GiB usable), upgraded from ~16 GB by adding a second module.
  * **Configuration:** Dual-channel, confirmed by `dmidecode -t memory`: two 16 GiB DDR5 SODIMMs (mixed vendors) on CHANNEL A and CHANNEL B, both at a configured 5600 MT/s. Matches the measured ~2× jump in sustained DRAM bandwidth.
  * **Measured bandwidth, external AVX-512 stream benchmark** (64 MB working set above L3, 16 threads): read+write 25.4 → **63.0 GB/s**, copy 17.8 → 36.1 GB/s after the upgrade. This is what established the ~2× dual-channel jump; it is not the figure the plots compare against.
  * **Measured bandwidth, this repo's roofline** (`make measure-roofline`, written to `bench/results/raw/roofline/roofline.json`): the `read`, `copy` and `cache_read` kernels. These are the numbers every bandwidth and roof-utilization plot uses, so quote **these** in the thesis, per kernel and per working-set size rather than as a single "wall". They are deliberately not transcribed here — read them off `bench/results/derived/roofline.csv` or `bench/results/plots/machine/cache_ladder.png` for the current run, because a hardcoded copy drifts the moment the machine is re-measured.
  * The two do not measure the same thing (different tool, different kernels, and `copy` counts both directions), so do not expect the 63.0 GB/s figure to reappear in `roofline.csv`. Cite one or the other and say which.
* **Storage:**
  * NVMe0: 476.9 GB WD PC SN7100S (root on ext4)
  * NVMe1: 931.5 GB Lexar NM710 (additional drive, NTFS partitions present)

## CPU: AMD Ryzen 7 260
* **Model:** AMD Ryzen 7 260 w/ Radeon 780M Graphics
* **Architecture:** x86_64 (microarchitecture details like Zen 4/Hawk Point are vendor-datasheet claims)
* **Cores/Threads:** 8 cores / 16 threads
* **Clock Speed (validated):** Max ~5.10 GHz
* **Cache:** 8 MiB L2 / 16 MiB L3
* **Instruction Sets:** AVX-512 flags
* **Power Profiling Interface:**
  * `/sys/class/powercap/intel-rapl*` exists as class entries.
  * `energy_uj` files are discovered in this session, and can be used for per-run CPU package-energy integration.

## GPU: NVIDIA GeForce RTX 5060 (Laptop)
* **Model:** NVIDIA GeForce RTX 5060 Laptop GPU
* **Driver:** 610.43.03
* **CUDA Version:** 13.3 (`nvcc` 13.3.73)
* **VRAM:** 8151 MiB (~8 GB)
* **Power Limits (validated):**
  * Default/Requested: 50.00 W
  * Max: 115.00 W
* **Architecture and bandwidth numbers:** Keep as vendor claims unless validated from NVIDIA technical documentation.
* **Integrated GPU (APU):** AMD Radeon 780M

## NPU: AMD XDNA (Ryzen AI)
* **Runtime Device:** RyzenAI-npu1 at BDF `0000:66:00.1`
* **Kernel Driver Module:** `amdxdna` loaded (in-tree, so its version tracks the running kernel)
* **NPU Firmware Version:** 1.5.5.391
* **XRT Version:** 2.21.75
* **Architecture / TOPS figures:** Keep as vendor claims unless tied to an AMD official source in this document.
* **Memory:** Likely shares system DDR5 memory

## Validation Status Summary
* **Fully validated from live commands:** host model, OS/kernel, desktop session, CPU topology/cache, GPU model/driver/CUDA/VRAM/power limits, NPU presence/firmware/runtime visibility, system RAM size (32 GB), dual-channel DDR5-5600 topology (two 16 GiB modules on CHANNEL A/B via `dmidecode`), and sustained DRAM bandwidth (~63 GB/s via an external AVX-512 stream benchmark; this repo's roofline measures the same wall per kernel and per working-set size, and is what the plots use).
* **Partially validated / vendor-claim fields:** process node, architecture marketing labels, TOPS and theoretical bandwidth, NPU shares system DDR5 memory.
