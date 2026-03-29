# Specifications of This Machine

## System Information
* **Host Model:** Lenovo Legion 5 15AHP10
* **Operating System:** Arch Linux x86_64
* **Kernel:** Linux 6.19.9-arch1-1
* **Desktop Environment:** KDE Plasma 6.6.3 on Wayland
* **System RAM:** ~16 GB installed (reported usable memory: ~14 GiB)
  * **Configuration:** Single-channel note kept as a physical-hardware assumption.
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
* **Driver:** 595.58.03
* **CUDA Version:** 13.2
* **VRAM:** 8151 MiB (~8 GB)
* **Power Limits (validated):**
  * Default/Requested: 50.00 W
  * Max: 115.00 W
* **Architecture and bandwidth numbers:** Keep as vendor claims unless validated from NVIDIA technical documentation.
* **Integrated GPU (APU):** AMD Radeon 780M

## NPU: AMD XDNA (Ryzen AI)
* **Runtime Device:** RyzenAI-npu1 at BDF `0000:66:00.1`
* **Kernel Driver Module:** `amdxdna` loaded
* **XRT-reported amdxdna version:** 6.19.9-arch1-1
* **NPU Firmware Version:** 1.5.2.380
* **Architecture / TOPS figures:** Keep as vendor claims unless tied to an AMD official source in this document.
* **Memory:** Likely shares system DDR5 memory

## Validation Status Summary
* **Fully validated from live commands:** host model, OS/kernel, desktop session, CPU topology/cache, GPU model/driver/CUDA/VRAM/power limits, NPU presence/firmware/runtime visibility.
* **Partially validated / vendor-claim fields:** process node, architecture marketing labels, TOPS and theoretical bandwidth, NPU shares system DDR5 memory.
