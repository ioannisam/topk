# Validation Guide

This document describes how to validate the toolchain and runtime stack after setup.

Prerequisites:
- All dependencies installed (see setup guide)
- NPU runtime configured (render/video groups, memlock limits, XRT path)

## 1. Runtime Sanity Checks

CPU and thermal tooling:

```bash
command -v cpupower
command -v sensors
```

GPU tooling:

```bash
command -v nvcc
nvcc --version
nvidia-smi
```

NPU runtime visibility:

```bash
xrt-smi examine --batch
```

Expected: RyzenAI-npu1 appears at BDF 0000:66:00.1.

## 2. MLIR-AIE Official Examples

Clone the official compiler repository:

```bash
git clone https://github.com/Xilinx/mlir-aie.git
cd mlir-aie
```

Activate your Python environment and make sure the compiler wheels are installed:

```bash
source /path/to/npu_venv/bin/activate
python3 -m pip list | grep -E 'mlir-aie|llvm-aie'
```

If you are on Arch Linux and building from source, ensure the compiler symlinks are in PATH:

```bash
export PATH="$HOME/.local/bin:$PATH"
```

Set up the MLIR-AIE environment (run from the repo root):

```bash
source utils/env_setup.sh
```

### Test 1: Passthrough Kernel (Basic Compilation)

```bash
cd programming_examples/basic/passthrough_kernel
make clean
make TARGET=strix
make run TARGET=strix
```

### Test 2: Matrix Multiplication (DMA Transfers)

```bash
cd ../matrix_multiplication
make clean
make TARGET=strix
make run TARGET=strix
```

Expected: both examples build the .xclbin, run on the NPU, and report PASS with timing output.

## 3. Project Binary Checks

Build all backends:

```bash
cd /path/to/Thesis
make build-all
```

Verify that binaries were generated:

```bash
ls -la build/CPU
ls -la build/GPU
ls -la build/NPU
```

Expected: a topk binary is present in each backend directory.
