# Quickstart: How to Run Each Backend

This file gives the shortest path to build and run for CPU, GPU and NPU backends using CMake + Ninja.

Repository root used below:
`topk/`

Prerequisites:
- CMake and Ninja installed
- Backend toolchains installed (see setup guide)

## CPU Backend

Path:
CPU/smoke_test

Build and run:
```
cd CPU/smoke_test
cmake --fresh -S . -B build -G Ninja
cmake --build build
./build/smoke
```

Expected success line:
CPU top-k smoke test: PASS

## GPU Backend

Path:
GPU/smoke_test

Optional pre-check:
```
command -v nvcc
nvidia-smi
```

Build and run:
```
cd GPU/smoke_test
cmake --fresh -S . -B build -G Ninja
cmake --build build
./build/smoke
```

Expected success line:
CUDA top-k smoke test: PASS

## NPU Backend (AMD XDNA)

Path:
NPU/smoke_test

Optional pre-check:
```
source ~/.bash/exports.sh
xrt-smi examine
```

Build and run:
```
cd NPU/smoke_test
cmake --fresh -S . -B build -G Ninja
cmake --build build
./build/smoke
```

Expected success line:
NPU top-k smoke test: PASS

## Rebuild Quickly After Code Changes

After editing source files, rebuild only:
```
cmake --build build
```

Run the backend executable again from the same folder.

## If CMake Cache Points to an Old Path

If you see a source/cache mismatch error, clean and reconfigure:

```
rm -rf build
cmake -S . -B build -G Ninja
cmake --build build
```
