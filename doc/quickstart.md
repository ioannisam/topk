# Quickstart: How to Run Each Backend

This file gives the shortest path to build and run for CPU, GPU and NPU backends using CMake + Ninja.

Repository root used below:
`topk/`

Prerequisites:
- CMake and Ninja installed
- Backend toolchains installed (see setup guide)

## CPU Backend

Path:
test/smoke/CPU

Build and run:
```
cd test/smoke/CPU
cmake --fresh -S . -B build -G Ninja
cmake --build build
./build/smoke
```

Expected success line:
CPU top-k smoke test: PASS

## GPU Backend

Path:
test/smoke/GPU

Optional pre-check:
```
command -v nvcc
nvidia-smi
```

Build and run:
```
cd test/smoke/GPU
cmake --fresh -S . -B build -G Ninja
cmake --build build
./build/smoke
```

Expected success line:
CUDA top-k smoke test: PASS

## NPU Backend (AMD XDNA)

Path:
test/smoke/NPU

Optional pre-check:
```
source ~/.bash/exports.sh
xrt-smi examine
```

Build and run:
```
cd test/smoke/NPU
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
