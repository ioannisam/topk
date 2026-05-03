# NPU Runtime Smoke Test

This test validates that user-space code can open the AMD XDNA NPU through XRT.

## Build

```bash
cd /home/ioannis/Development/Thesis/test/smoke/NPU
cmake --fresh -S . -B build -G Ninja
cmake --build build
```

## Run

```bash
source /home/ioannis/.bash/exports.sh
NPU_OFFLOAD_XCLBIN=./build/smoke.xclbin ./build/smoke
```

If successful, it prints:

- `NPU smoke test: PASS`
- Device BDF/name information

If it fails, verify:

- `id` includes `render` and `video`
- `ulimit -l` is `unlimited`
- `xrt-smi examine` succeeds
