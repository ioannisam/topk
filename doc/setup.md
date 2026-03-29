# Setup Guide

This document covers tool and environment setup specific to the current machine's specifications.

Scope:
- CPU backend in C++
- GPU backend in CUDA
- NPU backend using AMD XDNA + XRT + MLIR-AIE

All commands are written for Arch Linux and this repository layout.

## Dependencies

### CPU

```bash
sudo pacman -S gcc cmake ninja linux-tools lm_sensors
```

Notes:
- `linux-tools` provides utilities such as `cpupower` for runtime stability controls.
- `lm_sensors` helps track package/core temperatures.

### GPU

```bash
sudo pacman -S gcc cmake ninja cuda
```

Verify:

```bash
command -v nvcc
nvcc --version
nvidia-smi
```

### NPU

```bash
sudo pacman -S xrt xrt-plugin-amdxdna clang lld cmake ninja jq boost
```

What these provide:
- `xrt` and `xrt-plugin-amdxdna`: runtime/user-space plugin stack for the NPU
- `clang`, `lld`, `cmake`, `ninja`: toolchain/build support
- `boost`: needed if building `aiebu` from source

## CPU Tools

Check available power/thermal tooling:

```bash
command -v cpupower
command -v sensors
ls -la /sys/class/powercap
find -L /sys/class/powercap -maxdepth 6 -name energy_uj 2>/dev/null
```

Notes:
- On some AMD laptops, the powercap path can still be named `intel-rapl`.
- Reading `energy_uj` may require elevated permissions.

## GPU Tools

Verify telemetry tooling:

```bash
nvidia-smi --query-gpu=index,name,driver_version --format=csv,noheader
nvidia-smi --query-gpu=power.draw,temperature.gpu,utilization.gpu --format=csv,noheader
```

Optional controls (experiment configuration dependent):

```bash
# Enable persistence mode (if supported)
sudo nvidia-smi -pm 1

# Optional fixed power limit
sudo nvidia-smi -pl 60
```

## NPU Tools

### Runtime Stack (Required for C++ XRT Use)

Set XRT path:

```bash
export XILINX_XRT=/usr
```

Recommended persistent config: add `export XILINX_XRT=/usr` to your shell startup file.

Add required groups and memlock limits:

```bash
sudo usermod -aG render,video $USER
sudo tee /etc/security/limits.d/99-amdxdna-npu.conf >/dev/null <<'EOF'
@render soft memlock unlimited
@render hard memlock unlimited
@video  soft memlock unlimited
@video  hard memlock unlimited
EOF
```

Important: log out and log back in (or reboot) to apply group and PAM limit changes.

Runtime readiness checks:

```bash
source ~/.bashrc
id
ulimit -l
xrt-smi examine --batch
```

Expected:
- `id` includes `render` and `video`
- `ulimit -l` is `unlimited` (or very large)
- `xrt-smi examine --batch` lists your RyzenAI NPU device

### Python Toolchain

If you are using only C++ + XRT runtime code, you do not need this section.

Create and activate environment:

```bash
cd /home/ioannis/Development/Thesis/NPU
python3 -m venv npu_env
source npu_env/bin/activate
python3 -m pip install --upgrade pip
```

Install latest `mlir_aie` + `llvm-aie`:

```bash
latest_tag_with_v=$(curl -s https://api.github.com/repos/Xilinx/mlir-aie/releases/latest | jq -r '.tag_name')
latest_tag="${latest_tag_with_v#v}"

python3 -m pip install mlir_aie==${latest_tag} -f https://github.com/Xilinx/mlir-aie/releases/expanded_assets/${latest_tag_with_v}
python3 -m pip install llvm-aie -f https://github.com/Xilinx/llvm-aie/releases/expanded_assets/nightly
```

Verify:

```bash
python3 -m pip list | grep -E 'mlir-aie|llvm-aie'
```

### Install aiebu Tools (If Missing)

```bash
git clone --recursive https://github.com/Xilinx/aiebu.git
cd aiebu
mkdir build && cd build
cmake .. -DCMAKE_INSTALL_PREFIX=/usr
cmake --build . -j"$(nproc)"
sudo install -m 755 ../src/cpp/utils/asm/aiebu-asm /usr/bin/aiebu-asm
sudo install -m 755 ../src/cpp/utils/transform/aiebu-transform /usr/bin/aiebu-transform
```

---

This setup gives you the required toolchain/runtime foundation for CPU, GPU and NPU development.
