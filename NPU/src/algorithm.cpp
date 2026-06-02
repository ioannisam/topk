#include "../include/algorithm.hpp"

#include <algorithm>
#include <iostream>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <string>
#include <stdexcept>
#include <vector>
#include <limits>

#include <xrt/xrt_bo.h>
#include <xrt/xrt_device.h>
#include <xrt/xrt_hw_context.h>
#include <xrt/xrt_kernel.h>
#include <xrt/experimental/xrt_kernel.h>
#include <xrt/experimental/xrt_xclbin.h>

namespace npu::bitonic {

namespace {

xrt::device open_device() {
    try {
        return xrt::device{0};
    } catch (const std::exception& ex) {
        throw std::runtime_error(std::string("Failed to open NPU device 0 via XRT: ") + ex.what());
    }
}

struct OffloadConfig {
    bool enabled = false;
    std::string xclbin_path;
    std::string kernel_name = "MLIR_AIE";
};

const char* read_env(const char* key) {
    const char* value = std::getenv(key);
    return (value == nullptr || value[0] == '\0') ? nullptr : value;
}

OffloadConfig load_offload_config() {
    OffloadConfig cfg;
    if (const char* xclbin = read_env("NPU_OFFLOAD_XCLBIN")) {
        cfg.enabled = true;
        cfg.xclbin_path = xclbin;
    }
    if (const char* kernel = read_env("NPU_OFFLOAD_KERNEL")) {
        cfg.kernel_name = kernel;
    }
    return cfg;
}

std::size_t safe_group_id(const xrt::kernel& kernel, int arg_index) {
    try { 
        std::size_t id = static_cast<std::size_t>(kernel.group_id(arg_index)); 
        if (id == 65535 || id == static_cast<std::size_t>(-1)) {
            return 0; 
        }
        return id;
    } 
    catch (...) { 
        return 0; 
    }
}

unsigned int read_wait_timeout_ms() {
    if (const char* timeout = read_env("NPU_OFFLOAD_WAIT_MS")) return static_cast<unsigned int>(std::stoul(timeout));
    return 5000u;
}

void wait_for_runlist_or_throw(const xrt::runlist& rl, unsigned int timeout_ms) {
    if (rl.wait(std::chrono::milliseconds(timeout_ms)) == std::cv_status::timeout) {
        throw std::runtime_error("NPU command timed out");
    }
}

std::vector<uint32_t> load_instruction_sequence(const std::string& xclbin_path) {
    auto slash_idx = xclbin_path.find_last_of('/');
    std::string dir = (slash_idx != std::string::npos) ? xclbin_path.substr(0, slash_idx) : ".";
    std::string file_name = (slash_idx != std::string::npos) ? xclbin_path.substr(slash_idx + 1) : xclbin_path;
    
    auto dot_idx = file_name.find_last_of('.');
    std::string base_name = (dot_idx != std::string::npos) ? file_name.substr(0, dot_idx) : file_name;

    std::string path = dir + "/" + base_name + ".bin";
    
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    if (!file) {
        throw std::runtime_error("Could not find NPU instruction sequence at: " + path);
    }
    
    std::streamsize size = file.tellg();
    if (size <= 0) {
        throw std::runtime_error("Instruction file is empty (0 bytes): " + path);
    }
    file.seekg(0, std::ios::beg);

    std::vector<uint32_t> insts(size / sizeof(uint32_t));
    file.read(reinterpret_cast<char*>(insts.data()), size);

    return insts;
}

template <typename T>
RunStats run_network_offload_xrt(std::vector<T>& data, const std::vector<common::bitonic::Layer>& layers,
                                 const OffloadConfig& offload_cfg) {
    const std::size_t n = data.size();
    
    std::size_t padded_n = 1024;
    while (padded_n < n) {
        padded_n *= 2;
    }

    std::vector<T> padded_data(padded_n, std::numeric_limits<T>::lowest());
    std::memcpy(padded_data.data(), data.data(), n * sizeof(T));

    xrt::device dev = open_device();
    xrt::xclbin xclbin(offload_cfg.xclbin_path);
    xrt::uuid uuid = dev.register_xclbin(xclbin);
    xrt::hw_context hwctx(dev, uuid);
    xrt::kernel kernel(hwctx, offload_cfg.kernel_name);

    const std::size_t chunk_bytes = 1024 * sizeof(T);
    
    const std::size_t instr_group = safe_group_id(kernel, 1);
    const std::size_t cfg_group   = safe_group_id(kernel, 3);
    const std::size_t dst_group   = safe_group_id(kernel, 4);
    const std::size_t src_group   = safe_group_id(kernel, 5);

    std::vector<uint32_t> instr_v = load_instruction_sequence(offload_cfg.xclbin_path);
    xrt::bo instr_bo(dev, instr_v.size() * sizeof(uint32_t), xrt::bo::flags::cacheable, instr_group);
    std::memcpy(instr_bo.map<void*>(), instr_v.data(), instr_v.size() * sizeof(uint32_t));
    instr_bo.sync(XCL_BO_SYNC_BO_TO_DEVICE);

    xrt::bo cfg_bo(dev, 8 * sizeof(int32_t), xrt::bo::flags::host_only, cfg_group); 
    xrt::bo dst_bo(dev, chunk_bytes, xrt::bo::flags::host_only, dst_group);
    xrt::bo src_bo(dev, chunk_bytes, xrt::bo::flags::host_only, src_group);

    int32_t pad_val = static_cast<int32_t>(std::numeric_limits<T>::lowest());
    std::size_t total_dispatches = 0;

    auto t0 = std::chrono::high_resolution_clock::now();

    if (padded_n <= 1024) {
        std::memcpy(src_bo.map<void*>(), padded_data.data(), chunk_bytes);
        src_bo.sync(XCL_BO_SYNC_BO_TO_DEVICE);

        xrt::bo current_src = src_bo;
        xrt::bo current_dst = dst_bo;

        for (const auto& layer : layers) {
            int32_t* cfg_map = cfg_bo.map<int32_t*>();
            cfg_map[0] = static_cast<int32_t>(layer.j);
            cfg_map[1] = static_cast<int32_t>(layer.k);
            cfg_map[2] = static_cast<int32_t>(layer.type == common::bitonic::LayerType::Normal ? 0 : 1);
            cfg_map[3] = 0; 
            cfg_map[4] = pad_val;
            cfg_bo.sync(XCL_BO_SYNC_BO_TO_DEVICE);

            xrt::run run(kernel);
            run.set_arg(0, 3);
            run.set_arg(1, instr_bo);
            run.set_arg(2, static_cast<uint32_t>(instr_v.size()));
            run.set_arg(3, cfg_bo);
            run.set_arg(4, current_dst);
            run.set_arg(5, current_src);
            
            xrt::runlist rl(hwctx);
            rl.add(run);
            rl.execute();
            wait_for_runlist_or_throw(rl, read_wait_timeout_ms());
            total_dispatches++;

            current_dst.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
            std::memcpy(current_src.map<void*>(), current_dst.map<void*>(), chunk_bytes);
            current_src.sync(XCL_BO_SYNC_BO_TO_DEVICE);
        }
        current_src.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
        std::memcpy(padded_data.data(), current_src.map<void*>(), chunk_bytes);
    } 
    else {
        const std::size_t chunk_size = 1024;
        const std::size_t half_chunk = 512;
        T type_pad_val = std::numeric_limits<T>::lowest();

        for (const auto& layer : layers) {
            bool is_trunc = (layer.type == common::bitonic::LayerType::Truncate);
            std::vector<T> next_data;
            if (is_trunc) {
                next_data.assign(padded_n, type_pad_val);
            }
            
            if (layer.j < chunk_size) {
                for (std::size_t offset = 0; offset < padded_n; offset += chunk_size) {
                    std::memcpy(src_bo.map<void*>(), padded_data.data() + offset, chunk_size * sizeof(T));
                    src_bo.sync(XCL_BO_SYNC_BO_TO_DEVICE);

                    int32_t* cfg_map = cfg_bo.map<int32_t*>();
                    cfg_map[0] = static_cast<int32_t>(layer.j);
                    cfg_map[1] = static_cast<int32_t>(layer.k);
                    cfg_map[2] = is_trunc ? 1 : 0;
                    cfg_map[3] = static_cast<int32_t>(offset); 
                    cfg_map[4] = pad_val;
                    cfg_bo.sync(XCL_BO_SYNC_BO_TO_DEVICE);

                    xrt::run run(kernel);
                    run.set_arg(0, 3); run.set_arg(1, instr_bo);
                    run.set_arg(2, static_cast<uint32_t>(instr_v.size()));
                    run.set_arg(3, cfg_bo); run.set_arg(4, dst_bo); run.set_arg(5, src_bo);
                    
                    xrt::runlist rl(hwctx); rl.add(run); rl.execute();
                    wait_for_runlist_or_throw(rl, read_wait_timeout_ms());
                    total_dispatches++;

                    dst_bo.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
                    
                    if (is_trunc) {
                        std::memcpy(next_data.data() + (offset / 2), dst_bo.map<void*>(), half_chunk * sizeof(T));
                    } else {
                        std::memcpy(padded_data.data() + offset, dst_bo.map<void*>(), chunk_size * sizeof(T));
                    }
                }
            } 
            else {
                for (std::size_t i = 0; i < padded_n; i += (2 * layer.j)) {
                    for (std::size_t offset = 0; offset < layer.j; offset += half_chunk) {
                        std::size_t left_idx = i + offset;
                        std::size_t right_idx = i + layer.j + offset;

                        T* src_ptr = src_bo.map<T*>();
                        std::memcpy(src_ptr, padded_data.data() + left_idx, half_chunk * sizeof(T));
                        std::memcpy(src_ptr + half_chunk, padded_data.data() + right_idx, half_chunk * sizeof(T));
                        src_bo.sync(XCL_BO_SYNC_BO_TO_DEVICE);

                        int32_t* cfg_map = cfg_bo.map<int32_t*>();
                        cfg_map[0] = static_cast<int32_t>(layer.j);
                        cfg_map[1] = static_cast<int32_t>(layer.k);
                        cfg_map[2] = is_trunc ? 3 : 2;
                        cfg_map[3] = static_cast<int32_t>(left_idx); 
                        cfg_map[4] = pad_val;
                        cfg_bo.sync(XCL_BO_SYNC_BO_TO_DEVICE);

                        xrt::run run(kernel);
                        run.set_arg(0, 3); run.set_arg(1, instr_bo);
                        run.set_arg(2, static_cast<uint32_t>(instr_v.size()));
                        run.set_arg(3, cfg_bo); run.set_arg(4, dst_bo); run.set_arg(5, src_bo);
                        
                        xrt::runlist rl(hwctx); rl.add(run); rl.execute();
                        wait_for_runlist_or_throw(rl, read_wait_timeout_ms());
                        total_dispatches++;

                        dst_bo.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
                        T* dst_ptr = dst_bo.map<T*>();
                        
                        if (is_trunc) {
                            std::size_t out_base = (i / 2) + offset;
                            std::memcpy(next_data.data() + out_base, dst_ptr, half_chunk * sizeof(T));
                        } else {
                            std::memcpy(padded_data.data() + left_idx, dst_ptr, half_chunk * sizeof(T));
                            std::memcpy(padded_data.data() + right_idx, dst_ptr + half_chunk, half_chunk * sizeof(T));
                        }
                    }
                }
            }
            
            if (is_trunc) {
                padded_data = std::move(next_data);
            }
        }
    }

    auto t1 = std::chrono::high_resolution_clock::now();
    std::memcpy(data.data(), padded_data.data(), n * sizeof(T));

    return RunStats{std::chrono::duration<double, std::milli>(t1 - t0).count(), total_dispatches, 0, 1, true};
}

} // namespace

std::string query_device_name() {
    xrt::device dev = open_device();
    return dev.get_info<xrt::info::device::name>();
}

std::string query_device_bdf() {
    xrt::device dev = open_device();
    return dev.get_info<xrt::info::device::bdf>();
}

bool is_offload_configured() {
    return load_offload_config().enabled;
}

template <typename T>
RunStats run_network_npu(std::vector<T>& data, const std::vector<common::bitonic::Layer>& layers, std::size_t workers) {
    (void)workers;
    if (data.empty()) return RunStats{0.0, 0, 0, 1, true};

    (void)open_device();
    const OffloadConfig offload_cfg = load_offload_config();
    if (!offload_cfg.enabled) {
        throw std::runtime_error("NPU offload is required for this backend. Set NPU_OFFLOAD_XCLBIN to a valid xclbin path.");
    }
    return run_network_offload_xrt(data, layers, offload_cfg);
}

template RunStats run_network_npu<std::int32_t>(std::vector<std::int32_t>& data, const std::vector<common::bitonic::Layer>& layers, std::size_t workers);
template RunStats run_network_npu<std::uint32_t>(std::vector<std::uint32_t>& data, const std::vector<common::bitonic::Layer>& layers, std::size_t workers);
template RunStats run_network_npu<float>(std::vector<float>& data, const std::vector<common::bitonic::Layer>& layers, std::size_t workers);
template RunStats run_network_npu<double>(std::vector<double>& data, const std::vector<common::bitonic::Layer>& layers, std::size_t workers);

#if defined(__FLT16_MANT_DIG__)
template RunStats run_network_npu<_Float16>(std::vector<_Float16>& data, const std::vector<common::bitonic::Layer>& layers, std::size_t workers);
#endif

} // namespace npu::bitonic

namespace npu::map_reduce {

namespace {

template <typename T>
std::vector<T> run_map_reduce_offload_xrt(const std::vector<T>& data, std::size_t k, bool want_max, 
                                          const npu::bitonic::OffloadConfig& offload_cfg, 
                                          npu::bitonic::RunStats* stats) {
    if (k == 0 || data.empty()) return {};

    const std::size_t n = data.size();
    k = std::min(k, n);

    auto t0 = std::chrono::high_resolution_clock::now();

    using HeapCompare = std::function<bool(const T&, const T&)>;
    HeapCompare comp;
    if (want_max) {
        comp = std::greater<T>{};
    } else {
        comp = std::less<T>{};
    }

    std::vector<T> heap;
    heap.reserve(k);

    std::size_t initial_elements = std::min(k, n);
    heap.assign(data.begin(), data.begin() + initial_elements);
    std::make_heap(heap.begin(), heap.end(), comp);

    xrt::device dev = npu::bitonic::open_device();
    xrt::xclbin xclbin(offload_cfg.xclbin_path);
    xrt::uuid uuid = dev.register_xclbin(xclbin);
    xrt::hw_context hwctx(dev, uuid);
    xrt::kernel kernel(hwctx, offload_cfg.kernel_name);

    const std::size_t instr_group = npu::bitonic::safe_group_id(kernel, 1);
    std::vector<uint32_t> instr_v = npu::bitonic::load_instruction_sequence(offload_cfg.xclbin_path);
    xrt::bo instr_bo(dev, instr_v.size() * sizeof(uint32_t), xrt::bo::flags::cacheable, instr_group);
    std::memcpy(instr_bo.map<void*>(), instr_v.data(), instr_v.size() * sizeof(uint32_t));
    instr_bo.sync(XCL_BO_SYNC_BO_TO_DEVICE);

    const std::size_t chunk_size = 1024;
    const std::size_t chunk_bytes = chunk_size * sizeof(T);
    
    const std::size_t cfg_group = npu::bitonic::safe_group_id(kernel, 3);
    const std::size_t dst_group = npu::bitonic::safe_group_id(kernel, 4);
    const std::size_t src_group = npu::bitonic::safe_group_id(kernel, 5);

    xrt::bo cfg_bo(dev, 4 * sizeof(int32_t), xrt::bo::flags::host_only, cfg_group); 
    xrt::bo dst_bo(dev, chunk_bytes, xrt::bo::flags::host_only, dst_group);
    xrt::bo src_bo(dev, chunk_bytes, xrt::bo::flags::host_only, src_group);

    T pad_val = want_max ? std::numeric_limits<T>::lowest() : std::numeric_limits<T>::max();
    std::size_t total_dispatches = 0;

    for (std::size_t offset = initial_elements; offset < n; offset += chunk_size) {
        std::size_t current_chunk = std::min(chunk_size, n - offset);
        
        T* src_map = src_bo.map<T*>();
        std::memcpy(src_map, data.data() + offset, current_chunk * sizeof(T));
        if (current_chunk < chunk_size) {
             for(std::size_t p = current_chunk; p < chunk_size; ++p) src_map[p] = pad_val;
        }
        src_bo.sync(XCL_BO_SYNC_BO_TO_DEVICE);

        int32_t* cfg_map = cfg_bo.map<int32_t*>();
        cfg_map[0] = static_cast<int32_t>(heap.front()); 
        cfg_map[1] = want_max ? 1 : 0;
        cfg_bo.sync(XCL_BO_SYNC_BO_TO_DEVICE);

        xrt::run run(kernel);
        run.set_arg(0, 3);
        run.set_arg(1, instr_bo);
        run.set_arg(2, static_cast<uint32_t>(instr_v.size()));
        run.set_arg(3, cfg_bo);
        run.set_arg(4, dst_bo);
        run.set_arg(5, src_bo);
        
        xrt::runlist rl(hwctx);
        rl.add(run);
        rl.execute();
        npu::bitonic::wait_for_runlist_or_throw(rl, npu::bitonic::read_wait_timeout_ms());
        total_dispatches++;

        dst_bo.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
        T* dst_map = dst_bo.map<T*>();
        
        for (std::size_t i = 0; i < current_chunk; ++i) {
            T candidate = dst_map[i];
            if (candidate != pad_val) {
                // --- FIX 3: Re-verify against CURRENT threshold ---
                bool is_still_candidate = want_max ? (candidate > heap.front()) : (candidate < heap.front());
                
                if (is_still_candidate) {
                    std::pop_heap(heap.begin(), heap.end(), comp);
                    heap.back() = candidate;
                    std::push_heap(heap.begin(), heap.end(), comp);
                }
            }
        }
    }

    auto cmp_sort = [&want_max](const T& lhs, const T& rhs) {
        return want_max ? (lhs > rhs) : (lhs < rhs);
    };
    std::sort(heap.begin(), heap.end(), cmp_sort);

    auto t1 = std::chrono::high_resolution_clock::now();

    if (stats != nullptr) {
        stats->elapsed_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        stats->layer_dispatches = total_dispatches;
        stats->used_offload = true;
    }

    return heap;
}

} // namespace

template <typename T>
std::vector<T> run_topk_npu(const std::vector<T>& data, std::size_t k, bool want_max, std::size_t workers, npu::bitonic::RunStats* stats) {
    (void)workers; // Multi-threading not currently applied to the NPU dispatch loop
    
    npu::bitonic::OffloadConfig offload_cfg = npu::bitonic::load_offload_config();
    if (!offload_cfg.enabled) {
        throw std::runtime_error("NPU offload is required. Set NPU_OFFLOAD_XCLBIN to map_reduce.xclbin.");
    }
    
    return run_map_reduce_offload_xrt(data, k, want_max, offload_cfg, stats);
}

// Explicit template instantiations
template std::vector<std::int32_t> run_topk_npu<std::int32_t>(const std::vector<std::int32_t>& data, std::size_t k, bool want_max, std::size_t workers, npu::bitonic::RunStats* stats);
template std::vector<std::uint32_t> run_topk_npu<std::uint32_t>(const std::vector<std::uint32_t>& data, std::size_t k, bool want_max, std::size_t workers, npu::bitonic::RunStats* stats);
template std::vector<float> run_topk_npu<float>(const std::vector<float>& data, std::size_t k, bool want_max, std::size_t workers, npu::bitonic::RunStats* stats);
template std::vector<double> run_topk_npu<double>(const std::vector<double>& data, std::size_t k, bool want_max, std::size_t workers, npu::bitonic::RunStats* stats);

#if defined(__FLT16_MANT_DIG__)
template std::vector<_Float16> run_topk_npu<_Float16>(const std::vector<_Float16>& data, std::size_t k, bool want_max, std::size_t workers, npu::bitonic::RunStats* stats);
#endif

} // namespace npu::map_reduce
