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
#include <memory>

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

struct SharedXrtState {
    xrt::device dev;
    xrt::xclbin xclbin;
    xrt::uuid uuid;
    xrt::hw_context hwctx;
    xrt::kernel kernel;
    std::vector<uint32_t> instr_v;
    xrt::bo instr_bo;

    explicit SharedXrtState(const OffloadConfig& cfg) {
        dev = open_device();
        xclbin = xrt::xclbin(cfg.xclbin_path);
        uuid = dev.register_xclbin(xclbin);
        hwctx = xrt::hw_context(dev, uuid);
        kernel = xrt::kernel(hwctx, cfg.kernel_name);

        const std::size_t instr_group = safe_group_id(kernel, 1);
        instr_v = load_instruction_sequence(cfg.xclbin_path);
        instr_bo = xrt::bo(dev, instr_v.size() * sizeof(uint32_t), xrt::bo::flags::cacheable, instr_group);
        std::memcpy(instr_bo.map<void*>(), instr_v.data(), instr_v.size() * sizeof(uint32_t));
        instr_bo.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    }
};

SharedXrtState& get_shared_xrt_state(const OffloadConfig& offload_cfg) {
    static std::unique_ptr<SharedXrtState> instance = std::make_unique<SharedXrtState>(offload_cfg);
    return *instance;
}

template <typename T>
RunStats run_network_offload_xrt(std::vector<T>& data, const std::vector<common::bitonic::Layer>& layers,
                                 const OffloadConfig& offload_cfg) {
    const std::size_t n = data.size();
    
    std::size_t padded_n = n;
    if (n % 1024 != 0) {
        padded_n = ((n / 1024) + 1) * 1024;
    }

    std::vector<T> padded_data(padded_n, std::numeric_limits<T>::max());
    std::memcpy(padded_data.data(), data.data(), n * sizeof(T));

    // Fetch the hot XRT state instead of initializing from scratch
    SharedXrtState& state = get_shared_xrt_state(offload_cfg);

    const std::size_t chunk_bytes = 1024 * sizeof(T);
    
    xrt::bo src_bo(state.dev, padded_n * sizeof(T), xrt::bo::flags::host_only, safe_group_id(state.kernel, 3));
    xrt::bo dst_bo(state.dev, padded_n * sizeof(T), xrt::bo::flags::host_only, safe_group_id(state.kernel, 4));

    std::size_t total_dispatches = 0;
    auto t0 = std::chrono::high_resolution_clock::now();

    for (std::size_t offset = 0; offset < padded_n; offset += 1024) {
        std::memcpy(src_bo.map<void*>(), padded_data.data() + offset, chunk_bytes);
        src_bo.sync(XCL_BO_SYNC_BO_TO_DEVICE);

        xrt::run run(state.kernel);
        run.set_arg(0, 3);          
        run.set_arg(1, state.instr_bo);   
        run.set_arg(2, static_cast<uint32_t>(state.instr_v.size()));
        run.set_arg(3, src_bo);     
        run.set_arg(4, dst_bo);     
        
        xrt::runlist rl(state.hwctx);
        rl.add(run);
        rl.execute();
        wait_for_runlist_or_throw(rl, read_wait_timeout_ms());
        total_dispatches++;

        dst_bo.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
        std::memcpy(padded_data.data() + offset, dst_bo.map<void*>(), chunk_bytes);
        
        if ((offset / 1024) % 2 != 0) {
            std::reverse(padded_data.begin() + offset, padded_data.begin() + offset + 1024);
        }
    }

    for (std::size_t k = 2048; k <= padded_n; k *= 2) {
        for (std::size_t j = k / 2; j > 0; j /= 2) {
            for (std::size_t i = 0; i < padded_n; i++) {
                std::size_t ixj = i ^ j;
                if (ixj > i) {
                    bool ascending = ((i & k) == 0);
                    
                    if (ascending) {
                        if (padded_data[i] > padded_data[ixj]) {
                            std::swap(padded_data[i], padded_data[ixj]);
                        }
                    } else {
                        if (padded_data[i] < padded_data[ixj]) {
                            std::swap(padded_data[i], padded_data[ixj]);
                        }
                    }
                }
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
void prepare_npu_batch(xrt::bo& src_bo, xrt::bo& cfg_bo, const T* data_ptr,
                       std::size_t current_batch, std::size_t batch_size,
                       std::size_t batch_chunks, T current_threshold,
                       bool want_max, T pad_val) {
    T* src_map = src_bo.map<T*>();
    std::memcpy(src_map, data_ptr, current_batch * sizeof(T));
    if (current_batch < batch_size) {
        for (std::size_t p = current_batch; p < batch_size; ++p) src_map[p] = pad_val;
    }
    src_bo.sync(XCL_BO_SYNC_BO_TO_DEVICE);

    int32_t* cfg_map = cfg_bo.map<int32_t*>();
    for (std::size_t c = 0; c < batch_chunks; ++c) {
        cfg_map[c * 4 + 0] = static_cast<int32_t>(current_threshold);
        cfg_map[c * 4 + 1] = want_max ? 1 : 0;
        cfg_map[c * 4 + 2] = 0;
        cfg_map[c * 4 + 3] = 0;
    }
    cfg_bo.sync(XCL_BO_SYNC_BO_TO_DEVICE);
}

template <typename T, typename Compare>
void process_npu_results(T* dst_map, std::size_t batch_chunks, std::size_t chunk_size,
                         std::vector<T>& heap, Compare comp, bool want_max) {
    for (std::size_t c = 0; c < batch_chunks; ++c) {
        T* chunk_out = dst_map + (c * chunk_size);
        int32_t valid_count = *reinterpret_cast<int32_t*>(&chunk_out[0]);
        for (int i = 1; i <= valid_count; ++i) {
            T candidate = chunk_out[i];
            bool is_still_candidate = want_max ? (candidate > heap.front()) : (candidate < heap.front());
            if (is_still_candidate) {
                std::pop_heap(heap.begin(), heap.end(), comp);
                heap.back() = candidate;
                std::push_heap(heap.begin(), heap.end(), comp);
            }
        }
    }
}

template <typename T>
std::vector<T> run_map_reduce_offload_xrt(const std::vector<T>& data, std::size_t k, bool want_max,
                                          const npu::bitonic::OffloadConfig& offload_cfg,
                                          npu::bitonic::RunStats* stats) {
    if (k == 0 || data.empty()) return {};

    const std::size_t n = data.size();
    k = std::min(k, n);

    // Fetch the shared state (Initialized safely in runner's warmup block)
    npu::bitonic::SharedXrtState& state = npu::bitonic::get_shared_xrt_state(offload_cfg);
    
    auto t0 = std::chrono::high_resolution_clock::now();

    using HeapCompare = std::function<bool(const T&, const T&)>;
    HeapCompare comp = want_max ? HeapCompare(std::greater<T>{}) : HeapCompare(std::less<T>{});

    std::vector<T> heap;
    heap.reserve(k);
    std::size_t initial_elements = std::min(k, n);
    heap.assign(data.begin(), data.begin() + initial_elements);
    std::make_heap(heap.begin(), heap.end(), comp);

    const std::size_t BATCH_CHUNKS = 256;
    const std::size_t chunk_size = 1024;
    const std::size_t batch_size = BATCH_CHUNKS * chunk_size;
    const std::size_t batch_bytes = batch_size * sizeof(T);
    
    xrt::bo cfg_bo[2], dst_bo[2], src_bo[2];
    xrt::run run[2];
    xrt::runlist rl[2] = { xrt::runlist(state.hwctx), xrt::runlist(state.hwctx) };

    const std::size_t cfg_grp = npu::bitonic::safe_group_id(state.kernel, 3);
    const std::size_t dst_grp = npu::bitonic::safe_group_id(state.kernel, 4);
    const std::size_t src_grp = npu::bitonic::safe_group_id(state.kernel, 5);

    for (int i = 0; i < 2; ++i) {
        cfg_bo[i] = xrt::bo(state.dev, BATCH_CHUNKS * 4 * sizeof(int32_t), xrt::bo::flags::host_only, cfg_grp);
        dst_bo[i] = xrt::bo(state.dev, batch_bytes, xrt::bo::flags::host_only, dst_grp);
        src_bo[i] = xrt::bo(state.dev, batch_bytes, xrt::bo::flags::host_only, src_grp);

        run[i] = xrt::run(state.kernel);
        run[i].set_arg(0, 3);
        run[i].set_arg(1, state.instr_bo);
        run[i].set_arg(2, static_cast<uint32_t>(state.instr_v.size()));
        run[i].set_arg(3, cfg_bo[i]);
        run[i].set_arg(4, dst_bo[i]);
        run[i].set_arg(5, src_bo[i]);
        rl[i].add(run[i]);
    }

    T pad_val = want_max ? std::numeric_limits<T>::lowest() : std::numeric_limits<T>::max();
    std::size_t total_dispatches = 0;

    int active_idx = 0;
    int next_idx = 1;
    bool run_pending = false;

    for (std::size_t offset = initial_elements; offset < n; offset += batch_size) {
        std::size_t current_batch = std::min(batch_size, n - offset);

        // Prepare and dispatch
        prepare_npu_batch<T>(src_bo[active_idx], cfg_bo[active_idx], data.data() + offset,
                             current_batch, batch_size, BATCH_CHUNKS, heap.front(), want_max, pad_val);

        rl[active_idx].execute();
        total_dispatches++;

        if (run_pending) {
            npu::bitonic::wait_for_runlist_or_throw(rl[next_idx], npu::bitonic::read_wait_timeout_ms());
            dst_bo[next_idx].sync(XCL_BO_SYNC_BO_FROM_DEVICE);
            process_npu_results<T>(dst_bo[next_idx].map<T*>(), BATCH_CHUNKS, chunk_size, heap, comp, want_max);
        }

        run_pending = true;
        std::swap(active_idx, next_idx);
    }

    // Drain last pending run
    if (run_pending) {
        int pending_idx = active_idx ^ 1;
        npu::bitonic::wait_for_runlist_or_throw(rl[pending_idx], npu::bitonic::read_wait_timeout_ms());
        dst_bo[pending_idx].sync(XCL_BO_SYNC_BO_FROM_DEVICE);
        process_npu_results<T>(dst_bo[pending_idx].map<T*>(), BATCH_CHUNKS, chunk_size, heap, comp, want_max);
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
    (void)workers;
    
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
