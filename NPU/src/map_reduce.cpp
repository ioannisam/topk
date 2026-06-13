#include "../include/algorithm.hpp"
#include "xrt_utils.hpp"

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <functional>
#include <limits>
#include <stdexcept>
#include <vector>

namespace npu::map_reduce {
namespace {

template <typename T>
void prepare_npu_batch(xrt::bo& src_bo, xrt::bo& cfg_bo, const T* data_ptr,
                       std::size_t current_batch, std::size_t batch_size,
                       std::size_t batch_chunks, T current_threshold,
                       bool want_max, T pad_val) {
    
    std::size_t col_capacity = batch_size / 4;
    std::size_t chunks_per_col = batch_chunks / 4;

    int32_t sentinel_bits;
    T temp_pad = pad_val;
    std::memcpy(&sentinel_bits, &temp_pad, std::min(sizeof(T), sizeof(int32_t)));

    T* src_map = src_bo.map<T*>();
    int32_t* cfg_map = cfg_bo.map<int32_t*>();

    for (int c = 0; c < 4; ++c) {
        std::size_t offset = c * col_capacity;
        std::size_t elements_to_copy = (current_batch > offset) ? std::min(col_capacity, current_batch - offset) : 0;

        if (elements_to_copy > 0) {
            std::memcpy(src_map + offset, data_ptr + offset, elements_to_copy * sizeof(T));
        }
        if (elements_to_copy < col_capacity) {
            std::fill(src_map + offset + elements_to_copy, src_map + offset + col_capacity, pad_val);
        }

        std::size_t cfg_offset = c * chunks_per_col * 4;
        for (std::size_t ch = 0; ch < chunks_per_col; ++ch) {
            cfg_map[cfg_offset + ch * 4 + 0] = static_cast<int32_t>(current_threshold);
            cfg_map[cfg_offset + ch * 4 + 1] = want_max ? 1 : 0;
            cfg_map[cfg_offset + ch * 4 + 2] = sentinel_bits;
            cfg_map[cfg_offset + ch * 4 + 3] = 0;
        }
    }
    src_bo.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    cfg_bo.sync(XCL_BO_SYNC_BO_TO_DEVICE);
}

template <typename T, typename Compare>
void process_npu_results(xrt::bo& dst_bo, std::vector<T>& heap, std::size_t k, Compare comp, 
                         std::size_t batch_chunks) {
    T* dst_map = dst_bo.map<T*>();
    
    std::vector<T> local_candidates;
    local_candidates.reserve(8192); 
    
    const std::size_t chunk_size = 1024;
    
    for (std::size_t c = 0; c < batch_chunks; ++c) {
        T* chunk_ptr = dst_map + (c * chunk_size);
        
        int32_t count = static_cast<int32_t>(chunk_ptr[0]);
        if (count > 0 && count <= 1023) {
            // Read exactly 'count' elements, starting from index 1
            for (int32_t i = 1; i <= count; ++i) {
                local_candidates.push_back(chunk_ptr[i]);
            }
        }
    }

    if (!local_candidates.empty()) {
        heap.insert(heap.end(), local_candidates.begin(), local_candidates.end());
        std::nth_element(heap.begin(), heap.begin() + k - 1, heap.end(), comp);
        heap.resize(k);
        std::make_heap(heap.begin(), heap.end(), comp);
    }
}

template <typename T>
std::vector<T> run_map_reduce_offload_xrt(const std::vector<T>& data, std::size_t k, bool want_max,
                                          const npu::utils::OffloadConfig& offload_cfg,
                                          npu::map_reduce::RunStats* stats) {
    if (k == 0 || data.empty()) return {};

    const std::size_t n = data.size();
    k = std::min(k, n);

    npu::utils::SharedXrtState& state = npu::utils::get_shared_xrt_state(offload_cfg);
    
    auto t0 = std::chrono::high_resolution_clock::now();

    using HeapCompare = std::function<bool(const T&, const T&)>;
    HeapCompare comp = want_max ? HeapCompare(std::greater<T>{}) : HeapCompare(std::less<T>{});

    std::vector<T> heap(data.begin(), data.begin() + k);
    std::make_heap(heap.begin(), heap.end(), comp);

    const std::size_t BATCH_CHUNKS = 1024; // 1024 chunks total
    const std::size_t chunk_size = 1024;   // 1024 elements per chunk
    const std::size_t batch_size = BATCH_CHUNKS * chunk_size;
    const std::size_t batch_bytes = batch_size * sizeof(T);
    
    state.allocate_mr_buffers(batch_bytes);
    
    xrt::run run[2];
    xrt::runlist rl[2] = { xrt::runlist(state.hwctx), xrt::runlist(state.hwctx) };

    for (int i = 0; i < 2; ++i) {
        run[i] = xrt::run(state.kernel);
        run[i].set_arg(0, 3);
        run[i].set_arg(1, state.instr_bo);
        run[i].set_arg(2, static_cast<uint32_t>(state.instr_v.size()));
        run[i].set_arg(3, state.mr_cfg_bo[i]);
        run[i].set_arg(4, state.mr_dst_bo[i]);
        run[i].set_arg(5, state.mr_src_bo[i]);
        
        rl[i].add(run[i]);
    }

    T pad_val = want_max ? std::numeric_limits<T>::lowest() : std::numeric_limits<T>::max();
    std::size_t total_dispatches = 0;

    int active_idx = 0;
    int next_idx = 1;
    bool run_pending = false;

    for (std::size_t offset = k; offset < n; offset += batch_size) {
        std::size_t current_batch = std::min(batch_size, n - offset);

        prepare_npu_batch<T>(state.mr_src_bo[active_idx], state.mr_cfg_bo[active_idx], data.data() + offset,
                             current_batch, batch_size, BATCH_CHUNKS, heap.front(), want_max, pad_val);

        rl[active_idx].execute();
        total_dispatches++;

        if (run_pending) {
            npu::utils::wait_for_runlist_or_throw(rl[next_idx], npu::utils::read_wait_timeout_ms());
            state.mr_dst_bo[next_idx].sync(XCL_BO_SYNC_BO_FROM_DEVICE); 
            process_npu_results<T>(state.mr_dst_bo[next_idx], heap, k, comp, BATCH_CHUNKS);
        }

        run_pending = true;
        std::swap(active_idx, next_idx);
    }

    if (run_pending) {
        int pending_idx = active_idx ^ 1;
        npu::utils::wait_for_runlist_or_throw(rl[pending_idx], npu::utils::read_wait_timeout_ms());
        state.mr_dst_bo[pending_idx].sync(XCL_BO_SYNC_BO_FROM_DEVICE); 
        process_npu_results<T>(state.mr_dst_bo[pending_idx], heap, k, comp, BATCH_CHUNKS);
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
std::vector<T> run_topk_npu(const std::vector<T>& data, std::size_t k, bool want_max, std::size_t workers, npu::map_reduce::RunStats* stats) {
    (void)workers;
    
    npu::utils::OffloadConfig offload_cfg = npu::utils::load_offload_config();
    if (!offload_cfg.enabled) {
        throw std::runtime_error("NPU offload is required. Set NPU_OFFLOAD_XCLBIN to map_reduce.xclbin.");
    }
    
    return run_map_reduce_offload_xrt(data, k, want_max, offload_cfg, stats);
}

template std::vector<std::int32_t> run_topk_npu<std::int32_t>(const std::vector<std::int32_t>& data, std::size_t k, bool want_max, std::size_t workers, npu::map_reduce::RunStats* stats);
template std::vector<std::uint32_t> run_topk_npu<std::uint32_t>(const std::vector<std::uint32_t>& data, std::size_t k, bool want_max, std::size_t workers, npu::map_reduce::RunStats* stats);
template std::vector<float> run_topk_npu<float>(const std::vector<float>& data, std::size_t k, bool want_max, std::size_t workers, npu::map_reduce::RunStats* stats);
template std::vector<double> run_topk_npu<double>(const std::vector<double>& data, std::size_t k, bool want_max, std::size_t workers, npu::map_reduce::RunStats* stats);

#if defined(__FLT16_MANT_DIG__)
template std::vector<_Float16> run_topk_npu<_Float16>(const std::vector<_Float16>& data, std::size_t k, bool want_max, std::size_t workers, npu::map_reduce::RunStats* stats);
#endif

} // namespace npu::map_reduce
