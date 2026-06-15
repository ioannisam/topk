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
    
    T* src_map = src_bo.map<T*>();
    int32_t* cfg_map = cfg_bo.map<int32_t*>();

    if (current_batch > 0) {
        std::memcpy(src_map, data_ptr, current_batch * sizeof(T));
    }

    struct alignas(16) CfgWord {
        int32_t threshold;
        int32_t want_max;
        int32_t sentinel;
        int32_t padding;
    };

    int32_t sentinel_bits;
    T temp_pad = pad_val;
    std::memcpy(&sentinel_bits, &temp_pad, std::min(sizeof(T), sizeof(int32_t)));

    CfgWord current_cfg = {
        static_cast<int32_t>(current_threshold),
        want_max ? 1 : 0,
        sentinel_bits,
        0
    };

    CfgWord* cfg_words = reinterpret_cast<CfgWord*>(cfg_map);
    
    const std::size_t chunk_size = 1024;
    std::size_t full_chunks = current_batch / chunk_size;
    std::size_t remainder = current_batch % chunk_size;

    std::fill(cfg_words, cfg_words + full_chunks, current_cfg);

    if (remainder > 0) {
        std::fill(src_map + (full_chunks * chunk_size) + remainder, 
                  src_map + ((full_chunks + 1) * chunk_size), pad_val);
        cfg_words[full_chunks] = current_cfg;
        full_chunks++;
    }

    if (full_chunks < batch_chunks) {
        CfgWord dummy_cfg = current_cfg;
        dummy_cfg.threshold = want_max ? std::numeric_limits<int32_t>::max() : std::numeric_limits<int32_t>::lowest();
        std::fill(cfg_words + full_chunks, cfg_words + batch_chunks, dummy_cfg);
    }

    src_bo.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    cfg_bo.sync(XCL_BO_SYNC_BO_TO_DEVICE);
}

template <typename T, typename Compare>
void process_npu_results(xrt::bo& dst_bo, std::vector<T>& heap, std::size_t k, Compare comp, 
                         std::size_t batch_chunks, T pad_val) {
    T* dst_map = dst_bo.map<T*>();
    const std::size_t chunk_size = 1024;
    
    for (std::size_t c = 0; c < batch_chunks; ++c) {
        T* chunk_ptr = dst_map + (c * chunk_size);
        int32_t count = static_cast<int32_t>(chunk_ptr[0]);
        
        if (count > 0 && count <= 1008) {
            for (int32_t i = 1; i <= count; ++i) {
                T val = chunk_ptr[i];
                if (val != pad_val) {
                    if (comp(val, heap.front())) {
                        std::pop_heap(heap.begin(), heap.end(), comp);
                        heap.back() = val;
                        std::push_heap(heap.begin(), heap.end(), comp);
                    }
                }
            }
        }
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

    constexpr double SAMPLE_FRACTION = 0.02; 
    std::size_t sample_size = std::max(k, static_cast<std::size_t>(n * SAMPLE_FRACTION));
    sample_size = std::min(sample_size, n);

    std::vector<T> sample(data.begin(), data.begin() + sample_size);
    std::nth_element(sample.begin(), sample.begin() + k - 1, sample.end(), comp);

    std::vector<T> heap(sample.begin(), sample.begin() + k);
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

    int active_idx = 0;
    int next_idx = 1;

    // start after sample size to prevent redundant NPU work
    std::size_t offset = sample_size; 
    std::size_t total_dispatches = 0;

    if (offset < n) {
        std::size_t current_batch = std::min(batch_size, n - offset);
        prepare_npu_batch<T>(state.mr_src_bo[active_idx], state.mr_cfg_bo[active_idx], data.data() + offset,
                             current_batch, batch_size, BATCH_CHUNKS, heap.front(), want_max, pad_val);
        rl[active_idx].execute();
        total_dispatches++;
        offset += batch_size;
    }

    while (offset < n) {
        std::size_t current_batch = std::min(batch_size, n - offset);

        prepare_npu_batch<T>(state.mr_src_bo[next_idx], state.mr_cfg_bo[next_idx], data.data() + offset,
                             current_batch, batch_size, BATCH_CHUNKS, heap.front(), want_max, pad_val);

        npu::utils::wait_for_runlist_or_throw(rl[active_idx], npu::utils::read_wait_timeout_ms());
        
        rl[next_idx].execute();
        total_dispatches++;

        state.mr_dst_bo[active_idx].sync(XCL_BO_SYNC_BO_FROM_DEVICE); 
        process_npu_results<T>(state.mr_dst_bo[active_idx], heap, k, comp, BATCH_CHUNKS, pad_val);

        std::swap(active_idx, next_idx);
        offset += batch_size;
    }

    if (total_dispatches > 0) {
        npu::utils::wait_for_runlist_or_throw(rl[active_idx], npu::utils::read_wait_timeout_ms());
        state.mr_dst_bo[active_idx].sync(XCL_BO_SYNC_BO_FROM_DEVICE); 
        process_npu_results<T>(state.mr_dst_bo[active_idx], heap, k, comp, BATCH_CHUNKS, pad_val);
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
