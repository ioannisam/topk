#include "../include/algorithm.hpp"
#include "key_codec.hpp"
#include "xrt_utils.hpp"

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <type_traits>
#include <limits>
#include <stdexcept>
#include <vector>

namespace npu::map_reduce {
namespace {

using npu::utils::from_key;
using npu::utils::to_key;

template <bool WantMax> static void sift_down(std::vector<std::int32_t>& heap, std::size_t i) {
	using Cmp = std::conditional_t<WantMax, std::greater<std::int32_t>, std::less<std::int32_t>>;
	const std::size_t n = heap.size();
	while (true) {
		std::size_t best = i;
		std::size_t left = 2 * i + 1;
		std::size_t right = 2 * i + 2;

		if (left < n && Cmp{}(heap[best], heap[left]))
			best = left;
		if (right < n && Cmp{}(heap[best], heap[right]))
			best = right;
		if (best == i)
			break;
		std::swap(heap[i], heap[best]);
		i = best;
	}
}

template <bool WantMax, typename T>
std::size_t prepare_npu_batch(xrt::bo& src_bo, xrt::bo& cfg_bo, const T* data_ptr, std::size_t current_batch,
							  std::size_t batch_size, std::size_t batch_chunks, std::int32_t threshold_key,
							  std::int32_t sentinel_key) {

	std::int32_t* src_map = src_bo.map<std::int32_t*>();
	int32_t* cfg_map = cfg_bo.map<int32_t*>();

	if constexpr (std::is_same_v<T, std::int32_t>) {
		if (current_batch > 0)
			std::memcpy(src_map, data_ptr, current_batch * sizeof(std::int32_t));
	} else {
		for (std::size_t i = 0; i < current_batch; ++i)
			src_map[i] = to_key<T>(data_ptr[i]);
	}

	struct alignas(16) CfgWord {
		int32_t threshold;
		int32_t want_max;
		int32_t sentinel;
		int32_t padding;
	};

	CfgWord current_cfg = {threshold_key, WantMax ? 1 : 0, sentinel_key, 0};

	CfgWord* cfg_words = reinterpret_cast<CfgWord*>(cfg_map);
	const std::size_t chunk_size = 1024;
	std::size_t full_chunks = current_batch / chunk_size;
	std::size_t remainder = current_batch % chunk_size;

	std::fill(cfg_words, cfg_words + full_chunks, current_cfg);

	if (remainder > 0) {
		std::fill(src_map + (full_chunks * chunk_size) + remainder, src_map + ((full_chunks + 1) * chunk_size),
				  sentinel_key);
		cfg_words[full_chunks] = current_cfg;
		full_chunks++;
	}

	if (full_chunks < batch_chunks) {
		CfgWord dummy_cfg = current_cfg;
		dummy_cfg.threshold = WantMax ? std::numeric_limits<int32_t>::max() : std::numeric_limits<int32_t>::lowest();
		std::fill(cfg_words + full_chunks, cfg_words + batch_chunks, dummy_cfg);
	}

	src_bo.sync(XCL_BO_SYNC_BO_TO_DEVICE, full_chunks * chunk_size * sizeof(T), 0);
	cfg_bo.sync(XCL_BO_SYNC_BO_TO_DEVICE);
	return full_chunks;
}

template <bool WantMax>
void process_npu_results(xrt::bo& dst_bo, std::vector<std::int32_t>& heap, std::size_t batch_chunks,
						 std::int32_t sentinel_key) {
	using Cmp = std::conditional_t<WantMax, std::greater<std::int32_t>, std::less<std::int32_t>>;
	std::int32_t* dst_map = dst_bo.map<std::int32_t*>();
	const std::size_t chunk_size = 1024;

	for (std::size_t c = 0; c < batch_chunks; ++c) {
		std::int32_t* chunk_ptr = dst_map + (c * chunk_size);
		for (std::size_t i = 0; i < chunk_size; ++i) {
			std::int32_t key = chunk_ptr[i];
			if (key != sentinel_key && Cmp{}(key, heap.front())) {
				heap[0] = key;
				sift_down<WantMax>(heap, 0);
			}
		}
	}
}

template <bool WantMax, typename T>
std::vector<T> run_map_reduce_offload_xrt(const std::vector<T>& data, std::size_t k,
										  const npu::utils::OffloadConfig& offload_cfg,
										  npu::map_reduce::RunStats* stats) {
	using Cmp = std::conditional_t<WantMax, std::greater<std::int32_t>, std::less<std::int32_t>>;

	if (k == 0 || data.empty())
		return {};

	const std::size_t n = data.size();
	k = std::min(k, n);

	npu::utils::SharedXrtState& state = npu::utils::get_shared_xrt_state(offload_cfg);
	auto t0 = std::chrono::high_resolution_clock::now();

	constexpr double SAMPLE_FRACTION = 0.02;
	std::size_t sample_size = std::max(k, static_cast<std::size_t>(n * SAMPLE_FRACTION));
	sample_size = std::min(sample_size, n);

	std::vector<std::int32_t> heap(sample_size);
	for (std::size_t i = 0; i < sample_size; ++i)
		heap[i] = to_key<T>(data[i]);
	std::nth_element(heap.begin(), heap.begin() + k - 1, heap.end(), Cmp{});
	heap.resize(k);
	std::make_heap(heap.begin(), heap.end(), Cmp{});

	const std::size_t BATCH_CHUNKS = 1024;
	const std::size_t chunk_size = 1024;
	const std::size_t batch_size = BATCH_CHUNKS * chunk_size;
	const std::size_t batch_bytes = batch_size * sizeof(T);

	state.allocate_mr_buffers(batch_bytes);

	xrt::run run[2];
	xrt::runlist rl[2] = {xrt::runlist(state.hwctx), xrt::runlist(state.hwctx)};

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

	const T pad_val = WantMax ? std::numeric_limits<T>::lowest() : std::numeric_limits<T>::max();
	const std::int32_t sentinel_key = to_key<T>(pad_val);

	int active_idx = 0;
	int next_idx = 1;

	std::size_t offset = sample_size;
	std::size_t total_dispatches = 0;
	std::size_t valid_chunks[2] = {0, 0};

	if (offset < n) {
		std::size_t current_batch = std::min(batch_size, n - offset);
		valid_chunks[active_idx] = prepare_npu_batch<WantMax, T>(
			state.mr_src_bo[active_idx], state.mr_cfg_bo[active_idx], data.data() + offset, current_batch, batch_size,
			BATCH_CHUNKS, heap.front(), sentinel_key);
		rl[active_idx].execute();
		total_dispatches++;
		offset += batch_size;
	}

	while (offset < n) {
		std::size_t current_batch = std::min(batch_size, n - offset);

		valid_chunks[next_idx] =
			prepare_npu_batch<WantMax, T>(state.mr_src_bo[next_idx], state.mr_cfg_bo[next_idx], data.data() + offset,
										  current_batch, batch_size, BATCH_CHUNKS, heap.front(), sentinel_key);

		npu::utils::wait_for_runlist_or_throw(rl[active_idx], npu::utils::read_wait_timeout_ms());

		rl[next_idx].execute();
		total_dispatches++;

		state.mr_dst_bo[active_idx].sync(XCL_BO_SYNC_BO_FROM_DEVICE, valid_chunks[active_idx] * chunk_size * sizeof(T),
										 0);
		process_npu_results<WantMax>(state.mr_dst_bo[active_idx], heap, valid_chunks[active_idx], sentinel_key);

		std::swap(active_idx, next_idx);
		offset += batch_size;
	}

	if (total_dispatches > 0) {
		npu::utils::wait_for_runlist_or_throw(rl[active_idx], npu::utils::read_wait_timeout_ms());
		state.mr_dst_bo[active_idx].sync(XCL_BO_SYNC_BO_FROM_DEVICE, valid_chunks[active_idx] * chunk_size * sizeof(T),
										 0);
		process_npu_results<WantMax>(state.mr_dst_bo[active_idx], heap, valid_chunks[active_idx], sentinel_key);
	}

	std::vector<T> result(heap.size());
	for (std::size_t i = 0; i < heap.size(); ++i)
		result[i] = from_key<T>(heap[i]);
	std::sort(result.begin(), result.end(),
			  [](const T& lhs, const T& rhs) { return WantMax ? (lhs > rhs) : (lhs < rhs); });

	auto t1 = std::chrono::high_resolution_clock::now();

	if (stats != nullptr) {
		stats->elapsed_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
		stats->layer_dispatches = total_dispatches;
		stats->used_offload = true;
	}

	return result;
}

} // namespace

template <typename T>
std::vector<T> run_topk_npu(const std::vector<T>& data, std::size_t k, bool want_max, std::size_t workers,
							npu::map_reduce::RunStats* stats) {
	(void)workers;

	npu::utils::OffloadConfig offload_cfg = npu::utils::load_offload_config();
	if (!offload_cfg.enabled) {
		throw std::runtime_error("NPU offload is required. Set NPU_OFFLOAD_XCLBIN to map_reduce.xclbin.");
	}

	if (want_max)
		return run_map_reduce_offload_xrt<true>(data, k, offload_cfg, stats);
	return run_map_reduce_offload_xrt<false>(data, k, offload_cfg, stats);
}

template std::vector<std::int32_t> run_topk_npu<std::int32_t>(const std::vector<std::int32_t>& data, std::size_t k,
															  bool want_max, std::size_t workers,
															  npu::map_reduce::RunStats* stats);
template std::vector<std::uint32_t> run_topk_npu<std::uint32_t>(const std::vector<std::uint32_t>& data, std::size_t k,
																bool want_max, std::size_t workers,
																npu::map_reduce::RunStats* stats);
template std::vector<float> run_topk_npu<float>(const std::vector<float>& data, std::size_t k, bool want_max,
												std::size_t workers, npu::map_reduce::RunStats* stats);
template std::vector<double> run_topk_npu<double>(const std::vector<double>& data, std::size_t k, bool want_max,
												  std::size_t workers, npu::map_reduce::RunStats* stats);

#if defined(__FLT16_MANT_DIG__)
template std::vector<_Float16> run_topk_npu<_Float16>(const std::vector<_Float16>& data, std::size_t k, bool want_max,
													  std::size_t workers, npu::map_reduce::RunStats* stats);
#endif

} // namespace npu::map_reduce
