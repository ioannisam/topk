#include "../include/algorithm.hpp"
#include "key_codec.hpp"
#include "xrt_utils.hpp"

#include <algorithm>
#include <chrono>

#include "common/energy.hpp"
#include <cstdint>
#include <cstring>
#include <limits>
#include <memory>
#include <stdexcept>
#include <type_traits>
#include <vector>

namespace npu::bitonic {

namespace {

using npu::utils::from_key;
using npu::utils::to_key;

constexpr std::size_t kTile = 1024;
constexpr std::size_t kRunLen = 16;
constexpr std::size_t kBatchChunks = 1024;
constexpr std::size_t kBatchElems = kBatchChunks * kTile;

inline void sift_down_max(std::vector<std::int32_t>& heap, std::size_t i) {
	const std::size_t n = heap.size();
	while (true) {
		std::size_t best = i;
		const std::size_t left = 2 * i + 1;
		const std::size_t right = 2 * i + 2;
		if (left < n && heap[left] > heap[best])
			best = left;
		if (right < n && heap[right] > heap[best])
			best = right;
		if (best == i)
			break;
		std::swap(heap[i], heap[best]);
		i = best;
	}
}

std::size_t derive_kept_prefix(const std::vector<common::bitonic::Layer>& layers, std::size_t n) {
	for (const auto& layer : layers) {
		if (layer.type == common::bitonic::LayerType::Truncate) {
			return std::min(layer.j, n);
		}
	}
	return n;
}

constexpr std::int32_t kPadKey = std::numeric_limits<std::int32_t>::max();

template <typename T> std::size_t prepare_batch(xrt::bo& src_bo, const T* data_ptr, std::size_t batch_elems) {
	std::int32_t* src_map = src_bo.map<std::int32_t*>();
	npu::utils::encode_keys<T>(src_map, data_ptr, batch_elems);

	const std::size_t full_tiles = batch_elems / kTile;
	const std::size_t remainder = batch_elems % kTile;
	std::size_t valid_tiles = full_tiles;

	if (remainder > 0) {
		std::fill(src_map + batch_elems, src_map + (full_tiles + 1) * kTile, kPadKey);
		valid_tiles = full_tiles + 1;
	}

	src_bo.sync(XCL_BO_SYNC_BO_TO_DEVICE, valid_tiles * kTile * sizeof(std::int32_t), 0);
	return valid_tiles;
}

void reduce_batch(xrt::bo& dst_bo, std::vector<std::int32_t>& heap, std::size_t valid_tiles, std::size_t k,
				  std::size_t valid_elems) {
	const std::int32_t* dst_map = dst_bo.map<const std::int32_t*>();
	const std::size_t total = valid_tiles * kTile;

	for (std::size_t base = 0; base < total; base += kRunLen) {
		if (base >= valid_elems)
			break;

		const std::size_t run_len = std::min(kRunLen, valid_elems - base);
		const std::int32_t* run = dst_map + base;
		for (std::size_t i = 0; i < run_len; ++i) {
			const std::int32_t key = run[i];

			if (heap.size() < k) {
				heap.push_back(key);
				if (heap.size() == k)
					std::make_heap(heap.begin(), heap.end());
			} else if (key < heap.front()) {
				heap.front() = key;
				sift_down_max(heap, 0);
			} else {
				break;
			}
		}
	}
}

template <typename T>
RunStats run_network_offload_xrt(std::vector<T>& data, const std::vector<common::bitonic::Layer>& layers,
								 const npu::utils::OffloadConfig& offload_cfg) {
	const std::size_t n = data.size();
	const std::size_t kept = derive_kept_prefix(layers, n);

	npu::utils::SharedXrtState& state = npu::utils::get_shared_xrt_state(offload_cfg);

	std::vector<std::int32_t> heap;
	heap.reserve(kept);

	int active = 0;
	int next = 1;
	std::size_t offset = 0;
	std::size_t total_dispatches = 0;
	std::size_t valid_tiles[2] = {0, 0};
	std::size_t valid_elems[2] = {0, 0};
	std::size_t total_tiles = 0;
	double bytes_moved = 0.0;

	auto account_batch = [&](std::size_t batch, std::size_t tiles) {
		const double tile_bytes = static_cast<double>(tiles) * kTile * sizeof(std::int32_t);
		bytes_moved += static_cast<double>(batch) * sizeof(T) + 6.0 * tile_bytes;
		total_tiles += tiles;
	};

	auto t0 = std::chrono::high_resolution_clock::now();
	common::energy::Scope energy_scope(common::energy::Channel::Algo);

	npu::PhaseTimers phases;
	auto setup_timer = std::make_unique<npu::PhaseTimer>(phases.setup_ms);

	const std::size_t batch_bytes = kBatchElems * sizeof(std::int32_t);
	state.allocate_bit_buffers(batch_bytes);

	xrt::run run[2];
	xrt::runlist rl[2] = {xrt::runlist(state.hwctx), xrt::runlist(state.hwctx)};
	for (int i = 0; i < 2; ++i) {
		run[i] = xrt::run(state.kernel);
		run[i].set_arg(0, 3);
		run[i].set_arg(1, state.instr_bo);
		run[i].set_arg(2, static_cast<uint32_t>(state.instr_v.size()));
		run[i].set_arg(3, state.bit_dst_bo[i]);
		run[i].set_arg(4, state.bit_src_bo[i]);
		rl[i].add(run[i]);
	}

	setup_timer.reset();

	if (offset < n) {
		const std::size_t batch = std::min(kBatchElems, n - offset);
		{
			npu::PhaseTimer timer(phases.stage_ms);
			valid_tiles[active] = prepare_batch<T>(state.bit_src_bo[active], data.data() + offset, batch);
			valid_elems[active] = batch;
			account_batch(batch, valid_tiles[active]);
		}
		{
			npu::PhaseTimer timer(phases.dispatch_ms);
			rl[active].execute();
		}
		total_dispatches++;
		offset += batch;
	}

	while (offset < n) {
		const std::size_t batch = std::min(kBatchElems, n - offset);
		{
			npu::PhaseTimer timer(phases.stage_ms);
			valid_tiles[next] = prepare_batch<T>(state.bit_src_bo[next], data.data() + offset, batch);
			valid_elems[next] = batch;
			account_batch(batch, valid_tiles[next]);
		}
		{
			npu::PhaseTimer timer(phases.wait_ms);
			npu::utils::wait_for_runlist_or_throw(rl[active], npu::utils::read_wait_timeout_ms());
		}
		{
			npu::PhaseTimer timer(phases.dispatch_ms);
			rl[next].execute();
		}
		total_dispatches++;

		{
			npu::PhaseTimer timer(phases.merge_ms);
			state.bit_dst_bo[active].sync(XCL_BO_SYNC_BO_FROM_DEVICE,
										  valid_tiles[active] * kTile * sizeof(std::int32_t), 0);
			reduce_batch(state.bit_dst_bo[active], heap, valid_tiles[active], kept, valid_elems[active]);
		}

		std::swap(active, next);
		offset += batch;
	}

	if (total_dispatches > 0) {
		{
			npu::PhaseTimer timer(phases.wait_ms);
			npu::utils::wait_for_runlist_or_throw(rl[active], npu::utils::read_wait_timeout_ms());
		}
		{
			npu::PhaseTimer timer(phases.merge_ms);
			state.bit_dst_bo[active].sync(XCL_BO_SYNC_BO_FROM_DEVICE,
										  valid_tiles[active] * kTile * sizeof(std::int32_t), 0);
			reduce_batch(state.bit_dst_bo[active], heap, valid_tiles[active], kept, valid_elems[active]);
		}
	}

	{
		npu::PhaseTimer timer(phases.finalize_ms);
		if constexpr (sizeof(T) == 8) {
			bytes_moved += static_cast<double>(n) * sizeof(T);
			const std::size_t kept_k = heap.size();
			std::int32_t threshold = std::numeric_limits<std::int32_t>::min();
			for (const std::int32_t key : heap) {
				threshold = std::max(threshold, key);
			}
			std::vector<T> candidates;
			candidates.reserve(kept_k * 2);
			for (std::size_t j = 0; j < n; ++j) {
				if (to_key<T>(data[j]) <= threshold) {
					candidates.push_back(data[j]);
				}
			}
			const std::size_t take = std::min(kept_k, candidates.size());
			std::partial_sort(candidates.begin(), candidates.begin() + take, candidates.end());
			for (std::size_t i = 0; i < take; ++i) {
				data[i] = candidates[i];
			}
		} else {
			std::sort(heap.begin(), heap.end());
			for (std::size_t i = 0; i < heap.size(); ++i) {
				data[i] = from_key<T>(heap[i]);
			}
		}
	}

	energy_scope.close();
	auto t1 = std::chrono::high_resolution_clock::now();

	const std::size_t comparators = total_tiles * common::bitonic::count_full_comparators(kTile);
	return RunStats{std::chrono::duration<double, std::milli>(t1 - t0).count(),
					total_dispatches,
					comparators,
					1,
					true,
					phases,
					bytes_moved};
}

} // namespace

std::string query_device_name() {
	xrt::device dev = npu::utils::open_device();
	return dev.get_info<xrt::info::device::name>();
}

std::string query_device_bdf() {
	xrt::device dev = npu::utils::open_device();
	return dev.get_info<xrt::info::device::bdf>();
}

bool is_offload_configured() {
	return npu::utils::load_offload_config().enabled;
}

template <typename T> RunStats run_topk(std::vector<T>& data, const std::vector<common::bitonic::Layer>& layers) {
	if (data.empty())
		return RunStats{0.0, 0, 0, 1, true};

	(void)npu::utils::open_device();
	const npu::utils::OffloadConfig offload_cfg = npu::utils::load_offload_config();
	if (!offload_cfg.enabled) {
		throw std::runtime_error(
			"NPU offload is required for this backend. Set NPU_OFFLOAD_XCLBIN to a valid xclbin path.");
	}
	npu::utils::require_xclbin_for(offload_cfg, "bitonic");
	return run_network_offload_xrt(data, layers, offload_cfg);
}

template RunStats run_topk<std::int32_t>(std::vector<std::int32_t>& data,
										 const std::vector<common::bitonic::Layer>& layers);
template RunStats run_topk<std::uint32_t>(std::vector<std::uint32_t>& data,
										  const std::vector<common::bitonic::Layer>& layers);
template RunStats run_topk<float>(std::vector<float>& data, const std::vector<common::bitonic::Layer>& layers);
template RunStats run_topk<double>(std::vector<double>& data, const std::vector<common::bitonic::Layer>& layers);

#if defined(__FLT16_MANT_DIG__)
template RunStats run_topk<_Float16>(std::vector<_Float16>& data, const std::vector<common::bitonic::Layer>& layers);
#endif

} // namespace npu::bitonic
