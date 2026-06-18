#include "../include/algorithm.hpp"
#include "key_codec.hpp"
#include "xrt_utils.hpp"

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <limits>
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

template <typename T>
std::size_t prepare_batch(xrt::bo& src_bo, const T* data_ptr, std::size_t batch_elems, std::int32_t pad_key) {
	std::int32_t* src_map = src_bo.map<std::int32_t*>();
	for (std::size_t i = 0; i < batch_elems; ++i) {
		src_map[i] = to_key<T>(data_ptr[i]);
	}

	const std::size_t full_tiles = batch_elems / kTile;
	const std::size_t remainder = batch_elems % kTile;
	std::size_t valid_tiles = full_tiles;

	if (remainder > 0) {
		std::fill(src_map + batch_elems, src_map + (full_tiles + 1) * kTile, pad_key);
		valid_tiles = full_tiles + 1;
	}

	src_bo.sync(XCL_BO_SYNC_BO_TO_DEVICE, valid_tiles * kTile * sizeof(std::int32_t), 0);
	return valid_tiles;
}

void reduce_batch(xrt::bo& dst_bo, std::vector<std::int32_t>& heap, std::size_t valid_tiles, std::size_t k,
				  std::int32_t pad_key) {
	const std::int32_t* dst_map = dst_bo.map<const std::int32_t*>();
	const std::size_t total = valid_tiles * kTile;

	for (std::size_t base = 0; base < total; base += kRunLen) {
		const std::int32_t* run = dst_map + base;
		for (std::size_t i = 0; i < kRunLen; ++i) {
			const std::int32_t key = run[i];
			if (key == pad_key)
				break;

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

	const std::size_t batch_bytes = kBatchElems * sizeof(std::int32_t);
	state.allocate_bit_buffers(batch_bytes);

	const std::int32_t pad_key = to_key<T>(std::numeric_limits<T>::max());

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

	std::vector<std::int32_t> heap;
	heap.reserve(kept);

	int active = 0;
	int next = 1;
	std::size_t offset = 0;
	std::size_t total_dispatches = 0;
	std::size_t valid_tiles[2] = {0, 0};

	auto t0 = std::chrono::high_resolution_clock::now();

	if (offset < n) {
		const std::size_t batch = std::min(kBatchElems, n - offset);
		valid_tiles[active] = prepare_batch<T>(state.bit_src_bo[active], data.data() + offset, batch, pad_key);
		rl[active].execute();
		total_dispatches++;
		offset += batch;
	}

	while (offset < n) {
		const std::size_t batch = std::min(kBatchElems, n - offset);
		valid_tiles[next] = prepare_batch<T>(state.bit_src_bo[next], data.data() + offset, batch, pad_key);

		npu::utils::wait_for_runlist_or_throw(rl[active], npu::utils::read_wait_timeout_ms());
		rl[next].execute();
		total_dispatches++;

		state.bit_dst_bo[active].sync(XCL_BO_SYNC_BO_FROM_DEVICE, valid_tiles[active] * kTile * sizeof(std::int32_t),
									  0);
		reduce_batch(state.bit_dst_bo[active], heap, valid_tiles[active], kept, pad_key);

		std::swap(active, next);
		offset += batch;
	}

	if (total_dispatches > 0) {
		npu::utils::wait_for_runlist_or_throw(rl[active], npu::utils::read_wait_timeout_ms());
		state.bit_dst_bo[active].sync(XCL_BO_SYNC_BO_FROM_DEVICE, valid_tiles[active] * kTile * sizeof(std::int32_t),
									  0);
		reduce_batch(state.bit_dst_bo[active], heap, valid_tiles[active], kept, pad_key);
	}

	std::sort(heap.begin(), heap.end());
	for (std::size_t i = 0; i < heap.size(); ++i) {
		data[i] = from_key<T>(heap[i]);
	}

	auto t1 = std::chrono::high_resolution_clock::now();

	return RunStats{std::chrono::duration<double, std::milli>(t1 - t0).count(), total_dispatches, 0, 1, true};
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

template <typename T>
RunStats run_network_npu(std::vector<T>& data, const std::vector<common::bitonic::Layer>& layers, std::size_t workers) {
	(void)workers;
	if (data.empty())
		return RunStats{0.0, 0, 0, 1, true};

	(void)npu::utils::open_device();
	const npu::utils::OffloadConfig offload_cfg = npu::utils::load_offload_config();
	if (!offload_cfg.enabled) {
		throw std::runtime_error(
			"NPU offload is required for this backend. Set NPU_OFFLOAD_XCLBIN to a valid xclbin path.");
	}
	return run_network_offload_xrt(data, layers, offload_cfg);
}

template RunStats run_network_npu<std::int32_t>(std::vector<std::int32_t>& data,
												const std::vector<common::bitonic::Layer>& layers, std::size_t workers);
template RunStats run_network_npu<std::uint32_t>(std::vector<std::uint32_t>& data,
												 const std::vector<common::bitonic::Layer>& layers,
												 std::size_t workers);
template RunStats run_network_npu<float>(std::vector<float>& data, const std::vector<common::bitonic::Layer>& layers,
										 std::size_t workers);
template RunStats run_network_npu<double>(std::vector<double>& data, const std::vector<common::bitonic::Layer>& layers,
										  std::size_t workers);

#if defined(__FLT16_MANT_DIG__)
template RunStats run_network_npu<_Float16>(std::vector<_Float16>& data,
											const std::vector<common::bitonic::Layer>& layers, std::size_t workers);
#endif

} // namespace npu::bitonic
