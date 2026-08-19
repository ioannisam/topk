#include "../include/algorithm.hpp"
#include "cuda_utils.cuh"
#include "device_traits.cuh"

#include "common/energy.hpp"

#include <cstdint>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

#include <cuda_fp16.h>
#include <cuda_runtime.h>

namespace gpu::bitonic {

namespace {

using gpu::utils::DeviceBuffer;
using gpu::utils::EventGuard;
using gpu::utils::GraphExecGuard;
using gpu::utils::GraphGuard;
using gpu::utils::StreamGuard;

constexpr std::size_t BITONIC_BLOCK_SIZE = 256;
constexpr std::size_t BITONIC_TILE_LARGE = 8192;		  // working set > L2 (DRAM bound)
constexpr std::size_t BITONIC_TILE_SMALL = 4096;		  // working set <= L2 (occupancy bound)
constexpr std::size_t BITONIC_MAX_TILE_BYTES = 48 * 1024; // stay within default smem budget (no opt-in)
constexpr int FUSED_LAYER_CAP = 128;
constexpr int MULTISTEP_MAX_BITS = 5;

std::size_t l2_cache_bytes() {
	static const std::size_t bytes = []() {
		int v = 0;
		CUDA_CHECK(cudaDeviceGetAttribute(&v, cudaDevAttrL2CacheSize, 0));
		return static_cast<std::size_t>(v);
	}();
	return bytes;
}

template <typename T> std::size_t tile_target_elems(std::size_t n) {
	const std::size_t target = (n * sizeof(T) > l2_cache_bytes()) ? BITONIC_TILE_LARGE : BITONIC_TILE_SMALL;
	const std::size_t cap = BITONIC_MAX_TILE_BYTES / sizeof(T);
	const std::size_t limit = cap < target ? cap : target;
	std::size_t w = 1;
	while ((w << 1) <= limit) {
		w <<= 1;
	}
	return w;
}

struct FusedLayers {
	std::uint32_t stages[FUSED_LAYER_CAP];
	std::uint32_t steps[FUSED_LAYER_CAP];
	int count;
};

template <typename T, int TPB>
__global__ __launch_bounds__(TPB) void bitonic_fused_wide(
	T* __restrict__ data, std::size_t tile_elems, FusedLayers layers
) {
	extern __shared__ char smem[];
	T* s_data = reinterpret_cast<T*>(smem);

	const std::size_t block_offset = static_cast<std::size_t>(blockIdx.x) * tile_elems;
	const std::size_t comparisons = tile_elems >> 1;

	for (std::size_t k = threadIdx.x; k < tile_elems; k += TPB) {
		s_data[k] = data[block_offset + k];
	}
	__syncthreads();

	for (int l = 0; l < layers.count; l++) {
		const std::size_t stage = layers.stages[l];
		const std::size_t step = layers.steps[l];

		for (std::size_t c = threadIdx.x; c < comparisons; c += TPB) {
			const std::size_t low = c & (step - 1);
			const std::size_t local_i = ((c - low) << 1) + low;
			const std::size_t local_ixj = local_i + step;
			const bool ascending = ((block_offset + local_i) & stage) == 0;

			const T a = s_data[local_i];
			const T b = s_data[local_ixj];
			const T min_val = gpu::traits::DeviceTraits<T>::min(a, b);
			const T max_val = gpu::traits::DeviceTraits<T>::max(a, b);

			s_data[local_i] = ascending ? min_val : max_val;
			s_data[local_ixj] = ascending ? max_val : min_val;
		}
		__syncthreads();
	}

	for (std::size_t k = threadIdx.x; k < tile_elems; k += TPB) {
		data[block_offset + k] = s_data[k];
	}
}

template <typename T, int GROUP_BITS>
__global__ void bitonic_multistep(T* __restrict__ data, std::size_t num_groups, std::size_t stage, std::size_t j_top) {
	constexpr int G = 1 << GROUP_BITS;
	const std::size_t tid = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
	if (tid >= num_groups) {
		return;
	}

	const std::size_t j_bot = j_top >> (GROUP_BITS - 1);
	const std::size_t low = tid & (j_bot - 1);
	const std::size_t base = ((tid - low) << GROUP_BITS) + low;

	T reg[G];
#pragma unroll
	for (int p = 0; p < G; p++) {
		reg[p] = data[base + static_cast<std::size_t>(p) * j_bot];
	}

#pragma unroll
	for (int m = 0; m < GROUP_BITS; m++) {
		const int local_stride = 1 << (GROUP_BITS - 1 - m);
#pragma unroll
		for (int p = 0; p < G; p++) {
			if ((p & local_stride) == 0) {
				const int q = p + local_stride;
				const std::size_t gi = base + static_cast<std::size_t>(p) * j_bot;
				const bool ascending = (gi & stage) == 0;

				const T a = reg[p];
				const T b = reg[q];
				const T min_val = gpu::traits::DeviceTraits<T>::min(a, b);
				const T max_val = gpu::traits::DeviceTraits<T>::max(a, b);

				reg[p] = ascending ? min_val : max_val;
				reg[q] = ascending ? max_val : min_val;
			}
		}
	}

#pragma unroll
	for (int p = 0; p < G; p++) {
		data[base + static_cast<std::size_t>(p) * j_bot] = reg[p];
	}
}

__global__ void bitonic_layer_global_coalesced_half2(
	__half2* data, std::size_t total_vec_pairs, std::size_t stage_vec, std::size_t step_vec
) {
	const std::size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
	if (tid >= total_vec_pairs) {
		return;
	}

	const std::size_t low = tid & (step_vec - 1);
	const std::size_t i = ((tid - low) << 1) + low;
	const std::size_t ixj = i + step_vec;
	const bool ascending = (i & stage_vec) == 0;

	const __half2 a = data[i];
	const __half2 b = data[ixj];

	const __half2 min_val = __hmin2(a, b);
	const __half2 max_val = __hmax2(a, b);

	data[i] = ascending ? min_val : max_val;
	data[ixj] = ascending ? max_val : min_val;
}

template <typename T>
__global__ void bitonic_layer_truncate_kernel(
	const T* __restrict__ src, T* __restrict__ dst, std::size_t pairs, std::size_t step
) {
	const std::size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
	if (tid >= pairs) {
		return;
	}

	const std::size_t low = tid & (step - 1);
	const std::size_t i = ((tid - low) << 1) + low;
	const std::size_t ixj = i + step;

	const T a = src[i];
	const T b = src[ixj];

	dst[tid] = gpu::traits::DeviceTraits<T>::lt(a, b) ? a : b;
}

template <typename T>
T* execute_network_kernels(
	T* current_src,
	T* current_dst,
	std::size_t n,
	const std::vector<common::bitonic::Layer>& layers,
	std::size_t block_size,
	double& out_elapsed_ms,
	std::size_t& out_launches,
	std::size_t& out_comparators,
	std::size_t& out_final_n,
	double& out_bytes
) {

	StreamGuard stream_guard;
	EventGuard start_guard;
	EventGuard stop_guard;
	const cudaStream_t stream = stream_guard.get();
	const cudaEvent_t start = start_guard.get();
	const cudaEvent_t stop = stop_guard.get();

	CUDA_CHECK(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal));

	FusedLayers buffer;
	buffer.count = 0;
	out_final_n = n;

	const std::size_t tile_cap = tile_target_elems<T>(n);
	auto tile_for = [&](std::size_t active_n) { return active_n < tile_cap ? active_n : tile_cap; };
	auto fuses = [&](std::size_t active_n, std::size_t step) { return 2 * step <= tile_for(active_n); };
	auto flush = [&](std::size_t active_n) {
		if (buffer.count == 0) {
			return;
		}
		const std::size_t tile_elems = tile_for(active_n);
		const unsigned int grid = static_cast<unsigned int>(active_n / tile_elems);
		const std::size_t smem_size = tile_elems * sizeof(T);

		bitonic_fused_wide<T, BITONIC_BLOCK_SIZE>
			<<<grid, BITONIC_BLOCK_SIZE, smem_size, stream>>>(current_src, tile_elems, buffer);
		out_launches++;
		out_bytes += 2.0 * static_cast<double>(active_n) * sizeof(T);
		buffer.count = 0;
	};

	std::vector<std::size_t> large_steps;
	std::size_t large_stage = 0;
	std::size_t large_active = 0;
	auto flush_large = [&]() {
		if (large_steps.empty()) {
			return;
		}
		if constexpr (std::is_same_v<T, __half>) {
			const std::size_t vec_pairs = large_active / 4;
			const dim3 grid(static_cast<unsigned int>((vec_pairs + block_size - 1) / block_size));
			for (std::size_t s : large_steps) {
				bitonic_layer_global_coalesced_half2<<<grid, block_size, 0, stream>>>(
					reinterpret_cast<__half2*>(current_src), vec_pairs, large_stage / 2, s / 2
				);
				out_launches++;
				out_bytes += 2.0 * static_cast<double>(large_active) * sizeof(T);
			}
		} else {
			std::size_t idx = 0;
			while (idx < large_steps.size()) {
				int t = static_cast<int>(large_steps.size() - idx);
				if (t > MULTISTEP_MAX_BITS) {
					t = MULTISTEP_MAX_BITS;
				}
				const std::size_t j_top = large_steps[idx];
				const std::size_t num_groups = large_active >> t;
				const unsigned int grid = static_cast<unsigned int>((num_groups + block_size - 1) / block_size);
				switch (t) {
				case 1:
					bitonic_multistep<T, 1>
						<<<grid, block_size, 0, stream>>>(current_src, num_groups, large_stage, j_top);
					break;
				case 2:
					bitonic_multistep<T, 2>
						<<<grid, block_size, 0, stream>>>(current_src, num_groups, large_stage, j_top);
					break;
				case 3:
					bitonic_multistep<T, 3>
						<<<grid, block_size, 0, stream>>>(current_src, num_groups, large_stage, j_top);
					break;
				case 4:
					bitonic_multistep<T, 4>
						<<<grid, block_size, 0, stream>>>(current_src, num_groups, large_stage, j_top);
					break;
				default:
					bitonic_multistep<T, 5>
						<<<grid, block_size, 0, stream>>>(current_src, num_groups, large_stage, j_top);
					break;
				}
				out_launches++;
				out_bytes += 2.0 * static_cast<double>(large_active) * sizeof(T);
				idx += static_cast<std::size_t>(t);
			}
		}
		large_steps.clear();
	};

	for (std::size_t layer_idx = 0; layer_idx < layers.size(); layer_idx++) {
		const auto& layer = layers[layer_idx];
		const std::size_t stage = layer.stage;
		const std::size_t step = layer.stride;
		const std::size_t pairs = layer.active_n / 2;
		out_comparators += pairs;

		if (layer.type == common::bitonic::LayerType::Truncate) {
			out_final_n = pairs;

			flush_large();
			flush(layer.active_n);

			const dim3 grid(static_cast<unsigned int>((pairs + block_size - 1) / block_size));
			bitonic_layer_truncate_kernel<<<grid, block_size, 0, stream>>>(current_src, current_dst, pairs, step);
			out_launches++;
			out_bytes += 3.0 * static_cast<double>(pairs) * sizeof(T);

			T* temp = current_src;
			current_src = current_dst;
			current_dst = temp;
			continue;
		}

		out_final_n = layer.active_n;
		if (fuses(layer.active_n, step)) {
			flush_large();
			buffer.stages[buffer.count] = static_cast<std::uint32_t>(stage);
			buffer.steps[buffer.count] = static_cast<std::uint32_t>(step);
			buffer.count++;
		} else {
			large_steps.push_back(step);
			large_stage = stage;
			large_active = layer.active_n;
		}

		const bool is_last = (layer_idx == layers.size() - 1);
		const bool next_is_trunc = (!is_last && layers[layer_idx + 1].type == common::bitonic::LayerType::Truncate);
		const bool next_is_large =
			(!is_last && !next_is_trunc && !fuses(layers[layer_idx + 1].active_n, layers[layer_idx + 1].stride));
		const bool is_full = (buffer.count == FUSED_LAYER_CAP);

		if (buffer.count > 0 && (is_last || next_is_large || next_is_trunc || is_full)) {
			flush(layer.active_n);
		}
	}
	flush_large();

	GraphGuard graph_guard;
	CUDA_CHECK(cudaStreamEndCapture(stream, graph_guard.addr()));
	const cudaGraph_t graph = graph_guard.get();

	GraphExecGuard instance_guard;
#if __CUDACC_VER_MAJOR__ >= 12
	CUDA_CHECK(cudaGraphInstantiate(instance_guard.addr(), graph, 0));
#else
	CUDA_CHECK(cudaGraphInstantiate(instance_guard.addr(), graph, nullptr, nullptr, 0));
#endif
	const cudaGraphExec_t instance = instance_guard.get();

	common::energy::Scope energy_scope(common::energy::Channel::Algo);
	CUDA_CHECK(cudaEventRecord(start, stream));
	CUDA_CHECK(cudaGraphLaunch(instance, stream));
	CUDA_CHECK(cudaEventRecord(stop, stream));

	CUDA_CHECK(cudaEventSynchronize(stop));
	energy_scope.close();

	float elapsed_ms_f = 0.0f;
	CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms_f, start, stop));
	out_elapsed_ms = static_cast<double>(elapsed_ms_f);

	return current_src;
}

} // namespace

std::string query_device_name() {
	cudaDeviceProp prop{};
	CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
	return std::string(prop.name);
}

template <typename T>
RunStats run_topk(T* data, std::size_t n, std::size_t& final_n, const std::vector<common::bitonic::Layer>& layers) {
	using D = typename gpu::traits::DeviceType<T>::type;

	if (n == 0) {
		final_n = 0;
		return RunStats{0.0, 0, 0, 0};
	}

	const std::size_t block_size = BITONIC_BLOCK_SIZE;

	DeviceBuffer<D> d_data(n);
	DeviceBuffer<D> d_data_alt(n);

	CUDA_CHECK(cudaMemcpy(d_data.get(), data, n * sizeof(D), cudaMemcpyHostToDevice));

	double elapsed_ms = 0.0;
	std::size_t launches = 0;
	std::size_t comparators = 0;
	double bytes_moved = 0.0;
	final_n = n;

	D* final_src = execute_network_kernels(
		d_data.get(), d_data_alt.get(), n, layers, block_size, elapsed_ms, launches, comparators, final_n, bytes_moved
	);

	CUDA_CHECK(cudaMemcpy(data, final_src, final_n * sizeof(D), cudaMemcpyDeviceToHost));

	return RunStats{elapsed_ms, launches, comparators, block_size, bytes_moved};
}

// clang-format off
template RunStats run_topk<std::int32_t>(
	std::int32_t*, std::size_t, std::size_t&, const std::vector<common::bitonic::Layer>&
);
template RunStats run_topk<std::uint32_t>(
	std::uint32_t*, std::size_t, std::size_t&, const std::vector<common::bitonic::Layer>&
);
template RunStats run_topk<float>(
	float*, std::size_t, std::size_t&, const std::vector<common::bitonic::Layer>&
);
template RunStats run_topk<double>(
	double*, std::size_t, std::size_t&, const std::vector<common::bitonic::Layer>&
);
#if defined(__FLT16_MANT_DIG__)
template RunStats run_topk<_Float16>(
	_Float16*, std::size_t, std::size_t&, const std::vector<common::bitonic::Layer>&
);
#endif
// clang-format on

} // namespace gpu::bitonic
