#include "../include/algorithm.hpp"

#include <cstdint>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

#include <cuda_fp16.h>
#include <cuda_runtime.h>

namespace gpu::bitonic {

namespace {

#define CUDA_CHECK(expr)                                                                                               \
	do {                                                                                                               \
		cudaError_t _err = (expr);                                                                                     \
		if (_err != cudaSuccess) {                                                                                     \
			throw std::runtime_error(std::string("CUDA error: ") + cudaGetErrorString(_err));                          \
		}                                                                                                              \
	} while (false)

template <typename T> __device__ __forceinline__ bool greater_than(T a, T b) {
	if constexpr (std::is_same_v<T, __half>) {
		return __hgt(a, b);
	}
	return a > b;
}

template <typename T> __device__ __forceinline__ bool less_than(T a, T b) {
	if constexpr (std::is_same_v<T, __half>) {
		return __hlt(a, b);
	}
	return a < b;
}

struct FusedLayers {
	std::uint32_t stages[16];
	std::uint32_t steps[16];
	int count;
};

template <typename T>
__global__ void bitonic_fused_shared(T* data, std::size_t total_pairs, FusedLayers layers) {
	extern __shared__ char smem[];
	T* s_data = reinterpret_cast<T*>(smem);

	const std::size_t tid = threadIdx.x;
	const std::size_t block_size = blockDim.x;
	const std::size_t block_offset = blockIdx.x * (block_size * 2);

	const std::size_t idx1 = block_offset + tid;
	const std::size_t idx2 = block_offset + tid + block_size;
	const std::size_t n = total_pairs * 2;

	s_data[tid] = (idx1 < n) ? data[idx1] : data[0];
	s_data[tid + block_size] = (idx2 < n) ? data[idx2] : data[0];
	__syncthreads();

	const std::size_t global_tid = blockIdx.x * blockDim.x + threadIdx.x;

	for (int l = 0; l < layers.count; ++l) {
		if (global_tid < total_pairs) {
			const std::size_t stage = layers.stages[l];
			const std::size_t step = layers.steps[l];

			const std::size_t low = tid & (step - 1);
			const std::size_t local_i = ((tid - low) << 1) + low;
			const std::size_t local_ixj = local_i + step;

			const std::size_t global_i = block_offset + local_i;
			const bool ascending = (global_i & stage) == 0;

			const T a = s_data[local_i];
			const T b = s_data[local_ixj];
			if ((ascending && greater_than(a, b)) || (!ascending && less_than(a, b))) {
				s_data[local_i] = b;
				s_data[local_ixj] = a;
			}
		}
		__syncthreads();
	}

	if (idx1 < n) data[idx1] = s_data[tid];
	if (idx2 < n) data[idx2] = s_data[tid + block_size];
}

template <typename T>
__global__ void bitonic_layer_global_coalesced(T* data, std::size_t total_pairs, std::size_t stage, std::size_t step) {
	const std::size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
	if (tid >= total_pairs) {
		return;
	}

	const std::size_t low = tid & (step - 1);
	const std::size_t i = ((tid - low) << 1) + low;
	const std::size_t ixj = i + step;
	const bool ascending = (i & stage) == 0;

	const T a = data[i];
	const T b = data[ixj];
	if ((ascending && greater_than(a, b)) || (!ascending && less_than(a, b))) {
		data[i] = b;
		data[ixj] = a;
	}
}

template <typename T>
__global__ void bitonic_layer_truncate_kernel(const T* __restrict__ src, T* __restrict__ dst, std::size_t pairs,
											  std::size_t step) {
	const std::size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
	if (tid >= pairs) {
		return;
	}

	const std::size_t low = tid & (step - 1);
	const std::size_t i = ((tid - low) << 1) + low;
	const std::size_t ixj = i + step;

	const T a = src[i];
	const T b = src[ixj];

	// TBiS guarantees truncation always keeps the smaller element of the block.
	// Since data is pre-negated if want_max is true, the mathematical min is always right.
	// Output index is exactly tid, providing perfectly coalesced writes!
	dst[tid] = less_than(a, b) ? a : b;
}

std::size_t choose_block_size() {
	cudaDeviceProp prop{};
	CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
	return static_cast<std::size_t>(prop.warpSize * 8);
}

} // namespace

std::string query_device_name() {
	cudaDeviceProp prop{};
	CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
	return std::string(prop.name);
}

int query_device_sm_count() {
	cudaDeviceProp prop{};
	CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
	return prop.multiProcessorCount;
}

template <typename T>
RunStats run_network_cuda(std::vector<T>& data, const std::vector<common::bitonic::Layer>& layers) {
	if (data.empty()) {
		return RunStats{0.0, 0, 0, 0};
	}

	const std::size_t n = data.size();
	const std::size_t block_size = choose_block_size();

	T* d_data = nullptr;
	T* d_data_alt = nullptr; // Secondary buffer for ping-pong compaction
	CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_data), n * sizeof(T)));
	CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_data_alt), n * sizeof(T)));
	CUDA_CHECK(cudaMemcpy(d_data, data.data(), n * sizeof(T), cudaMemcpyHostToDevice));

	T* current_src = d_data;
	T* current_dst = d_data_alt;

	std::size_t active_comparators = 0;
	std::size_t kernel_launches = 0;

	cudaEvent_t start{};
	cudaEvent_t stop{};
	CUDA_CHECK(cudaEventCreate(&start));
	CUDA_CHECK(cudaEventCreate(&stop));
	CUDA_CHECK(cudaEventRecord(start));

	FusedLayers buffer;
	buffer.count = 0;

	for (std::size_t layer_idx = 0; layer_idx < layers.size(); ++layer_idx) {
		const auto& layer = layers[layer_idx];
		const std::size_t stage = layer.k;
		const std::size_t step = layer.j;
		const std::size_t pairs = layer.active_n / 2;
		active_comparators += pairs;

		if (layer.type == common::bitonic::LayerType::Truncate) {
			// Flush any pending fused normal layers before truncating
			if (buffer.count > 0) {
				const dim3 grid(static_cast<unsigned int>((pairs + block_size - 1) / block_size));
				const std::size_t smem_size = block_size * 2 * sizeof(T);

				bitonic_fused_shared<<<grid, block_size, smem_size>>>(current_src, pairs, buffer);
				CUDA_CHECK(cudaGetLastError());
				kernel_launches++;
				buffer.count = 0;
			}

			// Launch the compaction step
			const dim3 grid(static_cast<unsigned int>((pairs + block_size - 1) / block_size));
			bitonic_layer_truncate_kernel<<<grid, block_size>>>(current_src, current_dst, pairs, step);
			CUDA_CHECK(cudaGetLastError());
			kernel_launches++;

			// Ping-pong the active buffers
			T* temp = current_src;
			current_src = current_dst;
			current_dst = temp;
			continue;
		}

		// Normal Layer Logic
		if (step <= block_size) {
			buffer.stages[buffer.count] = static_cast<std::uint32_t>(stage);
			buffer.steps[buffer.count] = static_cast<std::uint32_t>(step);
			buffer.count++;
		}

		const bool is_last = (layer_idx == layers.size() - 1);
		const bool next_is_large = (!is_last && layers[layer_idx + 1].j > block_size);
		const bool next_is_trunc = (!is_last && layers[layer_idx + 1].type == common::bitonic::LayerType::Truncate);
		const bool is_full = (buffer.count == 16);

		if (buffer.count > 0 && (is_last || next_is_large || next_is_trunc || is_full)) {
			const dim3 grid(static_cast<unsigned int>((pairs + block_size - 1) / block_size));
			const std::size_t smem_size = block_size * 2 * sizeof(T);

			bitonic_fused_shared<<<grid, block_size, smem_size>>>(current_src, pairs, buffer);
			CUDA_CHECK(cudaGetLastError());
			kernel_launches++;
			buffer.count = 0;
		}

		if (step > block_size) {
			const dim3 grid(static_cast<unsigned int>((pairs + block_size - 1) / block_size));
			bitonic_layer_global_coalesced<<<grid, block_size>>>(current_src, pairs, stage, step);
			CUDA_CHECK(cudaGetLastError());
			kernel_launches++;
		}
	}

	CUDA_CHECK(cudaEventRecord(stop));
	CUDA_CHECK(cudaEventSynchronize(stop));
	float elapsed_ms_f = 0.0f;
	CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms_f, start, stop));

	// Even if current_src is d_data_alt, we just safely copy the 'n' elements back to host.
	// Only the first 'active_n' are valid, which is exactly what build_output reads.
	CUDA_CHECK(cudaMemcpy(data.data(), current_src, n * sizeof(T), cudaMemcpyDeviceToHost));

	CUDA_CHECK(cudaEventDestroy(start));
	CUDA_CHECK(cudaEventDestroy(stop));
	CUDA_CHECK(cudaFree(d_data));
	CUDA_CHECK(cudaFree(d_data_alt));

	return RunStats{static_cast<double>(elapsed_ms_f), kernel_launches, active_comparators, block_size};
}

RunStats run_network_cuda_fp16(std::vector<float>& data, const std::vector<common::bitonic::Layer>& layers) {
	std::vector<__half> half_data(data.size());
	for (std::size_t i = 0; i < data.size(); ++i) {
		half_data[i] = __float2half(data[i]);
	}

	const RunStats stats = run_network_cuda(half_data, layers);

	for (std::size_t i = 0; i < data.size(); ++i) {
		data[i] = __half2float(half_data[i]);
	}

	return stats;
}

template RunStats run_network_cuda<std::int32_t>(std::vector<std::int32_t>& data,
												 const std::vector<common::bitonic::Layer>& layers);
template RunStats run_network_cuda<std::uint32_t>(std::vector<std::uint32_t>& data,
												  const std::vector<common::bitonic::Layer>& layers);
template RunStats run_network_cuda<float>(std::vector<float>& data, const std::vector<common::bitonic::Layer>& layers);
template RunStats run_network_cuda<double>(std::vector<double>& data,
										   const std::vector<common::bitonic::Layer>& layers);
template RunStats run_network_cuda<__half>(std::vector<__half>& data,
										   const std::vector<common::bitonic::Layer>& layers);

} // namespace gpu::bitonic
