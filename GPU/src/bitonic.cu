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

// --- FP16 Cast Kernels ---
__global__ void cast_float_to_half_kernel(const float* __restrict__ src, __half* __restrict__ dst, std::size_t n) {
	const std::size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
	if (tid < n) {
		dst[tid] = __float2half(src[tid]);
	}
}

__global__ void cast_half_to_float_kernel(const __half* __restrict__ src, float* __restrict__ dst, std::size_t n) {
	const std::size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
	if (tid < n) {
		dst[tid] = __half2float(src[tid]);
	}
}

template <typename T> __global__ void bitonic_fused_shared(T* data, std::size_t total_pairs, FusedLayers layers) {
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

	if (idx1 < n)
		data[idx1] = s_data[tid];
	if (idx2 < n)
		data[idx2] = s_data[tid + block_size];
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

	dst[tid] = less_than(a, b) ? a : b;
}

std::size_t choose_block_size() {
	cudaDeviceProp prop{};
	CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
	return static_cast<std::size_t>(prop.warpSize * 8);
}

// --- Core execution loop extracted for code reuse ---
template <typename T>
T* execute_network_kernels(T* current_src, T* current_dst, std::size_t n,
						   const std::vector<common::bitonic::Layer>& layers, std::size_t block_size,
						   double& out_elapsed_ms, std::size_t& out_launches, std::size_t& out_comparators,
						   std::size_t& out_final_n) {

	cudaEvent_t start{};
	cudaEvent_t stop{};
	CUDA_CHECK(cudaEventCreate(&start));
	CUDA_CHECK(cudaEventCreate(&stop));
	CUDA_CHECK(cudaEventRecord(start));

	FusedLayers buffer;
	buffer.count = 0;
	out_final_n = n;

	for (std::size_t layer_idx = 0; layer_idx < layers.size(); ++layer_idx) {
		const auto& layer = layers[layer_idx];
		const std::size_t stage = layer.k;
		const std::size_t step = layer.j;
		const std::size_t pairs = layer.active_n / 2;
		out_comparators += pairs;

		if (layer.type == common::bitonic::LayerType::Truncate) {
			out_final_n = pairs; // Array is halved in a truncate layer

			if (buffer.count > 0) {
				const dim3 grid(static_cast<unsigned int>((pairs + block_size - 1) / block_size));
				const std::size_t smem_size = block_size * 2 * sizeof(T);

				bitonic_fused_shared<<<grid, block_size, smem_size>>>(current_src, pairs, buffer);
				CUDA_CHECK(cudaGetLastError());
				out_launches++;
				buffer.count = 0;
			}

			const dim3 grid(static_cast<unsigned int>((pairs + block_size - 1) / block_size));
			bitonic_layer_truncate_kernel<<<grid, block_size>>>(current_src, current_dst, pairs, step);
			CUDA_CHECK(cudaGetLastError());
			out_launches++;

			T* temp = current_src;
			current_src = current_dst;
			current_dst = temp;
			continue;
		}

		out_final_n = layer.active_n;
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
			out_launches++;
			buffer.count = 0;
		}

		if (step > block_size) {
			const dim3 grid(static_cast<unsigned int>((pairs + block_size - 1) / block_size));
			bitonic_layer_global_coalesced<<<grid, block_size>>>(current_src, pairs, stage, step);
			CUDA_CHECK(cudaGetLastError());
			out_launches++;
		}
	}

	CUDA_CHECK(cudaEventRecord(stop));
	CUDA_CHECK(cudaEventSynchronize(stop));

	float elapsed_ms_f = 0.0f;
	CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms_f, start, stop));
	out_elapsed_ms = static_cast<double>(elapsed_ms_f);

	CUDA_CHECK(cudaEventDestroy(start));
	CUDA_CHECK(cudaEventDestroy(stop));

	return current_src;
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
	T* d_data_alt = nullptr;
	CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_data), n * sizeof(T)));
	CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_data_alt), n * sizeof(T)));
	CUDA_CHECK(cudaMemcpy(d_data, data.data(), n * sizeof(T), cudaMemcpyHostToDevice));

	double elapsed_ms = 0.0;
	std::size_t launches = 0;
	std::size_t comparators = 0;
	std::size_t final_n = n;

	T* final_src =
		execute_network_kernels(d_data, d_data_alt, n, layers, block_size, elapsed_ms, launches, comparators, final_n);

	CUDA_CHECK(cudaMemcpy(data.data(), final_src, final_n * sizeof(T), cudaMemcpyDeviceToHost));

	CUDA_CHECK(cudaFree(d_data));
	CUDA_CHECK(cudaFree(d_data_alt));

	return RunStats{elapsed_ms, launches, comparators, block_size};
}

RunStats run_network_cuda_fp16(std::vector<float>& data, const std::vector<common::bitonic::Layer>& layers) {
	if (data.empty()) {
		return RunStats{0.0, 0, 0, 0};
	}

	const std::size_t n = data.size();
	const std::size_t block_size = choose_block_size();

	float* d_float_data = nullptr;
	__half* d_half_data = nullptr;
	__half* d_half_alt = nullptr;

	CUDA_CHECK(cudaMalloc(&d_float_data, n * sizeof(float)));
	CUDA_CHECK(cudaMalloc(&d_half_data, n * sizeof(__half)));
	CUDA_CHECK(cudaMalloc(&d_half_alt, n * sizeof(__half)));

	CUDA_CHECK(cudaMemcpy(d_float_data, data.data(), n * sizeof(float), cudaMemcpyHostToDevice));

	const dim3 grid((n + block_size - 1) / block_size);
	cast_float_to_half_kernel<<<grid, block_size>>>(d_float_data, d_half_data, n);
	CUDA_CHECK(cudaGetLastError());

	double elapsed_ms = 0.0;
	std::size_t launches = 0;
	std::size_t comparators = 0;
	std::size_t final_n = n;

	__half* final_src = execute_network_kernels(d_half_data, d_half_alt, n, layers, block_size, elapsed_ms, launches,
												comparators, final_n);

	const dim3 final_grid((final_n + block_size - 1) / block_size);
	cast_half_to_float_kernel<<<final_grid, block_size>>>(final_src, d_float_data, final_n);
	CUDA_CHECK(cudaGetLastError());
	CUDA_CHECK(cudaDeviceSynchronize());

	CUDA_CHECK(cudaMemcpy(data.data(), d_float_data, final_n * sizeof(float), cudaMemcpyDeviceToHost));

	CUDA_CHECK(cudaFree(d_float_data));
	CUDA_CHECK(cudaFree(d_half_data));
	CUDA_CHECK(cudaFree(d_half_alt));

	return RunStats{elapsed_ms, launches, comparators, block_size};
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
