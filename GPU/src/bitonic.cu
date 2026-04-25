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
__global__ void bitonic_layer_trunc(T* data, const std::uint32_t* pair_i, std::size_t count, std::size_t stage,
									std::size_t step) {
	const std::size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
	if (tid >= count) {
		return;
	}

	const std::size_t i = static_cast<std::size_t>(pair_i[tid]);
	const std::size_t ixj = i ^ step;
	const bool ascending = (i & stage) == 0;

	const T a = data[i];
	const T b = data[ixj];
	if ((ascending && greater_than(a, b)) || (!ascending && less_than(a, b))) {
		data[i] = b;
		data[ixj] = a;
	}
}

std::vector<std::size_t> build_offsets(const std::vector<std::vector<unsigned char>>& keep,
									   const std::vector<common::bitonic::Layer>& layers,
									   std::vector<std::uint32_t>& pairs_out, std::size_t n) {
	std::vector<std::size_t> offsets;
	offsets.reserve(layers.size() + 1);
	offsets.push_back(0);

	pairs_out.reserve((n >> 1) * layers.size());

	for (std::size_t layer_idx = 0; layer_idx < layers.size(); ++layer_idx) {
		const std::size_t j = layers[layer_idx].j;
		for (std::size_t i = 0; i < n; ++i) {
			const std::size_t ixj = i ^ j;
			if (ixj <= i) {
				continue;
			}
			if (keep[layer_idx][i] || keep[layer_idx][ixj]) {
				pairs_out.push_back(static_cast<std::uint32_t>(i));
			}
		}
		offsets.push_back(pairs_out.size());
	}

	return offsets;
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
RunStats run_network_cuda(std::vector<T>& data, const std::vector<common::bitonic::Layer>& layers,
						  const std::vector<std::vector<unsigned char>>& keep, bool trunc) {
	if (data.empty()) {
		return RunStats{0.0, 0, 0, 0};
	}

	const std::size_t n = data.size();
	const std::size_t block_size = choose_block_size();
	const std::size_t full_pairs = n >> 1;

	T* d_data = nullptr;
	CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_data), n * sizeof(T)));
	CUDA_CHECK(cudaMemcpy(d_data, data.data(), n * sizeof(T), cudaMemcpyHostToDevice));

	std::uint32_t* d_pairs = nullptr;
	std::vector<std::uint32_t> pair_i;
	std::vector<std::size_t> offsets;
	std::size_t active_comparators = full_pairs * layers.size();
	
	if (trunc) {
		offsets = build_offsets(keep, layers, pair_i, n);
		active_comparators = pair_i.size();
		if (!pair_i.empty()) {
			CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_pairs), pair_i.size() * sizeof(std::uint32_t)));
			CUDA_CHECK(cudaMemcpy(d_pairs, pair_i.data(), pair_i.size() * sizeof(std::uint32_t), cudaMemcpyHostToDevice));
		}
	}

	cudaEvent_t start{};
	cudaEvent_t stop{};
	CUDA_CHECK(cudaEventCreate(&start));
	CUDA_CHECK(cudaEventCreate(&stop));
	CUDA_CHECK(cudaEventRecord(start));

	std::size_t kernel_launches = 0;
	FusedLayers buffer;
	buffer.count = 0;

	for (std::size_t layer_idx = 0; layer_idx < layers.size(); ++layer_idx) {
		const std::size_t stage = layers[layer_idx].k;
		const std::size_t step = layers[layer_idx].j;

		if (trunc) {
			const std::size_t begin = offsets[layer_idx];
			const std::size_t end = offsets[layer_idx + 1];
			const std::size_t count = end - begin;
			if (count == 0) continue;

			const dim3 grid(static_cast<unsigned int>((count + block_size - 1) / block_size));
			const dim3 block(static_cast<unsigned int>(block_size));
			bitonic_layer_trunc<<<grid, block>>>(d_data, d_pairs + begin, count, stage, step);
			CUDA_CHECK(cudaGetLastError());
			kernel_launches++;
			continue;
		}

		if (step <= block_size) {
			buffer.stages[buffer.count] = static_cast<std::uint32_t>(stage);
			buffer.steps[buffer.count] = static_cast<std::uint32_t>(step);
			buffer.count++;
		}

		const bool is_last = (layer_idx == layers.size() - 1);
		const bool next_is_large = (!is_last && layers[layer_idx + 1].j > block_size);
		const bool is_full = (buffer.count == 16);

		if (buffer.count > 0 && (is_last || next_is_large || is_full)) {
			const dim3 grid(static_cast<unsigned int>((full_pairs + block_size - 1) / block_size));
			const dim3 block(static_cast<unsigned int>(block_size));
			const std::size_t smem_size = block_size * 2 * sizeof(T);
			
			bitonic_fused_shared<<<grid, block, smem_size>>>(d_data, full_pairs, buffer);
			CUDA_CHECK(cudaGetLastError());
			kernel_launches++;
			buffer.count = 0;
		}

		if (step > block_size) {
			const dim3 grid(static_cast<unsigned int>((full_pairs + block_size - 1) / block_size));
			const dim3 block(static_cast<unsigned int>(block_size));
			
			bitonic_layer_global_coalesced<<<grid, block>>>(d_data, full_pairs, stage, step);
			CUDA_CHECK(cudaGetLastError());
			kernel_launches++;
		}
	}

	CUDA_CHECK(cudaEventRecord(stop));
	CUDA_CHECK(cudaEventSynchronize(stop));
	float elapsed_ms_f = 0.0f;
	CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms_f, start, stop));

	CUDA_CHECK(cudaMemcpy(data.data(), d_data, n * sizeof(T), cudaMemcpyDeviceToHost));

	CUDA_CHECK(cudaEventDestroy(start));
	CUDA_CHECK(cudaEventDestroy(stop));
	if (d_pairs != nullptr) {
		CUDA_CHECK(cudaFree(d_pairs));
	}
	CUDA_CHECK(cudaFree(d_data));

	return RunStats{static_cast<double>(elapsed_ms_f), kernel_launches, active_comparators, block_size};
}

RunStats run_network_cuda_fp16(std::vector<float>& data, const std::vector<common::bitonic::Layer>& layers,
							   const std::vector<std::vector<unsigned char>>& keep, bool trunc) {
	std::vector<__half> half_data(data.size());
	for (std::size_t i = 0; i < data.size(); ++i) {
		half_data[i] = __float2half(data[i]);
	}

	const RunStats stats = run_network_cuda(half_data, layers, keep, trunc);

	for (std::size_t i = 0; i < data.size(); ++i) {
		data[i] = __half2float(half_data[i]);
	}

	return stats;
}

template RunStats run_network_cuda<std::int32_t>(std::vector<std::int32_t>& data,
											 const std::vector<common::bitonic::Layer>& layers,
											 const std::vector<std::vector<unsigned char>>& keep, bool trunc);
template RunStats run_network_cuda<std::uint32_t>(std::vector<std::uint32_t>& data,
											  const std::vector<common::bitonic::Layer>& layers,
											  const std::vector<std::vector<unsigned char>>& keep, bool trunc);
template RunStats run_network_cuda<float>(std::vector<float>& data, const std::vector<common::bitonic::Layer>& layers,
										  const std::vector<std::vector<unsigned char>>& keep, bool trunc);
template RunStats run_network_cuda<double>(std::vector<double>& data, const std::vector<common::bitonic::Layer>& layers,
										   const std::vector<std::vector<unsigned char>>& keep, bool trunc);
template RunStats run_network_cuda<__half>(std::vector<__half>& data, const std::vector<common::bitonic::Layer>& layers,
										   const std::vector<std::vector<unsigned char>>& keep, bool trunc);

} // namespace gpu::bitonic
