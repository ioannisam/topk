#include "../include/algorithm.hpp"
#include "../include/device_traits.cuh"

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

template <typename T>
struct DeviceBuffer {
	T* ptr = nullptr;
	std::size_t size = 0;

	explicit DeviceBuffer(std::size_t num_elements) : size(num_elements) {
		if (size > 0) {
			CUDA_CHECK(cudaMalloc(&ptr, size * sizeof(T)));
		}
	}

	~DeviceBuffer() {
		if (ptr) {
			cudaFree(ptr);  // Silently ignore errors on destruction
			ptr = nullptr;
		}
	}

	// Deleted copy operations
	DeviceBuffer(const DeviceBuffer&) = delete;
	DeviceBuffer& operator=(const DeviceBuffer&) = delete;

	// Move operations
	DeviceBuffer(DeviceBuffer&& other) noexcept : ptr(other.ptr), size(other.size) {
		other.ptr = nullptr;
		other.size = 0;
	}

	DeviceBuffer& operator=(DeviceBuffer&& other) noexcept {
		if (this != &other) {
			if (ptr) cudaFree(ptr);
			ptr = other.ptr;
			size = other.size;
			other.ptr = nullptr;
			other.size = 0;
		}
		return *this;
	}

	T* get() const { return ptr; }
	T* operator->() const { return ptr; }
	T& operator[](std::size_t idx) const { return ptr[idx]; }
};

struct FusedLayers {
	std::uint32_t stages[16];
	std::uint32_t steps[16];
	int count;
};

__global__ void cast_float_to_half_vec2(const float2* __restrict__ src, __half2* __restrict__ dst, std::size_t num_vecs) {
	const std::size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
	if (tid < num_vecs) {
		dst[tid] = __float22half2_rn(src[tid]);
	}
}

__global__ void cast_float_to_half_scalar_tail(const float* __restrict__ src, __half* __restrict__ dst, std::size_t n) {
	if (n % 2 != 0) {
		dst[n - 1] = __float2half(src[n - 1]);
	}
}

__global__ void cast_half_to_float_vec2(const __half2* __restrict__ src, float2* __restrict__ dst, std::size_t num_vecs) {
	const std::size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
	if (tid < num_vecs) {
		dst[tid] = __half22float2(src[tid]);
	}
}

__global__ void cast_half_to_float_scalar_tail(const __half* __restrict__ src, float* __restrict__ dst, std::size_t n) {
	if (n % 2 != 0) {
		dst[n - 1] = __half2float(src[n - 1]);
	}
}

template <typename T> 
__global__ __launch_bounds__(256, 4) 
void bitonic_fused_shared(T* data, std::size_t total_pairs, FusedLayers layers) {
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
			if ((ascending && gpu::traits::DeviceTraits<T>::gt(a, b)) || 
                (!ascending && gpu::traits::DeviceTraits<T>::lt(a, b))) {
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

	const T min_val = gpu::traits::DeviceTraits<T>::min(a, b); 
	const T max_val = gpu::traits::DeviceTraits<T>::max(a, b); 

	data[i]   = ascending ? min_val : max_val;
	data[ixj] = ascending ? max_val : min_val;
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

	dst[tid] = gpu::traits::DeviceTraits<T>::lt(a, b) ? a : b;
}

std::size_t choose_block_size() {
	cudaDeviceProp prop{};
	CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
	return static_cast<std::size_t>(prop.warpSize * 8);
}

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
			out_final_n = pairs;

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

	DeviceBuffer<T> d_data(n);
	DeviceBuffer<T> d_data_alt(n);

	CUDA_CHECK(cudaMemcpy(d_data.get(), data.data(), n * sizeof(T), cudaMemcpyHostToDevice));

	double elapsed_ms = 0.0;
	std::size_t launches = 0;
	std::size_t comparators = 0;
	std::size_t final_n = n;

	T* final_src =
		execute_network_kernels(d_data.get(), d_data_alt.get(), n, layers, block_size, elapsed_ms, launches, comparators, final_n);

	CUDA_CHECK(cudaMemcpy(data.data(), final_src, final_n * sizeof(T), cudaMemcpyDeviceToHost));

	data.resize(final_n);

	return RunStats{elapsed_ms, launches, comparators, block_size};
}

RunStats run_network_cuda_fp16(std::vector<float>& data, const std::vector<common::bitonic::Layer>& layers) {
	if (data.empty()) {
		return RunStats{0.0, 0, 0, 0};
	}

	const std::size_t n = data.size();
	const std::size_t block_size = choose_block_size();

	DeviceBuffer<float> d_float_data(n);
	DeviceBuffer<__half> d_half_data(n);
	DeviceBuffer<__half> d_half_alt(n);

	CUDA_CHECK(cudaMemcpy(d_float_data.get(), data.data(), n * sizeof(float), cudaMemcpyHostToDevice));

	const std::size_t n_vec = n / 2;
	const dim3 grid_vec((n_vec + block_size - 1) / block_size);
	cast_float_to_half_vec2<<<grid_vec, block_size>>>(
		reinterpret_cast<const float2*>(d_float_data.get()), 
		reinterpret_cast<__half2*>(d_half_data.get()), 
		n_vec
	);
	CUDA_CHECK(cudaGetLastError());

	// Handle odd-sized input
	if (n % 2 != 0) {
		cast_float_to_half_scalar_tail<<<1, 1>>>(d_float_data.get(), d_half_data.get(), n);
		CUDA_CHECK(cudaGetLastError());
	}

	double elapsed_ms = 0.0;
	std::size_t launches = 0;
	std::size_t comparators = 0;
	std::size_t final_n = n;

	__half* final_src = execute_network_kernels(d_half_data.get(), d_half_alt.get(), n, layers, block_size, elapsed_ms, launches,
												comparators, final_n);

	const std::size_t final_n_vec = final_n / 2;
	const dim3 final_grid_vec((final_n_vec + block_size - 1) / block_size);
	cast_half_to_float_vec2<<<final_grid_vec, block_size>>>(
		reinterpret_cast<const __half2*>(final_src), 
		reinterpret_cast<float2*>(d_float_data.get()), 
		final_n_vec
	);
	CUDA_CHECK(cudaGetLastError());

	// Handle odd-sized output
	if (final_n % 2 != 0) {
		cast_half_to_float_scalar_tail<<<1, 1>>>(final_src, d_float_data.get(), final_n);
		CUDA_CHECK(cudaGetLastError());
	}

	CUDA_CHECK(cudaMemcpy(data.data(), d_float_data.get(), final_n * sizeof(float), cudaMemcpyDeviceToHost));
	data.resize(final_n);

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
