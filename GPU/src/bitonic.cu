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

constexpr std::size_t BITONIC_BLOCK_SIZE = 256;
constexpr std::size_t BITONIC_TILE_LARGE = 8192;		  // working set > L2 (DRAM bound)
constexpr std::size_t BITONIC_TILE_SMALL = 4096;		  // working set <= L2 (occupancy bound)
constexpr std::size_t BITONIC_MAX_TILE_BYTES = 48 * 1024; // stay within default smem budget (no opt-in)
constexpr int FUSED_LAYER_CAP = 128;
constexpr int MULTISTEP_MAX_BITS = 5;

#define CUDA_CHECK(expr)                                                                                               \
	do {                                                                                                               \
		cudaError_t _err = (expr);                                                                                     \
		if (_err != cudaSuccess) {                                                                                     \
			throw std::runtime_error(std::string("CUDA error: ") + cudaGetErrorString(_err));                          \
		}                                                                                                              \
	} while (false)

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

template <typename T> struct DeviceBuffer {
	T* ptr = nullptr;
	std::size_t size = 0;

	explicit DeviceBuffer(std::size_t num_elements) : size(num_elements) {
		if (size > 0) {
			CUDA_CHECK(cudaMalloc(&ptr, size * sizeof(T)));
		}
	}

	~DeviceBuffer() {
		cudaFree(ptr);
		ptr = nullptr;
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
			cudaFree(ptr);
			ptr = other.ptr;
			size = other.size;
			other.ptr = nullptr;
			other.size = 0;
		}
		return *this;
	}

	T* get() const {
		return ptr;
	}
	T* operator->() const {
		return ptr;
	}
	T& operator[](std::size_t idx) const {
		return ptr[idx];
	}
};

struct FusedLayers {
	std::uint32_t stages[FUSED_LAYER_CAP];
	std::uint32_t steps[FUSED_LAYER_CAP];
	int count;
};

__global__ void cast_float_to_half_vec2(const float2* __restrict__ src, __half2* __restrict__ dst,
										std::size_t num_vecs) {
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

__global__ void cast_half_to_float_vec2(const __half2* __restrict__ src, float2* __restrict__ dst,
										std::size_t num_vecs) {
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

template <typename T, int TPB>
__global__ __launch_bounds__(TPB) void bitonic_fused_wide(T* __restrict__ data, std::size_t tile_elems,
														  FusedLayers layers) {
	extern __shared__ char smem[];
	T* s_data = reinterpret_cast<T*>(smem);

	const std::size_t block_offset = static_cast<std::size_t>(blockIdx.x) * tile_elems;
	const std::size_t comparisons = tile_elems >> 1;

	for (std::size_t k = threadIdx.x; k < tile_elems; k += TPB) {
		s_data[k] = data[block_offset + k];
	}
	__syncthreads();

	for (int l = 0; l < layers.count; ++l) {
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
	for (int p = 0; p < G; ++p) {
		reg[p] = data[base + static_cast<std::size_t>(p) * j_bot];
	}

#pragma unroll
	for (int m = 0; m < GROUP_BITS; ++m) {
		const int local_stride = 1 << (GROUP_BITS - 1 - m);
#pragma unroll
		for (int p = 0; p < G; ++p) {
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
	for (int p = 0; p < G; ++p) {
		data[base + static_cast<std::size_t>(p) * j_bot] = reg[p];
	}
}

__global__ void bitonic_layer_global_coalesced_half2(__half2* data, std::size_t total_vec_pairs, std::size_t stage_vec,
													 std::size_t step_vec) {
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

template <typename T>
T* execute_network_kernels(T* current_src, T* current_dst, std::size_t n,
						   const std::vector<common::bitonic::Layer>& layers, std::size_t block_size,
						   double& out_elapsed_ms, std::size_t& out_launches, std::size_t& out_comparators,
						   std::size_t& out_final_n) {

	cudaStream_t stream;
	CUDA_CHECK(cudaStreamCreate(&stream));

	cudaEvent_t start{};
	cudaEvent_t stop{};
	CUDA_CHECK(cudaEventCreate(&start));
	CUDA_CHECK(cudaEventCreate(&stop));

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
					reinterpret_cast<__half2*>(current_src), vec_pairs, large_stage / 2, s / 2);
				out_launches++;
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
				idx += static_cast<std::size_t>(t);
			}
		}
		large_steps.clear();
	};

	for (std::size_t layer_idx = 0; layer_idx < layers.size(); ++layer_idx) {
		const auto& layer = layers[layer_idx];
		const std::size_t stage = layer.k;
		const std::size_t step = layer.j;
		const std::size_t pairs = layer.active_n / 2;
		out_comparators += pairs;

		if (layer.type == common::bitonic::LayerType::Truncate) {
			out_final_n = pairs;

			flush_large();
			flush(layer.active_n);

			const dim3 grid(static_cast<unsigned int>((pairs + block_size - 1) / block_size));
			bitonic_layer_truncate_kernel<<<grid, block_size, 0, stream>>>(current_src, current_dst, pairs, step);
			out_launches++;

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
			(!is_last && !next_is_trunc && !fuses(layers[layer_idx + 1].active_n, layers[layer_idx + 1].j));
		const bool is_full = (buffer.count == FUSED_LAYER_CAP);

		if (buffer.count > 0 && (is_last || next_is_large || next_is_trunc || is_full)) {
			flush(layer.active_n);
		}
	}
	flush_large();

	cudaGraph_t graph;
	CUDA_CHECK(cudaStreamEndCapture(stream, &graph));

	cudaGraphExec_t instance;
#if __CUDACC_VER_MAJOR__ >= 12
	CUDA_CHECK(cudaGraphInstantiate(&instance, graph, 0));
#else
	CUDA_CHECK(cudaGraphInstantiate(&instance, graph, nullptr, nullptr, 0));
#endif

	CUDA_CHECK(cudaEventRecord(start, stream));
	CUDA_CHECK(cudaGraphLaunch(instance, stream));
	CUDA_CHECK(cudaEventRecord(stop, stream));

	CUDA_CHECK(cudaEventSynchronize(stop));

	float elapsed_ms_f = 0.0f;
	CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms_f, start, stop));
	out_elapsed_ms = static_cast<double>(elapsed_ms_f);

	CUDA_CHECK(cudaGraphExecDestroy(instance));
	CUDA_CHECK(cudaGraphDestroy(graph));
	CUDA_CHECK(cudaEventDestroy(start));
	CUDA_CHECK(cudaEventDestroy(stop));
	CUDA_CHECK(cudaStreamDestroy(stream));

	return current_src;
}

} // namespace

std::string query_device_name() {
	cudaDeviceProp prop{};
	CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
	return std::string(prop.name);
}

template <typename T>
RunStats run_network_cuda(std::vector<T>& data, const std::vector<common::bitonic::Layer>& layers) {
	if (data.empty()) {
		return RunStats{0.0, 0, 0, 0};
	}

	const std::size_t n = data.size();
	const std::size_t block_size = BITONIC_BLOCK_SIZE;

	DeviceBuffer<T> d_data(n);
	DeviceBuffer<T> d_data_alt(n);

	CUDA_CHECK(cudaMemcpy(d_data.get(), data.data(), n * sizeof(T), cudaMemcpyHostToDevice));

	double elapsed_ms = 0.0;
	std::size_t launches = 0;
	std::size_t comparators = 0;
	std::size_t final_n = n;

	T* final_src = execute_network_kernels(d_data.get(), d_data_alt.get(), n, layers, block_size, elapsed_ms, launches,
										   comparators, final_n);

	CUDA_CHECK(cudaMemcpy(data.data(), final_src, final_n * sizeof(T), cudaMemcpyDeviceToHost));

	data.resize(final_n);

	return RunStats{elapsed_ms, launches, comparators, block_size};
}

RunStats run_network_cuda_fp16(std::vector<float>& data, const std::vector<common::bitonic::Layer>& layers) {
	if (data.empty()) {
		return RunStats{0.0, 0, 0, 0};
	}

	const std::size_t n = data.size();
	const std::size_t block_size = BITONIC_BLOCK_SIZE;

	DeviceBuffer<float> d_float_data(n);
	DeviceBuffer<__half> d_half_data(n);
	DeviceBuffer<__half> d_half_alt(n);

	CUDA_CHECK(cudaMemcpy(d_float_data.get(), data.data(), n * sizeof(float), cudaMemcpyHostToDevice));

	const std::size_t n_vec = n / 2;
	const dim3 grid_vec((n_vec + block_size - 1) / block_size);
	cast_float_to_half_vec2<<<grid_vec, block_size>>>(reinterpret_cast<const float2*>(d_float_data.get()),
													  reinterpret_cast<__half2*>(d_half_data.get()), n_vec);
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

	__half* final_src = execute_network_kernels(d_half_data.get(), d_half_alt.get(), n, layers, block_size, elapsed_ms,
												launches, comparators, final_n);

	const std::size_t final_n_vec = final_n / 2;
	const dim3 final_grid_vec((final_n_vec + block_size - 1) / block_size);
	cast_half_to_float_vec2<<<final_grid_vec, block_size>>>(reinterpret_cast<const __half2*>(final_src),
															reinterpret_cast<float2*>(d_float_data.get()), final_n_vec);
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
