#include "../include/algorithm.hpp"

#include <algorithm>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

#include <cuda_runtime.h>
#include <thrust/device_ptr.h>
#include <thrust/execution_policy.h>
#include <thrust/sort.h>

namespace gpu::map_reduce {

namespace {

#define CUDA_CHECK(expr)                                                                                               \
	do {                                                                                                               \
		cudaError_t _err = (expr);                                                                                     \
		if (_err != cudaSuccess) {                                                                                     \
			throw std::runtime_error(std::string("CUDA error: ") + cudaGetErrorString(_err));                          \
		}                                                                                                              \
	} while (false)

template <bool WantMax, typename T> __device__ __forceinline__ bool candidate(T value, T threshold) {
	if constexpr (WantMax) {
		return value > threshold;
	}
	return value < threshold;
}

template <bool WantMax, typename T> __host__ __forceinline__ T worst_value() {
	if constexpr (WantMax) {
		return std::numeric_limits<T>::lowest();
	}
	return std::numeric_limits<T>::max();
}

template <bool WantMax, typename T> struct DeviceOrder {
	__host__ __device__ bool operator()(const T& lhs, const T& rhs) const {
		if constexpr (WantMax) {
			return lhs > rhs;
		}
		return lhs < rhs;
	}
};

template <bool WantMax, typename T> __device__ __forceinline__ bool heap_less(T lhs, T rhs) {
	if constexpr (WantMax) {
		return lhs > rhs;
	}
	return lhs < rhs;
}

template <bool WantMax, typename T> __device__ void sift_down(T* heap, std::size_t size, std::size_t i) {
	while (true) {
		std::size_t best = i;
		const std::size_t left = 2 * i + 1;
		const std::size_t right = 2 * i + 2;

		if (left < size && heap_less<WantMax>(heap[left], heap[best])) {
			best = left;
		}
		if (right < size && heap_less<WantMax>(heap[right], heap[best])) {
			best = right;
		}

		if (best == i) {
			break;
		}

		const T tmp = heap[i];
		heap[i] = heap[best];
		heap[best] = tmp;
		i = best;
	}
}

template <bool WantMax, typename T> __device__ void make_heap(T* heap, std::size_t size) {
	if (size < 2) {
		return;
	}
	for (std::size_t i = size / 2; i-- > 0;) {
		sift_down<WantMax>(heap, size, i);
	}
}

template <bool WantMax, typename T>
__global__ void map_topk_kernel(const T* input, std::size_t n, std::size_t k, std::size_t tiles, T* tile_values,
								T pad_value) {
	const std::size_t tid = static_cast<std::size_t>(blockIdx.x);
	if (tid >= tiles) {
		return;
	}

	if (threadIdx.x != 0) {
		return;
	}

	const std::size_t begin = (n * tid) / tiles;
	const std::size_t end = (n * (tid + 1)) / tiles;
	T* heap = tile_values + tid * k;
	std::size_t heap_size = 0;

	for (std::size_t i = begin; i < end; ++i) {
		const T value = input[i];
		if (heap_size < k) {
			heap[heap_size++] = value;
			if (heap_size == k) {
				make_heap<WantMax>(heap, heap_size);
			}
			continue;
		}

		const T threshold = heap[0];
		if (!candidate<WantMax>(value, threshold)) {
			continue;
		}

		heap[0] = value;
		sift_down<WantMax>(heap, heap_size, 0);
	}

	for (std::size_t i = heap_size; i < k; ++i) {
		heap[i] = pad_value;
	}
}

template <bool WantMax, typename T>
std::vector<T> run_topk_impl(const std::vector<T>& data, std::size_t k, std::size_t workers, RunStats* stats) {
	const std::size_t n = data.size();
	if (k == 0 || n == 0) {
		if (stats != nullptr) {
			*stats = RunStats{0.0, 0, 0, 0};
		}
		return {};
	}

	k = std::min(k, n);
	const std::size_t sm_count = static_cast<std::size_t>(gpu::bitonic::query_device_sm_count());
	const std::size_t preferred_tiles = std::max<std::size_t>(workers, sm_count * 8);
	const std::size_t tiles = std::max<std::size_t>(1, std::min(preferred_tiles, n));
	const std::size_t block_size = 256;

	T* d_input = nullptr;
	T* d_tile_values = nullptr;

	CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_input), n * sizeof(T)));
	CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_tile_values), tiles * k * sizeof(T)));
	CUDA_CHECK(cudaMemcpy(d_input, data.data(), n * sizeof(T), cudaMemcpyHostToDevice));

	cudaEvent_t start{};
	cudaEvent_t stop{};
	CUDA_CHECK(cudaEventCreate(&start));
	CUDA_CHECK(cudaEventCreate(&stop));
	CUDA_CHECK(cudaEventRecord(start));

	const dim3 grid(static_cast<unsigned int>(tiles));
	const dim3 block(static_cast<unsigned int>(block_size));
	map_topk_kernel<WantMax, T><<<grid, block>>>(d_input, n, k, tiles, d_tile_values, worst_value<WantMax, T>());
	CUDA_CHECK(cudaGetLastError());

	auto d_begin = thrust::device_pointer_cast(d_tile_values);
	auto d_end = d_begin + static_cast<std::ptrdiff_t>(tiles * k);
	DeviceOrder<WantMax, T> order{};
	thrust::sort(thrust::device, d_begin, d_end, order);

	CUDA_CHECK(cudaEventRecord(stop));
	CUDA_CHECK(cudaEventSynchronize(stop));

	float elapsed_ms_f = 0.0f;
	CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms_f, start, stop));

	std::vector<T> output(k);
	CUDA_CHECK(cudaMemcpy(output.data(), d_tile_values, k * sizeof(T), cudaMemcpyDeviceToHost));

	std::size_t aggregated_candidates = 0;
	for (std::size_t t = 0; t < tiles; ++t) {
		const std::size_t begin = (n * t) / tiles;
		const std::size_t end = (n * (t + 1)) / tiles;
		aggregated_candidates += std::min<std::size_t>(k, end - begin);
	}

	CUDA_CHECK(cudaEventDestroy(start));
	CUDA_CHECK(cudaEventDestroy(stop));
	CUDA_CHECK(cudaFree(d_tile_values));
	CUDA_CHECK(cudaFree(d_input));

	if (stats != nullptr) {
		*stats = RunStats{static_cast<double>(elapsed_ms_f), tiles, aggregated_candidates, block_size};
	}

	return output;
}

} // namespace

template <typename T>
std::vector<T> run_topk(const std::vector<T>& data, std::size_t k, bool want_max, std::size_t workers,
						RunStats* stats) {
	if (want_max) {
		return run_topk_impl<true>(data, k, workers, stats);
	}
	return run_topk_impl<false>(data, k, workers, stats);
}

template std::vector<std::int32_t> run_topk<std::int32_t>(const std::vector<std::int32_t>& data, std::size_t k,
										   bool want_max, std::size_t workers, RunStats* stats);
template std::vector<std::uint32_t> run_topk<std::uint32_t>(const std::vector<std::uint32_t>& data, std::size_t k,
											bool want_max, std::size_t workers, RunStats* stats);
template std::vector<float> run_topk<float>(const std::vector<float>& data, std::size_t k, bool want_max,
										std::size_t workers, RunStats* stats);
template std::vector<double> run_topk<double>(const std::vector<double>& data, std::size_t k, bool want_max,
										  std::size_t workers, RunStats* stats);

} // namespace gpu::map_reduce
