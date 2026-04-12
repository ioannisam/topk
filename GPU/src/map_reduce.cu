#include "../include/algorithm.hpp"

#include <algorithm>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

#include <cuda_runtime.h>

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
								std::size_t* tile_sizes) {
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

	tile_sizes[tid] = heap_size;
}

template <bool WantMax, typename T> std::vector<T> finalize_topk(std::vector<T> aggregated, std::size_t k) {
	if (k == 0 || aggregated.empty()) {
		return {};
	}

	auto cmp = [](const T& lhs, const T& rhs) {
		if constexpr (WantMax) {
			return lhs > rhs;
		}
		return lhs < rhs;
	};

	if (aggregated.size() > k) {
		std::nth_element(aggregated.begin(), aggregated.begin() + static_cast<std::ptrdiff_t>(k), aggregated.end(), cmp);
		aggregated.resize(k);
	}

	std::sort(aggregated.begin(), aggregated.end(), cmp);
	return aggregated;
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
	const std::size_t tiles = std::max<std::size_t>(1, std::min(workers, n));
	const std::size_t block_size = 256;

	T* d_input = nullptr;
	T* d_tile_values = nullptr;
	std::size_t* d_tile_sizes = nullptr;

	CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_input), n * sizeof(T)));
	CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_tile_values), tiles * k * sizeof(T)));
	CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_tile_sizes), tiles * sizeof(std::size_t)));
	CUDA_CHECK(cudaMemcpy(d_input, data.data(), n * sizeof(T), cudaMemcpyHostToDevice));

	cudaEvent_t start{};
	cudaEvent_t stop{};
	CUDA_CHECK(cudaEventCreate(&start));
	CUDA_CHECK(cudaEventCreate(&stop));
	CUDA_CHECK(cudaEventRecord(start));

	const dim3 grid(static_cast<unsigned int>(tiles));
	const dim3 block(static_cast<unsigned int>(block_size));
	map_topk_kernel<WantMax, T><<<grid, block>>>(d_input, n, k, tiles, d_tile_values, d_tile_sizes);
	CUDA_CHECK(cudaGetLastError());
	CUDA_CHECK(cudaEventRecord(stop));
	CUDA_CHECK(cudaEventSynchronize(stop));

	float elapsed_ms_f = 0.0f;
	CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms_f, start, stop));

	std::vector<std::size_t> tile_sizes(tiles);
	std::vector<T> tile_values(tiles * k);
	CUDA_CHECK(cudaMemcpy(tile_sizes.data(), d_tile_sizes, tiles * sizeof(std::size_t), cudaMemcpyDeviceToHost));
	CUDA_CHECK(cudaMemcpy(tile_values.data(), d_tile_values, tiles * k * sizeof(T), cudaMemcpyDeviceToHost));

	std::size_t aggregated_candidates = 0;
	for (std::size_t t = 0; t < tiles; ++t) {
		aggregated_candidates += tile_sizes[t];
	}

	std::vector<T> aggregated;
	aggregated.reserve(aggregated_candidates);
	for (std::size_t t = 0; t < tiles; ++t) {
		const T* tile_ptr = tile_values.data() + t * k;
		aggregated.insert(aggregated.end(), tile_ptr, tile_ptr + tile_sizes[t]);
	}

	CUDA_CHECK(cudaEventDestroy(start));
	CUDA_CHECK(cudaEventDestroy(stop));
	CUDA_CHECK(cudaFree(d_tile_sizes));
	CUDA_CHECK(cudaFree(d_tile_values));
	CUDA_CHECK(cudaFree(d_input));

	if (stats != nullptr) {
		*stats = RunStats{static_cast<double>(elapsed_ms_f), tiles, aggregated_candidates, block_size};
	}

	return finalize_topk<WantMax>(std::move(aggregated), k);
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
