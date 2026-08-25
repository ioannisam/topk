#include "../include/algorithm.hpp"
#include "cuda_utils.cuh"
#include "device_traits.cuh"

#include "common/energy.hpp"

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#include <cuda_fp16.h>
#include <cuda_runtime.h>

namespace gpu::map_reduce {

namespace {

using gpu::utils::DeviceBuffer;
using gpu::utils::EventGuard;

constexpr int MAP_BLOCK_SIZE = 256;
constexpr std::size_t MAP_STRATEGY_A_MAX_K = 256;
constexpr int MAP_BLOCK_BUDGET = 128;
constexpr int MAP_MIN_BLOCKS_PER_SM = 2;

inline int map_blocks_per_sm(int occupancy_blocks, int sm_count, std::size_t n, std::size_t k) {
	const std::size_t full_threads = static_cast<std::size_t>(sm_count) * occupancy_blocks * MAP_BLOCK_SIZE;
	const bool heaps_fill = full_threads > 0 && (n / full_threads) >= k;
	if (k > MAP_STRATEGY_A_MAX_K || !heaps_fill) {
		return occupancy_blocks;
	}
	int budget = static_cast<int>(MAP_BLOCK_BUDGET / k);
	if (budget < MAP_MIN_BLOCKS_PER_SM) {
		budget = MAP_MIN_BLOCKS_PER_SM;
	}
	return occupancy_blocks < budget ? occupancy_blocks : budget;
}

template <typename T> __device__ __forceinline__ bool beats_threshold(T candidate, T threshold, bool want_max) {
	return want_max ? gpu::traits::DeviceTraits<T>::gt(candidate, threshold)
					: gpu::traits::DeviceTraits<T>::lt(candidate, threshold);
}

template <typename T> __device__ void sift_down(T* heap, int size, int root, bool want_max) {
	int current = root;
	while (true) {
		int left_child = 2 * current + 1;
		int right_child = 2 * current + 2;
		int extreme = current;

		if (left_child < size && beats_threshold(heap[extreme], heap[left_child], want_max)) {
			extreme = left_child;
		}
		if (right_child < size && beats_threshold(heap[extreme], heap[right_child], want_max)) {
			extreme = right_child;
		}

		if (extreme == current) {
			break;
		}

		T temp = heap[current];
		heap[current] = heap[extreme];
		heap[extreme] = temp;
		current = extreme;
	}
}

template <typename T>
__global__ __launch_bounds__(256, 4) void topk_map_kernel(
	const T* __restrict__ input,
	std::size_t n,
	int k,
	bool want_max,
	T* __restrict__ thread_workspaces,
	int* __restrict__ thread_counts,
	T* __restrict__ block_outputs
) {
	if (k <= 0) {
		return;
	}

	// Fetch sentinel strictly on the device!
	const T sentinel = gpu::traits::DeviceTraits<T>::sentinel(want_max);

	const std::size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
	const std::size_t stride = blockDim.x * gridDim.x;

	T* local_heap = thread_workspaces + (tid * k);
	int current_size = 0;
	T thresh = sentinel;

	for (std::size_t i = tid; i < n; i += stride) {
		T val = input[i];

		if (current_size < k) {
			local_heap[current_size] = val;
			current_size++;
			if (current_size == k) {
				for (int j = k / 2 - 1; j >= 0; j--) {
					sift_down(local_heap, k, j, want_max);
				}
				thresh = local_heap[0];
			}
		} else if (beats_threshold(val, thresh, want_max)) {
			local_heap[0] = val;
			sift_down(local_heap, k, 0, want_max);
			thresh = local_heap[0];
		}
	}

	thread_counts[tid] = current_size;
	__syncthreads();

	// PHASE 2: HYBRID REDUCE
	if (k <= 256) {
		// STRATEGY A: Parallel Tree Reduction
		for (int step = 1; step < blockDim.x; step *= 2) {
			int index = 2 * step * threadIdx.x;

			if (index < blockDim.x) {
				const std::size_t my_global_tid = blockIdx.x * blockDim.x + index;
				const std::size_t other_global_tid = blockIdx.x * blockDim.x + index + step;

				T* my_heap = thread_workspaces + (my_global_tid * k);
				T* other_heap = thread_workspaces + (other_global_tid * k);

				int c_my = thread_counts[my_global_tid];
				const int c_other = thread_counts[other_global_tid];
				T my_thresh = (c_my == k) ? my_heap[0] : sentinel;

				for (int j = 0; j < c_other; j++) {
					const T val = other_heap[j];

					if (c_my < k) {
						my_heap[c_my] = val;
						c_my++;
						if (c_my == k) {
							for (int h = k / 2 - 1; h >= 0; h--) {
								sift_down(my_heap, k, h, want_max);
							}
							my_thresh = my_heap[0];
						}
					} else if (beats_threshold(val, my_thresh, want_max)) {
						my_heap[0] = val;
						sift_down(my_heap, k, 0, want_max);
						my_thresh = my_heap[0];
					}
				}
				thread_counts[my_global_tid] = c_my;
			}
			__syncthreads();
		}

		if (threadIdx.x == 0) {
			const std::size_t my_global_tid = blockIdx.x * blockDim.x;
			T* final_block_heap = thread_workspaces + (my_global_tid * k);
			int final_count = thread_counts[my_global_tid];

			for (int j = 0; j < final_count; j++) {
				block_outputs[blockIdx.x * k + j] = final_block_heap[j];
			}
			for (int j = final_count; j < k; j++) {
				block_outputs[blockIdx.x * k + j] = sentinel;
			}
		}

	} else {
		// STRATEGY B: Sequential Merge in Global Memory
		if (threadIdx.x == 0) {
			const std::size_t block_start_tid = blockIdx.x * blockDim.x;

			T* block_heap = thread_workspaces + (block_start_tid * k);
			int block_heap_size = thread_counts[block_start_tid];
			T block_thresh = (block_heap_size == k) ? block_heap[0] : sentinel;

			for (int t = 1; t < blockDim.x; t++) {
				const std::size_t t_global_tid = block_start_tid + t;
				T* t_heap = thread_workspaces + (t_global_tid * k);
				const int c_t = thread_counts[t_global_tid];

				for (int j = 0; j < c_t; j++) {
					const T val = t_heap[j];

					if (block_heap_size < k) {
						block_heap[block_heap_size] = val;
						block_heap_size++;
						if (block_heap_size == k) {
							for (int h = k / 2 - 1; h >= 0; h--) {
								sift_down(block_heap, k, h, want_max);
							}
							block_thresh = block_heap[0];
						}
					} else if (beats_threshold(val, block_thresh, want_max)) {
						block_heap[0] = val;
						sift_down(block_heap, k, 0, want_max);
						block_thresh = block_heap[0];
					}
				}
			}

			for (int j = 0; j < block_heap_size; j++) {
				block_outputs[blockIdx.x * k + j] = block_heap[j];
			}
			for (int j = block_heap_size; j < k; j++) {
				block_outputs[blockIdx.x * k + j] = sentinel;
			}
		}
	}
}

} // namespace

template <typename T>
std::size_t run_topk(const T* input, std::size_t n, std::size_t k, bool want_max, T* out, RunStats* stats) {
	using D = typename gpu::traits::DeviceType<T>::type;

	if (k == 0 || n == 0) {
		return 0;
	}

	const int block_size = MAP_BLOCK_SIZE;
	const size_t shared_mem_size = 0;

	struct DeviceInfo {
		int multiprocessor_count;
		int max_active_blocks;
	};
	static const DeviceInfo device_info = [&]() {
		cudaDeviceProp prop{};
		CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
		int blocks = 0;
		CUDA_CHECK(
			cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks, topk_map_kernel<D>, block_size, shared_mem_size)
		);
		return DeviceInfo{prop.multiProcessorCount, blocks};
	}();

	int num_blocks = map_blocks_per_sm(device_info.max_active_blocks, device_info.multiprocessor_count, n, k);
	int grid_size = device_info.multiprocessor_count * num_blocks;

	const size_t max_workspace_bytes = 1024ULL * 1024ULL * 512ULL;
	const size_t bytes_per_thread = k * sizeof(D);

	if (bytes_per_thread > max_workspace_bytes / block_size) {
		throw std::invalid_argument(
			"k is too large for thread-local heap MapReduce. Reduce k or use a different algorithm."
		);
	}

	const int max_allowed_threads = std::max<int>(block_size, static_cast<int>(max_workspace_bytes / bytes_per_thread));
	const int max_allowed_blocks = max_allowed_threads / block_size;

	if (grid_size > max_allowed_blocks) {
		grid_size = max_allowed_blocks;
	}

	const int total_threads = grid_size * block_size;

	DeviceBuffer<D> d_input(n);
	DeviceBuffer<D> d_thread_workspaces(static_cast<std::size_t>(total_threads) * k);
	DeviceBuffer<int> d_thread_counts(total_threads);
	DeviceBuffer<D> d_block_outputs(static_cast<std::size_t>(grid_size) * k);

	CUDA_CHECK(cudaMemcpy(d_input.get(), input, n * sizeof(D), cudaMemcpyHostToDevice));

	EventGuard start_guard;
	EventGuard stop_guard;
	const cudaEvent_t start = start_guard.get();
	const cudaEvent_t stop = stop_guard.get();
	common::energy::Scope energy_scope(common::energy::Channel::Algo);
	CUDA_CHECK(cudaEventRecord(start));

	topk_map_kernel<<<grid_size, block_size, shared_mem_size>>>(
		d_input.get(),
		n,
		static_cast<int>(k),
		want_max,
		d_thread_workspaces.get(),
		d_thread_counts.get(),
		d_block_outputs.get()
	);
	CUDA_CHECK(cudaGetLastError());

	CUDA_CHECK(cudaEventRecord(stop));
	CUDA_CHECK(cudaEventSynchronize(stop));
	energy_scope.close();

	float elapsed_ms;
	CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, start, stop));

	std::vector<D> block_results(static_cast<std::size_t>(grid_size) * k);
	CUDA_CHECK(cudaMemcpy(
		block_results.data(),
		d_block_outputs.get(),
		static_cast<std::size_t>(grid_size) * k * sizeof(D),
		cudaMemcpyDeviceToHost
	));

	if (stats != nullptr) {
		stats->elapsed_ms = elapsed_ms;
		stats->tiles_used = grid_size;
		stats->aggregated_candidates = grid_size * k;
		stats->block_size = block_size;
		const std::size_t threads = static_cast<std::size_t>(total_threads);
		const std::size_t heap_fill = std::min(k, (n + threads - 1) / threads);
		stats->bytes_moved =
			(static_cast<double>(n) + 2.0 * static_cast<double>(threads) * static_cast<double>(heap_fill) +
			 static_cast<double>(grid_size) * static_cast<double>(k)) *
			sizeof(D);
	}

	if (block_results.size() > k) {
		auto mid = block_results.begin() + static_cast<std::ptrdiff_t>(k);
		if (want_max) {
			std::nth_element(block_results.begin(), mid, block_results.end(), std::greater<D>());
			block_results.resize(k);
			std::sort(block_results.begin(), block_results.end(), std::greater<D>());
		} else {
			std::nth_element(block_results.begin(), mid, block_results.end(), std::less<D>());
			block_results.resize(k);
			std::sort(block_results.begin(), block_results.end(), std::less<D>());
		}
	} else {
		if (want_max) {
			std::sort(block_results.begin(), block_results.end(), std::greater<D>());
		} else {
			std::sort(block_results.begin(), block_results.end(), std::less<D>());
		}
	}

	std::memcpy(out, block_results.data(), block_results.size() * sizeof(D));
	return block_results.size();
}

// clang-format off
template std::size_t run_topk<float>(
	const float*, std::size_t, std::size_t, bool, float*, RunStats*
);
template std::size_t run_topk<std::int32_t>(
	const std::int32_t*, std::size_t, std::size_t, bool, std::int32_t*, RunStats*
);
template std::size_t run_topk<std::uint32_t>(
	const std::uint32_t*, std::size_t, std::size_t, bool, std::uint32_t*, RunStats*
);
template std::size_t run_topk<double>(
	const double*, std::size_t, std::size_t, bool, double*, RunStats*
);
#if defined(__FLT16_MANT_DIG__)
template std::size_t run_topk<_Float16>(
	const _Float16*, std::size_t, std::size_t, bool, _Float16*, RunStats*
);
#endif
// clang-format on

} // namespace gpu::map_reduce
