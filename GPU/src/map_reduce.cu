#include "../include/algorithm.hpp"
#include "../include/device_traits.cuh"

#include <algorithm>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#include <cuda_fp16.h>
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
			cudaFree(ptr); 
			ptr = nullptr;
		}
	}

	DeviceBuffer(const DeviceBuffer&) = delete;
	DeviceBuffer& operator=(const DeviceBuffer&) = delete;

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
__global__ 
__launch_bounds__(256, 4) 
void topk_map_kernel(const T* __restrict__ input, std::size_t n, int k, bool want_max,
								T* __restrict__ thread_workspaces, int* __restrict__ thread_counts,
								T* __restrict__ block_outputs) {
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
				for (int j = k / 2 - 1; j >= 0; --j) {
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

				for (int j = 0; j < c_other; ++j) {
					const T val = other_heap[j];

					if (c_my < k) {
						my_heap[c_my] = val;
						c_my++;
						if (c_my == k) {
							for (int h = k / 2 - 1; h >= 0; --h)
								sift_down(my_heap, k, h, want_max);
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

			for (int j = 0; j < final_count; ++j) {
				block_outputs[blockIdx.x * k + j] = final_block_heap[j];
			}
			for (int j = final_count; j < k; ++j) {
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

			for (int t = 1; t < blockDim.x; ++t) {
				const std::size_t t_global_tid = block_start_tid + t;
				T* t_heap = thread_workspaces + (t_global_tid * k);
				const int c_t = thread_counts[t_global_tid];

				for (int j = 0; j < c_t; ++j) {
					const T val = t_heap[j];

					if (block_heap_size < k) {
						block_heap[block_heap_size] = val;
						block_heap_size++;
						if (block_heap_size == k) {
							for (int h = k / 2 - 1; h >= 0; --h)
								sift_down(block_heap, k, h, want_max);
							block_thresh = block_heap[0];
						}
					} else if (beats_threshold(val, block_thresh, want_max)) {
						block_heap[0] = val;
						sift_down(block_heap, k, 0, want_max);
						block_thresh = block_heap[0];
					}
				}
			}

			for (int j = 0; j < block_heap_size; ++j) {
				block_outputs[blockIdx.x * k + j] = block_heap[j];
			}
			for (int j = block_heap_size; j < k; ++j) {
				block_outputs[blockIdx.x * k + j] = sentinel;
			}
		}
	}
}

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

} // namespace

template <typename T>
std::vector<T> run_topk(const std::vector<T>& input, std::size_t k, bool want_max, std::size_t ex_threads,
						RunStats* stats) {
	const std::size_t n = input.size();

	if (k == 0 || input.empty()) {
		return {};
	}

	cudaDeviceProp prop{};
	CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
	const int block_size = 256;

	const size_t shared_mem_size = 0;

	int num_blocks;
	CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&num_blocks, topk_map_kernel<T>, block_size, shared_mem_size));
	num_blocks = map_blocks_per_sm(num_blocks, prop.multiProcessorCount, n, k);
	int grid_size = prop.multiProcessorCount * num_blocks;

	const size_t max_workspace_bytes = 1024ULL * 1024ULL * 512ULL;
	const size_t bytes_per_thread = k * sizeof(T);

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

	DeviceBuffer<T> d_input(n);
	DeviceBuffer<T> d_thread_workspaces(static_cast<std::size_t>(total_threads) * k);
	DeviceBuffer<int> d_thread_counts(total_threads);
	DeviceBuffer<T> d_block_outputs(static_cast<std::size_t>(grid_size) * k);

	CUDA_CHECK(cudaMemcpy(d_input.get(), input.data(), n * sizeof(T), cudaMemcpyHostToDevice));

	cudaEvent_t start, stop;
	CUDA_CHECK(cudaEventCreate(&start));
	CUDA_CHECK(cudaEventCreate(&stop));
	CUDA_CHECK(cudaEventRecord(start));

	topk_map_kernel<<<grid_size, block_size, shared_mem_size>>>(
		d_input.get(), n, static_cast<int>(k), want_max, d_thread_workspaces.get(), d_thread_counts.get(), d_block_outputs.get());

	CUDA_CHECK(cudaEventRecord(stop));
	CUDA_CHECK(cudaEventSynchronize(stop));

	float elapsed_ms;
	CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, start, stop));

	std::vector<T> block_results(static_cast<std::size_t>(grid_size) * k);
	CUDA_CHECK(cudaMemcpy(block_results.data(), d_block_outputs.get(), static_cast<std::size_t>(grid_size) * k * sizeof(T), cudaMemcpyDeviceToHost));

	CUDA_CHECK(cudaEventDestroy(start));
	CUDA_CHECK(cudaEventDestroy(stop));

	if (stats != nullptr) {
		stats->elapsed_ms = elapsed_ms;
		stats->tiles_used = grid_size;
		stats->aggregated_candidates = grid_size * k;
		stats->block_size = block_size;
	}

	if (block_results.size() > k) {
		auto mid = block_results.begin() + static_cast<std::ptrdiff_t>(k);
		if (want_max) {
			std::nth_element(block_results.begin(), mid, block_results.end(), std::greater<T>());
			block_results.resize(k);
			std::sort(block_results.begin(), block_results.end(), std::greater<T>());
		} else {
			std::nth_element(block_results.begin(), mid, block_results.end(), std::less<T>());
			block_results.resize(k);
			std::sort(block_results.begin(), block_results.end(), std::less<T>());
		}
	} else {
		if (want_max) {
			std::sort(block_results.begin(), block_results.end(), std::greater<T>());
		} else {
			std::sort(block_results.begin(), block_results.end(), std::less<T>());
		}
	}

	return block_results;
}

std::vector<float> run_topk_fp16(const std::vector<float>& input, std::size_t k, bool want_max, std::size_t ex_threads,
                                 RunStats* stats) {
	const std::size_t n = input.size();
	if (k == 0 || input.empty()) {
		return {};
	}

	cudaDeviceProp prop{};
	CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
	const int block_size = 256;
	const size_t shared_mem_size = 0;

	int num_blocks;
	CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&num_blocks, topk_map_kernel<__half>, block_size, shared_mem_size));
	num_blocks = map_blocks_per_sm(num_blocks, prop.multiProcessorCount, n, k);
	int grid_size = prop.multiProcessorCount * num_blocks;

	const size_t max_workspace_bytes = 1024ULL * 1024ULL * 512ULL;
	const size_t bytes_per_thread = k * sizeof(__half);

	if (bytes_per_thread > max_workspace_bytes / block_size) {
		throw std::invalid_argument(
			"k is too large for thread-local heap MapReduce. Reduce k or use a different algorithm."
		);
	}

	const int max_allowed_threads = std::max<int>(block_size, static_cast<int>(max_workspace_bytes / bytes_per_thread));
	const int max_allowed_blocks = max_allowed_threads / block_size;

	if (grid_size > max_allowed_blocks) grid_size = max_allowed_blocks;
	const int total_threads = grid_size * block_size;

	DeviceBuffer<float> d_input_float(n);
	DeviceBuffer<__half> d_input_half(n);
	DeviceBuffer<__half> d_thread_workspaces(static_cast<std::size_t>(total_threads) * k);
	DeviceBuffer<int> d_thread_counts(total_threads);
	DeviceBuffer<__half> d_block_outputs(static_cast<std::size_t>(grid_size) * k);

	CUDA_CHECK(cudaMemcpy(d_input_float.get(), input.data(), n * sizeof(float), cudaMemcpyHostToDevice));

	const std::size_t n_vec = n / 2;
	if (n_vec > 0) {
		const dim3 grid_vec((n_vec + block_size - 1) / block_size);
		cast_float_to_half_vec2<<<grid_vec, block_size>>>(
			reinterpret_cast<const float2*>(d_input_float.get()), 
			reinterpret_cast<__half2*>(d_input_half.get()), 
			n_vec
		);
		CUDA_CHECK(cudaGetLastError());
	}
	if (n % 2 != 0) {
		cast_float_to_half_scalar_tail<<<1, 1>>>(d_input_float.get(), d_input_half.get(), n);
		CUDA_CHECK(cudaGetLastError());
	}

	cudaEvent_t start, stop;
	CUDA_CHECK(cudaEventCreate(&start));
	CUDA_CHECK(cudaEventCreate(&stop));
	CUDA_CHECK(cudaEventRecord(start));

	topk_map_kernel<<<grid_size, block_size, shared_mem_size>>>(
		d_input_half.get(), n, static_cast<int>(k), want_max, d_thread_workspaces.get(), d_thread_counts.get(), d_block_outputs.get());

	CUDA_CHECK(cudaEventRecord(stop));
	CUDA_CHECK(cudaEventSynchronize(stop));

	float elapsed_ms;
	CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, start, stop));

	std::vector<__half> block_results_half(static_cast<std::size_t>(grid_size) * k);
	CUDA_CHECK(cudaMemcpy(block_results_half.data(), d_block_outputs.get(), static_cast<std::size_t>(grid_size) * k * sizeof(__half), cudaMemcpyDeviceToHost));

	CUDA_CHECK(cudaEventDestroy(start));
	CUDA_CHECK(cudaEventDestroy(stop));

	if (stats != nullptr) {
		stats->elapsed_ms = elapsed_ms;
		stats->tiles_used = grid_size;
		stats->aggregated_candidates = grid_size * k;
		stats->block_size = block_size;
	}

	std::vector<float> block_results(block_results_half.size());
	for (size_t i = 0; i < block_results_half.size(); ++i) {
		block_results[i] = __half2float(block_results_half[i]);
	}

	if (block_results.size() > k) {
		auto mid = block_results.begin() + static_cast<std::ptrdiff_t>(k);
		if (want_max) {
			std::nth_element(block_results.begin(), mid, block_results.end(), std::greater<float>());
			block_results.resize(k);
			std::sort(block_results.begin(), block_results.end(), std::greater<float>());
		} else {
			std::nth_element(block_results.begin(), mid, block_results.end(), std::less<float>());
			block_results.resize(k);
			std::sort(block_results.begin(), block_results.end(), std::less<float>());
		}
	} else {
		if (want_max) {
			std::sort(block_results.begin(), block_results.end(), std::greater<float>());
		} else {
			std::sort(block_results.begin(), block_results.end(), std::less<float>());
		}
	}

	return block_results;
}

template std::vector<float> run_topk<float>(const std::vector<float>&, std::size_t, bool, std::size_t, RunStats*);
template std::vector<std::int32_t> run_topk<std::int32_t>(const std::vector<std::int32_t>&, std::size_t, bool,
														  std::size_t, RunStats*);
template std::vector<std::uint32_t> run_topk<std::uint32_t>(const std::vector<std::uint32_t>&, std::size_t, bool,
															std::size_t, RunStats*);
template std::vector<double> run_topk<double>(const std::vector<double>&, std::size_t, bool, std::size_t, RunStats*);

} // namespace gpu::map_reduce
