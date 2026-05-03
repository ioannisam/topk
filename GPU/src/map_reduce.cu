#include "../include/algorithm.hpp"

#include <algorithm>
#include <cstdint>
#include <limits>
#include <vector>

#include <cuda_runtime.h>

namespace gpu::map_reduce {

namespace {

template <typename T> 
__device__ __forceinline__ bool beats_threshold(T candidate, T threshold, bool want_max) {
	return want_max ? (candidate > threshold) : (candidate < threshold);
}

template <typename T> 
__device__ void sift_down(T* heap, int size, int root, bool want_max) {
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
__global__ void topk_map_kernel(const T* __restrict__ input, std::size_t n, int k, bool want_max,
							    T* __restrict__ thread_workspaces, int* __restrict__ thread_counts, 
                                T* __restrict__ block_outputs, T sentinel) {
	if (k <= 0) {
		return;
	}

	const std::size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
	const std::size_t stride = blockDim.x * gridDim.x;

	T* local_heap = thread_workspaces + (tid * k);
	int current_size = 0;
	
	// PHASE 1: MAP
	for (std::size_t i = tid; i < n; i += stride) {
		T val = input[i];
		
		if (current_size < k) {
			local_heap[current_size] = val;
			current_size++;
			if (current_size == k) {
				for (int j = k / 2 - 1; j >= 0; --j) {
					sift_down(local_heap, k, j, want_max);
				}
			}
		} else if (beats_threshold(val, local_heap[0], want_max)) {
			local_heap[0] = val;
			sift_down(local_heap, k, 0, want_max);
		}
	}

    // Explicitly track how many valid elements this thread processed
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
				
                // Only iterate over strictly valid elements in the other heap
				int c_other = thread_counts[other_global_tid];
				for (int j = 0; j < c_other; ++j) {
					T val = other_heap[j];
					int c_my = thread_counts[my_global_tid];
					
					if (c_my < k) {
						my_heap[c_my] = val;
						thread_counts[my_global_tid]++;
						if (thread_counts[my_global_tid] == k) {
							for (int h = k / 2 - 1; h >= 0; --h) sift_down(my_heap, k, h, want_max);
						}
					} else if (beats_threshold(val, my_heap[0], want_max)) {
						my_heap[0] = val;
						sift_down(my_heap, k, 0, want_max);
					}
				}
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
            // Pad remaining space with sentinels so the CPU reduction doesn't sort garbage memory
			for (int j = final_count; j < k; ++j) {
				block_outputs[blockIdx.x * k + j] = sentinel;
			}
		}

	} else {
		// STRATEGY B: Sequential Merge
		extern __shared__ char shared_mem[];
		T* block_heap = reinterpret_cast<T*>(shared_mem);

		if (threadIdx.x == 0) {
			int block_heap_size = 0;
			const std::size_t block_start_tid = blockIdx.x * blockDim.x;

			for (int t = 0; t < blockDim.x; ++t) {
				const std::size_t t_global_tid = block_start_tid + t;
				T* t_heap = thread_workspaces + (t_global_tid * k);
				int c_t = thread_counts[t_global_tid];

				for (int j = 0; j < c_t; ++j) {
					T val = t_heap[j];
					
					if (block_heap_size < k) {
						block_heap[block_heap_size] = val;
						block_heap_size++;
						if (block_heap_size == k) {
							for (int h = k / 2 - 1; h >= 0; --h) sift_down(block_heap, k, h, want_max);
						}
					} else if (beats_threshold(val, block_heap[0], want_max)) {
						block_heap[0] = val;
						sift_down(block_heap, k, 0, want_max);
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

} // namespace

template <typename T>
std::vector<T> run_topk(const std::vector<T>& input, std::size_t k, bool want_max, std::size_t ex_threads,
						RunStats* stats) {
	const std::size_t n = input.size();

	if (k == 0 || input.empty()) {
		return {};
	}

	cudaDeviceProp prop{};
	cudaGetDeviceProperties(&prop, 0);
	const int block_size = 256;

	size_t shared_mem_size = (k > 256) ? (k * sizeof(T)) : 0;

	int num_blocks;
	cudaOccupancyMaxActiveBlocksPerMultiprocessor(&num_blocks, topk_map_kernel<T>, block_size, shared_mem_size);
	const int grid_size = prop.multiProcessorCount * num_blocks;
	const int total_threads = grid_size * block_size;
	const T sentinel = want_max ? std::numeric_limits<T>::lowest() : std::numeric_limits<T>::max();

	T* d_input = nullptr;
	T* d_thread_workspaces = nullptr;
	int* d_thread_counts = nullptr;
	T* d_block_outputs = nullptr;

	cudaMalloc(&d_input, n * sizeof(T));
	cudaMemcpy(d_input, input.data(), n * sizeof(T), cudaMemcpyHostToDevice);

	cudaMalloc(&d_thread_workspaces, total_threads * k * sizeof(T));
	cudaMalloc(&d_thread_counts, total_threads * sizeof(int));
	cudaMalloc(&d_block_outputs, grid_size * k * sizeof(T));

	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start);

	topk_map_kernel<<<grid_size, block_size, shared_mem_size>>>(d_input, n, static_cast<int>(k), want_max,
											   d_thread_workspaces, d_thread_counts, d_block_outputs, sentinel);

	cudaEventRecord(stop);
	cudaEventSynchronize(stop);

	float elapsed_ms;
	cudaEventElapsedTime(&elapsed_ms, start, stop);

	std::vector<T> block_results(grid_size * k);
	cudaMemcpy(block_results.data(), d_block_outputs, grid_size * k * sizeof(T), cudaMemcpyDeviceToHost);

	cudaFree(d_input);
	cudaFree(d_thread_workspaces);
	cudaFree(d_thread_counts);
	cudaFree(d_block_outputs);
	cudaEventDestroy(start);
	cudaEventDestroy(stop);

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

template std::vector<float> run_topk<float>(const std::vector<float>&, std::size_t, bool, std::size_t, RunStats*);
template std::vector<std::int32_t> run_topk<std::int32_t>(const std::vector<std::int32_t>&, std::size_t, bool,
														  std::size_t, RunStats*);
template std::vector<std::uint32_t> run_topk<std::uint32_t>(const std::vector<std::uint32_t>&, std::size_t, bool,
															std::size_t, RunStats*);
template std::vector<double> run_topk<double>(const std::vector<double>&, std::size_t, bool, std::size_t, RunStats*);

} // namespace gpu::map_reduce
