#include "../include/algorithm.hpp"

#include <cstdint>
#include <vector>

#include <cuda_runtime.h>

namespace gpu::map_reduce {

namespace {

template <typename T> __device__ __forceinline__ bool beats_threshold(T candidate, T threshold, bool want_max) {
	return want_max ? (candidate > threshold) : (candidate < threshold);
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
__global__ void topk_map_kernel(const T* __restrict__ input, std::size_t n, int k, bool want_max,
								T* __restrict__ thread_workspaces, T* __restrict__ block_outputs) {
	const std::size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
	const std::size_t stride = blockDim.x * gridDim.x;

	T* local_heap = thread_workspaces + (tid * k);
	int current_size = 0;

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
		} else {
			if (beats_threshold(val, local_heap[0], want_max)) {
				local_heap[0] = val;
				sift_down(local_heap, k, 0, want_max);
			}
		}
	}

	__syncthreads();

	extern __shared__ char shared_mem[];
	T* block_heap = reinterpret_cast<T*>(shared_mem);

	if (threadIdx.x == 0) {
		int block_heap_size = 0;
		const std::size_t block_start_tid = blockIdx.x * blockDim.x;

		for (int t = 0; t < blockDim.x; ++t) {
			T* t_heap = thread_workspaces + ((block_start_tid + t) * k);

			for (int j = 0; j < k; ++j) {
				if (t_heap[j] == 0 && current_size < k) {
					continue;
				}

				T val = t_heap[j];
				if (block_heap_size < k) {
					block_heap[block_heap_size] = val;
					block_heap_size++;
					if (block_heap_size == k) {
						for (int h = k / 2 - 1; h >= 0; --h) {
							sift_down(block_heap, k, h, want_max);
						}
					}
				} else if (beats_threshold(val, block_heap[0], want_max)) {
					block_heap[0] = val;
					sift_down(block_heap, k, 0, want_max);
				}
			}
		}

		for (int j = 0; j < k; ++j) {
			block_outputs[blockIdx.x * k + j] = block_heap[j];
		}
	}
}

} // namespace

template <typename T>
std::vector<T> run_topk(const std::vector<T>& input, std::size_t k, bool want_max, std::size_t ex_threads,
						RunStats* stats) {
	const std::size_t n = input.size();

	cudaDeviceProp prop{};
	cudaGetDeviceProperties(&prop, 0);
	const int block_size = 256;

	int num_blocks;
	cudaOccupancyMaxActiveBlocksPerMultiprocessor(&num_blocks, topk_map_kernel<T>, block_size, k * sizeof(T));
	const int grid_size = prop.multiProcessorCount * num_blocks;
	const int total_threads = grid_size * block_size;

	T* d_input = nullptr;
	T* d_thread_workspaces = nullptr;
	T* d_block_outputs = nullptr;

	cudaMalloc(&d_input, n * sizeof(T));
	cudaMemcpy(d_input, input.data(), n * sizeof(T), cudaMemcpyHostToDevice);

	cudaMalloc(&d_thread_workspaces, total_threads * k * sizeof(T));
	cudaMalloc(&d_block_outputs, grid_size * k * sizeof(T));

	size_t shared_mem_size = k * sizeof(T);

	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start);

	topk_map_kernel<<<grid_size, block_size, shared_mem_size>>>(d_input, n, static_cast<int>(k), want_max,
																d_thread_workspaces, d_block_outputs);

	cudaEventRecord(stop);
	cudaEventSynchronize(stop);

	float elapsed_ms;
	cudaEventElapsedTime(&elapsed_ms, start, stop);

	std::vector<T> block_results(grid_size * k);
	cudaMemcpy(block_results.data(), d_block_outputs, grid_size * k * sizeof(T), cudaMemcpyDeviceToHost);

	cudaFree(d_input);
	cudaFree(d_thread_workspaces);
	cudaFree(d_block_outputs);
	cudaEventDestroy(start);
	cudaEventDestroy(stop);

	if (stats != nullptr) {
		stats->elapsed_ms = elapsed_ms;
		stats->tiles_used = grid_size;
		stats->aggregated_candidates = grid_size * k;
		stats->block_size = block_size;
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
