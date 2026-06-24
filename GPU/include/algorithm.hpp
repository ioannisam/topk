#pragma once

#include <cstddef>
#include <string>
#include <vector>

#include "common/bitonic.hpp"

namespace gpu::bitonic {

struct RunStats {
	double elapsed_ms;
	std::size_t kernel_launches;
	std::size_t active_comparators;
	std::size_t block_size;
};

std::string query_device_name();

template <typename T>
RunStats run_network_cuda(T* data, std::size_t n, std::size_t& final_n,
						  const std::vector<common::bitonic::Layer>& layers);

#if defined(__FLT16_MANT_DIG__)
RunStats run_network_cuda_fp16(_Float16* data, std::size_t n, std::size_t& final_n,
							   const std::vector<common::bitonic::Layer>& layers);
#endif

} // namespace gpu::bitonic

namespace gpu::map_reduce {

struct RunStats {
	double elapsed_ms;
	std::size_t tiles_used;
	std::size_t aggregated_candidates;
	std::size_t block_size;
};

template <typename T>
std::vector<T> run_topk(const T* data, std::size_t n, std::size_t k, bool want_max, std::size_t workers,
						RunStats* stats = nullptr);

#if defined(__FLT16_MANT_DIG__)
std::size_t run_topk_fp16(const _Float16* input, std::size_t n, std::size_t k, bool want_max, std::size_t ex_threads,
						  _Float16* out, RunStats* stats = nullptr);
#endif

} // namespace gpu::map_reduce

namespace gpu::ground_truth {

template <typename T> double run_topk(std::vector<T>& data, std::size_t k, bool want_max);

} // namespace gpu::ground_truth
