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
RunStats run_topk(T* data, std::size_t n, std::size_t& final_n, const std::vector<common::bitonic::Layer>& layers);

} // namespace gpu::bitonic

namespace gpu::map_reduce {

struct RunStats {
	double elapsed_ms;
	std::size_t tiles_used;
	std::size_t aggregated_candidates;
	std::size_t block_size;
};

template <typename T>
std::size_t run_topk(const T* data, std::size_t n, std::size_t k, bool want_max, std::size_t workers, T* out,
					 RunStats* stats = nullptr);

} // namespace gpu::map_reduce

namespace gpu::ground_truth {

template <typename T> double run_topk(T* data, std::size_t n, std::size_t k, bool want_max);

} // namespace gpu::ground_truth
