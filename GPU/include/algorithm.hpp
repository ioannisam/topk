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
int query_device_sm_count();

template <typename T>
RunStats run_network_cuda(std::vector<T>& data, const std::vector<common::bitonic::Layer>& layers);

RunStats run_network_cuda_fp16(std::vector<float>& data, const std::vector<common::bitonic::Layer>& layers);

} // namespace gpu::bitonic

namespace gpu::map_reduce {

struct RunStats {
	double elapsed_ms;
	std::size_t tiles_used;
	std::size_t aggregated_candidates;
	std::size_t block_size;
};

template <typename T>
std::vector<T> run_topk(const std::vector<T>& data, std::size_t k, bool want_max, std::size_t workers,
						RunStats* stats = nullptr);

std::vector<float> run_topk_fp16(const std::vector<float>& input, std::size_t k, bool want_max, std::size_t ex_threads,
                                 RunStats* stats = nullptr);

} // namespace gpu::map_reduce
