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
RunStats run_network_cuda(std::vector<T>& data, const std::vector<common::bitonic::Layer>& layers,
						  const std::vector<std::vector<unsigned char>>& keep, bool trunc);

RunStats run_network_cuda_fp16(std::vector<float>& data, const std::vector<common::bitonic::Layer>& layers,
							   const std::vector<std::vector<unsigned char>>& keep, bool trunc);

} // namespace gpu::bitonic
