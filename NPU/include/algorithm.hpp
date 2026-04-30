#pragma once

#include <cstddef>
#include <string>
#include <vector>

#include "common/bitonic.hpp"

namespace npu::bitonic {

struct RunStats {
	double elapsed_ms;
	std::size_t layer_dispatches;
	std::size_t active_comparators;
	std::size_t workers;
	bool used_offload;
};

std::string query_device_name();
std::string query_device_bdf();
bool is_offload_configured();

template <typename T>
RunStats run_network_npu(std::vector<T>& data, const std::vector<common::bitonic::Layer>& layers, std::size_t workers);

} // namespace npu::bitonic
