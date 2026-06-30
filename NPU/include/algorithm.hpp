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
RunStats run_topk(std::vector<T>& data, const std::vector<common::bitonic::Layer>& layers, std::size_t workers);

} // namespace npu::bitonic

namespace npu::map_reduce {

struct RunStats {
	double elapsed_ms;
	std::size_t layer_dispatches;
	bool used_offload;
};

template <typename T>
std::vector<T> run_topk(const std::vector<T>& data, std::size_t k, bool want_max, std::size_t workers,
						npu::map_reduce::RunStats* stats);

} // namespace npu::map_reduce

namespace npu::ground_truth {

template <typename T> void run_topk(std::vector<T>& data, std::size_t k, bool want_max);

} // namespace npu::ground_truth
