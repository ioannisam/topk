#pragma once

#include <cstddef>
#include <string>

#include "algorithm.hpp"
#include "common/config.hpp"

namespace npu::reporting {

void print_configuration(const common::config::Config& cfg, std::size_t n, std::size_t ex_threads,
						 const std::string& device_name, const std::string& device_bdf, bool offload_enabled);

void print_debug_metrics(const common::config::Config& cfg, const std::string& device_name,
						 const npu::bitonic::RunStats* full_stats, const npu::bitonic::RunStats* trunc_stats,
						 std::size_t layer_count, std::size_t full_cmp, std::size_t trunc_cmp);

} // namespace npu::reporting
