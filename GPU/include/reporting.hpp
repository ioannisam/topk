#pragma once

#include <cstddef>
#include <string>

#include "algorithm.hpp"
#include "common/config.hpp"

namespace gpu::reporting {

void print_configuration(const common::config::Config& cfg, std::size_t n, const std::string& device_name);

void print_debug_metrics(const common::config::Config& cfg, const std::string& device_name,
						 const gpu::bitonic::RunStats* full_stats, const gpu::bitonic::RunStats* trunc_stats,
						 std::size_t layer_count, std::size_t full_cmp, std::size_t trunc_cmp);

} // namespace gpu::reporting
