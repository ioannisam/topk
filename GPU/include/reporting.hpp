#pragma once

#include <cstddef>
#include <string>

#include "algorithm.hpp"
#include "common/config.hpp"
#include "common/runner.hpp"

namespace gpu::reporting {

void print_configuration(const common::config::Config& cfg, std::size_t n, const std::string& device_name);

void print_bitonic_debug_metrics(const common::config::Config& cfg, const std::string& device_name,
								 const common::topk::BitonicRunStats& stats,
								 const gpu::bitonic::RunStats* full_gpu_stats,
								 const gpu::bitonic::RunStats* trunc_gpu_stats);

void print_map_reduce_debug_metrics(const common::config::Config& cfg, const std::string& device_name,
									const common::topk::MapReduceRunStats& stats,
									const gpu::map_reduce::RunStats& gpu_stats);

} // namespace gpu::reporting
