#pragma once

#include <cstddef>
#include <string>

#include "algorithm.hpp"
#include "common/config.hpp"
#include "common/runner.hpp"

namespace npu::reporting {

void print_configuration(const common::config::Config& cfg, std::size_t n, std::size_t ex_threads,
						 const std::string& device_name, const std::string& device_bdf, bool offload_enabled);

void print_bitonic_debug_metrics(const common::config::Config& cfg, const std::string& device_name,
                                 const common::topk::BitonicRunStats& stats,
						         const npu::bitonic::RunStats* full_stats, 
                                 const npu::bitonic::RunStats* trunc_stats);

void print_map_reduce_debug_metrics(const common::config::Config& cfg, const std::string& device_name, 
                                    const std::string& device_bdf, bool offload_enabled,
                                    const common::topk::MapReduceRunStats& stats, 
                                    const npu::map_reduce::RunStats& npu_stats);

} // namespace npu::reporting
