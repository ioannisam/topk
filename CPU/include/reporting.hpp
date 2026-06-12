#pragma once

#include <cstddef>

#include "common/config.hpp"
#include "common/runner.hpp"

namespace cpu::reporting {

void print_configuration(const common::config::Config& cfg, std::size_t ex_threads, std::size_t n);

void print_bitonic_debug_metrics(const common::config::Config& cfg, std::size_t hw_threads, std::size_t ex_threads,
								 const common::topk::BitonicRunStats& stats);

void print_map_reduce_debug_metrics(const common::config::Config& cfg, std::size_t hw_threads, std::size_t ex_threads,
									const common::topk::MapReduceRunStats& stats);

} // namespace cpu::reporting
