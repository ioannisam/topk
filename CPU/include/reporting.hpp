#pragma once

#include <cstddef>

#include "common/config.hpp"

namespace cpu::reporting {

void print_configuration(const common::config::Config& cfg, std::size_t ex_threads, std::size_t n);

void print_debug_metrics(const common::config::Config& cfg, std::size_t hw_threads, std::size_t ex_threads,
							 std::size_t layer_count,
						 std::size_t full_cmp, std::size_t trunc_cmp, double full_ms, double trunc_ms);

} // namespace cpu::reporting
