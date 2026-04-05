#pragma once

#include <cstddef>
#include <string>
#include <vector>

#include "config.hpp"

void print_configuration(const Config& cfg, std::size_t ex_threads, std::size_t n);

void print_timing_summary(bool ran_full, bool ran_trunc, double full_ms, double trunc_ms);

void print_skipped_summary(bool ran_trunc, std::size_t full_cmp, std::size_t trunc_cmp);

void print_debug_metrics(const Config& cfg, std::size_t hw_threads, std::size_t ex_threads, std::size_t layer_count,
						 std::size_t full_cmp, std::size_t trunc_cmp, double full_ms, double trunc_ms);

void print_correctness_summary(bool run_check, bool ok);

void print_topk_output(const Config& cfg, const std::vector<std::string>& output);
