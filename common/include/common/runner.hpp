#pragma once

#include <algorithm>
#include <cstddef>
#include <iostream>
#include <limits>
#include <optional>
#include <string>
#include <type_traits>
#include <vector>

#include "common/bitonic.hpp"
#include "common/config.hpp"
#include "common/random.hpp"
#include "common/reporting.hpp"
#include "common/validation.hpp"

namespace common::topk {

constexpr int kDefaultRandMin = 0;
constexpr int kDefaultRandMax = 1000;

struct BasicRunStats {
	double elapsed_ms = 0.0;
};

template <typename T> class RunnerHooks {
  public:
	virtual ~RunnerHooks() = default;

	virtual void print_configuration(const common::config::Config& cfg, std::size_t n) = 0;
	virtual BasicRunStats run_network(std::vector<T>& data, const std::vector<common::bitonic::Layer>& layers,
										  const std::vector<std::vector<unsigned char>>& keep,
										  bool trunc) = 0;
	virtual void print_debug_metrics(const common::config::Config& cfg, std::size_t layer_count,
									 std::size_t full_cmp, std::size_t trunc_cmp,
									 const BasicRunStats* full_stats,
									 const BasicRunStats* trunc_stats) = 0;
};

template <typename T> T transform_for_max(T value) {
	if constexpr (std::is_unsigned_v<T>) {
		return static_cast<T>(std::numeric_limits<T>::max() - value);
	}
	return static_cast<T>(-value);
}

template <typename T> T restore_from_max(T value) {
	if constexpr (std::is_unsigned_v<T>) {
		return static_cast<T>(std::numeric_limits<T>::max() - value);
	}
	return static_cast<T>(-value);
}

template <typename T> void apply_mode_transform(std::vector<T>& data, bool want_max) {
	if (!want_max) {
		return;
	}
	for (std::size_t i = 0; i < data.size(); ++i) {
		data[i] = transform_for_max(data[i]);
	}
}

template <typename T> bool compare_topk_prefix(const std::vector<T>& lhs, const std::vector<T>& rhs, std::size_t k) {
	for (std::size_t i = 0; i < k; ++i) {
		if (lhs[i] != rhs[i]) {
			return false;
		}
	}
	return true;
}

template <typename T> std::vector<T> build_output(const std::vector<T>& network_out, const common::config::Config& cfg) {
	std::vector<T> output(cfg.k);
	for (std::size_t i = 0; i < cfg.k; ++i) {
		output[i] = cfg.want_max ? restore_from_max(network_out[i]) : network_out[i];
	}

	if (cfg.want_max) {
		std::sort(output.begin(), output.end(), std::greater<T>());
	} else {
		std::sort(output.begin(), output.end());
	}
	return output;
}

template <typename T> int execute(const common::config::Config& cfg, RunnerHooks<T>& hooks) {
	const std::size_t n = std::size_t{1} << cfg.q;
	hooks.print_configuration(cfg, n);

	auto layers = common::bitonic::build_layers(n);
	auto keep = common::bitonic::build_masks(layers, n, cfg.k);
	const std::size_t full_cmp = common::bitonic::count_full_comparators(layers, n);
	const std::size_t trunc_cmp = common::bitonic::count_trunc_comparators(layers, keep, n);

	std::vector<T> input = common::utils::generate_random_input<T>(n, cfg.seed, kDefaultRandMin, kDefaultRandMax);
	apply_mode_transform(input, cfg.want_max);

	const bool run_trunc = cfg.run_mode != common::config::RunMode::Full;
	const bool run_full = cfg.run_mode != common::config::RunMode::Trunc;

	std::vector<T> trunc;
	std::vector<T> full;
	BasicRunStats trunc_stats{};
	BasicRunStats full_stats{};

	if (run_trunc) {
		trunc = input;
		trunc_stats = hooks.run_network(trunc, layers, keep, true);
	}
	if (run_full) {
		full = input;
		full_stats = hooks.run_network(full, layers, keep, false);
	}

	const bool run_both = cfg.run_mode == common::config::RunMode::Both;
	const bool both_ok = run_both ? compare_topk_prefix(trunc, full, cfg.k) : true;

	common::reporting::print_timing_lines({
		{"Full bitonic time (ms)", run_full ? std::optional<double>(full_stats.elapsed_ms) : std::nullopt},
		{"Trunc bitonic time (ms)", run_trunc ? std::optional<double>(trunc_stats.elapsed_ms) : std::nullopt},
	});

	if (run_trunc) {
		const std::size_t skipped = full_cmp >= trunc_cmp ? (full_cmp - trunc_cmp) : 0;
		const double skipped_pct =
			full_cmp == 0 ? 0.0 : (100.0 * static_cast<double>(skipped) / static_cast<double>(full_cmp));
		common::reporting::print_key_value(
			"Skipped comparators",
			std::to_string(skipped) + "/" + std::to_string(full_cmp) + " (" +
				common::reporting::format_fixed(skipped_pct, 2, "%") + ")");
	}

	hooks.print_debug_metrics(cfg, layers.size(), full_cmp, trunc_cmp,
							 run_full ? &full_stats : nullptr,
							 run_trunc ? &trunc_stats : nullptr);

	if (run_both) {
		common::reporting::print_check_result("Top-k correctness vs full network", true, both_ok);
		if (!both_ok) {
			return 2;
		}
	}

	const std::vector<T>& chosen_network_output = (cfg.run_mode == common::config::RunMode::Full) ? full : trunc;
	std::vector<T> output = build_output(chosen_network_output, cfg);
	if (cfg.run_check && cfg.has_expected_output) {
		const bool expected_ok = common::utils::validate_expected_output(cfg, output);
		std::cout << "Top-k correctness vs testcase answer: " << (expected_ok ? "OK" : "FAIL") << "\n";
		if (!expected_ok) {
			return 3;
		}
	}

	common::reporting::print_output(cfg, common::utils::format_output(output));
	return 0;
}

} // namespace common::topk
