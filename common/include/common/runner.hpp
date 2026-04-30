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

struct MapReduceRunStats {
	double elapsed_ms = 0.0;
	std::size_t tiles_used = 0;
	std::size_t aggregated_candidates = 0;
};

template <typename T> class BitonicRunnerHooks {
  public:
	virtual ~BitonicRunnerHooks() = default;

	virtual void print_configuration(const common::config::Config& cfg, std::size_t n) = 0;
	virtual BasicRunStats run(std::vector<T>& data, const std::vector<common::bitonic::Layer>& layers) = 0;
    
	virtual void print_debug_metrics(const common::config::Config& cfg, std::size_t layer_count, std::size_t full_cmp,
									 std::size_t trunc_cmp, const BasicRunStats* full_stats,
									 const BasicRunStats* trunc_stats) = 0;
};

template <typename T> class MapReduceRunnerHooks {
  public:
	virtual ~MapReduceRunnerHooks() = default;

	virtual void print_configuration(const common::config::Config& cfg, std::size_t n) = 0;
	virtual std::vector<T> run(const std::vector<T>& input, const common::config::Config& cfg,
							   MapReduceRunStats* stats) = 0;
	virtual void print_debug_metrics(const common::config::Config& cfg, const MapReduceRunStats& stats) = 0;
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

template <typename T> std::vector<T> build_reference_topk(const std::vector<T>& input, std::size_t k, bool want_max) {
	std::vector<T> ref = input;
	k = std::min(k, ref.size());
	if (k == 0) {
		return {};
	}

	if (want_max) {
		std::nth_element(ref.begin(), ref.begin() + static_cast<std::ptrdiff_t>(k), ref.end(), std::greater<T>());
		ref.resize(k);
		std::sort(ref.begin(), ref.end(), std::greater<T>());
	} else {
		std::nth_element(ref.begin(), ref.begin() + static_cast<std::ptrdiff_t>(k), ref.end(), std::less<T>());
		ref.resize(k);
		std::sort(ref.begin(), ref.end(), std::less<T>());
	}

	return ref;
}

template <typename T> bool equal_output(const std::vector<T>& lhs, const std::vector<T>& rhs) {
	if (lhs.size() != rhs.size()) {
		return false;
	}

	for (std::size_t i = 0; i < lhs.size(); ++i) {
		if (!common::utils::value_equal(lhs[i], rhs[i])) {
			return false;
		}
	}
	return true;
}

template <typename T>
std::vector<T> build_output(const std::vector<T>& network_out, const common::config::Config& cfg) {
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

template <typename T> int execute_bitonic(const common::config::Config& cfg, BitonicRunnerHooks<T>& hooks) {
	const std::size_t n = std::size_t{1} << cfg.q;
	hooks.print_configuration(cfg, n);

    // Build the dynamic instruction sets
	auto trunc_layers = common::bitonic::build_layers(n, cfg.k);
	auto full_layers = common::bitonic::build_layers(n, n); // topk = n yields a full sort
    
	const std::size_t full_cmp = common::bitonic::count_full_comparators(n);
	const std::size_t trunc_cmp = common::bitonic::count_trunc_comparators(trunc_layers);

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
		trunc_stats = hooks.run(trunc, trunc_layers);
	}
	if (run_full) {
		full = input;
		full_stats = hooks.run(full, full_layers);
	}

	const bool run_both = cfg.run_mode == common::config::RunMode::Both;
	const bool both_ok = run_both ? compare_topk_prefix(trunc, full, cfg.k) : true;

	// Print timing lines; ensure an explicit newline after the Full timing
	common::reporting::print_section_header("Timing");
	if (run_full) {
		common::reporting::print_key_value("Full bitonic time (ms)", full_stats.elapsed_ms, 3);
	} else {
		common::reporting::print_key_value("Full bitonic time (ms)", "skipped");
	}

	// Ensure a separating newline after the Full bitonic time to avoid
	// accidental concatenation with subsequent output (fixes parsing bugs).
	std::cout << std::endl;

	if (run_trunc) {
		common::reporting::print_key_value("Trunc bitonic time (ms)", trunc_stats.elapsed_ms, 3);
	} else {
		common::reporting::print_key_value("Trunc bitonic time (ms)", "skipped");
	}

	if (run_trunc) {
		const std::size_t skipped = full_cmp >= trunc_cmp ? (full_cmp - trunc_cmp) : 0;
		const double skipped_pct =
			full_cmp == 0 ? 0.0 : (100.0 * static_cast<double>(skipped) / static_cast<double>(full_cmp));
		common::reporting::print_key_value("Skipped comparators",
										   std::to_string(skipped) + "/" + std::to_string(full_cmp) + " (" +
											   common::reporting::format_fixed(skipped_pct, 2, "%") + ")");
	}

	hooks.print_debug_metrics(cfg, trunc_layers.size(), full_cmp, trunc_cmp, run_full ? &full_stats : nullptr,
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

template <typename T> int execute_map_reduce(const common::config::Config& cfg, MapReduceRunnerHooks<T>& hooks) {
	const std::size_t n = std::size_t{1} << cfg.q;
	hooks.print_configuration(cfg, n);

	std::vector<T> input = common::utils::generate_random_input<T>(n, cfg.seed, kDefaultRandMin, kDefaultRandMax);

	MapReduceRunStats run_stats{};
	std::vector<T> output = hooks.run(input, cfg, &run_stats);

	common::reporting::print_timing_lines({
		{"Map-reduce top-k time (ms)", std::optional<double>(run_stats.elapsed_ms)},
	});

	const bool check_vs_reference = cfg.run_mode == common::config::RunMode::Both;
	bool reference_ok = true;
	if (check_vs_reference) {
		const std::vector<T> ref = build_reference_topk(input, cfg.k, cfg.want_max);
		reference_ok = equal_output(output, ref);
	}
	common::reporting::print_check_result("Top-k correctness vs nth_element reference", check_vs_reference,
										  reference_ok);
	if (!reference_ok) {
		return 2;
	}

	hooks.print_debug_metrics(cfg, run_stats);

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
