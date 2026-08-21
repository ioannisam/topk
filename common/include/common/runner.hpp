#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
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
#include "common/stats.hpp"
#include "common/validation.hpp"

namespace common::topk {

template <typename Fn> int dispatch_by_dtype(common::config::DataType dtype, Fn&& fn) {
	switch (dtype) {
	case common::config::DataType::Int:
		return fn(std::int32_t{});
	case common::config::DataType::UInt:
		return fn(std::uint32_t{});
	case common::config::DataType::Float:
		return fn(float{});
	case common::config::DataType::Double:
		return fn(double{});
	case common::config::DataType::Half:
#if defined(__FLT16_MANT_DIG__)
		return fn(_Float16{});
#else
		throw std::invalid_argument("dtype=half is not supported by this compiler target");
#endif
	}
	throw std::invalid_argument("Unsupported dtype");
}

template <typename T> class BitonicRunnerHooks {
  public:
	virtual ~BitonicRunnerHooks() = default;

	virtual void print_configuration(const common::config::Config& cfg, std::size_t n) = 0;
	virtual BasicRunStats run(std::vector<T>& data, const std::vector<common::bitonic::Layer>& layers) = 0;

	virtual void print_debug_metrics(const common::config::Config& cfg, const BitonicRunStats& stats) = 0;
};

template <typename T> class MapReduceRunnerHooks {
  public:
	virtual ~MapReduceRunnerHooks() = default;

	virtual void print_configuration(const common::config::Config& cfg, std::size_t n) = 0;
	virtual std::vector<T> run(
		const std::vector<T>& input, const common::config::Config& cfg, MapReduceRunStats* stats
	) = 0;
	virtual void print_debug_metrics(const common::config::Config& cfg, const MapReduceRunStats& stats) = 0;
};

template <typename T> class GroundTruthRunnerHooks {
  public:
	virtual ~GroundTruthRunnerHooks() = default;

	virtual void print_configuration(const common::config::Config& cfg, std::size_t n) = 0;
	virtual std::vector<T> run(
		const std::vector<T>& input, const common::config::Config& cfg, GroundTruthRunStats* stats
	) = 0;
	virtual void print_debug_metrics(const common::config::Config& cfg, const GroundTruthRunStats& stats) = 0;
};

template <typename T> T transform_for_max(T value) {
	if constexpr (std::is_unsigned_v<T>) {
		return static_cast<T>(std::numeric_limits<T>::max() - value);
	} else if constexpr (std::is_integral_v<T>) {
		using U = std::make_unsigned_t<T>;
		constexpr U msb = static_cast<U>(U{1} << (std::numeric_limits<U>::digits - 1));
		const U key = static_cast<U>(static_cast<U>(value) ^ msb);
		return static_cast<T>(static_cast<U>(std::numeric_limits<U>::max() - key) ^ msb);
	} else {
		return static_cast<T>(-value);
	}
}

template <typename T> T restore_from_max(T value) {
	return transform_for_max(value);
}

template <typename T> void apply_mode_transform(std::vector<T>& data, bool want_max) {
	if (!want_max) {
		return;
	}
	for (std::size_t i = 0; i < data.size(); i++) {
		data[i] = transform_for_max(data[i]);
	}
}

template <typename T> bool compare_topk_prefix(const std::vector<T>& lhs, const std::vector<T>& rhs, std::size_t k) {
	for (std::size_t i = 0; i < k; i++) {
		if (!common::utils::value_equal(lhs[i], rhs[i])) {
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
		std::sort(ref.begin(), ref.end(), std::greater<T>());
	} else {
		std::sort(ref.begin(), ref.end(), std::less<T>());
	}
	ref.resize(k);

	return ref;
}

template <typename T> bool equal_output(const std::vector<T>& lhs, const std::vector<T>& rhs) {
	if (lhs.size() != rhs.size()) {
		return false;
	}

	for (std::size_t i = 0; i < lhs.size(); i++) {
		if (!common::utils::value_equal(lhs[i], rhs[i])) {
			return false;
		}
	}
	return true;
}

template <typename T>
std::vector<T> build_output(const std::vector<T>& network_out, const common::config::Config& cfg) {
	std::vector<T> output(cfg.k);
	for (std::size_t i = 0; i < cfg.k; i++) {
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

	auto trunc_layers = common::bitonic::build_layers(n, cfg.k);
	auto full_layers = common::bitonic::build_layers(n, n); // topk = n yields a full sort

	const std::size_t full_cmp = common::bitonic::count_full_comparators(n);
	const std::size_t trunc_cmp = common::bitonic::count_trunc_comparators(trunc_layers);

	const std::vector<T> raw_input =
		common::utils::generate_random_input<T>(n, cfg.seed, cfg.rand_min, cfg.rand_max, cfg.dist);
	std::vector<T> input = raw_input;
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
		if (trunc_stats.traffic.compare_ops == 0.0) {
			trunc_stats.traffic.compare_ops = static_cast<double>(trunc_cmp);
			trunc_stats.traffic.ops_exact = true;
		}
	}
	if (run_full) {
		full = input;
		full_stats = hooks.run(full, full_layers);
		if (full_stats.traffic.compare_ops == 0.0) {
			full_stats.traffic.compare_ops = static_cast<double>(full_cmp);
			full_stats.traffic.ops_exact = true;
		}
	}

	const bool run_both = cfg.run_mode == common::config::RunMode::Both;
	const bool both_ok = run_both ? compare_topk_prefix(trunc, full, cfg.k) : true;

	common::reporting::print_section_header("Timing");
	if (run_full) {
		common::reporting::print_key_value("Full bitonic end-to-end time (ms)", full_stats.end_to_end_ms, 3);
		common::reporting::print_key_value("Full bitonic end-to-end stdev (ms)", full_stats.end_to_end_stdev_ms, 3);
		common::reporting::print_key_value("Full bitonic end-to-end min (ms)", full_stats.end_to_end_min_ms, 3);
		common::reporting::print_key_value("Full bitonic algorithmic time (ms)", full_stats.algorithm_ms, 3);
		common::reporting::print_key_value("Full bitonic algorithmic stdev (ms)", full_stats.algorithm_stdev_ms, 3);
		common::reporting::print_key_value("Full bitonic algorithmic min (ms)", full_stats.algorithm_min_ms, 3);
	} else {
		common::reporting::print_key_value("Full bitonic end-to-end time (ms)", "skipped");
		common::reporting::print_key_value("Full bitonic algorithmic time (ms)", "skipped");
	}

	std::cout << std::endl;

	if (run_trunc) {
		common::reporting::print_key_value("Trunc bitonic end-to-end time (ms)", trunc_stats.end_to_end_ms, 3);
		common::reporting::print_key_value("Trunc bitonic end-to-end stdev (ms)", trunc_stats.end_to_end_stdev_ms, 3);
		common::reporting::print_key_value("Trunc bitonic end-to-end min (ms)", trunc_stats.end_to_end_min_ms, 3);
		common::reporting::print_key_value("Trunc bitonic algorithmic time (ms)", trunc_stats.algorithm_ms, 3);
		common::reporting::print_key_value("Trunc bitonic algorithmic stdev (ms)", trunc_stats.algorithm_stdev_ms, 3);
		common::reporting::print_key_value("Trunc bitonic algorithmic min (ms)", trunc_stats.algorithm_min_ms, 3);
	} else {
		common::reporting::print_key_value("Trunc bitonic end-to-end time (ms)", "skipped");
		common::reporting::print_key_value("Trunc bitonic algorithmic time (ms)", "skipped");
	}

	if (run_full) {
		common::reporting::print_energy_lines(full_stats.energy, run_both ? "full" : nullptr);
		common::reporting::print_traffic_lines(full_stats.traffic, run_both ? "full" : nullptr);
	}
	if (run_trunc) {
		common::reporting::print_energy_lines(trunc_stats.energy, run_both ? "trunc" : nullptr);
		common::reporting::print_traffic_lines(trunc_stats.traffic, run_both ? "trunc" : nullptr);
	}

	if (run_trunc) {
		common::reporting::print_skipped_comparators(full_cmp, trunc_cmp);
	}

	BitonicRunStats debug_stats{};
	debug_stats.layer_count = trunc_layers.size();
	debug_stats.full_comparators = full_cmp;
	debug_stats.trunc_comparators = trunc_cmp;
	debug_stats.full_run_stats = run_full ? &full_stats : nullptr;
	debug_stats.trunc_run_stats = run_trunc ? &trunc_stats : nullptr;

	hooks.print_debug_metrics(cfg, debug_stats);

	if (run_both) {
		common::reporting::print_check_result("Top-k correctness vs full network", true, both_ok);
		if (!both_ok) {
			return 2;
		}
	}

	const std::vector<T>& chosen_network_output = (cfg.run_mode == common::config::RunMode::Full) ? full : trunc;
	std::vector<T> output = build_output(chosen_network_output, cfg);
	if (cfg.verify_output) {
		const std::vector<T> ref = build_reference_topk(raw_input, cfg.k, cfg.want_max);
		const bool ref_ok = equal_output(output, ref);
		common::reporting::print_check_result("Top-k correctness vs CPU sorted reference", true, ref_ok);
		if (!ref_ok) {
			return 2;
		}
	}

	common::reporting::print_output(cfg, common::utils::format_output(output));
	return 0;
}

template <typename T> int execute_map_reduce(const common::config::Config& cfg, MapReduceRunnerHooks<T>& hooks) {
	const std::size_t n = std::size_t{1} << cfg.q;
	hooks.print_configuration(cfg, n);

	std::vector<T> input = common::utils::generate_random_input<T>(n, cfg.seed, cfg.rand_min, cfg.rand_max, cfg.dist);

	MapReduceRunStats run_stats{};
	std::vector<T> output = hooks.run(input, cfg, &run_stats);
	if (run_stats.traffic.compare_ops == 0.0) {
		run_stats.traffic.compare_ops = static_cast<double>(n);
	}

	common::reporting::print_timing_lines({
		{"Map-reduce top-k end-to-end time (ms)", std::optional<double>(run_stats.end_to_end_ms)},
		{"Map-reduce top-k end-to-end stdev (ms)", std::optional<double>(run_stats.end_to_end_stdev_ms)},
		{"Map-reduce top-k end-to-end min (ms)", std::optional<double>(run_stats.end_to_end_min_ms)},
		{"Map-reduce top-k algorithmic time (ms)", std::optional<double>(run_stats.algorithm_ms)},
		{"Map-reduce top-k algorithmic stdev (ms)", std::optional<double>(run_stats.algorithm_stdev_ms)},
		{"Map-reduce top-k algorithmic min (ms)", std::optional<double>(run_stats.algorithm_min_ms)},
	});

	common::reporting::print_energy_lines(run_stats.energy);
	common::reporting::print_traffic_lines(run_stats.traffic);

	const bool check_vs_reference = cfg.verify_output || cfg.run_mode == common::config::RunMode::Both;
	bool reference_ok = true;
	if (check_vs_reference) {
		const std::vector<T> ref = build_reference_topk(input, cfg.k, cfg.want_max);
		reference_ok = equal_output(output, ref);
	}
	common::reporting::print_check_result(
		"Top-k correctness vs CPU sorted reference", check_vs_reference, reference_ok
	);
	if (!reference_ok) {
		return 2;
	}

	hooks.print_debug_metrics(cfg, run_stats);

	common::reporting::print_output(cfg, common::utils::format_output(output));
	return 0;
}

template <typename T> int execute_ground_truth(const common::config::Config& cfg, GroundTruthRunnerHooks<T>& hooks) {
	const std::size_t n = std::size_t{1} << cfg.q;
	hooks.print_configuration(cfg, n);

	std::vector<T> input = common::utils::generate_random_input<T>(n, cfg.seed, cfg.rand_min, cfg.rand_max, cfg.dist);

	GroundTruthRunStats run_stats{};
	std::vector<T> output = hooks.run(input, cfg, &run_stats);
	if (run_stats.traffic.compare_ops == 0.0) {
		run_stats.traffic.compare_ops = static_cast<double>(n);
	}

	common::reporting::print_timing_lines({
		{"Ground truth end-to-end time (ms)", std::optional<double>(run_stats.end_to_end_ms)},
		{"Ground truth end-to-end stdev (ms)", std::optional<double>(run_stats.end_to_end_stdev_ms)},
		{"Ground truth end-to-end min (ms)", std::optional<double>(run_stats.end_to_end_min_ms)},
		{"Ground truth algorithmic time (ms)", std::optional<double>(run_stats.algorithm_ms)},
		{"Ground truth algorithmic stdev (ms)", std::optional<double>(run_stats.algorithm_stdev_ms)},
		{"Ground truth algorithmic min (ms)", std::optional<double>(run_stats.algorithm_min_ms)},
	});

	common::reporting::print_energy_lines(run_stats.energy);
	common::reporting::print_traffic_lines(run_stats.traffic);

	const bool check_vs_reference = cfg.verify_output;
	bool reference_ok = true;
	if (check_vs_reference) {
		const std::vector<T> ref = build_reference_topk(input, cfg.k, cfg.want_max);
		reference_ok = equal_output(output, ref);
	}
	common::reporting::print_check_result(
		"Top-k correctness vs CPU sorted reference", check_vs_reference, reference_ok
	);
	if (!reference_ok) {
		return 2;
	}

	hooks.print_debug_metrics(cfg, run_stats);

	common::reporting::print_output(cfg, common::utils::format_output(output));
	return 0;
}

} // namespace common::topk
