#include "runner.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <limits>
#include <iostream>
#include <sstream>
#include <string>
#include <thread>
#include <type_traits>
#include <vector>

#include "algorithm.hpp"
#include "layers.hpp"
#include "reporting.hpp"
#include "utils.hpp"

namespace {

constexpr int randMin = 0;
constexpr int randMax = 1000;

struct Context {
	std::size_t n;
	std::size_t hw_threads;
	std::size_t ex_threads;
};

Context build_context(const Config& cfg) {
	const std::size_t n = std::size_t{1} << cfg.q;
	const std::size_t hw_threads = std::max<std::size_t>(1, std::thread::hardware_concurrency());
	std::size_t ex_threads = cfg.ex_threads == 0 ? hw_threads : cfg.ex_threads;
	ex_threads = std::max<std::size_t>(1, std::min(ex_threads, n));

	return Context{n, hw_threads, ex_threads};
}

template <typename T> T transform_for_max(T value) {
	if constexpr (std::is_unsigned_v<T>) {
		return static_cast<T>(std::numeric_limits<T>::max() - value);
	} else {
		return static_cast<T>(-value);
	}
}

template <typename T> T restore_from_max(T value) {
	if constexpr (std::is_unsigned_v<T>) {
		return static_cast<T>(std::numeric_limits<T>::max() - value);
	} else {
		return static_cast<T>(-value);
	}
}

template <typename T> void apply_mode_transform(std::vector<T>& data, bool want_max) {
	if (!want_max) {
		return;
	}

	for (std::size_t i = 0; i < data.size(); i++) {
		data[i] = transform_for_max(data[i]);
	}
}

template <typename T>
double run_trunc_network(std::vector<T>& trunc, const std::vector<Layer>& layers,
						 const std::vector<std::vector<unsigned char>>& keep, std::size_t ex_threads) {

	auto t0 = std::chrono::high_resolution_clock::now();
	run_network_parallel(trunc, layers, keep, true, ex_threads);
	auto t1 = std::chrono::high_resolution_clock::now();
	return std::chrono::duration<double, std::milli>(t1 - t0).count();
}

template <typename T>
double run_full_network(std::vector<T>& full, const std::vector<Layer>& layers,
						const std::vector<std::vector<unsigned char>>& keep, std::size_t ex_threads) {
	auto t0 = std::chrono::high_resolution_clock::now();
	run_network_parallel(full, layers, keep, false, ex_threads);
	auto t1 = std::chrono::high_resolution_clock::now();
	return std::chrono::duration<double, std::milli>(t1 - t0).count();
}

template <typename T> bool compare_topk_prefix(const std::vector<T>& lhs, const std::vector<T>& rhs, std::size_t k) {
	for (std::size_t i = 0; i < k; i++) {
		if (lhs[i] != rhs[i]) {
			return false;
		}
	}
	return true;
}

template <typename T> std::vector<T> build_output(const std::vector<T>& trunc, const Config& cfg) {
	std::vector<T> output(cfg.k);
	for (std::size_t i = 0; i < cfg.k; i++) {
		output[i] = cfg.want_max ? restore_from_max(trunc[i]) : trunc[i];
	}

	if (cfg.want_max) {
		std::sort(output.begin(), output.end(), std::greater<T>());
	} else {
		std::sort(output.begin(), output.end());
	}

	return output;
}

template <typename T> std::string format_value(T value) {
	if constexpr (std::is_integral_v<T>) {
		if constexpr (std::is_unsigned_v<T>) {
			return std::to_string(static_cast<unsigned long long>(value));
		}
		return std::to_string(static_cast<long long>(value));
	} else {
		std::ostringstream out;
		out << static_cast<double>(value);
		return out.str();
	}
}

template <typename T> std::vector<std::string> format_output(const std::vector<T>& values) {
	std::vector<std::string> out;
	out.reserve(values.size());
	for (const T value : values) {
		out.push_back(format_value(value));
	}
	return out;
}

template <typename T> T parse_expected_value(const std::string& token) {
	if constexpr (std::is_integral_v<T>) {
		if constexpr (std::is_unsigned_v<T>) {
			return static_cast<T>(std::stoull(token));
		}
		return static_cast<T>(std::stoll(token));
	} else {
		return static_cast<T>(std::stod(token));
	}
}

template <typename T> bool value_equal(T lhs, T rhs) {
	if constexpr (std::is_floating_point_v<T>) {
		const double a = static_cast<double>(lhs);
		const double b = static_cast<double>(rhs);
		const double diff = std::fabs(a - b);
		const double scale = std::max(1.0, std::max(std::fabs(a), std::fabs(b)));
		return diff <= 1e-6 * scale;
	}
	return lhs == rhs;
}

template <typename T> bool validate_output(const Config& cfg, const std::vector<T>& output) {
	if (!cfg.has_expected_output) {
		return true;
	}

	if (cfg.expected_output_tokens.size() != output.size()) {
		return false;
	}

	for (std::size_t i = 0; i < output.size(); ++i) {
		const T expected = parse_expected_value<T>(cfg.expected_output_tokens[i]);
		if (!value_equal(output[i], expected)) {
			return false;
		}
	}

	return true;
}

template <typename T> int run_topk_typed(const Config& cfg) {
	const Context ctx = build_context(cfg);

	print_configuration(cfg, ctx.ex_threads, ctx.n);
	auto layers = build_layers(ctx.n);
	auto keep = build_masks(layers, ctx.n, cfg.k);

	const std::size_t full_cmp = count_full_comparators(layers, ctx.n);
	const std::size_t trunc_cmp = count_trunc_comparators(layers, keep, ctx.n);

	std::vector<T> input = generate_random_input<T>(ctx.n, cfg.seed, randMin, randMax);
	apply_mode_transform(input, cfg.want_max);

	const bool run_trunc = cfg.run_mode != RunMode::Full;
	const bool run_full = cfg.run_mode != RunMode::Trunc;

	std::vector<T> trunc;
	std::vector<T> full;
	double trunc_ms = 0.0;
	double full_ms = 0.0;

	if (run_trunc) {
		trunc = input;
		trunc_ms = run_trunc_network(trunc, layers, keep, ctx.ex_threads);
	}

	if (run_full) {
		full = input;
		full_ms = run_full_network(full, layers, keep, ctx.ex_threads);
	}

	const bool run_both = cfg.run_mode == RunMode::Both;
	const bool both_ok = run_both ? compare_topk_prefix(trunc, full, cfg.k) : true;

	print_timing_summary(run_full, run_trunc, full_ms, trunc_ms);
	print_skipped_summary(run_trunc, full_cmp, trunc_cmp);
	print_debug_metrics(cfg, ctx.hw_threads, ctx.ex_threads, layers.size(), full_cmp, trunc_cmp, full_ms, trunc_ms);

	if (run_both) {
		print_correctness_summary(true, both_ok);
		if (!both_ok) {
			return 2;
		}
	}

	const std::vector<T>& chosen_network_output = (cfg.run_mode == RunMode::Full) ? full : trunc;
	std::vector<T> output = build_output(chosen_network_output, cfg);
	if (cfg.run_check && cfg.has_expected_output) {
		const bool expected_ok = validate_output(cfg, output);
		std::cout << "Top-k correctness vs testcase answer: " << (expected_ok ? "OK" : "FAIL") << "\n";
		if (!expected_ok) {
			return 3;
		}
	}
	print_topk_output(cfg, format_output(output));

	return 0;
}

} // namespace

int run_topk(const Config& cfg) {
	switch (cfg.dtype) {
	case DataType::Int:
		return run_topk_typed<std::int32_t>(cfg);
	case DataType::UInt:
		return run_topk_typed<std::uint32_t>(cfg);
	case DataType::Float:
		return run_topk_typed<float>(cfg);
	case DataType::Double:
		return run_topk_typed<double>(cfg);
	case DataType::Fp16:
#if defined(__FLT16_MANT_DIG__)
		return run_topk_typed<_Float16>(cfg);
#else
		throw std::invalid_argument("dtype=fp16 is not supported by this compiler target");
#endif
	}

	throw std::invalid_argument("Unsupported dtype");
}
