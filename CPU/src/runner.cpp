#include "runner.hpp"

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <limits>
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

struct Check {
	bool ok;
	double full_ms;
};

Context build_context(const Config& cfg) {
	const std::size_t n = std::size_t{1} << cfg.q;
	const std::size_t hw_threads = std::max<std::size_t>(1, std::thread::hardware_concurrency());
	std::size_t ex_threads = cfg.ex_threads == 0 ? hw_threads : cfg.ex_threads;
	ex_threads = std::max<std::size_t>(1, std::min(ex_threads, n));

	return Context{n, hw_threads, ex_threads};
}

template <typename T>
T transform_for_max(T value) {
	if constexpr (std::is_unsigned_v<T>) {
		return static_cast<T>(std::numeric_limits<T>::max() - value);
	} else {
		return static_cast<T>(-value);
	}
}

template <typename T>
T restore_from_max(T value) {
	if constexpr (std::is_unsigned_v<T>) {
		return static_cast<T>(std::numeric_limits<T>::max() - value);
	} else {
		return static_cast<T>(-value);
	}
}

template <typename T>
void apply_mode_transform(std::vector<T>& data, bool want_max) {
	if (!want_max) {
		return;
	}

	for (std::size_t i = 0; i < data.size(); i++) {
		data[i] = transform_for_max(data[i]);
	}
}

template <typename T>
double run_truncated_network(std::vector<T>& truncated, const std::vector<Layer>& layers,
							 const std::vector<std::vector<unsigned char>>& keep, std::size_t ex_threads) {

	auto t0 = std::chrono::high_resolution_clock::now();
	run_network_parallel(truncated, layers, keep, true, ex_threads);
	auto t1 = std::chrono::high_resolution_clock::now();
	return std::chrono::duration<double, std::milli>(t1 - t0).count();
}

template <typename T>
Check run_check(const Config& cfg, const std::vector<T>& input, const std::vector<T>& truncated,
				const std::vector<Layer>& layers, const std::vector<std::vector<unsigned char>>& keep,
				std::size_t ex_threads) {

	if (!cfg.run_check) {
		return Check{true, 0.0};
	}

	std::vector<T> full = input;
	auto t0 = std::chrono::high_resolution_clock::now();
	run_network_parallel(full, layers, keep, false, ex_threads);
	auto t1 = std::chrono::high_resolution_clock::now();
	const double full_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

	for (std::size_t i = 0; i < cfg.k; i++) {
		if (truncated[i] != full[i]) {
			return Check{false, full_ms};
		}
	}

	return Check{true, full_ms};
}

template <typename T>
std::vector<T> build_output(const std::vector<T>& truncated, const Config& cfg) {
	std::vector<T> output(cfg.k);
	for (std::size_t i = 0; i < cfg.k; i++) {
		output[i] = cfg.want_max ? restore_from_max(truncated[i]) : truncated[i];
	}

	if (cfg.sort_output) {
		if (cfg.want_max) {
			std::sort(output.begin(), output.end(), std::greater<int>());
		} else {
			std::sort(output.begin(), output.end());
		}
	}

	return output;
}

template <typename T>
std::string format_value(T value) {
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

template <typename T>
std::vector<std::string> format_output(const std::vector<T>& values) {
	std::vector<std::string> out;
	out.reserve(values.size());
	for (const T value : values) {
		out.push_back(format_value(value));
	}
	return out;
}

template <typename T>
int run_topk_typed(const Config& cfg) {
	const Context ctx = build_context(cfg);

	print_configuration(cfg, ctx.ex_threads, ctx.n);
	auto layers = build_layers(ctx.n);
	auto keep = build_masks(layers, ctx.n, cfg.k);

	const std::size_t full_cmp = count_full_comparators(layers, ctx.n);
	const std::size_t trunc_cmp = count_truncated_comparators(layers, keep, ctx.n);

	std::vector<T> input = generate_random_input<T>(ctx.n, cfg.seed, randMin, randMax);
	apply_mode_transform(input, cfg.want_max);

	std::vector<T> truncated = input;
	const double trunc_ms = run_truncated_network(truncated, layers, keep, ctx.ex_threads);
	const Check check = run_check(cfg, input, truncated, layers, keep, ctx.ex_threads);

	print_timing_summary(cfg.run_check, check.full_ms, trunc_ms);
	print_debug_metrics(cfg, ctx.hw_threads, ctx.ex_threads, layers.size(), full_cmp, trunc_cmp,
						check.full_ms, trunc_ms);

	if (cfg.run_check) {
		print_correctness_summary(cfg.run_check, check.ok);
		if (!check.ok) {
			return 2;
		}
	}

	std::vector<T> output = build_output(truncated, cfg);
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
