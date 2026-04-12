#include "../include/runner.hpp"

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

#include "common/random.hpp"
#include "common/reporting.hpp"
#include "common/validation.hpp"

namespace {

using common::config::Config;
using common::config::DataType;

constexpr int kRandMin = 0;
constexpr int kRandMax = 1000;

template <typename T> std::vector<T> topk_ground_truth(std::vector<T> values, std::size_t k, bool want_max) {
	if (k >= values.size()) {
		if (want_max) {
			std::sort(values.begin(), values.end(), std::greater<T>());
		} else {
			std::sort(values.begin(), values.end());
		}
		return values;
	}

	if (want_max) {
		std::nth_element(values.begin(), values.begin() + static_cast<std::ptrdiff_t>(k), values.end(),
						 std::greater<T>());
		values.resize(k);
		std::sort(values.begin(), values.end(), std::greater<T>());
	} else {
		std::nth_element(values.begin(), values.begin() + static_cast<std::ptrdiff_t>(k), values.end());
		values.resize(k);
		std::sort(values.begin(), values.end());
	}

	return values;
}

template <typename T> int topk_typed(const Config& cfg) {
	const std::size_t n = std::size_t{1} << cfg.q;
	const std::size_t k = std::min(cfg.k, n);
	common::reporting::print_configuration(cfg, n, std::nullopt, "Run mode", "gt");

	std::vector<T> input = common::utils::generate_random_input<T>(n, cfg.seed, kRandMin, kRandMax);

	const auto t0 = std::chrono::high_resolution_clock::now();
	std::vector<T> output = topk_ground_truth(std::move(input), k, cfg.want_max);
	const auto t1 = std::chrono::high_resolution_clock::now();
	const double select_sort_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

	common::reporting::print_timing_lines({{"Ground truth select/sort time (ms)", select_sort_ms}});

	if (cfg.run_check && cfg.has_expected_output) {
		const bool expected_ok = common::utils::validate_expected_output(cfg, output);
		std::cout << "Top-k correctness vs testcase answer: " << (expected_ok ? "OK" : "FAIL") << "\n";
		if (!expected_ok) {
			return 3;
		}
	}

	std::vector<std::string> formatted = common::utils::format_output(output);
	common::reporting::print_output(cfg, formatted);

	return 0;
}

} // namespace

namespace gt::topk {

int execute(const common::config::Config& cfg) {
	switch (cfg.dtype) {
	case common::config::DataType::Int:
		return topk_typed<std::int32_t>(cfg);
	case common::config::DataType::UInt:
		return topk_typed<std::uint32_t>(cfg);
	case common::config::DataType::Float:
		return topk_typed<float>(cfg);
	case common::config::DataType::Double:
		return topk_typed<double>(cfg);
	case common::config::DataType::Fp16:
#if defined(__FLT16_MANT_DIG__)
		return topk_typed<_Float16>(cfg);
#else
		throw std::invalid_argument("dtype=fp16 is not supported by this compiler target");
#endif
	}

	throw std::invalid_argument("Unsupported dtype");
}

} // namespace gt::topk
