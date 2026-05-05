#include "../include/runner.hpp"

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

#include "common/benchmark.hpp"
#include "common/random.hpp"
#include "common/reporting.hpp"
#include "common/validation.hpp"

namespace {

using common::config::Config;
using common::config::DataType;

template <typename T> std::vector<T> topk_ground_truth(std::vector<T> values, std::size_t k, bool want_max) {
	if (k >= values.size()) {
		if (want_max) {
			std::sort(values.begin(), values.end(), std::greater<T>());
		} else {
			std::sort(values.begin(), values.end());
		}
		return values;
	}
	if (k == 0) {
		return {};
	}

	if (want_max) {
		std::partial_sort(values.begin(), values.begin() + static_cast<std::ptrdiff_t>(k), values.end(),
						  std::greater<T>());
		values.resize(k);
	} else {
		std::partial_sort(values.begin(), values.begin() + static_cast<std::ptrdiff_t>(k), values.end(),
						  std::less<T>());
		values.resize(k);
	}

	return values;
}

template <typename T> std::vector<T> sorted_topk_reference(std::vector<T> values, std::size_t k, bool want_max) {
	if (want_max) {
		std::sort(values.begin(), values.end(), std::greater<T>());
	} else {
		std::sort(values.begin(), values.end());
	}
	if (k < values.size()) {
		values.resize(k);
	}
	return values;
}

template <typename T> bool equal_typed(const std::vector<T>& lhs, const std::vector<T>& rhs) {
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

template <typename T> int topk_typed(const Config& cfg) {
	const std::size_t n = std::size_t{1} << cfg.q;
	const std::size_t k = std::min(cfg.k, n);
	common::reporting::print_configuration(cfg, n, std::nullopt, "Run mode", "gt");

	const std::vector<T> input = common::utils::generate_random_input<T>(n, cfg.seed, cfg.rand_min, cfg.rand_max);

	const int warmup_iters = std::max(1, common::benchmark::kWarmupIters);
	const int measure_iters = std::max(1, common::benchmark::kMeasureIters);

	common::benchmark::warmup(warmup_iters, [&]() {
		std::vector<T> temp = input;
		(void)topk_ground_truth(std::move(temp), k, cfg.want_max);
	});

	double total_ms = 0.0;
	std::vector<T> output;
	for (int i = 0; i < measure_iters; ++i) {
		std::vector<T> temp = input;
		const auto t0 = std::chrono::high_resolution_clock::now();
		std::vector<T> current = topk_ground_truth(std::move(temp), k, cfg.want_max);
		const auto t1 = std::chrono::high_resolution_clock::now();
		total_ms += std::chrono::duration<double, std::milli>(t1 - t0).count();
		if (i == measure_iters - 1) {
			output = std::move(current);
		}
	}

	const double avg_select_sort_ms = total_ms / static_cast<double>(measure_iters);
	common::reporting::print_timing_lines({{"Ground truth average partial-sort time (ms)", avg_select_sort_ms}});

	if (cfg.verify_output) {
		const std::vector<T> ref = sorted_topk_reference(input, k, cfg.want_max);
		const bool ok = equal_typed(output, ref);
		common::reporting::print_check_result("Top-k correctness vs CPU sorted reference", true, ok);
		if (!ok) {
			return 2;
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
